"""
Script to run LIME-based error analysis on the GoEmotions test set
Generates comprehensive explainability reports for model predictions
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

from model_utils import load_all_models, EMOTION_LABELS, MAX_WORDS, MAX_LEN
from data_loader import load_dataset
from explainability_utils import (
    EmotionExplainer,
    batch_explain_samples,
    analyze_prediction_errors
)


def create_binary_labels(df, labels_list):
    """Convert emotion labels to binary matrix"""
    Y = np.zeros((len(df), len(labels_list)))
    for idx, row in df.iterrows():
        if isinstance(row['emotions'], str):
            emotion_ids = [int(e) for e in row['emotions'].split(',') if e.strip()]
        else:
            emotion_ids = row['emotions']
        for eid in emotion_ids:
            if eid < len(labels_list):
                Y[idx, eid] = 1
    return Y


def run_error_analysis(model_name='lstm', num_errors=30):
    """
    Run comprehensive error analysis for a specific model

    Args:
        model_name: Name of model to analyze
        num_errors: Number of errors to analyze in detail
    """
    print("="*80)
    print(f"ERROR ANALYSIS FOR {model_name.upper()} MODEL")
    print("="*80)

    # Load data
    print("\n[1/5] Loading dataset...")
    df_train, df_dev, df_test = load_dataset()

    # Prepare tokenizer
    print("\n[2/5] Preparing tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(df_train['text'])

    # Load model
    print("\n[3/5] Loading model...")
    models = load_all_models()
    if model_name not in models:
        print(f"Error: Model {model_name} not found!")
        return

    model = models[model_name]

    # Prepare test data
    print("\n[4/5] Preparing test data...")
    X_test_seq = tokenizer.texts_to_sequences(df_test['text'])
    X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    Y_test = create_binary_labels(df_test, EMOTION_LABELS)

    # Get predictions
    print(f"\n[5/5] Generating predictions for {len(df_test)} test samples...")
    Y_pred_proba = model.predict(X_test, verbose=1)

    # Apply threshold
    threshold = 0.3 if model_name == 'lstm' else 0.5
    Y_pred = (Y_pred_proba > threshold).astype(int)

    # Calculate basic metrics
    from sklearn.metrics import hamming_loss, f1_score

    hamming = hamming_loss(Y_test, Y_pred)
    f1_micro = f1_score(Y_test, Y_pred, average='micro')
    f1_macro = f1_score(Y_test, Y_pred, average='macro')

    print(f"\nModel Performance:")
    print(f"  Hamming Loss: {hamming:.4f}")
    print(f"  F1-Micro: {f1_micro:.4f}")
    print(f"  F1-Macro: {f1_macro:.4f}")

    # Run LIME-based error analysis
    print(f"\n[Error Analysis] Analyzing misclassifications with LIME...")
    error_analysis = analyze_prediction_errors(
        df_test, Y_test, Y_pred, model, tokenizer,
        model_name, num_errors=num_errors
    )

    # Save results
    output_dir = 'explainability'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{model_name}_error_analysis.json')

    # Prepare JSON-serializable output
    output_data = {
        'model_name': model_name,
        'threshold': threshold,
        'metrics': {
            'hamming_loss': float(hamming),
            'f1_micro': float(f1_micro),
            'f1_macro': float(f1_macro)
        },
        'error_analysis': {
            'total_errors': error_analysis['total_errors'],
            'analyzed_errors': error_analysis['analyzed_errors'],
            'false_positives': error_analysis['false_positives'],
            'false_negatives': error_analysis['false_negatives'],
            'error_examples': [
                {
                    'text': ex['text'],
                    'true_emotions': ex['true_emotions'],
                    'pred_emotions': ex['pred_emotions'],
                    'lime_explanations': {
                        emotion: {
                            'prediction': data['prediction'],
                            'top_words': data['word_weights'][:5]  # Top 5 words
                        }
                        for emotion, data in ex['explanation']['explanations'].items()
                    }
                }
                for ex in error_analysis['error_examples'][:10]  # Save top 10
            ]
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Error analysis saved to: {output_file}")

    # Print summary
    print("\n" + "="*80)
    print("ERROR PATTERN SUMMARY")
    print("="*80)

    print("\nMost Common False Positives:")
    fp_sorted = sorted(error_analysis['false_positives'].items(),
                      key=lambda x: x[1], reverse=True)[:5]
    for emotion, count in fp_sorted:
        print(f"  {emotion}: {count} times")

    print("\nMost Common False Negatives:")
    fn_sorted = sorted(error_analysis['false_negatives'].items(),
                      key=lambda x: x[1], reverse=True)[:5]
    for emotion, count in fn_sorted:
        print(f"  {emotion}: {count} times")

    print("\n" + "="*80)
    print("EXAMPLE ERRORS (with LIME explanations)")
    print("="*80)

    for idx, example in enumerate(error_analysis['error_examples'][:3], 1):
        print(f"\nExample {idx}:")
        print(f"  Text: {example['text'][:100]}...")
        print(f"  True: {example['true_emotions']}")
        print(f"  Predicted: {example['pred_emotions']}")

        print("  LIME Key Words:")
        for emotion, exp_data in example['explanation']['explanations'].items():
            top_words = [f"{word} ({weight:.3f})"
                        for word, weight in exp_data['word_weights'][:3]]
            print(f"    {emotion}: {', '.join(top_words)}")

    print("\n" + "="*80)

    return error_analysis


def run_batch_explanation(model_name='lstm', num_samples=50):
    """
    Run batch LIME explanations on random test samples
    Helps identify common word-emotion patterns

    Args:
        model_name: Name of model
        num_samples: Number of samples to explain
    """
    print("="*80)
    print(f"BATCH LIME EXPLANATION FOR {model_name.upper()}")
    print("="*80)

    # Load data
    print("\n[1/3] Loading dataset...")
    df_train, df_dev, df_test = load_dataset()

    # Prepare tokenizer
    print("\n[2/3] Preparing tokenizer...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(df_train['text'])

    # Load model
    print("\n[3/3] Loading model...")
    models = load_all_models()
    model = models[model_name]

    # Sample random texts
    sample_texts = df_test['text'].sample(n=num_samples, random_state=42).tolist()

    # Run batch explanation
    results = batch_explain_samples(sample_texts, model, tokenizer, model_name, max_samples=num_samples)

    # Save results
    output_dir = 'explainability'
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{model_name}_batch_explanations.json')

    # Prepare output (without full explanations to save space)
    output_data = {
        'model_name': results['model_name'],
        'num_samples': results['num_samples'],
        'emotion_keywords': results['emotion_keywords']
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Batch explanations saved to: {output_file}")

    # Print emotion keywords
    print("\n" + "="*80)
    print("DISCOVERED EMOTION KEYWORDS (LIME-based)")
    print("="*80)

    for emotion, keywords in results['emotion_keywords'].items():
        if keywords:  # Only show emotions with keywords
            print(f"\n{emotion.upper()}:")
            for word, weight in keywords:
                print(f"  {word}: {weight:.3f}")

    return results


if __name__ == '__main__':
    import sys

    print("\n" + "="*80)
    print("GoEmotions LIME-based Explainability Analysis")
    print("="*80)

    # Determine which models to analyze
    if len(sys.argv) > 1:
        model_names = sys.argv[1].split(',')
    else:
        model_names = ['lstm', 'bilstm', 'cnn-bilstm']

    print(f"\nAnalyzing models: {', '.join(model_names)}")
    print("\nNote: This will take several minutes per model...")

    choice = input("\nRun [1] Error Analysis, [2] Batch Explanations, or [3] Both? (default=1): ").strip() or '1'

    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Processing {model_name.upper()}")
        print(f"{'='*80}")

        try:
            if choice in ['1', '3']:
                run_error_analysis(model_name, num_errors=30)

            if choice in ['2', '3']:
                run_batch_explanation(model_name, num_samples=50)

        except Exception as e:
            print(f"\nError processing {model_name}: {e}")
            import traceback
            traceback.print_exc()

        print("\n")

    print("="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nResults saved in 'explainability/' directory")
    print("Files generated:")
    print("  - {model}_error_analysis.json")
    print("  - {model}_batch_explanations.json")
