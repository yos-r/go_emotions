"""
Flask Web Application for GoEmotions Multi-Label Emotion Classification
Provides data visualization, model predictions, and comparative analysis
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Import custom model utilities
from model_utils import (
    Attention,
    weighted_loss_factory,
    load_all_models,
    predict_with_model,
    EMOTION_LABELS,
    NUM_LABELS,
    MAX_WORDS,
    MAX_LEN
)
from data_loader import load_dataset, get_dataset_statistics
from explainability_utils import EmotionExplainer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'goemotion-secret-key-2024'

# Global variables for models and tokenizer
models = {}
tokenizer = None
dataset_stats = None
explainers = {}  # LIME explainers for each model

def initialize_app():
    """Initialize models, tokenizer, and dataset statistics"""
    global models, tokenizer, dataset_stats, explainers

    print("Loading models...")
    models = load_all_models()

    print("Loading tokenizer...")
    # Load training data to fit tokenizer
    df_train, df_dev, df_test = load_dataset()
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(df_train['text'])

    print("Computing dataset statistics...")
    dataset_stats = get_dataset_statistics(df_train, df_dev, df_test)

    print("Initializing LIME explainers...")
    for model_name, model in models.items():
        if model_name != 'bert':  # Skip BERT for now (different tokenization)
            explainers[model_name] = EmotionExplainer(model, tokenizer, model_name)
            print(f"  ✓ Created explainer for {model_name}")

    print("App initialized successfully!")


@app.route('/')
def index():
    """Home page with navigation"""
    return render_template('index.html')


@app.route('/data-visualization')
def data_visualization():
    """Data visualization page showing dataset statistics"""
    if dataset_stats is None:
        return "Dataset not loaded", 500

    # Create visualizations
    plots = {}

    # 1. Emotion Distribution (Bar Chart)
    emotion_counts = dataset_stats['emotion_distribution']
    fig_distribution = go.Figure(data=[
        go.Bar(
            x=list(emotion_counts.keys()),
            y=list(emotion_counts.values()),
            marker_color='lightblue'
        )
    ])
    fig_distribution.update_layout(
        title='Emotion Distribution in Training Data',
        xaxis_title='Emotion',
        yaxis_title='Count',
        xaxis_tickangle=-45,
        height=500
    )
    plots['distribution'] = json.dumps(fig_distribution, cls=PlotlyJSONEncoder)

    # 2. Dataset Split (Pie Chart)
    fig_split = go.Figure(data=[
        go.Pie(
            labels=['Train', 'Dev', 'Test'],
            values=[
                dataset_stats['splits']['train'],
                dataset_stats['splits']['dev'],
                dataset_stats['splits']['test']
            ],
            hole=0.3
        )
    ])
    fig_split.update_layout(title='Dataset Split Distribution', height=400)
    plots['split'] = json.dumps(fig_split, cls=PlotlyJSONEncoder)

    # 3. Class Imbalance (Log Scale)
    sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
    fig_imbalance = go.Figure(data=[
        go.Bar(
            x=[e[0] for e in sorted_emotions],
            y=[e[1] for e in sorted_emotions],
            marker_color='coral'
        )
    ])
    fig_imbalance.update_layout(
        title='Class Imbalance (Sorted by Frequency)',
        xaxis_title='Emotion',
        yaxis_title='Count (Log Scale)',
        yaxis_type='log',
        xaxis_tickangle=-45,
        height=500
    )
    plots['imbalance'] = json.dumps(fig_imbalance, cls=PlotlyJSONEncoder)

    # 4. Multi-Label Statistics
    multi_label_stats = dataset_stats['multi_label_stats']
    fig_labels_per_sample = go.Figure(data=[
        go.Bar(
            x=list(multi_label_stats['labels_per_sample'].keys()),
            y=list(multi_label_stats['labels_per_sample'].values()),
            marker_color='lightgreen'
        )
    ])
    fig_labels_per_sample.update_layout(
        title='Number of Labels per Sample',
        xaxis_title='Number of Labels',
        yaxis_title='Count',
        height=400
    )
    plots['labels_per_sample'] = json.dumps(fig_labels_per_sample, cls=PlotlyJSONEncoder)

    return render_template(
        'data_visualization.html',
        stats=dataset_stats,
        plots=plots
    )


@app.route('/model-performance')
def model_performance():
    """Display model performance metrics and comparisons"""
    # Filter out LSTM embedding variants and hybrid variants - they'll be shown within their parent models
    main_models = [m for m in models.keys() if not (m.startswith('lstm_') or m.startswith('hybrid_'))]
    return render_template('model_performance.html', models=main_models)


@app.route('/api/evaluate-model/<model_name>', methods=['POST'])
def evaluate_model(model_name):
    """Evaluate a specific model on test data"""
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 404

    try:
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name.upper()} model on test data...")
        print(f"{'='*70}")

        # Special handling for BERT: load pre-calculated metrics from bert.txt
        if model_name == 'bert':
            print("Loading pre-calculated BERT metrics from bert.txt...")
            bert_results_path = 'bert.txt'

            if not os.path.exists(bert_results_path):
                return jsonify({'error': 'BERT results file (bert.txt) not found'}), 500

            # Parse bert.txt file
            with open(bert_results_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Extract overall metrics (lines 24-31)
            metrics = {
                'hamming_loss': float(lines[23].split()[2]),  # Line 24: Hamming Loss
                'precision_micro': float(lines[24].split()[2]),  # Line 25: Precision (Micro)
                'recall_micro': float(lines[25].split()[2]),  # Line 26: Recall (Micro)
                'f1_micro': float(lines[26].split()[2]),  # Line 27: F1-Score (Micro)
                'precision_macro': float(lines[27].split()[2]),  # Line 28: Precision (Macro)
                'recall_macro': float(lines[28].split()[2]),  # Line 29: Recall (Macro)
                'f1_macro': float(lines[29].split()[2]),  # Line 30: F1-Score (Macro)
                'auc_roc_macro': float(lines[30].split()[2]),  # Line 31: AUC-ROC (Macro)
                'threshold': 0.5
            }

            # Extract per-emotion metrics (lines 39-66)
            per_emotion_metrics = {}
            for i in range(39, 67):  # Lines 39-66 contain emotion metrics
                parts = lines[i-1].split()
                if len(parts) >= 6:
                    emotion = parts[0]
                    per_emotion_metrics[emotion] = {
                        'f1': float(parts[1]),
                        'precision': float(parts[2]),
                        'recall': float(parts[3])
                    }

            print(f"\nBERT evaluation loaded from file!")
            print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
            print(f"  AUC-ROC: {metrics['auc_roc_macro']:.4f}")
            print(f"  F1-Micro: {metrics['f1_micro']:.4f}")
            print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
            print(f"{'='*70}\n")

            return jsonify({
                'model': model_name,
                'metrics': metrics,
                'per_emotion': per_emotion_metrics
            })

        # For non-BERT models: run normal evaluation
        # Load test data
        _, _, df_test = load_dataset()
        print(f"Loaded test data: {len(df_test)} samples")

        # Preprocess
        from model_utils import preprocess_for_prediction
        is_bert = model_name == 'bert'
        print(f"Preprocessing data (for_bert={is_bert})...")
        X_test, Y_test = preprocess_for_prediction(df_test, tokenizer, for_bert=is_bert)
        print(f"Preprocessed X_test shape: {X_test.shape if hasattr(X_test, 'shape') else len(X_test)}")

        # Get predictions
        print(f"Running predictions with {model_name}...")
        model = models[model_name]
        Y_pred_proba = model.predict(X_test)
        print(f"Predictions complete! Shape: {Y_pred_proba.shape}")

        # Calculate metrics
        from sklearn.metrics import (
            hamming_loss, roc_auc_score, f1_score,
            precision_score, recall_score
        )

        # Determine threshold based on model
        # All LSTM variants use 0.3 threshold
        threshold = 0.3 if model_name.startswith('lstm') else 0.5
        Y_pred_binary = (Y_pred_proba > threshold).astype(int)

        metrics = {
            'hamming_loss': float(hamming_loss(Y_test, Y_pred_binary)),
            'auc_roc_macro': float(roc_auc_score(Y_test, Y_pred_proba, average='macro')),
            'f1_micro': float(f1_score(Y_test, Y_pred_binary, average='micro')),
            'f1_macro': float(f1_score(Y_test, Y_pred_binary, average='macro')),
            'precision_micro': float(precision_score(Y_test, Y_pred_binary, average='micro', zero_division=0)),
            'precision_macro': float(precision_score(Y_test, Y_pred_binary, average='macro', zero_division=0)),
            'recall_micro': float(recall_score(Y_test, Y_pred_binary, average='micro', zero_division=0)),
            'recall_macro': float(recall_score(Y_test, Y_pred_binary, average='macro', zero_division=0)),
            'threshold': threshold
        }

        # Per-emotion metrics
        per_emotion_metrics = {}
        for i, emotion in enumerate(EMOTION_LABELS):
            per_emotion_metrics[emotion] = {
                'precision': float(precision_score(Y_test[:, i], Y_pred_binary[:, i], zero_division=0)),
                'recall': float(recall_score(Y_test[:, i], Y_pred_binary[:, i], zero_division=0)),
                'f1': float(f1_score(Y_test[:, i], Y_pred_binary[:, i], zero_division=0))
            }

        print(f"\nEvaluation complete for {model_name}!")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc_macro']:.4f}")
        print(f"  F1-Micro: {metrics['f1_micro']:.4f}")
        print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        print(f"{'='*70}\n")

        return jsonify({
            'model': model_name,
            'metrics': metrics,
            'per_emotion': per_emotion_metrics
        })

    except Exception as e:
        import traceback
        print(f"\nError evaluating {model_name}:")
        print(traceback.format_exc())
        return jsonify({'error': f"{model_name} evaluation failed: {str(e)}"}), 500


@app.route('/predict')
def predict_page():
    """Interactive prediction page for user input"""
    # Filter out LSTM embedding variants and hybrid variants - they'll be shown within parent models
    main_models = [m for m in models.keys() if not (m.startswith('lstm_') or m.startswith('hybrid_'))]
    return render_template('predict.html', models=main_models)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict emotions for input text using all models"""
    data = request.get_json()
    text = data.get('text', '')
    specific_model = data.get('model', None)  # Optional: request specific model only

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        results = {}

        # If specific model requested, only predict for that model
        if specific_model:
            print(f"\n[Predict API] Predicting for specific model: {specific_model}")
            models_to_process = {specific_model: models.get(specific_model)}
            if models_to_process[specific_model] is None:
                return jsonify({'error': f'Model {specific_model} not found'}), 404
        else:
            # Include all models (including LSTM variants) for prediction
            models_to_process = models

        for model_name, model in models_to_process.items():
            # Preprocess text (BERT handles its own tokenization)
            if model_name == 'bert':
                # BERT uses its own tokenizer
                prediction = model.predict(text, verbose=0)[0]
            else:
                # Keras models use shared tokenizer
                sequence = tokenizer.texts_to_sequences([text])
                padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
                prediction = model.predict(padded, verbose=0)[0]

            # Determine threshold
            # All LSTM variants use 0.3 threshold
            threshold = 0.3 if model_name.startswith('lstm') else 0.5

            # Get top emotions
            emotion_scores = {
                EMOTION_LABELS[i]: float(prediction[i])
                for i in range(NUM_LABELS)
            }

            # Get predicted emotions (above threshold)
            predicted_emotions = [
                emotion for emotion, score in emotion_scores.items()
                if score > threshold
            ]

            # Get top 5 emotions by probability
            top_emotions = sorted(
                emotion_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            results[model_name] = {
                'all_scores': emotion_scores,
                'predicted_emotions': predicted_emotions,
                'top_emotions': dict(top_emotions),
                'threshold': threshold
            }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/explain', methods=['POST'])
def explain_prediction():
    """Generate LIME-based explanation for a prediction"""
    data = request.get_json()
    text = data.get('text', '')
    model_name = data.get('model', 'lstm')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    if model_name not in explainers:
        return jsonify({'error': f'Explainer not available for {model_name}'}), 400

    try:
        print(f"\n[Explain API] Generating LIME explanation for {model_name}...")

        explainer = explainers[model_name]

        # Generate LIME explanation (faster with fewer samples for API)
        explanation = explainer.explain_with_lime(
            text,
            num_features=8,  # Top 8 words
            num_samples=300  # Reduced for faster response
        )

        # Also get word importance scores
        word_importance = explainer.analyze_word_importance(text, explanation)

        # Get attention weights if available
        attention_data = explainer.get_attention_weights(text)

        result = {
            'model': model_name,
            'text': text,
            'lime_explanation': explanation,
            'word_importance': word_importance,
            'attention_weights': attention_data
        }

        print(f"  ✓ Explanation generated successfully")

        return jsonify(result)

    except Exception as e:
        import traceback
        print(f"[Error] Failed to generate explanation:")
        print(traceback.format_exc())
        return jsonify({'error': f'Explanation failed: {str(e)}'}), 500


@app.route('/api/get-hybrid-variants', methods=['GET'])
def get_hybrid_variants():
    """Get list of available hybrid model variants organized by group"""
    try:
        import json
        import os

        # Load ablation study config
        config_path = 'models/hybrid/ablation_study_config.json'
        if not os.path.exists(config_path):
            return jsonify({'error': 'Ablation study config not found'}), 404

        with open(config_path, 'r') as f:
            variant_descriptions = json.load(f)

        # Organize variants by group
        groups = {
            'attention': {
                'name': 'Attention Mechanism',
                'description': 'Ablate attention mechanism to measure its contribution',
                'variants': []
            },
            'components': {
                'name': 'CNN vs LSTM Components',
                'description': 'Test individual component importance',
                'variants': []
            },
            'embeddings': {
                'name': 'Embedding Types',
                'description': 'Test different pre-trained and random embeddings',
                'variants': []
            },
            'regularization': {
                'name': 'Regularization Techniques',
                'description': 'Test different regularization strategies',
                'variants': []
            },
            'reference': {
                'name': 'Reference Baseline',
                'description': 'Proven baseline for comparison',
                'variants': []
            }
        }

        # Categorize variants
        for variant_id, description in variant_descriptions.items():
            model_key = f'hybrid_{variant_id}'
            if model_key not in models:
                continue

            variant_info = {
                'id': variant_id,
                'model_key': model_key,
                'description': description,
                'number': int(variant_id.split('_')[0])
            }

            # Categorize by number
            num = variant_info['number']
            if 1 <= num <= 4:
                groups['attention']['variants'].append(variant_info)
            elif 5 <= num <= 7:
                groups['components']['variants'].append(variant_info)
            elif 8 <= num <= 12:
                groups['embeddings']['variants'].append(variant_info)
            elif 13 <= num <= 19:
                groups['regularization']['variants'].append(variant_info)
            elif num == 20:
                groups['reference']['variants'].append(variant_info)

        return jsonify(groups)

    except Exception as e:
        import traceback
        print(f"\nError getting hybrid variants:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare-hybrid-variants', methods=['POST'])
def compare_hybrid_variants():
    """Compare selected hybrid model variants"""
    data = request.get_json()
    variant_keys = data.get('variants', [])

    if not variant_keys:
        return jsonify({'error': 'No variants selected'}), 400

    try:
        print(f"\n{'='*70}")
        print(f"Comparing {len(variant_keys)} hybrid variants...")
        print(f"{'='*70}")

        # Load test data
        _, _, df_test = load_dataset()
        print(f"Loaded test data: {len(df_test)} samples")

        # Preprocess
        from model_utils import preprocess_for_prediction
        X_test, Y_test = preprocess_for_prediction(df_test, tokenizer, for_bert=False)

        results = {}

        for model_key in variant_keys:
            if model_key not in models:
                continue

            print(f"\n  Evaluating {model_key}...")
            model = models[model_key]

            Y_pred_proba = model.predict(X_test, verbose=0)
            print(f"    Predictions complete!")

            # Calculate metrics
            from sklearn.metrics import (
                hamming_loss, roc_auc_score, f1_score,
                precision_score, recall_score
            )

            threshold = 0.5
            Y_pred_binary = (Y_pred_proba > threshold).astype(int)

            metrics = {
                'hamming_loss': float(hamming_loss(Y_test, Y_pred_binary)),
                'auc_roc_macro': float(roc_auc_score(Y_test, Y_pred_proba, average='macro')),
                'f1_micro': float(f1_score(Y_test, Y_pred_binary, average='micro')),
                'f1_macro': float(f1_score(Y_test, Y_pred_binary, average='macro')),
                'precision_micro': float(precision_score(Y_test, Y_pred_binary, average='micro', zero_division=0)),
                'precision_macro': float(precision_score(Y_test, Y_pred_binary, average='macro', zero_division=0)),
                'recall_micro': float(recall_score(Y_test, Y_pred_binary, average='micro', zero_division=0)),
                'recall_macro': float(recall_score(Y_test, Y_pred_binary, average='macro', zero_division=0)),
                'threshold': threshold
            }

            # Per-emotion metrics
            per_emotion_metrics = {}
            for i, emotion in enumerate(EMOTION_LABELS):
                per_emotion_metrics[emotion] = {
                    'precision': float(precision_score(Y_test[:, i], Y_pred_binary[:, i], zero_division=0)),
                    'recall': float(recall_score(Y_test[:, i], Y_pred_binary[:, i], zero_division=0)),
                    'f1': float(f1_score(Y_test[:, i], Y_pred_binary[:, i], zero_division=0))
                }

            results[model_key] = {
                'model': model_key,
                'metrics': metrics,
                'per_emotion': per_emotion_metrics
            }

            print(f"    ✓ {model_key} - F1-Macro: {metrics['f1_macro']:.4f}")

        print(f"\n{'='*70}\n")
        return jsonify(results)

    except Exception as e:
        import traceback
        print(f"\nError comparing hybrid variants:")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare-lstm-embeddings', methods=['POST'])
def compare_lstm_embeddings():
    """Compare LSTM models with different embedding types"""
    try:
        print(f"\n{'='*70}")
        print(f"Comparing LSTM models with different embeddings...")
        print(f"{'='*70}")

        # Find all LSTM models
        lstm_models = {k: v for k, v in models.items() if k.startswith('lstm')}

        if not lstm_models:
            return jsonify({'error': 'No LSTM models found'}), 404

        # Load test data
        _, _, df_test = load_dataset()
        print(f"Loaded test data: {len(df_test)} samples")

        # Preprocess
        from model_utils import preprocess_for_prediction
        X_test, Y_test = preprocess_for_prediction(df_test, tokenizer, for_bert=False)
        print(f"Preprocessed X_test shape: {X_test.shape}")

        results = {}

        # Evaluate each LSTM variant
        for model_name, model in lstm_models.items():
            print(f"\n  Evaluating {model_name}...")

            Y_pred_proba = model.predict(X_test, verbose=0)
            print(f"    Predictions complete! Shape: {Y_pred_proba.shape}")

            # Calculate metrics
            from sklearn.metrics import (
                hamming_loss, roc_auc_score, f1_score,
                precision_score, recall_score
            )

            threshold = 0.3  # All LSTM models use 0.3
            Y_pred_binary = (Y_pred_proba > threshold).astype(int)

            metrics = {
                'hamming_loss': float(hamming_loss(Y_test, Y_pred_binary)),
                'auc_roc_macro': float(roc_auc_score(Y_test, Y_pred_proba, average='macro')),
                'f1_micro': float(f1_score(Y_test, Y_pred_binary, average='micro')),
                'f1_macro': float(f1_score(Y_test, Y_pred_binary, average='macro')),
                'precision_micro': float(precision_score(Y_test, Y_pred_binary, average='micro', zero_division=0)),
                'precision_macro': float(precision_score(Y_test, Y_pred_binary, average='macro', zero_division=0)),
                'recall_micro': float(recall_score(Y_test, Y_pred_binary, average='micro', zero_division=0)),
                'recall_macro': float(recall_score(Y_test, Y_pred_binary, average='macro', zero_division=0)),
                'threshold': threshold
            }

            # Per-emotion metrics
            per_emotion_metrics = {}
            for i, emotion in enumerate(EMOTION_LABELS):
                per_emotion_metrics[emotion] = {
                    'precision': float(precision_score(Y_test[:, i], Y_pred_binary[:, i], zero_division=0)),
                    'recall': float(recall_score(Y_test[:, i], Y_pred_binary[:, i], zero_division=0)),
                    'f1': float(f1_score(Y_test[:, i], Y_pred_binary[:, i], zero_division=0))
                }

            # Extract embedding type from model name
            embedding_type = model_name.replace('lstm_', '').replace('lstm', 'learned').upper()
            if embedding_type == 'LEARNED':
                embedding_type = 'Learned'
            elif embedding_type == 'TFIDF':
                embedding_type = 'TF-IDF'

            results[model_name] = {
                'model': model_name,
                'embedding_type': embedding_type,
                'metrics': metrics,
                'per_emotion': per_emotion_metrics
            }

            print(f"    ✓ {model_name} - F1-Macro: {metrics['f1_macro']:.4f}, AUC-ROC: {metrics['auc_roc_macro']:.4f}")

        print(f"\n{'='*70}")
        print(f"LSTM embedding comparison complete!")
        print(f"{'='*70}\n")

        return jsonify(results)

    except Exception as e:
        import traceback
        print(f"\nError comparing LSTM embeddings:")
        print(traceback.format_exc())
        return jsonify({'error': f"LSTM embedding comparison failed: {str(e)}"}), 500


if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
