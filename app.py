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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'goemotion-secret-key-2024'

# Global variables for models and tokenizer
models = {}
tokenizer = None
dataset_stats = None

def initialize_app():
    """Initialize models, tokenizer, and dataset statistics"""
    global models, tokenizer, dataset_stats

    print("Loading models...")
    models = load_all_models()

    print("Loading tokenizer...")
    # Load training data to fit tokenizer
    df_train, df_dev, df_test = load_dataset()
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(df_train['text'])

    print("Computing dataset statistics...")
    dataset_stats = get_dataset_statistics(df_train, df_dev, df_test)

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
    return render_template('model_performance.html', models=list(models.keys()))


@app.route('/api/evaluate-model/<model_name>', methods=['POST'])
def evaluate_model(model_name):
    """Evaluate a specific model on test data"""
    if model_name not in models:
        return jsonify({'error': 'Model not found'}), 404

    try:
        print(f"\n{'='*70}")
        print(f"Evaluating {model_name.upper()} model on test data...")
        print(f"{'='*70}")

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
        threshold = 0.3 if model_name == 'lstm' else 0.5
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
        print(f"\n❌ Error evaluating {model_name}:")
        print(traceback.format_exc())
        return jsonify({'error': f"{model_name} evaluation failed: {str(e)}"}), 500


@app.route('/predict')
def predict_page():
    """Interactive prediction page for user input"""
    return render_template('predict.html', models=list(models.keys()))


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict emotions for input text using all models"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        results = {}

        for model_name, model in models.items():
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
            threshold = 0.3 if model_name == 'lstm' else 0.5

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


@app.route('/compare')
def compare_page():
    """Model comparison page"""
    return render_template('compare.html', models=list(models.keys()))


@app.route('/api/compare-models', methods=['POST'])
def compare_models():
    """Compare all models on sample texts"""
    try:
        # Load a sample of test data
        _, _, df_test = load_dataset()
        sample_size = min(100, len(df_test))
        df_sample = df_test.sample(n=sample_size, random_state=42)

        # Preprocess
        from model_utils import preprocess_for_prediction

        comparison_results = {}

        for model_name, model in models.items():
            # Preprocess for each model
            is_bert = model_name == 'bert'
            X_sample, Y_sample = preprocess_for_prediction(df_sample, tokenizer, for_bert=is_bert)

            # Predict
            Y_pred = model.predict(X_sample, verbose=0)
            threshold = 0.3 if model_name == 'lstm' else 0.5
            Y_pred_binary = (Y_pred > threshold).astype(int)

            # Calculate metrics
            from sklearn.metrics import hamming_loss, f1_score, roc_auc_score

            comparison_results[model_name] = {
                'hamming_loss': float(hamming_loss(Y_sample, Y_pred_binary)),
                'f1_micro': float(f1_score(Y_sample, Y_pred_binary, average='micro')),
                'f1_macro': float(f1_score(Y_sample, Y_pred_binary, average='macro')),
                'auc_roc': float(roc_auc_score(Y_sample, Y_pred, average='macro'))
            }

        return jsonify(comparison_results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
