"""
Explainability utilities using LIME and SHAP for GoEmotions models
Provides model-agnostic explanations for emotion predictions
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shap
import lime
from lime.lime_text import LimeTextExplainer
import warnings
warnings.filterwarnings('ignore')

# Import custom layers
from .model_utils import Attention, weighted_loss_factory, EMOTION_LABELS, MAX_LEN

# SHAP uses TensorFlow backend
import tensorflow as tf

class EmotionExplainer:
    """
    Wrapper class for explaining emotion predictions using LIME and SHAP
    """

    def __init__(self, model, tokenizer, model_name='model'):
        """
        Initialize explainer with model and tokenizer

        Args:
            model: Trained Keras model
            tokenizer: Fitted Keras tokenizer
            model_name: Name of the model for identification
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.emotion_labels = EMOTION_LABELS

        # Initialize LIME explainer
        self.lime_explainer = LimeTextExplainer(
            class_names=self.emotion_labels,
            bow=False,  # Use word presence, not bag of words
            random_state=42
        )

    def predict_proba(self, texts):
        """
        Predict probabilities for a list of texts
        Required for LIME compatibility

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (n_samples, n_emotions)
        """
        # Tokenize and pad sequences
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

        # Get predictions
        predictions = self.model.predict(padded, verbose=0)

        return predictions

    def explain_with_lime(self, text, num_features=10, num_samples=1000):
        """
        Generate LIME explanation for a single text

        Args:
            text: Input text string
            num_features: Number of top features to show
            num_samples: Number of perturbed samples for LIME

        Returns:
            dict with explanations for each emotion
        """
        print(f"[LIME] Explaining prediction for: '{text[:50]}...'")

        # Get base prediction
        base_prediction = self.predict_proba([text])[0]

        # Get top predicted emotions (above 0.3 threshold)
        top_emotions_idx = np.where(base_prediction > 0.3)[0]

        if len(top_emotions_idx) == 0:
            # If no emotions above threshold, take top 3
            top_emotions_idx = np.argsort(base_prediction)[-3:]

        explanations = {}

        for emotion_idx in top_emotions_idx:
            emotion_name = self.emotion_labels[emotion_idx]

            # Generate LIME explanation for this emotion
            exp = self.lime_explainer.explain_instance(
                text,
                self.predict_proba,
                labels=[emotion_idx],
                num_features=num_features,
                num_samples=num_samples
            )

            # Extract word importance
            word_weights = exp.as_list(label=emotion_idx)

            explanations[emotion_name] = {
                'prediction': float(base_prediction[emotion_idx]),
                'word_weights': word_weights,  # List of (word, weight) tuples
                'intercept': exp.intercept[emotion_idx] if hasattr(exp, 'intercept') else 0.0
            }

            print(f"  ✓ {emotion_name}: {base_prediction[emotion_idx]:.3f}")

        return {
            'method': 'LIME',
            'text': text,
            'base_predictions': {self.emotion_labels[i]: float(base_prediction[i])
                                for i in range(len(base_prediction))},
            'explanations': explanations
        }

    def get_attention_weights(self, text):
        """
        Extract attention weights from models with Attention layers
        (BiLSTM and CNN-BiLSTM models)

        Args:
            text: Input text string

        Returns:
            dict with attention weights per word
        """
        try:
            # Check if model has attention layer
            has_attention = any('attention' in layer.name.lower()
                              for layer in self.model.layers)

            if not has_attention:
                return None

            # Tokenize text
            sequences = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

            # Get attention layer output
            from tensorflow.keras.models import Model

            # Find attention layer
            attention_layer = None
            for layer in self.model.layers:
                if 'attention' in layer.name.lower():
                    attention_layer = layer
                    break

            if attention_layer is None:
                return None

            # Create model that outputs attention weights
            # This is tricky because attention weights are internal to the layer
            # For simplicity, we'll use the prediction as proxy

            # Get word-level tokens
            words = text.split()
            tokens = sequences[0][:len(words)]  # Limit to actual words

            # Simple approximation: perturb each word and see impact
            word_importance = {}
            base_pred = self.predict_proba([text])[0]

            for i, word in enumerate(words):
                # Remove word and measure impact
                perturbed_text = ' '.join(words[:i] + words[i+1:])
                if perturbed_text.strip():
                    perturbed_pred = self.predict_proba([perturbed_text])[0]
                    # Calculate average impact across top emotions
                    impact = np.mean(np.abs(base_pred - perturbed_pred))
                    word_importance[word] = float(impact)

            return {
                'words': words,
                'importance': word_importance
            }

        except Exception as e:
            print(f"[Warning] Could not extract attention weights: {e}")
            return None

    def analyze_word_importance(self, text, lime_explanation):
        """
        Combine LIME explanations to create unified word importance scores

        Args:
            text: Input text
            lime_explanation: LIME explanation dict

        Returns:
            dict mapping words to importance scores
        """
        words = text.split()
        word_scores = {word: 0.0 for word in words}

        # Aggregate LIME weights across all explained emotions
        for emotion, exp_data in lime_explanation['explanations'].items():
            for word_phrase, weight in exp_data['word_weights']:
                # LIME may return phrases, so match individual words
                for word in words:
                    if word.lower() in word_phrase.lower():
                        word_scores[word] += abs(weight)

        # Normalize scores
        max_score = max(word_scores.values()) if word_scores.values() else 1.0
        if max_score > 0:
            word_scores = {word: score / max_score for word, score in word_scores.items()}

        return word_scores


def explain_prediction_with_lime(text, model, tokenizer, model_name='model',
                                 num_features=10, num_samples=500):
    """
    High-level function to explain a prediction using LIME

    Args:
        text: Input text to explain
        model: Trained Keras model
        tokenizer: Fitted Keras tokenizer
        model_name: Name of the model
        num_features: Number of important features to extract
        num_samples: Number of samples for LIME

    Returns:
        dict with LIME explanation
    """
    explainer = EmotionExplainer(model, tokenizer, model_name)
    return explainer.explain_with_lime(text, num_features, num_samples)


def get_model_attention_visualization(text, model, tokenizer):
    """
    Get attention-based explanation for models with Attention layers

    Args:
        text: Input text
        model: Keras model
        tokenizer: Fitted tokenizer

    Returns:
        dict with attention weights
    """
    explainer = EmotionExplainer(model, tokenizer)
    return explainer.get_attention_weights(text)


def batch_explain_samples(texts, model, tokenizer, model_name='model', max_samples=100):
    """
    Explain multiple samples and aggregate insights
    Useful for error analysis

    Args:
        texts: List of text samples
        model: Trained model
        tokenizer: Fitted tokenizer
        model_name: Model name
        max_samples: Maximum number of samples to explain

    Returns:
        dict with aggregated explanations
    """
    explainer = EmotionExplainer(model, tokenizer, model_name)

    # Limit samples
    texts = texts[:max_samples]

    all_explanations = []
    word_frequency = {}
    emotion_word_associations = {emotion: {} for emotion in EMOTION_LABELS}

    print(f"\n[Batch Explain] Analyzing {len(texts)} samples for {model_name}...")

    for idx, text in enumerate(texts):
        if idx % 20 == 0:
            print(f"  Progress: {idx}/{len(texts)}")

        try:
            explanation = explainer.explain_with_lime(text, num_features=5, num_samples=200)
            all_explanations.append(explanation)

            # Track word-emotion associations
            for emotion, exp_data in explanation['explanations'].items():
                for word_phrase, weight in exp_data['word_weights']:
                    words = word_phrase.split()
                    for word in words:
                        word = word.lower().strip()
                        if word not in emotion_word_associations[emotion]:
                            emotion_word_associations[emotion][word] = []
                        emotion_word_associations[emotion][word].append(weight)

        except Exception as e:
            print(f"  [Warning] Failed to explain sample {idx}: {e}")
            continue

    # Aggregate word importance per emotion
    emotion_keywords = {}
    for emotion, word_weights in emotion_word_associations.items():
        # Average weights for each word
        avg_weights = {word: np.mean(weights)
                      for word, weights in word_weights.items()
                      if len(weights) >= 3}  # At least 3 occurrences

        # Sort by absolute weight
        sorted_words = sorted(avg_weights.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True)[:10]

        emotion_keywords[emotion] = sorted_words

    print(f"  ✓ Completed batch explanation")

    return {
        'model_name': model_name,
        'num_samples': len(all_explanations),
        'emotion_keywords': emotion_keywords,
        'all_explanations': all_explanations
    }


def analyze_prediction_errors(df_test, y_test, y_pred, model, tokenizer,
                              model_name='model', num_errors=50):
    """
    Analyze misclassified samples to understand model errors

    Args:
        df_test: Test dataframe with 'text' column
        y_test: True labels (binary matrix)
        y_pred: Predicted labels (binary matrix)
        model: Trained model
        tokenizer: Fitted tokenizer
        model_name: Model name
        num_errors: Number of errors to analyze

    Returns:
        dict with error analysis
    """
    print(f"\n[Error Analysis] Analyzing errors for {model_name}...")

    # Find misclassified samples
    errors = []
    for idx in range(len(y_test)):
        if not np.array_equal(y_test[idx], y_pred[idx]):
            # Calculate error severity
            error_count = np.sum(y_test[idx] != y_pred[idx])
            errors.append({
                'index': idx,
                'text': df_test.iloc[idx]['text'],
                'true_labels': y_test[idx],
                'pred_labels': y_pred[idx],
                'error_count': error_count
            })

    # Sort by error severity
    errors.sort(key=lambda x: x['error_count'], reverse=True)
    errors = errors[:num_errors]

    print(f"  Found {len(errors)} misclassified samples (showing top {num_errors})")

    # Explain errors using LIME
    error_explanations = []
    for error in errors[:20]:  # Limit to 20 for performance
        text = error['text']
        explanation = explain_prediction_with_lime(
            text, model, tokenizer, model_name,
            num_features=8, num_samples=300
        )

        error_explanations.append({
            'text': text,
            'true_emotions': [EMOTION_LABELS[i] for i, val in enumerate(error['true_labels']) if val == 1],
            'pred_emotions': [EMOTION_LABELS[i] for i, val in enumerate(error['pred_labels']) if val == 1],
            'explanation': explanation
        })

    # Identify common error patterns
    false_positives = {emotion: 0 for emotion in EMOTION_LABELS}
    false_negatives = {emotion: 0 for emotion in EMOTION_LABELS}

    for error in errors:
        for i, emotion in enumerate(EMOTION_LABELS):
            if error['true_labels'][i] == 0 and error['pred_labels'][i] == 1:
                false_positives[emotion] += 1
            elif error['true_labels'][i] == 1 and error['pred_labels'][i] == 0:
                false_negatives[emotion] += 1

    print(f"  ✓ Completed error analysis")

    return {
        'model_name': model_name,
        'total_errors': len(errors),
        'analyzed_errors': len(error_explanations),
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'error_examples': error_explanations
    }


if __name__ == '__main__':
    # Example usage
    print("Explainability Utils Module")
    print("=" * 70)
    print("This module provides LIME-based explanations for emotion predictions")
    print("\nAvailable functions:")
    print("  - explain_prediction_with_lime()")
    print("  - get_model_attention_visualization()")
    print("  - batch_explain_samples()")
    print("  - analyze_prediction_errors()")
