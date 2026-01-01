"""
Model utilities for loading and using trained emotion classification models
Includes custom layers and loss functions
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# PyTorch and BERT imports
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Constants
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
NUM_LABELS = len(EMOTION_LABELS)
MAX_WORDS = 50000
MAX_LEN = 70
BERT_MAX_LEN = 128


class Attention(Layer):
    """
    Custom Attention layer for Keras models
    Used in BiLSTM and CNN-BiLSTM architectures
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W (Weights): learns importance of each dimension
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], 1),
            initializer='random_normal',
            trainable=True
        )
        # b (Bias)
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[1], 1),
            initializer='zeros',
            trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        # Calculate scores (e = tanh(X * W + b))
        e = K.tanh(K.dot(x, self.W) + self.b)
        # Normalize scores (a = softmax(e)) -> attention weights
        a = K.softmax(e, axis=1)
        # Weighted context (output = X * a)
        output = x * a
        # Aggregation (sum of weighted sequence)
        return K.sum(output, axis=1)

    def get_config(self):
        return super().get_config()


def weighted_loss_factory(weights):
    """
    Creates a weighted Binary Cross-Entropy loss function
    Used for handling class imbalance in multi-label classification
    """
    weights_tensor = K.constant(weights, dtype='float32')

    def weighted_binary_crossentropy(y_true, y_pred):
        # Avoid log(0) by clipping predictions
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # Calculate standard binary cross-entropy
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        # Apply class weights
        weighted_bce = bce * weights_tensor
        # Average over all classes and samples
        return K.mean(weighted_bce)

    weighted_binary_crossentropy.__name__ = 'weighted_binary_crossentropy'
    return weighted_binary_crossentropy


def calculate_class_weights(Y_train):
    """Calculate class weights for imbalanced dataset"""
    class_counts = Y_train.sum(axis=0)
    total_samples = len(Y_train)
    class_frequencies = class_counts / total_samples

    # Calculate inverse frequencies, avoid division by zero
    weights = np.where(class_frequencies > 0, 1.0 / class_frequencies, 1.0)
    # Normalize weights
    weights = weights / weights.sum() * NUM_LABELS

    return weights


# ============================================================================
# BERT MODEL CLASS (PyTorch)
# ============================================================================

class BertMultiLabelClassifier(nn.Module):
    """
    BERT model for multi-label emotion classification
    Uses bert-base-uncased with dropout and linear classifier
    """
    def __init__(self, num_labels=28, dropout=0.3):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        dropout_output = self.dropout(cls_output)
        logits = self.classifier(dropout_output)
        return logits


class BertModelWrapper:
    """Wrapper for BERT PyTorch model to match Keras interface"""
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def predict(self, texts, verbose=0, batch_size=32):
        """
        Predict emotions for texts with batching for efficiency
        Args:
            texts: Single text string, list of text strings, or numpy array of strings
            verbose: Ignored (for Keras compatibility)
            batch_size: Number of texts to process at once
        Returns:
            numpy array of shape (n_samples, 28) with probabilities
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        # Handle numpy array
        elif isinstance(texts, np.ndarray):
            texts = texts.tolist()

        all_predictions = []

        # Process in batches for efficiency
        num_batches = (len(texts) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]

                # Tokenize batch
                encoding = self.tokenizer.batch_encode_plus(
                    batch_texts,
                    add_special_tokens=True,
                    max_length=BERT_MAX_LEN,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                # Forward pass
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)
                batch_predictions = probs.cpu().numpy()
                all_predictions.append(batch_predictions)

                # Print progress for large batches (always show for >100 samples)
                if len(texts) > 100 and (batch_idx + 1) % 10 == 0:
                    print(f"  BERT Progress: {end_idx}/{len(texts)} samples ({100*end_idx/len(texts):.1f}%)")

        return np.vstack(all_predictions)


def load_bert_model(model_path, device=None):
    """Load BERT PyTorch model"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create model
    model = BertMultiLabelClassifier(num_labels=NUM_LABELS)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Wrap in interface
    return BertModelWrapper(model, tokenizer, device)


def load_all_models():
    """
    Load all trained models with appropriate custom objects
    Returns dictionary of model_name -> model
    """
    models = {}
    models_dir = 'models'

    # Calculate weights for custom loss (using dummy data)
    # In production, load actual training data
    dummy_weights = np.ones(NUM_LABELS)  # Placeholder
    custom_loss = weighted_loss_factory(dummy_weights)

    # Custom objects for loading Keras models
    custom_objects = {
        'Attention': Attention,
        'weighted_binary_crossentropy': custom_loss
    }

    # Model 1: Simple LSTM (default - learned embeddings)
    lstm_path = os.path.join(models_dir, 'modele_lstm_simple.h5')
    if os.path.exists(lstm_path):
        try:
            models['lstm'] = load_model(lstm_path)
            print(f"[OK] Loaded LSTM model from {lstm_path}")
        except Exception as e:
            print(f"[ERROR] Error loading LSTM: {e}")

    # Model 1a: LSTM with FastText embeddings
    lstm_fasttext_path = os.path.join(models_dir, 'modele_lstm_simple_fasttext.h5')
    if os.path.exists(lstm_fasttext_path):
        try:
            models['lstm_fasttext'] = load_model(lstm_fasttext_path)
            print(f"[OK] Loaded LSTM (FastText) model from {lstm_fasttext_path}")
        except Exception as e:
            print(f"[ERROR] Error loading LSTM (FastText): {e}")

    # Model 1b: LSTM with GloVe embeddings
    lstm_glove_path = os.path.join(models_dir, 'modele_lstm_simple_glove.h5')
    if os.path.exists(lstm_glove_path):
        try:
            models['lstm_glove'] = load_model(lstm_glove_path)
            print(f"[OK] Loaded LSTM (GloVe) model from {lstm_glove_path}")
        except Exception as e:
            print(f"[ERROR] Error loading LSTM (GloVe): {e}")

    # Model 1c: LSTM with TF-IDF embeddings
    # lstm_tfidf_path = os.path.join(models_dir, 'modele_lstm_simple_tf-idf.h5')
    # if os.path.exists(lstm_tfidf_path):
    #     try:
    #         models['lstm_tfidf'] = load_model(lstm_tfidf_path)
    #         print(f"[OK] Loaded LSTM (TF-IDF) model from {lstm_tfidf_path}")
    #     except Exception as e:
    #         print(f"[ERROR] Error loading LSTM (TF-IDF): {e}")

    # Model 2: BiLSTM with Attention
    bilstm_path = os.path.join(models_dir, 'modele_bilstm_attention_final.keras')
    if os.path.exists(bilstm_path):
        try:
            models['bilstm_attention'] = load_model(bilstm_path, custom_objects=custom_objects)
            print(f"[OK] Loaded BiLSTM+Attention model from {bilstm_path}")
        except Exception as e:
            print(f"[ERROR] Error loading BiLSTM+Attention: {e}")

    # Model 3: CNN-BiLSTM with Attention (Main model)
    cnn_bilstm_path = os.path.join(models_dir, 'modele_cnn_bilstm_attention_final.keras')
    if os.path.exists(cnn_bilstm_path):
        try:
            models['cnn_bilstm_attention'] = load_model(cnn_bilstm_path, custom_objects=custom_objects)
            print(f"[OK] Loaded CNN-BiLSTM+Attention model from {cnn_bilstm_path}")
        except Exception as e:
            print(f"[ERROR] Error loading CNN-BiLSTM+Attention: {e}")

    # Model 3 Variants: Hybrid ablation study models
    hybrid_dir = os.path.join(models_dir, 'hybrid')
    if os.path.exists(hybrid_dir):
        # Load ablation study config
        config_path = os.path.join(hybrid_dir, 'ablation_study_config.json')
        hybrid_config = {}
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                hybrid_config = json.load(f)

        # Load all hybrid variants
        for variant_file in os.listdir(hybrid_dir):
            if variant_file.endswith('.h5'):
                variant_name = variant_file.replace('.h5', '')
                variant_path = os.path.join(hybrid_dir, variant_file)
                model_key = f'hybrid_{variant_name}'

                try:
                    models[model_key] = load_model(variant_path, custom_objects=custom_objects)
                    description = hybrid_config.get(variant_name, variant_name)
                    print(f"[OK] Loaded hybrid variant: {variant_name} - {description}")
                except Exception as e:
                    print(f"[ERROR] Error loading hybrid variant {variant_name}: {e}")

    # Model 4: BERT (PyTorch)
    bert_path = os.path.join(models_dir, 'best_bert_model_pytorch.pt')
    if os.path.exists(bert_path):
        try:
            models['bert'] = load_bert_model(bert_path)
            print(f"[OK] Loaded BERT model from {bert_path}")
        except Exception as e:
            print(f"[ERROR] Error loading BERT: {e}")
            print("   Note: BERT requires PyTorch and transformers library")

    if not models:
        raise Exception("No models found! Please train models first.")

    return models


def preprocess_for_prediction(df, tokenizer, for_bert=False):
    """
    Preprocess dataframe for prediction
    Returns X (padded sequences or raw texts for BERT) and Y (binary labels)

    Args:
        df: DataFrame with 'text' and 'emotion_ids' columns
        tokenizer: Keras Tokenizer (ignored if for_bert=True)
        for_bert: If True, returns raw texts for BERT's own tokenization
    """
    # Convert emotion_ids to binary labels
    label_matrix = np.zeros((len(df), NUM_LABELS))

    for idx, emotion_ids in enumerate(df['emotion_ids']):
        ids = str(emotion_ids).split(',')
        try:
            int_ids = [int(i) for i in ids if i.isdigit()]
            for emotion_idx in int_ids:
                if 0 <= emotion_idx < NUM_LABELS:
                    label_matrix[idx, emotion_idx] = 1
        except ValueError:
            pass

    Y = label_matrix

    if for_bert:
        # For BERT, return raw texts (BERT will do its own tokenization)
        return df['text'].values, Y
    else:
        # For Keras models, tokenize and pad
        sequences = tokenizer.texts_to_sequences(df['text'])
        X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
        return X, Y


def predict_with_model(model, text, tokenizer, threshold=0.5):
    """
    Predict emotions for a single text using a model
    Returns dictionary of emotion probabilities
    """
    # Preprocess
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    # Predict
    prediction = model.predict(padded, verbose=0)[0]

    # Create result dictionary
    result = {
        'probabilities': {EMOTION_LABELS[i]: float(prediction[i]) for i in range(NUM_LABELS)},
        'predicted': [EMOTION_LABELS[i] for i in range(NUM_LABELS) if prediction[i] > threshold],
        'threshold': threshold
    }

    return result


def get_model_summary(model):
    """Get model architecture summary as string"""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return '\n'.join(string_list)
