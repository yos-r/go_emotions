"""
BERT Model Evaluation Script
Evaluates the trained BERT model on the test dataset
Calculates comprehensive metrics for multi-label classification
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (
    roc_auc_score, f1_score, hamming_loss,
    precision_score, recall_score
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
NUM_LABELS = len(EMOTION_LABELS)
COLUMN_NAMES = ['text', 'emotion_ids', 'comment_id']
    BERT_MAX_LEN = 128
    BATCH_SIZE = 32
    THRESHOLD = 0.5

# ============================================================================
# LOAD DATA (same as bert_pytorch.py)
# ============================================================================

print("="*70)
print(" LOADING TEST DATA")
print("="*70)

tsv_path = 'dataset/data/'
df_test = pd.read_csv(
    os.path.join(tsv_path, 'test.tsv'),
    sep='\t',
    header=None,
    names=COLUMN_NAMES,
    encoding='utf-8'
)

print(f"Test dataset: {len(df_test)} samples")

def create_binary_labels(df, labels_list):
    """Convert emotion_ids column to binary matrix (28 columns)"""
    label_matrix = pd.DataFrame(0, index=df.index, columns=labels_list)

    for index, row in df.iterrows():
        ids = str(row['emotion_ids']).split(',')
        try:
            int_ids = [int(i) for i in ids if i.isdigit()]
        except ValueError:
            int_ids = []

        for emotion_index in int_ids:
            if 0 <= emotion_index < NUM_LABELS:
                label_matrix.loc[index, labels_list[emotion_index]] = 1

    df_result = pd.concat([df[['text', 'comment_id']], label_matrix], axis=1)
    return df_result

# Transform labels
df_test_final = create_binary_labels(df_test, EMOTION_LABELS)
Y_test = df_test_final[EMOTION_LABELS].values.astype(np.float32)

print(f"Y_test shape: {Y_test.shape}")

# ============================================================================
# BERT MODEL DEFINITION (same as training)
# ============================================================================

class BertMultiLabelClassifier(nn.Module):
    """
    BERT model for multi-label classification

    Architecture:
        - Pre-trained BERT (bert-base-uncased)
        - Dropout (0.3)
        - Dense layer (28 units, sigmoid activation)
    """

    def __init__(self, num_labels, dropout=0.3):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token (first position)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Dropout + Classification
        dropout_output = self.dropout(cls_output)
        logits = self.classifier(dropout_output)

        return logits

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================

print("\n" + "="*70)
print(" LOADING TRAINED BERT MODEL")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load tokenizer
print("\nLoading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load model
print("Loading trained model...")
model_path = 'models/best_bert_model_pytorch.pt'

if not os.path.exists(model_path):
    print(f"ERROR: Model file not found: {model_path}")
    exit(1)

checkpoint = torch.load(model_path, map_location=device)
model = BertMultiLabelClassifier(num_labels=NUM_LABELS)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"Model loaded successfully!")
print(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
print(f"Best validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")

# ============================================================================
# RUN PREDICTIONS ON TEST SET
# ============================================================================

print("\n" + "="*70)
print(" RUNNING PREDICTIONS ON TEST SET")
print("="*70)

all_predictions = []
texts = df_test_final['text'].values

print(f"Processing {len(texts)} test samples in batches of {BATCH_SIZE}...")

num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

with torch.no_grad():
    for batch_idx in tqdm(range(num_batches), desc="Predicting"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(texts))
        batch_texts = texts[start_idx:end_idx].tolist()

        # Tokenize batch
        encoding = tokenizer.batch_encode_plus(
            batch_texts,
            add_special_tokens=True,
            max_length=BERT_MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        batch_predictions = probs.cpu().numpy()
        all_predictions.append(batch_predictions)

y_pred_proba = np.vstack(all_predictions)
y_pred_binary = (y_pred_proba >= THRESHOLD).astype(int)

print(f"\nPredictions complete!")
print(f"  Predictions shape: {y_pred_proba.shape}")
print(f"  Binary predictions shape: {y_pred_binary.shape}")

# ============================================================================
# CALCULATE OVERALL METRICS
# ============================================================================

print("\n" + "="*70)
print(" OVERALL METRICS ON TEST SET")
print("="*70)

# Calculate metrics
hamming = hamming_loss(Y_test, y_pred_binary)
precision_micro = precision_score(Y_test, y_pred_binary, average='micro', zero_division=0)
recall_micro = recall_score(Y_test, y_pred_binary, average='micro', zero_division=0)
f1_micro = f1_score(Y_test, y_pred_binary, average='micro', zero_division=0)

precision_macro = precision_score(Y_test, y_pred_binary, average='macro', zero_division=0)
recall_macro = recall_score(Y_test, y_pred_binary, average='macro', zero_division=0)
f1_macro = f1_score(Y_test, y_pred_binary, average='macro', zero_division=0)

# AUC-ROC (macro)
try:
    auc_roc = roc_auc_score(Y_test, y_pred_proba, average='macro')
except ValueError as e:
    print(f"Warning: Cannot calculate AUC-ROC: {e}")
    auc_roc = None

# Display results
print(f"\n{'Metric':<25} {'Score':>10}")
print("-"*70)
print(f"{'Hamming Loss':<25} {hamming:>10.4f}")
print(f"{'Precision (Micro)':<25} {precision_micro:>10.4f}")
print(f"{'Recall (Micro)':<25} {recall_micro:>10.4f}")
print(f"{'F1-Score (Micro)':<25} {f1_micro:>10.4f}")
print(f"{'Precision (Macro)':<25} {precision_macro:>10.4f}")
print(f"{'Recall (Macro)':<25} {recall_macro:>10.4f}")
print(f"{'F1-Score (Macro)':<25} {f1_macro:>10.4f}")
if auc_roc is not None:
    print(f"{'AUC-ROC (Macro)':<25} {auc_roc:>10.4f}")
print("="*70)

# ============================================================================
# PER-EMOTION METRICS
# ============================================================================

print("\n" + "="*70)
print(" PER-EMOTION METRICS")
print("="*70)

print(f"{'Emotion':<20} {'F1-Score':>12} {'Precision':>12} {'Recall':>12} {'AUC-ROC':>12} {'Support':>10}")
print("-"*70)

for i, emotion in enumerate(EMOTION_LABELS):
    f1 = f1_score(Y_test[:, i], y_pred_binary[:, i], zero_division=0)
    precision = precision_score(Y_test[:, i], y_pred_binary[:, i], zero_division=0)
    recall = recall_score(Y_test[:, i], y_pred_binary[:, i], zero_division=0)
    support = int(Y_test[:, i].sum())

    try:
        auc = roc_auc_score(Y_test[:, i], y_pred_proba[:, i])
    except ValueError:
        auc = 0.0

    print(f"{emotion:<20} {f1:>12.4f} {precision:>12.4f} {recall:>12.4f} {auc:>12.4f} {support:>10}")

print("="*70)

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*70)
print(" SUMMARY STATISTICS")
print("="*70)

# Count predictions
total_predictions = y_pred_binary.sum()
avg_labels_per_sample = total_predictions / len(y_pred_binary)
total_true_labels = Y_test.sum()
avg_true_labels_per_sample = total_true_labels / len(Y_test)

print(f"\n{'Statistic':<40} {'Value':>10}")
print("-"*70)
print(f"{'Test samples':<40} {len(Y_test):>10}")
print(f"{'Total emotions (28)':<40} {NUM_LABELS:>10}")
print(f"{'Threshold used':<40} {THRESHOLD:>10.2f}")
print(f"{'True labels (total)':<40} {int(total_true_labels):>10}")
print(f"{'Predicted labels (total)':<40} {int(total_predictions):>10}")
print(f"{'Avg true labels per sample':<40} {avg_true_labels_per_sample:>10.2f}")
print(f"{'Avg predicted labels per sample':<40} {avg_labels_per_sample:>10.2f}")
print("="*70)

print("\n EVALUATION COMPLETE!")
print("="*70)
