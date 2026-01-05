import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.utils import class_weight
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss, precision_score, recall_score
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
NUM_LABELS = len(EMOTION_LABELS)

# Column names for TSV files (No header, Tab separator)
COLUMN_NAMES = ['text', 'emotion_ids', 'comment_id']
tsv_path = '/kaggle/input/goemotions/dataset/data/'
df_train = pd.read_csv(os.path.join(tsv_path, 'train.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')
df_dev = pd.read_csv(os.path.join(tsv_path, 'dev.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')
df_test = pd.read_csv(os.path.join(tsv_path, 'test.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')

print(f"Training dataset: {len(df_train)} rows")
print(f"Validation dataset: {len(df_dev)} rows")
print(f"Test dataset: {len(df_test)} rows")

print("\nPreview of emotion_ids column before transformation:")
print(df_train['emotion_ids'].head())


def create_binary_labels(df, labels_list):
    """
    Converts 'emotion_ids' column (comma-separated ID strings)
    into 28 binary columns (0 or 1) for multi-label classification.
    """
    # Initialize empty binary DataFrame with size (num_rows x 28)
    label_matrix = pd.DataFrame(0, index=df.index, columns=labels_list)

    # Iterate through each row
    for index, row in df.iterrows():
        # Split emotion IDs (strings like '2,5' -> ['2', '5'])
        ids = str(row['emotion_ids']).split(',')

        # Convert IDs to integers
        try:
            int_ids = [int(i) for i in ids if i.isdigit()]
        except ValueError:
            int_ids = []

        # Set corresponding columns to 1
        for emotion_index in int_ids:
            # Ensure index is valid (between 0 and 27)
            if 0 <= emotion_index < NUM_LABELS:
                label_matrix.loc[index, labels_list[emotion_index]] = 1

    # Concatenate binary DataFrame with original
    df_result = pd.concat([df[['text', 'comment_id']], label_matrix], axis=1)

    return df_result


# Apply transformation to all three DataFrames
df_train_final = create_binary_labels(df_train, EMOTION_LABELS)
df_dev_final = create_binary_labels(df_dev, EMOTION_LABELS)
df_test_final = create_binary_labels(df_test, EMOTION_LABELS)

print("\nPreview of final training DataFrame:")
print(df_train_final.head())

# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

MAX_WORDS = 50000    # Maximum vocabulary size (top 50000 most frequent words)
MAX_LEN = 70         # Maximum sequence length (comment length)

# Instantiate and fit Tokenizer ONLY on training data
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
tokenizer.fit_on_texts(df_train_final['text'])

# Transform texts to sequences of integers (word indices)
train_sequences = tokenizer.texts_to_sequences(df_train_final['text'])
dev_sequences = tokenizer.texts_to_sequences(df_dev_final['text'])
test_sequences = tokenizer.texts_to_sequences(df_test_final['text'])

# Padding (standardize sequence lengths)
X_train = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
X_dev = pad_sequences(dev_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"X_train shape (after padding): {X_train.shape}")

Y_train = df_train_final[EMOTION_LABELS].values
Y_dev = df_dev_final[EMOTION_LABELS].values
Y_test = df_test_final[EMOTION_LABELS].values

print(f"Y_train shape: {Y_train.shape}")  # Should be (num_rows, 28)


# ============================================================================
# CLASS WEIGHTS FOR IMBALANCED DATA
# ============================================================================

class_counts = Y_train.sum(axis=0)  # Count per class (28,)
total_samples = len(Y_train)
class_frequencies = class_counts / total_samples

# Calculate inverse frequencies, avoid division by zero
WEIGHTS_NUMPY = np.where(class_frequencies > 0,
                          1.0 / class_frequencies,
                          1.0)
# Normalize weights to maintain loss scale
WEIGHTS_NUMPY = WEIGHTS_NUMPY / WEIGHTS_NUMPY.sum() * NUM_LABELS

print("\n" + "="*70)
print("CLASS WEIGHTS FOR WEIGHTED LOSS")
print("="*70)
print(f"{'Emotion':<20} {'Frequency':>12} {'Weight':>12}")
print("-"*70)
for i in range(NUM_LABELS):
    print(f"{EMOTION_LABELS[i]:<20} {class_frequencies[i]:>12.4f} {WEIGHTS_NUMPY[i]:>12.4f}")
print("="*70)


# ============================================================================
# CUSTOM ATTENTION LAYER
# ============================================================================

class Attention(Layer):
    """
    Attention layer for Keras/TensorFlow.
    Condenses BiLSTM sequential output into a single weighted vector
    based on word importance.
    """
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W (Weights): learns importance of each dimension
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        # b (Bias)
        self.b = self.add_weight(name='attention_bias',
                                 shape=(input_shape[1], 1),
                                 initializer='zeros',
                                 trainable=True)
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


# ============================================================================
# WEIGHTED LOSS FUNCTION FOR MULTI-LABEL CLASSIFICATION
# ============================================================================

def weighted_loss_factory(weights):
    """
    Creates a weighted Binary Cross-Entropy loss function for multi-label classification.
    
    Addresses class imbalance by applying different weights to each emotion.
    Rare emotions (e.g., 'grief') receive higher weights, frequent emotions 
    (e.g., 'neutral') receive lower weights.
    
    Args:
        weights: numpy array of shape (NUM_LABELS,) containing class weights
        
    Returns:
        Loss function compatible with Keras/TensorFlow
    """
    weights_tensor = K.constant(weights, dtype='float32')
    
    def weighted_binary_crossentropy(y_true, y_pred):
        """
        Weighted Binary Cross-Entropy for multi-label classification.
        
        Formula: Loss = -Σ[w_i * (y_i * log(p_i) + (1-y_i) * log(1-p_i))]
        where:
            - y_i: true label (0 or 1)
            - p_i: prediction (probability between 0 and 1)
            - w_i: weight of class i
            
        Args:
            y_true: True labels (batch_size, NUM_LABELS)
            y_pred: Model predictions (batch_size, NUM_LABELS)
            
        Returns:
            Weighted average loss (scalar)
        """
        # Avoid log(0) by clipping predictions
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Calculate standard binary cross-entropy
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        
        # Apply class weights (broadcasting over batch)
        weighted_bce = bce * weights_tensor
        
        # Average over all classes and samples
        return K.mean(weighted_bce)
    
    weighted_binary_crossentropy.__name__ = 'weighted_binary_crossentropy'
    
    return weighted_binary_crossentropy


# Create custom loss function with calculated weights
custom_loss_function = weighted_loss_factory(WEIGHTS_NUMPY)

print(f"\n✅ Weighted loss function created successfully!")
print(f"   Penalizes errors on rare classes more heavily.")
print(f"   Example: 'grief' (weight: {WEIGHTS_NUMPY[EMOTION_LABELS.index('grief')]:.2f}) vs 'neutral' (weight: {WEIGHTS_NUMPY[EMOTION_LABELS.index('neutral')]:.2f})\n")


# ============================================================================
# BUILD BILSTM ATTENTION MODEL
# ============================================================================

def build_bilstm_attention_model(vocab_size, max_len, embedding_dim=128, 
                                  lstm_units=128, dropout_rate=0.3):
    """
    Builds BiLSTM + Attention model for multi-label emotion classification.
    
    Args:
        vocab_size: Size of vocabulary
        max_len: Maximum sequence length
        embedding_dim: Embedding dimension
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, 
                 output_dim=embedding_dim, 
                 input_length=max_len,
                 name='embedding'),
        Dropout(dropout_rate, name='dropout_embedding'),
        Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm'),
        Dropout(dropout_rate, name='dropout_bilstm'),
        Attention(name='attention'),
        Dense(64, activation='relu', name='dense_1'),
        Dropout(dropout_rate, name='dropout_dense'),
        Dense(NUM_LABELS, activation='sigmoid', name='output')
    ])
    
    return model


# Build model
model_bilstm_att = build_bilstm_attention_model(
    vocab_size=MAX_WORDS,
    max_len=MAX_LEN,
    embedding_dim=128,
    lstm_units=128,
    dropout_rate=0.3
)

print("\n" + "="*70)
print("MODEL ARCHITECTURE")
print("="*70)
model_bilstm_att.summary()

# Compile model with custom loss and appropriate metrics
model_bilstm_att.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=custom_loss_function,
    metrics=['binary_accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("\n✅ Model compiled successfully with weighted loss!")


# ============================================================================
# TRAIN MODEL
# ============================================================================

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

history = model_bilstm_att.fit(
    X_train, Y_train,
    validation_data=(X_dev, Y_dev),
    epochs=15,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

print("\n✅ Training completed!")


# ============================================================================
# SAVE MODEL
# ============================================================================

chemin_sauvegarde_final = '/kaggle/working/modele_bilstm_attention_final.keras'

# Save with custom_objects for Attention layer and loss function
model_bilstm_att.save(
    chemin_sauvegarde_final,
    save_format='keras'
)

print(f"✅ Model saved to: {chemin_sauvegarde_final}")


# ============================================================================
# EVALUATION ON TEST SET
# ============================================================================

# --- 1. GENERATE PREDICTIONS ---

# Probability predictions (Sigmoid output)
Y_pred_proba = model_bilstm_att.predict(X_test)

# THRESHOLD should be defined - using 0.5 as default
THRESHOLD = 0.5
Y_pred_binary = (Y_pred_proba > THRESHOLD).astype(int)

print(f"\n✅ Predictions generated. Test data size: {Y_pred_proba.shape[0]}")


# --- 2. CALCULATE MULTI-LABEL METRICS ---

# Metric 1: Hamming Loss (H-Loss) - Average error per label
h_loss = hamming_loss(Y_test, Y_pred_binary)

# Metric 2: AUC-ROC (Macro) - Based on probabilities
auc_roc = roc_auc_score(Y_test, Y_pred_proba, average='macro')

# Metric 3: Micro-Averaged (Prioritizes frequent classes)
precision_micro = precision_score(Y_test, Y_pred_binary, average='micro', zero_division=0)
recall_micro = recall_score(Y_test, Y_pred_binary, average='micro', zero_division=0)
f1_micro = f1_score(Y_test, Y_pred_binary, average='micro', zero_division=0)

# Metric 4: Macro-Averaged (Prioritizes rare classes)
precision_macro = precision_score(Y_test, Y_pred_binary, average='macro', zero_division=0)
recall_macro = recall_score(Y_test, Y_pred_binary, average='macro', zero_division=0)
f1_macro = f1_score(Y_test, Y_pred_binary, average='macro', zero_division=0)


# --- 3. DISPLAY RESULTS ---

print("\n" + "="*70)
print("  MODEL RESULTS (BiLSTM + Attention)")
print("="*70)
print(f"Hamming Loss (H-Loss)  : {h_loss:.4f} (Closer to 0 is better)")
print(f"AUC-ROC (Macro)        : {auc_roc:.4f} (Closer to 1 is better)")
print("-" * 70)
print("   **Micro-Averaged (Global/Frequent Classes)**")
print(f"   Precision (Micro)    : {precision_micro:.4f}")
print(f"   Recall (Micro)       : {recall_micro:.4f}")
print(f"   F1-score (Micro)     : {f1_micro:.4f}")
print("-" * 70)
print("   **Macro-Averaged (Rare Classes)**")
print(f"   Precision (Macro)    : {precision_macro:.4f}")
print(f"   Recall (Macro)       : {recall_macro:.4f}")
print(f"   F1-score (Macro)     : {f1_macro:.4f}")
print("="*70)

# ==========================================================================
# MODÈLE 3: ARCHITECTURE HYBRIDE CNN-BiLSTM + ATTENTION
# ==========================================================================

from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import Input

print("="*70)
print("CONSTRUCTION DU MODÈLE 3: CNN-BiLSTM + ATTENTION")
print("="*70)

# --- 1. CONSTRUCTION DU MODÈLE HYBRIDE (API FONCTIONNELLE) ---

# Input layer
input_layer = Input(shape=(MAX_LEN,), name='input')

# Couche Embedding partagée
embedding_layer = Embedding(
    input_dim=MAX_WORDS,
    output_dim=128,
    name='embedding'
)(input_layer)
embedding_dropout = Dropout(0.3, name='dropout_embedding')(embedding_layer)

# --- BRANCHE CNN: Extraction de features locales avec plusieurs tailles de filtres ---
# Utilisation de 3 tailles de filtres différentes pour capturer différents n-grams

# Filtres de taille 3 (trigrams)
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', name='conv_3')(embedding_dropout)
pool1 = MaxPooling1D(pool_size=2, name='pool_3')(conv1)

# Filtres de taille 4 (4-grams)
conv2 = Conv1D(filters=64, kernel_size=4, activation='relu', padding='same', name='conv_4')(embedding_dropout)
pool2 = MaxPooling1D(pool_size=2, name='pool_4')(conv2)

# Filtres de taille 5 (5-grams)
conv3 = Conv1D(filters=64, kernel_size=5, activation='relu', padding='same', name='conv_5')(embedding_dropout)
pool3 = MaxPooling1D(pool_size=2, name='pool_5')(conv3)

# Concaténer les 3 branches CNN
cnn_concat = Concatenate(axis=-1, name='cnn_concat')([pool1, pool2, pool3])
cnn_dropout = Dropout(0.3, name='dropout_cnn')(cnn_concat)

# --- BRANCHE BiLSTM: Capture des dépendances séquentielles ---
bilstm_layer = Bidirectional(
    LSTM(units=64, 
         return_sequences=True,  # Nécessaire pour l'attention
         dropout=0.2,
         recurrent_dropout=0.2),
    name='bilstm'
)(cnn_dropout)

# --- MÉCANISME D'ATTENTION: Pondération des features importantes ---
attention_layer = Attention(name='attention')(bilstm_layer)

# --- COUCHES DENSES FINALES ---
dense_hidden = Dense(128, activation='relu', name='dense_hidden')(attention_layer)
dense_dropout = Dropout(0.5, name='dropout_hidden')(dense_hidden)

# Couche de sortie (classification multi-label)
output_layer = Dense(28, activation='sigmoid', name='output')(dense_dropout)

# Créer le modèle
model_cnn_bilstm_att = Model(inputs=input_layer, outputs=output_layer, name='CNN_BiLSTM_Attention')

print(model_cnn_bilstm_att.summary())


# --- 2. COMPILATION AVEC WEIGHTED LOSS ---
model_cnn_bilstm_att.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=custom_loss_function,  # Weighted Binary Cross-Entropy
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc', multi_label=True),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print("\nModèle compilé avec Weighted Binary Cross-Entropy")


# --- 3. CALLBACKS POUR L'ENTRAÎNEMENT ---
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks_cnn = [
    EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='models/modele_cnn_bilstm_attention_best.keras',
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]


# --- 4. ENTRAÎNEMENT DU MODÈLE ---
print("\n" + "="*70)
print("DÉBUT DE L'ENTRAÎNEMENT DU CNN-BiLSTM + ATTENTION")
print("="*70)

history_cnn_bilstm_att = model_cnn_bilstm_att.fit(
    X_train,
    Y_train,
    epochs=15,
    batch_size=64,
    validation_data=(X_dev, Y_dev),
        class_weight=class_weights_dict,

    callbacks=callbacks_cnn,
    verbose=1
)

print("\n" + "="*70)
print("ENTRAÎNEMENT TERMINÉ")
print("="*70)


# --- 5. SAUVEGARDE DU MODÈLE ---
chemin_sauvegarde_cnn = 'models/modele_cnn_bilstm_attention_final.keras'
model_cnn_bilstm_att.save(chemin_sauvegarde_cnn, save_format='keras')

print(f"\nModèle CNN-BiLSTM+Attention sauvegardé dans: {chemin_sauvegarde_cnn}")
print(f"Meilleur modèle sauvegardé dans: models/modele_cnn_bilstm_attention_best.keras")


# --- 6. ÉVALUATION SUR LE JEU DE TEST ---
print("\n" + "="*70)
print("ÉVALUATION SUR LE JEU DE TEST")
print("="*70)

# Prédictions
Y_pred_proba_cnn = model_cnn_bilstm_att.predict(X_test)
Y_pred_binary_cnn = (Y_pred_proba_cnn > 0.5).astype(int)

# Calcul des métriques
h_loss_cnn = hamming_loss(Y_test, Y_pred_binary_cnn)
auc_roc_cnn = roc_auc_score(Y_test, Y_pred_proba_cnn, average='macro')
precision_micro_cnn = precision_score(Y_test, Y_pred_binary_cnn, average='micro', zero_division=0)
recall_micro_cnn = recall_score(Y_test, Y_pred_binary_cnn, average='micro', zero_division=0)
f1_micro_cnn = f1_score(Y_test, Y_pred_binary_cnn, average='micro', zero_division=0)
precision_macro_cnn = precision_score(Y_test, Y_pred_binary_cnn, average='macro', zero_division=0)
recall_macro_cnn = recall_score(Y_test, Y_pred_binary_cnn, average='macro', zero_division=0)
f1_macro_cnn = f1_score(Y_test, Y_pred_binary_cnn, average='macro', zero_division=0)

# Affichage des résultats
print("\nRÉSULTATS DU MODÈLE 3 (CNN-BiLSTM + Attention)")
print("="*70)
print(f"Hamming Loss (H-Loss) : {h_loss_cnn:.4f} (Proche de 0 : Mieux)")
print(f"AUC-ROC (Macro)       : {auc_roc_cnn:.4f} (Proche de 1 : Mieux)")
print("-" * 35)
print("   Micro-Averaged (Global/Fréquent)")
print(f"   Precision (Micro)   : {precision_micro_cnn:.4f}")
print(f"   Recall (Micro)      : {recall_micro_cnn:.4f}")
print(f"   F1-score (Micro)    : {f1_micro_cnn:.4f}")
print("-" * 35)
print("   Macro-Averaged (Classes Rares)")
print(f"   Precision (Macro)   : {precision_macro_cnn:.4f}")
print(f"   Recall (Macro)      : {recall_macro_cnn:.4f}")
print(f"   F1-score (Macro)    : {f1_macro_cnn:.4f}")
print("="*70)