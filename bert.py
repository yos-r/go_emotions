import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.utils import class_weight
import pandas as pd
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Layer

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
NUM_LABELS = len(EMOTION_LABELS)

# 3. Noms des colonnes pour les TSV (Pas de header, sparateur Tab)
COLUMN_NAMES = ['text', 'emotion_ids', 'comment_id']
tsv_path = 'dataset/data/'
df_train = pd.read_csv(os.path.join(tsv_path, 'train.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')
df_dev = pd.read_csv(os.path.join(tsv_path, 'dev.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')
df_test = pd.read_csv(os.path.join(tsv_path, 'test.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')

print(f"Jeu de donnes d'entranement : {len(df_train)} lignes")
print(f"Jeu de donnes de validation : {len(df_dev)} lignes")
print(f"Jeu de donnes de test : {len(df_test)} lignes")

# Vrification (doit contenir des chanes d'IDs comme '2' ou '1,17')
print("\nAperu de la colonne des IDs avant transformation :")
print(df_train['emotion_ids'].head())

def create_binary_labels(df, labels_list):
    """
    Convertit la colonne 'emotion_ids' (chane d'IDs spares par des virgules)
    en 28 colonnes binaires (0 ou 1) pour la classification multi-label.
    """

    # Initialise un DataFrame binaire vide de taille (Nb_lignes x 28)
    label_matrix = pd.DataFrame(0, index=df.index, columns=labels_list)

    # Parcourt chaque ligne du DataFrame
    for index, row in df.iterrows():
        # Spare les IDs d'motions (sont des chanes, ex: '2,5' -> ['2', '5'])
        ids = str(row['emotion_ids']).split(',')

        # Convertit les IDs en entiers pour les utiliser comme index
        # L'index 0 correspond  la premire motion de votre liste, etc.
        try:
            # S'assure de ne traiter que des IDs valides (chiffres)
            int_ids = [int(i) for i in ids if i.isdigit()]
        except ValueError:
            # Cas d'erreur (rare) ou ID non numrique
            int_ids = []

        # Met  1 les colonnes correspondantes dans la matrice
        for emotion_index in int_ids:
            # S'assurer que l'index est valide (entre 0 et 27)
            if 0 <= emotion_index < NUM_LABELS:
                label_matrix.loc[index, labels_list[emotion_index]] = 1

    # Concatne le DataFrame binaire avec le DataFrame original
    df_result = pd.concat([df[['text', 'comment_id']], label_matrix], axis=1)

    return df_result

# Appliquer la transformation aux trois DataFrames
df_train_final = create_binary_labels(df_train, EMOTION_LABELS)
df_dev_final = create_binary_labels(df_dev, EMOTION_LABELS)
df_test_final = create_binary_labels(df_test, EMOTION_LABELS)

print("\nAperu du DataFrame d'entranement FINAL :")
print(df_train_final.head())

# 1. Dfinir les hyperparamtres ( ajuster)

MAX_WORDS = 50000    # Taille maximale du vocabulaire (les 20000 mots les plus frquents)
MAX_LEN = 70         # Longueur maximale d'une squence (un commentaire)

# 2. Instancier et adapter le Tokenizer UNIQUEMENT sur les donnes d'ENTRANEMENT
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
tokenizer.fit_on_texts(df_train_final['text'])

# 3. Transformer les textes en squences d'entiers (indices de mots)
train_sequences = tokenizer.texts_to_sequences(df_train_final['text'])
dev_sequences = tokenizer.texts_to_sequences(df_dev_final['text'])
test_sequences = tokenizer.texts_to_sequences(df_test_final['text'])

# 4. Padding (uniformisation de la longueur des squences)
# Remplissage  MAX_LEN (ajout de zros au dbut ou  la fin)
X_train = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
X_dev = pad_sequences(dev_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"La taille du X_train (aprs padding) est : {X_train.shape}")
Y_train = df_train_final[EMOTION_LABELS].values
Y_dev = df_dev_final[EMOTION_LABELS].values
Y_test = df_test_final[EMOTION_LABELS].values

print(f"La taille du Y_train est : {Y_train.shape}") # Devrait tre (Nb_lignes, 28)
class_weights_dict = {}
for i in range(NUM_LABELS):
    y_col = Y_train[:, i]
    try:
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_col),
            y=y_col
        )
        class_weights_dict[i] = weights[1]
    except ValueError:
        class_weights_dict[i] = 1.0
    

# ============================================================================
# WEIGHTED LOSS FUNCTION FOR MULTI-LABEL CLASSIFICATION
# ============================================================================

# --- 1. CALCUL DES POIDS DE CLASSE (Class Weights) ---
# Pour grer le dsquilibre svre des classes (neutral: 14219 vs grief: 77)
# Calculer les poids inversement proportionnels  la frquence de chaque classe

class_counts = Y_train.sum(axis=0)  # Nombre d'occurrences par classe (28,)
total_samples = len(Y_train)
class_frequencies = class_counts / total_samples

# viter division par zro et calculer les poids inverss
WEIGHTS_NUMPY = np.where(class_frequencies > 0,
                          1.0 / class_frequencies,
                          1.0)
# Normalisation des poids pour maintenir l'chelle de la loss
WEIGHTS_NUMPY = WEIGHTS_NUMPY / WEIGHTS_NUMPY.sum() * NUM_LABELS

print("="*70)
print("POIDS DE CLASSE POUR WEIGHTED LOSS")
print("="*70)
print(f"{'motion':<20} {'Frquence':>12} {'Poids':>12}")
print("-"*70)
for i in range(NUM_LABELS):
    print(f"{EMOTION_LABELS[i]:<20} {class_frequencies[i]:>12.4f} {WEIGHTS_NUMPY[i]:>12.4f}")
print("="*70)


# --- 2. WEIGHTED LOSS FUNCTION FACTORY ---
def weighted_loss_factory(weights):
    """
    Cre une fonction de perte Binary Cross-Entropy pondre pour multi-label.
    
    Cette fonction rsout le problme de dsquilibre des classes en appliquant
    des poids diffrents  chaque motion. Les motions rares (comme 'grief')
    reoivent un poids plus lev, tandis que les motions frquentes (comme 'neutral')
    reoivent un poids plus faible.
    
    Args:
        weights: numpy array de shape (NUM_LABELS,) contenant les poids de chaque classe
        
    Returns:
        Fonction de perte compatible avec Keras/TensorFlow
    """
    weights_tensor = K.constant(weights, dtype='float32')
    
    def weighted_binary_crossentropy(y_true, y_pred):
        """
        Binary Cross-Entropy pondre pour classification multi-label.
        
        Formule: Loss = -[w_i * (y_i * log(p_i) + (1-y_i) * log(1-p_i))]
        o:
            - y_i: label rel (0 ou 1)
            - p_i: prdiction (probabilit entre 0 et 1)
            - w_i: poids de la classe i
            
        Args:
            y_true: Labels rels (batch_size, NUM_LABELS)
            y_pred: Prdictions du modle (batch_size, NUM_LABELS)
            
        Returns:
            Perte moyenne pondre (scalaire)
        """
        # viter log(0) en clippant les prdictions entre epsilon et 1-epsilon
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Calcul de la binary cross-entropy standard
        # BCE = -(y_true * log(y_pred) + (1-y_true) * log(1-y_pred))
        bce = -(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
        
        # Application des poids de classe (broadcasting sur le batch)
        weighted_bce = bce * weights_tensor
        
        # Moyenne sur toutes les classes et tous les chantillons du batch
        return K.mean(weighted_bce)
    
    # Dfinir le nom de la fonction pour le chargement du modle
    weighted_binary_crossentropy.__name__ = 'weighted_binary_crossentropy'
    
    return weighted_binary_crossentropy


# Crer la fonction de perte personnalise avec les poids calculs
custom_loss_function = weighted_loss_factory(WEIGHTS_NUMPY)

print(f"\n[OK] Weighted loss function creee avec succes!")
print(f"   Cette fonction penalise davantage les erreurs sur les classes rares.")
print(f"   Exemple: 'grief' (poids: {WEIGHTS_NUMPY[EMOTION_LABELS.index('grief')]:.2f}) vs 'neutral' (poids: {WEIGHTS_NUMPY[EMOTION_LABELS.index('neutral')]:.2f})\n")
# ========================================================================== #
# MODLE 4: BERT-BASE TRANSFORMER (Fine-tuning) - FIX SUBCLASSING API
# ========================================================================== #
import numpy as np
import pandas as pd
import transformers
from transformers import BertTokenizer, TFBertModel, BertConfig
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

# Dsactiver les avertissements TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("="*70)
print("MODLE 4: BERT-BASE POUR CLASSIFICATION MULTI-LABEL")
print("="*70)
print(f"Version transformers: {transformers.__version__}")
print(f"Version TensorFlow: {tf.__version__}")

# ============================================================================
# TAPE 1: TOKENIZATION BERT (diffrente de la tokenization Keras)
# ============================================================================

# Charger le tokenizer BERT pr-entran
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Hyperparamtres BERT
BERT_MAX_LEN = 128  # Longueur maximale pour BERT (gnralement 128 ou 512)
BATCH_SIZE = 16     # Batch size plus petit pour BERT (gourmand en mmoire)
EPOCHS = 5          # BERT converge rapidement (3-5 epochs suffisent souvent)
LEARNING_RATE = 2e-5  # Learning rate recommand pour fine-tuning BERT

print(f"\n Tokenization BERT avec longueur maximale: {BERT_MAX_LEN}")

def bert_encode_batch(texts, tokenizer, max_len):
    """
    Encode un batch de textes avec le tokenizer BERT.

    Args:
        texts: Liste de chanes de caractres
        tokenizer: BertTokenizer
        max_len: Longueur maximale de la squence

    Returns:
        dict contenant input_ids, attention_mask, token_type_ids
    """
    encoding = tokenizer.batch_encode_plus(
        texts.tolist(),
        add_special_tokens=True,      # Ajoute [CLS] et [SEP]
        max_length=max_len,
        padding='max_length',          # Padding  max_len
        truncation=True,               # Tronque si trop long
        return_attention_mask=True,    # Retourne le masque d'attention
        return_token_type_ids=True,    # Retourne les token type IDs
        return_tensors='np'            # Retourne des arrays numpy
    )
    return encoding

# Encoder les trois ensembles de donnes
print("Encodage du jeu d'entranement...")
train_encoding = bert_encode_batch(df_train_final['text'], bert_tokenizer, BERT_MAX_LEN)

print("Encodage du jeu de validation...")
dev_encoding = bert_encode_batch(df_dev_final['text'], bert_tokenizer, BERT_MAX_LEN)

print("Encodage du jeu de test...")
test_encoding = bert_encode_batch(df_test_final['text'], bert_tokenizer, BERT_MAX_LEN)

print(f" Tokenization termine!")
print(f"   Train input_ids shape: {train_encoding['input_ids'].shape}")
print(f"   Train attention_mask shape: {train_encoding['attention_mask'].shape}")

# ============================================================================
# TAPE 2: CONSTRUCTION DU MODLE BERT
# ============================================================================

def build_bert_model(num_labels, learning_rate=2e-5):
    """
    Construit un modle BERT pour classification multi-label.

    Architecture:
        - BERT pr-entran (bert-base-uncased)
        - Dropout layer (0.3)
        - Dense layer (28 units, activation sigmoid)

    Args:
        num_labels: Nombre de labels (28 pour GoEmotions)
        learning_rate: Taux d'apprentissage

    Returns:
        Modle Keras compil
    """
    # Dfinir les inputs (3 entres pour BERT)
    input_ids = Input(shape=(BERT_MAX_LEN,), dtype=tf.int32, name='input_ids')
    attention_mask = Input(shape=(BERT_MAX_LEN,), dtype=tf.int32, name='attention_mask')
    token_type_ids = Input(shape=(BERT_MAX_LEN,), dtype=tf.int32, name='token_type_ids')

    # Charger BERT pr-entran
    print("   Chargement du modle BERT...")

    try:
        # Mthode 1: Essayer de charger les poids TF natifs
        bert_model = TFBertModel.from_pretrained('bert-base-uncased', from_pt=False)
        print("    Poids TensorFlow natifs chargs")
    except Exception as e1:
        print(f"    Impossible de charger les poids TF natifs")
        try:
            # Mthode 2: Crer le modle depuis la config et initialiser avec poids alatoires
            print("   Cration du modle BERT avec config...")
            config = BertConfig.from_pretrained('bert-base-uncased')
            bert_model = TFBertModel(config)
            # Construire le modle avec des inputs dummy pour initialiser les poids
            dummy_input_ids = tf.constant([[0]], dtype=tf.int32)
            dummy_attention_mask = tf.constant([[1]], dtype=tf.int32)
            dummy_token_type_ids = tf.constant([[0]], dtype=tf.int32)
            _ = bert_model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask, token_type_ids=dummy_token_type_ids, training=False)
            print("    Modle BERT cr avec poids ALATOIRES (non pr-entran)")
            print("    Les performances seront rduites. Pour de meilleurs rsultats:")
            print("    1) Installez: pip install transformers==4.30.0 tensorflow==2.13.0")
            print("    2) Ou utilisez PyTorch au lieu de TensorFlow")
        except Exception as e2:
            print(f"    Erreur: {e2}")
            raise RuntimeError("Impossible de crer le modle BERT. Vrifiez l'installation de transformers.")

    # Passer les inputs  BERT
    bert_output = bert_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )

    # Utiliser le [CLS] token (premire position) pour la classification
    cls_token = bert_output.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)

    # Ajouter un Dropout pour rgularisation
    dropout = Dropout(0.3)(cls_token)

    # Couche de sortie: 28 neurones avec activation sigmoid (multi-label)
    output = Dense(num_labels, activation='sigmoid', name='output')(dropout)

    # Crer le modle
    model = Model(
        inputs=[input_ids, attention_mask, token_type_ids],
        outputs=output
    )

    # Compiler le modle
    # Utiliser la weighted loss personnalise ou binary_crossentropy standard
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=custom_loss_function,  # Ou 'binary_crossentropy'
        metrics=['accuracy']
    )

    return model

print("\n Construction du modle BERT...")
bert_model = build_bert_model(NUM_LABELS, LEARNING_RATE)

print("\n Rsum du modle BERT:")
bert_model.summary()

# ============================================================================
# TAPE 3: ENTRANEMENT DU MODLE
# ============================================================================

print("\n Dbut de l'entranement du modle BERT...")

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        'best_bert_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Entranement
history_bert = bert_model.fit(
    x={
        'input_ids': train_encoding['input_ids'],
        'attention_mask': train_encoding['attention_mask'],
        'token_type_ids': train_encoding['token_type_ids']
    },
    y=Y_train,
    validation_data=(
        {
            'input_ids': dev_encoding['input_ids'],
            'attention_mask': dev_encoding['attention_mask'],
            'token_type_ids': dev_encoding['token_type_ids']
        },
        Y_dev
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\n Entranement termin!")

# ============================================================================
# TAPE 4: VALUATION SUR L'ENSEMBLE DE TEST
# ============================================================================

print("\n VALUATION DU MODLE BERT SUR L'ENSEMBLE DE TEST")
print("="*70)

# Prdictions sur l'ensemble de test
y_pred_proba = bert_model.predict({
    'input_ids': test_encoding['input_ids'],
    'attention_mask': test_encoding['attention_mask'],
    'token_type_ids': test_encoding['token_type_ids']
})

# Seuil de dcision (0.5 par dfaut, peut tre ajust)
THRESHOLD = 0.5
y_pred_binary = (y_pred_proba >= THRESHOLD).astype(int)

# Calcul des mtriques
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
    print(f" Attention: Impossible de calculer l'AUC-ROC: {e}")
    auc_roc = None

# Affichage des rsultats
print(f"\n{'Mtrique':<25} {'Score':>10}")
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
# TAPE 5: SAUVEGARDE DU MODLE
# ============================================================================

print("\n Sauvegarde du modle BERT...")

# Sauvegarder le modle complet
bert_model.save('models/bert_emotion_classifier.keras')

# Sauvegarder les poids sparment (optionnel)
bert_model.save_weights('models/bert_emotion_weights.h5')

print(" Modle sauvegard:")
print("   - bert_emotion_classifier.keras (modle complet)")
print("   - bert_emotion_weights.h5 (poids uniquement)")

# ============================================================================
# TAPE 6: FONCTION DE PRDICTION POUR DMONSTRATION
# ============================================================================

def predict_emotions_bert(text, model, tokenizer, threshold=0.5):
    """
    Prdit les motions pour un texte donn avec BERT.

    Args:
        text: Texte  analyser (string)
        model: Modle BERT entran
        tokenizer: BertTokenizer
        threshold: Seuil de dcision (par dfaut 0.5)

    Returns:
        dict: {emotion: probabilit} pour les motions dtectes
    """
    # Encoder le texte
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=BERT_MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='np'
    )

    # Prdiction
    pred = model.predict({
        'input_ids': encoding['input_ids'],
        'attention_mask': encoding['attention_mask'],
        'token_type_ids': encoding['token_type_ids']
    }, verbose=0)[0]

    # Filtrer les motions au-dessus du seuil
    emotions = {}
    for i, emotion in enumerate(EMOTION_LABELS):
        if pred[i] >= threshold:
            emotions[emotion] = float(pred[i])

    # Trier par probabilit dcroissante
    emotions = dict(sorted(emotions.items(), key=lambda x: x[1], reverse=True))

    return emotions

# Test de la fonction de prdiction
print("\n TEST DE PRDICTION")
print("="*70)
test_texts = [
    "I'm so happy and excited about this!",
    "This makes me really angry and disappointed.",
    "I'm confused and not sure what to think."
]

for text in test_texts:
    print(f"\nTexte: \"{text}\"")
    predictions = predict_emotions_bert(text, bert_model, bert_tokenizer)
    if predictions:
        print("motions dtectes:")
        for emotion, prob in predictions.items():
            print(f"  - {emotion}: {prob:.3f}")
    else:
        print("  Aucune motion dtecte (toutes en dessous du seuil)")

print("\n" + "="*70)
print(" SCRIPT BERT TERMIN AVEC SUCCS!")
print("="*70)
