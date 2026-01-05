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

# 3. Noms des colonnes pour les TSV (Pas de header, séparateur Tab)
COLUMN_NAMES = ['text', 'emotion_ids', 'comment_id']
tsv_path = 'dataset/data/'
df_train = pd.read_csv(os.path.join(tsv_path, 'train.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')
df_dev = pd.read_csv(os.path.join(tsv_path, 'dev.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')
df_test = pd.read_csv(os.path.join(tsv_path, 'test.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')

print(f"Jeu de données d'entraînement : {len(df_train)} lignes")
print(f"Jeu de données de validation : {len(df_dev)} lignes")
print(f"Jeu de données de test : {len(df_test)} lignes")

# Vérification (doit contenir des chaînes d'IDs comme '2' ou '1,17')
print("\nAperçu de la colonne des IDs avant transformation :")
print(df_train['emotion_ids'].head())

def create_binary_labels(df, labels_list):
    """
    Convertit la colonne 'emotion_ids' (chaîne d'IDs séparées par des virgules)
    en 28 colonnes binaires (0 ou 1) pour la classification multi-label.
    """

    # Initialise un DataFrame binaire vide de taille (Nb_lignes x 28)
    label_matrix = pd.DataFrame(0, index=df.index, columns=labels_list)

    # Parcourt chaque ligne du DataFrame
    for index, row in df.iterrows():
        # Sépare les IDs d'émotions (sont des chaînes, ex: '2,5' -> ['2', '5'])
        ids = str(row['emotion_ids']).split(',')

        # Convertit les IDs en entiers pour les utiliser comme index
        # L'index 0 correspond à la première émotion de votre liste, etc.
        try:
            # S'assure de ne traiter que des IDs valides (chiffres)
            int_ids = [int(i) for i in ids if i.isdigit()]
        except ValueError:
            # Cas d'erreur (rare) ou ID non numérique
            int_ids = []

        # Met à 1 les colonnes correspondantes dans la matrice
        for emotion_index in int_ids:
            # S'assurer que l'index est valide (entre 0 et 27)
            if 0 <= emotion_index < NUM_LABELS:
                label_matrix.loc[index, labels_list[emotion_index]] = 1

    # Concatène le DataFrame binaire avec le DataFrame original
    df_result = pd.concat([df[['text', 'comment_id']], label_matrix], axis=1)

    return df_result

# Appliquer la transformation aux trois DataFrames
df_train_final = create_binary_labels(df_train, EMOTION_LABELS)
df_dev_final = create_binary_labels(df_dev, EMOTION_LABELS)
df_test_final = create_binary_labels(df_test, EMOTION_LABELS)

print("\nAperçu du DataFrame d'entraînement FINAL :")
print(df_train_final.head())

# 1. Définir les hyperparamètres (à ajuster)

MAX_WORDS = 50000    # Taille maximale du vocabulaire (les 20000 mots les plus fréquents)
MAX_LEN = 70         # Longueur maximale d'une séquence (un commentaire)

# 2. Instancier et adapter le Tokenizer UNIQUEMENT sur les données d'ENTRAÎNEMENT
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
tokenizer.fit_on_texts(df_train_final['text'])

# 3. Transformer les textes en séquences d'entiers (indices de mots)
train_sequences = tokenizer.texts_to_sequences(df_train_final['text'])
dev_sequences = tokenizer.texts_to_sequences(df_dev_final['text'])
test_sequences = tokenizer.texts_to_sequences(df_test_final['text'])

# 4. Padding (uniformisation de la longueur des séquences)
# Remplissage à MAX_LEN (ajout de zéros au début ou à la fin)
X_train = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
X_dev = pad_sequences(dev_sequences, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"La taille du X_train (après padding) est : {X_train.shape}")
Y_train = df_train_final[EMOTION_LABELS].values
Y_dev = df_dev_final[EMOTION_LABELS].values
Y_test = df_test_final[EMOTION_LABELS].values

print(f"La taille du Y_train est : {Y_train.shape}") # Devrait être (Nb_lignes, 28)
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


# SEQUENTIAL ET MODEL_LSTM 
# 1. Création du modèle séquentiel
model_lstm = Sequential()
# 2. Couche Embedding (Option A : Apprentissage des poid
# Cette couche transforme les indices de mots (X_train) en vecteurs denses.
model_lstm.add(Embedding(
    input_dim=MAX_WORDS,
    output_dim=100,
    input_length=MAX_LEN
))
model_lstm.add(Dropout(0.2))  # ✅ AJOUT: Dropout après embedding

# 3. Couche LSTM (Réseau de neurones récurrents)# C'est le cœur du modèle, il lit la séquence et capture les dépendances.
# - units: Nombre de neurones/unités internes
model_lstm.add(LSTM(units=128))
model_lstm.add(Dropout(0.5))  # ✅ AJOUT: Dropout avant la couche Dense

# Note: On n'utilise pas return_sequences=True ici car nous voulons un seul vecteur
# de sortie pour toute la séquence, nécessaire pour la classification finale.
# 4. Couche de Classification Finale units: Doit être égal au nombre de classes (28) et activation: 'sigmoid' .
#   Chaque neurone de sortie prédit indépendamment la probabilité d'une émotion.
model_lstm.add(Dense(NUM_LABELS, activation='sigmoid'))


# 2. COMPILATION 

# Définition de l'optimiseur (Adam est un bon choix par défaut)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Compilation
model_lstm.compile(
    optimizer=optimizer,
    loss='binary_crossentropy', # ESSENTIEL pour le multi-label
    metrics=[   
        'accuracy',
        tf.keras.metrics.AUC(name='auc'),
    ]
)

# 3- FIT AVEC EPOCHS / BATCH SIZE
EPOCHS = 10
BATCH_SIZE = 64

print("\n--- DÉBUT DE L'ENTRAÎNEMENT DU LSTM SIMPLE ---")

lstm_simple = model_lstm.fit(
    X_train,
    Y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_dev, Y_dev),
        class_weight=class_weights_dict,

    verbose=1
)

print("\n--- ENTRAÎNEMENT TERMINÉ ---")

#sauvegarde
chemin_sauvegarde = 'models/modele_lstm_simple.h5'
model_lstm.save(chemin_sauvegarde)

# Vous devez spécifier le chemin exact où vous l'avez sauvegardé
model_lstm_loaded = load_model('models/modele_lstm_simple.h5')
Y_pred = model_lstm_loaded.predict(X_test)
# Prédictions Binaires (avec Seuil)
THRESHOLD = 0.3

# Y_pred_binary: Transformation des probabilités en 0 ou 1
y_pred_binary= (Y_pred > THRESHOLD).astype(int)
print(f"Forme des prédictions binaires : {Y_pred_binary.shape}")
h_loss = hamming_loss(Y_test, y_pred_binary)

f1_micro = f1_score(Y_test, y_pred_binary, average='micro')
f1_macro = f1_score(Y_test, y_pred_binary, average='macro')
auc_roc = roc_auc_score(Y_test, Y_pred, average='macro')
print("\n" + "="*50)
print("  RÉSULTATS DE LA BASELINE (LSTM SIMPLE)   ")
print("="*50)
print(f"1. Hamming Loss (H-Loss) : {h_loss:.4f} (Doit être proche de 0)")
print(f"2. F1-score (Micro)      : {f1_micro:.4f}")
print(f"3. F1-score (Macro)      : {f1_macro:.4f}")
print(f"4. AUC-ROC (Macro)       : {auc_roc:.4f} (Doit être proche de 1)")
print("="*50)

