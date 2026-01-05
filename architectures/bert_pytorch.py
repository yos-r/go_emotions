import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss, precision_score, recall_score
from sklearn.utils import class_weight
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION ET CHARGEMENT DES DONNÉES
# ============================================================================

EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude',
    'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride',
    'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]
NUM_LABELS = len(EMOTION_LABELS)

# Noms des colonnes pour les TSV
COLUMN_NAMES = ['text', 'emotion_ids', 'comment_id']
tsv_path = 'dataset/data/'

print("="*70)
print(" CHARGEMENT DES DONNÉES")
print("="*70)

df_train = pd.read_csv(os.path.join(tsv_path, 'train.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')
df_dev = pd.read_csv(os.path.join(tsv_path, 'dev.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')
df_test = pd.read_csv(os.path.join(tsv_path, 'test.tsv'), sep='\t', header=None, names=COLUMN_NAMES, encoding='utf-8')

print(f"Jeu de données d'entraînement : {len(df_train)} lignes")
print(f"Jeu de données de validation : {len(df_dev)} lignes")
print(f"Jeu de données de test : {len(df_test)} lignes")

def create_binary_labels(df, labels_list):
    """
    Convertit la colonne 'emotion_ids' en matrice binaire (28 colonnes).
    """
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

# Transformation des labels
df_train_final = create_binary_labels(df_train, EMOTION_LABELS)
df_dev_final = create_binary_labels(df_dev, EMOTION_LABELS)
df_test_final = create_binary_labels(df_test, EMOTION_LABELS)

# Extraire les labels Y
Y_train = df_train_final[EMOTION_LABELS].values.astype(np.float32)
Y_dev = df_dev_final[EMOTION_LABELS].values.astype(np.float32)
Y_test = df_test_final[EMOTION_LABELS].values.astype(np.float32)

print(f"Y_train shape: {Y_train.shape}")
print(f"Y_dev shape: {Y_dev.shape}")
print(f"Y_test shape: {Y_test.shape}")

# ============================================================================
# CALCUL DES POIDS DE CLASSE POUR LOSS PONDÉRÉE
# ============================================================================

print("\n" + "="*70)
print(" POIDS DE CLASSE POUR WEIGHTED LOSS")
print("="*70)

# Calculer la fréquence de chaque émotion
emotion_frequencies = Y_train.sum(axis=0) / len(Y_train)
# Calculer les poids (inverse de la fréquence)
class_weights = 1.0 / (emotion_frequencies + 1e-6)
# Normaliser les poids
class_weights = class_weights / class_weights.sum() * NUM_LABELS

print(f"{'Émotion':<20} {'Fréquence':>15} {'Poids':>15}")
print("-"*70)
for i, emotion in enumerate(EMOTION_LABELS):
    print(f"{emotion:<20} {emotion_frequencies[i]:>15.4f} {class_weights[i]:>15.4f}")
print("="*70)

# Convertir en tenseur PyTorch
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# ============================================================================
# DATASET PYTORCH POUR BERT
# ============================================================================

class EmotionDataset(Dataset):
    """Dataset PyTorch pour GoEmotions avec tokenization BERT"""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels[idx]

        # Tokenization BERT
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

# ============================================================================
# MODÈLE BERT POUR CLASSIFICATION MULTI-LABEL
# ============================================================================

class BertMultiLabelClassifier(nn.Module):
    """
    Modèle BERT pour classification multi-label.

    Architecture:
        - BERT pré-entraîné (bert-base-uncased)
        - Dropout (0.3)
        - Dense layer (28 units, sigmoid activation)
    """

    def __init__(self, num_labels, dropout=0.3):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # Obtenir les sorties BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Utiliser le [CLS] token (première position)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Dropout + Classification
        dropout_output = self.dropout(cls_output)
        logits = self.classifier(dropout_output)

        return logits

# ============================================================================
# LOSS PONDÉRÉE POUR DÉSÉQUILIBRE DE CLASSES
# ============================================================================

class WeightedBCELoss(nn.Module):
    """Binary Cross-Entropy Loss pondérée par classe"""

    def __init__(self, weights):
        super(WeightedBCELoss, self).__init__()
        self.weights = weights

    def forward(self, logits, labels):
        # Calculer BCE loss
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction='none'
        )
        # Appliquer les poids
        weighted_loss = bce_loss * self.weights.to(logits.device)
        return weighted_loss.mean()

# ============================================================================
# FONCTIONS D'ENTRAÎNEMENT ET D'ÉVALUATION
# ============================================================================

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device):
    """Entraîne le modèle pour une époque"""
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)

def eval_model(model, data_loader, loss_fn, device):
    """Évalue le modèle sur un ensemble de validation"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            # Obtenir les probabilités avec sigmoid
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return total_loss / len(data_loader), all_preds, all_labels

# ============================================================================
# CONFIGURATION ET ENTRAÎNEMENT
# ============================================================================

print("\n" + "="*70)
print(" CONFIGURATION DU MODÈLE BERT")
print("="*70)

# Hyperparamètres
BERT_MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

# Détecter le device (GPU si disponible)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilisé: {device}")

# Charger le tokenizer BERT
print("\n Chargement du tokenizer BERT...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Créer les datasets
print(" Création des datasets...")
train_dataset = EmotionDataset(
    texts=df_train_final['text'],
    labels=Y_train,
    tokenizer=tokenizer,
    max_len=BERT_MAX_LEN
)

dev_dataset = EmotionDataset(
    texts=df_dev_final['text'],
    labels=Y_dev,
    tokenizer=tokenizer,
    max_len=BERT_MAX_LEN
)

test_dataset = EmotionDataset(
    texts=df_test_final['text'],
    labels=Y_test,
    tokenizer=tokenizer,
    max_len=BERT_MAX_LEN
)

# Créer les dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f" Datasets créés:")
print(f"   Train: {len(train_dataset)} échantillons, {len(train_loader)} batches")
print(f"   Dev: {len(dev_dataset)} échantillons, {len(dev_loader)} batches")
print(f"   Test: {len(test_dataset)} échantillons, {len(test_loader)} batches")

# Créer le modèle
print("\n Construction du modèle BERT...")
model = BertMultiLabelClassifier(num_labels=NUM_LABELS)
model = model.to(device)

# Déplacer les poids sur le device
class_weights_tensor = class_weights_tensor.to(device)

# Loss function
loss_fn = WeightedBCELoss(weights=class_weights_tensor)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

print(f" Modèle créé avec succès!")
print(f"   Nombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Nombre de paramètres entraînables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# ============================================================================
# ENTRAÎNEMENT
# ============================================================================

print("\n" + "="*70)
print(" ENTRAÎNEMENT DU MODÈLE BERT")
print("="*70)

best_val_loss = float('inf')
best_model_state = None

for epoch in range(EPOCHS):
    print(f"\nÉpoque {epoch + 1}/{EPOCHS}")
    print("-"*70)

    # Entraînement
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, scheduler, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Validation
    val_loss, val_preds, val_labels = eval_model(model, dev_loader, loss_fn, device)
    print(f"Val Loss: {val_loss:.4f}")

    # Sauvegarder le meilleur modèle
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }, 'best_bert_model_pytorch.pt')
        print(f" Meilleur modèle sauvegardé! (Val Loss: {val_loss:.4f})")

print("\n Entraînement terminé!")

# Charger le meilleur modèle
model.load_state_dict(best_model_state)

# ============================================================================
# ÉVALUATION SUR L'ENSEMBLE DE TEST
# ============================================================================

print("\n" + "="*70)
print(" ÉVALUATION DU MODÈLE BERT SUR L'ENSEMBLE DE TEST")
print("="*70)

# Prédictions sur le test set
test_loss, y_pred_proba, Y_test_actual = eval_model(model, test_loader, loss_fn, device)

# Seuil de décision
THRESHOLD = 0.5
y_pred_binary = (y_pred_proba >= THRESHOLD).astype(int)

# Calcul des métriques
hamming = hamming_loss(Y_test_actual, y_pred_binary)
precision_micro = precision_score(Y_test_actual, y_pred_binary, average='micro', zero_division=0)
recall_micro = recall_score(Y_test_actual, y_pred_binary, average='micro', zero_division=0)
f1_micro = f1_score(Y_test_actual, y_pred_binary, average='micro', zero_division=0)

precision_macro = precision_score(Y_test_actual, y_pred_binary, average='macro', zero_division=0)
recall_macro = recall_score(Y_test_actual, y_pred_binary, average='macro', zero_division=0)
f1_macro = f1_score(Y_test_actual, y_pred_binary, average='macro', zero_division=0)

# AUC-ROC (macro)
try:
    auc_roc = roc_auc_score(Y_test_actual, y_pred_proba, average='macro')
except ValueError as e:
    print(f" Attention: Impossible de calculer l'AUC-ROC: {e}")
    auc_roc = None

# Affichage des résultats
print(f"\n{'Métrique':<25} {'Score':>10}")
print("-"*70)
print(f"{'Test Loss':<25} {test_loss:>10.4f}")
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
# ANALYSE PAR ÉMOTION
# ============================================================================

print("\n" + "="*70)
print(" MÉTRIQUES PAR ÉMOTION")
print("="*70)

print(f"{'Émotion':<20} {'F1-Score':>12} {'Precision':>12} {'Recall':>12} {'AUC-ROC':>12}")
print("-"*70)

for i, emotion in enumerate(EMOTION_LABELS):
    f1 = f1_score(Y_test_actual[:, i], y_pred_binary[:, i], zero_division=0)
    precision = precision_score(Y_test_actual[:, i], y_pred_binary[:, i], zero_division=0)
    recall = recall_score(Y_test_actual[:, i], y_pred_binary[:, i], zero_division=0)

    try:
        auc = roc_auc_score(Y_test_actual[:, i], y_pred_proba[:, i])
    except ValueError:
        auc = 0.0

    print(f"{emotion:<20} {f1:>12.4f} {precision:>12.4f} {recall:>12.4f} {auc:>12.4f}")

print("="*70)
print("\n ÉVALUATION TERMINÉE!")
print("="*70)
