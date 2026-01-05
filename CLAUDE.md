# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **GoEmotions Multi-Label Text Classification** project that detects 28 emotion categories from Reddit comments using deep learning. The project implements four different neural network architectures for comparative evaluation of multi-label classification with significant class imbalance.

**Key Context**: Academic project (mini-projet 3IDL) with deliverables including technical report, trained models, and interactive demonstration.

## Development Environment

### Setup and Running Models

```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run individual model training scripts
python architectures/lstm.py                # Model 1: Simple LSTM
python architectures/bi-lstm.py            # Model 2: BiLSTM with Attention
python architectures/cnn-bi-lstm.py        # Model 3: CNN-BiLSTM with Attention
python architectures/bert_pytorch.py       # Model 4: BERT-base (PyTorch)

# Or work interactively with the notebook
jupyter notebook architectures/notebook.ipynb

# Run Flask web application
python app.py                 # Starts web interface at http://localhost:5000
```

### Flask Web Application

A comprehensive web interface has been built for exploring datasets, evaluating models, and making predictions:

**Key Features**:
- **Data Visualization**: Interactive Plotly charts showing emotion distributions, class imbalance, multi-label statistics
- **Model Performance**: Evaluate models on test data with comprehensive metrics (Hamming Loss, AUC-ROC, F1, Precision, Recall)
- **Interactive Prediction**: Enter custom text and see predictions from all models simultaneously
- **Model Comparison**: Side-by-side performance comparison with rankings

**File Structure**:
- `app.py` - Main Flask application with routes
- `utils/model_utils.py` - Model loading utilities with custom Attention layer
- `utils/data_loader.py` - Dataset loading and statistics computation
- `utils/explainability_utils.py` - LIME-based prediction explanations
- `templates/` - HTML templates (base, index, data_visualization, model_performance, predict, compare)
- `static/` - CSS, JavaScript, and static assets

**Running the App**:
```bash
python app.py
# Navigate to http://localhost:5000 in browser
```

### Data Path Configuration

All Python training scripts expect data at one of these paths:
- Local: `dataset/data/` (for local runs)
- Kaggle: `/kaggle/input/goemotions/dataset/data/` (some scripts)

**Important**: When running locally, ensure scripts use `dataset/data/` path. Some files (architectures/bi-lstm.py, architectures/cnn-bi-lstm.py) hardcode Kaggle paths and need adjustment for local execution.

### LaTeX Report

The technical report is located in the `report/` directory:
```bash
cd report
pdflatex main.tex    # Compile LaTeX to PDF
```

**Files**:
- `main.tex` - Main report document
- `main.pdf` - Compiled PDF output
- `references.bib` - Bibliography
- `sn-jnl.cls` - Springer journal class file

## Dataset Structure

- **Source**: GoEmotions (58,000 Reddit comments)
- **Label Format**: Multi-label (each comment can have 0-N emotions)
- **Data Split**: Train: 43,410 | Dev: 5,426 | Test: 5,427
- **TSV Format**: text | emotion_ids | comment_id (tab-separated, no header)
- **Emotion IDs**: Comma-separated integers (0-27), e.g., "2,5" or "27"
- **28 Emotions**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

## Critical Implementation Details

### Multi-Label vs Multi-Class
- **NOT multi-class** - each comment can have 0 to multiple emotions simultaneously
- Loss function: **Binary Cross-Entropy** (not Categorical Cross-Entropy)
- Output activation: **Sigmoid** (not Softmax) - independent probability per emotion
- Y labels: Binary matrix (43410, 28) where each cell is 0 or 1

### Class Imbalance Handling
- Severe imbalance: "neutral" appears 14,219 times, "grief" only 77 times
- Apply `class_weight='balanced'` when training to penalize frequent classes less
- Custom weighted loss function is documented in the notebook for advanced control
- Threshold for binary predictions: typically 0.5, but may need adjustment (e.g., 0.3) based on recall/precision trade-offs

### Text Preprocessing
- Average text length: ~68 characters
- Vocabulary size: ~50,000 most frequent words
- Sequence padding: 70 tokens maximum length (post-padding)
- Tokenizer fit only on training data, then applied to dev/test

## Codebase Architecture

### Project Structure
```
GOEMOTION/
├── dataset/
│   ├── data/                    # TSV data files (train.tsv, dev.tsv, test.tsv)
│   ├── README.md                # GoEmotions dataset documentation
│   ├── analyze_data.py          # Dataset statistics and correlation analysis
│   ├── extract_words.py         # Extract emotion-associated words
│   └── calculate_metrics.py     # Metrics computation utilities
├── architectures/               # Model training scripts
│   ├── lstm.py                  # Model 1: Simple LSTM baseline
│   ├── bi-lstm.py               # Model 2: BiLSTM + Attention
│   ├── cnn-bi-lstm.py           # Model 3: CNN-BiLSTM + Attention (hybrid)
│   ├── bert_pytorch.py          # Model 4: BERT-base fine-tuning
│   ├── notebook.ipynb           # Interactive experimentation notebook
│   └── bert.txt                 # BERT model training output/logs
├── utils/                       # Shared utility modules
│   ├── model_utils.py           # Model loading, custom layers, prediction
│   ├── data_loader.py           # Dataset loading and statistics
│   └── explainability_utils.py  # LIME explanations and interpretability
├── models/                      # Saved trained models (.h5, .keras)
├── report/                      # LaTeX technical report
│   ├── main.tex                 # Report source
│   ├── main.pdf                 # Compiled PDF
│   ├── references.bib           # Bibliography
│   └── sn-jnl.cls               # Springer class file
├── templates/                   # Flask HTML templates
├── static/                      # CSS, JS, static assets
├── app.py                       # Flask web application
└── requirements.txt             # Python dependencies
```

### Model Implementations

**All models share common preprocessing pipeline**:
1. `create_binary_labels(df, labels_list)` - Converts comma-separated emotion IDs to 28-column binary matrix (located in `utils/data_loader.py`)
2. Keras Tokenizer (MAX_WORDS=50000) fitted only on training data
3. Pad sequences to MAX_LEN=70 tokens (post-padding)
4. Class weight calculation using inverse frequency normalization

**Important**: Model training scripts are in `architectures/` directory, while shared utilities are in `utils/` for code reusability.

#### Model 1: Simple LSTM ([architectures/lstm.py](architectures/lstm.py))
- **Architecture**: Embedding(100) → Dropout(0.2) → LSTM(128) → Dropout(0.5) → Dense(28, sigmoid)
- **Training**: Binary crossentropy loss, class_weight dictionary, THRESHOLD=0.3
- **Saved**: `models/modele_lstm_simple.h5`

#### Model 2: BiLSTM with Attention ([architectures/bi-lstm.py](architectures/bi-lstm.py))
- **Architecture**: Embedding(128) → Dropout(0.3) → Bidirectional(LSTM(128, return_sequences=True)) → Dropout(0.3) → **Attention** → Dense(64, relu) → Dropout(0.3) → Dense(28, sigmoid)
- **Custom Components**:
  - `Attention` layer: Computes `tanh(X·W + b)`, applies softmax, aggregates weighted context (now centralized in `utils/model_utils.py`)
  - `weighted_loss_factory(weights)`: Creates weighted BCE loss (penalizes rare emotions more heavily, in `utils/model_utils.py`)
- **Training**: Custom weighted loss, early stopping (patience=3), THRESHOLD=0.5
- **Saved**: `models/modele_bilstm_attention_final.keras`

#### Model 3: CNN-BiLSTM Hybrid ([architectures/cnn-bi-lstm.py](architectures/cnn-bi-lstm.py))
- **Architecture** (Functional API):
  - Embedding(128) → Dropout(0.3)
  - **CNN Branch**: 3 parallel Conv1D layers (filters=64, kernels=3/4/5) → MaxPooling1D → Concatenate
  - **Sequential Component**: Dropout(0.3) → Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
  - **Attention**: Attention layer aggregates BiLSTM outputs
  - **Dense**: Dense(128, relu) → Dropout(0.5) → Dense(28, sigmoid)
- **Training**: Same weighted loss, 3 callbacks (EarlyStopping on val_auc, ModelCheckpoint, ReduceLROnPlateau)
- **Saved**: `models/modele_cnn_bilstm_attention_final.keras` and `models/modele_cnn_bilstm_attention_best.keras`
- **Outputs**: Per-emotion performance summary table

#### Model 4: BERT-base ([architectures/bert_pytorch.py](architectures/bert_pytorch.py))
- Uses PyTorch + HuggingFace Transformers
- Different tokenization: BertTokenizer (WordPiece), not Keras Tokenizer
- Fine-tunes pretrained `bert-base-uncased` for multi-label classification
- Saved model loaded via `utils/model_utils.py` with custom wrapper for Flask integration

## Evaluation Metrics

Must compute all metrics on test set:
- **Hamming Loss**: fraction of labels incorrectly predicted (lower is better)
- **Precision/Recall/F1** (micro): treats all label occurrences equally
- **Precision/Recall/F1** (macro): treats each emotion equally regardless of frequency
- **AUC-ROC** (macro): area under ROC curve for each emotion, then averaged
- All computed via `sklearn.metrics`

## Ablation Study (Model 3)

For the hybrid CNN-BiLSTM architecture, evaluate impact of:
- Attention mechanism (with/without)
- CNN vs LSTM layers (different combinations)
- Embedding types (learned vs. pre-trained)
- Regularization techniques (dropout, L2)

## Critical Implementation Patterns

### Custom Components and Model Loading

**Attention Layer** (used in Models 2 & 3):
- Custom Keras Layer implementing additive attention mechanism
- Requires `custom_objects={'Attention': Attention}` when loading with `load_model()`
- **Centralized in `utils/model_utils.py`** for code reusability (previously duplicated across training scripts)

**Weighted Loss Function**:
- `weighted_loss_factory(weights)` creates custom BCE loss
- Requires `custom_objects={'weighted_binary_crossentropy': custom_loss_function}` for loading
- Addresses severe class imbalance (grief: 77 examples vs neutral: 14,219)

**Loading saved models**:
```python
from tensorflow.keras.models import load_model
from utils.model_utils import Attention, weighted_loss_factory

# For Model 1 (no custom objects)
model_lstm = load_model('models/modele_lstm_simple.h5')

# For Models 2 & 3 (with custom Attention layer)
custom_objects = {
    'Attention': Attention,
    'weighted_binary_crossentropy': weighted_loss_factory(class_weights)
}
model = load_model('models/modele_bilstm_attention_final.keras',
                   custom_objects=custom_objects)

# Or use the centralized loader
from utils.model_utils import load_all_models
models = load_all_models()  # Loads all trained models
```

### Threshold Tuning for Multi-Label Predictions

Different models use different thresholds:
- Model 1 (LSTM): `THRESHOLD = 0.3` (lower threshold increases recall for rare classes)
- Models 2 & 3: `THRESHOLD = 0.5` (standard sigmoid threshold)

Conversion: `y_pred_binary = (y_pred_proba > THRESHOLD).astype(int)`

### Path Compatibility Issues

**Known Issue**: architectures/bi-lstm.py and architectures/cnn-bi-lstm.py may hardcode Kaggle paths:
```python
# Hardcoded Kaggle path
tsv_path = '/kaggle/input/goemotions/dataset/data/'
```

**Fix for local execution**: Change to relative path:
```python
tsv_path = 'dataset/data/'
```

### Utility Modules (`utils/` package)

The codebase now uses modular utilities to avoid code duplication:

- **`utils/model_utils.py`**: Model loading, custom Attention layer, weighted loss, BERT wrapper, prediction utilities
- **`utils/data_loader.py`**: TSV loading, binary label creation, dataset statistics, emotion distribution
- **`utils/explainability_utils.py`**: LIME-based explanations, attention visualization, error analysis

Import pattern:
```python
from utils.model_utils import Attention, weighted_loss_factory, load_all_models, EMOTION_LABELS
from utils.data_loader import load_dataset, get_dataset_statistics
from utils.explainability_utils import EmotionExplainer
```

## Evaluation Workflow

All models follow this evaluation pattern:
```python
# 1. Generate probability predictions
Y_pred_proba = model.predict(X_test)

# 2. Apply threshold
Y_pred_binary = (Y_pred_proba > THRESHOLD).astype(int)

# 3. Compute metrics
from sklearn.metrics import hamming_loss, roc_auc_score, f1_score, precision_score, recall_score

hamming_loss(Y_test, Y_pred_binary)                    # Lower is better
roc_auc_score(Y_test, Y_pred_proba, average='macro')  # Uses probabilities
f1_score(Y_test, Y_pred_binary, average='micro')      # Global performance
f1_score(Y_test, Y_pred_binary, average='macro')      # Rare class performance
```

**Metric Interpretation**:
- **Hamming Loss**: Fraction of incorrect labels (expect 0.02-0.05)
- **Micro F1**: Weighted by label frequency (favors common emotions)
- **Macro F1**: Unweighted average (sensitive to rare emotion performance)
- **AUC-ROC Macro**: Best overall metric for imbalanced multi-label tasks

## Key Hyperparameters by Model

| Hyperparameter    | Model 1 (LSTM) | Model 2 (BiLSTM+Att) | Model 3 (CNN-BiLSTM+Att) |
|-------------------|----------------|----------------------|--------------------------|
| Embedding Dim     | 100            | 128                  | 128                      |
| LSTM Units        | 128            | 128 (bidirectional)  | 64 (bidirectional)       |
| Dropout           | 0.2, 0.5       | 0.3 (uniform)        | 0.2-0.5 (varies)         |
| Batch Size        | 64             | 32                   | 64                       |
| Epochs            | 10             | 15                   | 15                       |
| Learning Rate     | 1e-3           | 1e-3                 | 1e-3                     |
| Prediction Thresh | 0.3            | 0.5                  | 0.5                      |

## Project Status

**Completed**:
- ✅ Model 1: Simple LSTM (trained, saved in models/)
- ✅ Model 2: BiLSTM + Attention (trained, saved in models/)
- ✅ Model 3: CNN-BiLSTM + Attention (trained, saved in models/)
- ✅ Model 4: BERT-base (script in architectures/bert_pytorch.py)
- ✅ Flask web application (app.py with full visualization and prediction features)
- ✅ Utility modules (utils/ package for code reusability)
- ✅ Technical report (report/main.tex, compiled to main.pdf)
- ✅ Explainability integration (LIME support in utils/explainability_utils.py)

**Remaining Work (if any)**:
- 📋 Ablation study for Model 3 (attention vs no-attention, CNN vs LSTM variants)
- 📋 Comprehensive comparative results table across all 4 models
- 📋 Final report revisions and proofreading

## Important Notes for Development

1. **Code Organization**:
   - Model training scripts: `architectures/` directory
   - Shared utilities: `utils/` package (model_utils, data_loader, explainability_utils)
   - Flask web app: `app.py` (root) with `templates/` and `static/`
   - Report: `report/` directory with LaTeX source

2. **Model Naming Convention**:
   - `.h5` format: Model 1 only (older Keras format)
   - `.keras` format: Models 2 & 3 (newer format, supports custom objects better)
   - PyTorch `.pt` or `.pth`: BERT model
   - Use `.keras` for all new Keras/TensorFlow models

3. **Custom Components**: Attention layer and weighted_loss_factory are now centralized in `utils/model_utils.py` to avoid duplication

4. **Notebook vs Scripts**: `architectures/notebook.ipynb` contains experimentation. Production training uses standalone Python scripts in `architectures/`

5. **Flask Integration**: The web app uses `utils/` modules for model loading and predictions, ensuring consistency between training and inference

6. **Dataset Utilities**: Use scripts in `dataset/` directory for data analysis:
   - `python dataset/analyze_data.py` - Statistics and correlation
   - `python dataset/extract_words.py` - Emotion-associated words
