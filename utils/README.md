# Utils Package

This directory contains utility modules for the GoEmotions Flask web application.

## Module Overview

### 1. `model_utils.py`
**Purpose**: Model loading, custom layers, and prediction utilities

**Key Components**:
- `Attention` - Custom Keras attention layer for BiLSTM and CNN-BiLSTM models
- `weighted_loss_factory()` - Creates weighted BCE loss for class imbalance handling
- `BertMultiLabelClassifier` - PyTorch BERT model class
- `BertModelWrapper` - Wrapper to make BERT compatible with Keras interface
- `load_all_models()` - Loads all trained models (LSTM, BiLSTM, CNN-BiLSTM, BERT, hybrid variants)
- `preprocess_for_prediction()` - Preprocesses text data for model inference
- `predict_with_model()` - Generate predictions for single text samples

**Constants**:
- `EMOTION_LABELS` - List of 28 emotion labels
- `NUM_LABELS = 28`
- `MAX_WORDS = 50000` - Vocabulary size
- `MAX_LEN = 70` - Sequence padding length for Keras models
- `BERT_MAX_LEN = 128` - Sequence length for BERT

### 2. `data_loader.py`
**Purpose**: Dataset loading and statistics computation

**Key Functions**:
- `load_dataset(tsv_path)` - Load train/dev/test TSV files
- `create_binary_labels(df, labels_list)` - Convert emotion IDs to binary matrix
- `get_emotion_distribution(df)` - Calculate emotion frequency counts
- `get_multi_label_statistics(df)` - Compute multi-label metrics
- `get_text_statistics(df)` - Analyze text length and word count stats
- `get_dataset_statistics()` - Comprehensive statistics for all splits
- `get_sample_texts(df, n)` - Extract sample texts with labels

### 3. `explainability_utils.py`
**Purpose**: LIME-based explanations for emotion predictions

**Key Components**:
- `EmotionExplainer` - Main class for generating explanations
  - `explain_with_lime()` - Generate LIME explanation for text
  - `get_attention_weights()` - Extract attention weights (for BiLSTM/CNN-BiLSTM)
  - `analyze_word_importance()` - Compute unified word importance scores
- `explain_prediction_with_lime()` - High-level function for single prediction
- `get_model_attention_visualization()` - Attention-based visualization
- `batch_explain_samples()` - Explain multiple samples and aggregate insights
- `analyze_prediction_errors()` - Error analysis on misclassified samples

## Dependencies

### Common Dependencies (all modules)
- `numpy`
- `pandas`
- `tensorflow` / `keras`

### Model-Specific Dependencies
- **model_utils.py**: `torch`, `transformers` (for BERT model)
- **explainability_utils.py**: `lime`, `shap`

## Usage Examples

### Loading Models and Making Predictions

```python
from utils.model_utils import load_all_models, EMOTION_LABELS
from utils.data_loader import load_dataset

# Load all trained models
models = load_all_models()

# Load dataset
df_train, df_dev, df_test = load_dataset()

# Make prediction
text = "I'm so happy today!"
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(df_train['text'])

sequences = tokenizer.texts_to_sequences([text])
padded = pad_sequences(sequences, maxlen=70)

predictions = models['lstm'].predict(padded)
print("Predicted emotions:", [EMOTION_LABELS[i] for i, p in enumerate(predictions[0]) if p > 0.3])
```

### Generating Explanations

```python
from utils.explainability_utils import EmotionExplainer

explainer = EmotionExplainer(models['lstm'], tokenizer, 'lstm')
explanation = explainer.explain_with_lime(
    "I can't believe you did this!",
    num_features=10,
    num_samples=500
)

print("Explanation:", explanation)
```

### Computing Dataset Statistics

```python
from utils.data_loader import get_dataset_statistics

stats = get_dataset_statistics(df_train, df_dev, df_test)
print(f"Total samples: {stats['splits']['total']}")
print(f"Most common emotion: {stats['most_common_emotion']}")
print(f"Class imbalance ratio: {stats['imbalance_ratio']:.2f}")
```

## Integration with Flask App

The main Flask application ([app.py](../app.py)) imports from these modules:

```python
from utils.model_utils import (
    Attention, weighted_loss_factory, load_all_models,
    EMOTION_LABELS, NUM_LABELS, MAX_WORDS, MAX_LEN
)
from utils.data_loader import load_dataset, get_dataset_statistics
from utils.explainability_utils import EmotionExplainer
```

## File Organization

```
utils/
├── __init__.py                  # Package initialization
├── README.md                    # This file
├── model_utils.py               # Model loading and prediction
├── data_loader.py               # Dataset loading and statistics
└── explainability_utils.py      # LIME explanations
```

## Notes

- All utility modules use relative imports (e.g., `from .model_utils import ...`)
- The `__init__.py` uses lazy imports to avoid loading heavy dependencies (PyTorch, LIME) until needed
- Custom Keras objects (`Attention` layer, weighted loss) require `custom_objects` parameter when loading models
- BERT model uses PyTorch instead of TensorFlow/Keras
