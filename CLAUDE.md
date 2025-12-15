# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **GoEmotions Multi-Label Text Classification** project that detects 28 emotion categories from Reddit comments using deep learning. The project requires implementing multiple neural network architectures for comparative evaluation against a multi-label classification task with significant class imbalance.

**Key Context**: This is a 5-week academic project (mini-projet 3IDL) with specific deliverables including a technical report, evaluated models, and an interactive demonstration.

## Dataset Structure

- **Source**: GoEmotions dataset (58,000 Reddit comments)
- **Label Format**: Multi-label (each comment can have multiple emotions)
- **Emotion Categories**: 28 (27 emotions + 1 neutral)
- **Data Split**: 80-10-10 (train: 43,410 | dev: 5,426 | test: 5,427)
- **Files**: `train.tsv`, `dev.tsv`, `test.tsv` (tab-separated, no header)
- **Column Structure**: text | emotion_ids | comment_id
- **Emotion List**: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

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

## Architecture Implementations

Four architectures must be implemented and evaluated:

### Model 1: Simple LSTM
- Embedding (learned) → LSTM (128 units) → Dense (28, sigmoid)
- Baseline model for comparison
- Status: Working baseline exists in notebook

### Model 2: BiLSTM with Attention
- Embedding → Bidirectional LSTM (64 units per direction, return_sequences=True) → Attention → Dense (28, sigmoid)
- Custom Attention layer: computes softmax weights over sequence, aggregates context vector
- Status: Implemented and showing strong results (AUC: 0.8617)

### Model 3: Hybrid CNN-BiLSTM with Attention
- Embedding → CNN (multiple filter sizes) → BiLSTM → Attention → Dense
- Allows evaluation of CNN feature extraction vs. pure recurrent approach
- Required for ablation study

### Model 4: BERT-base
- Use HuggingFace transformers library (`transformers` package)
- Fine-tune pre-trained BERT for multi-label classification
- Different tokenization pipeline (WordPieceTokenizer, not Keras Tokenizer)

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

## Model Saving and Loading

- Save models using Keras: `model.save(path)` → `.h5` or `.keras` format
- When custom layers (Attention) or custom loss functions are used, save with `custom_objects` parameter
- Models should be easily loadable for later testing and demonstration

## Expected Challenges

1. **Class Imbalance**: Unbalanced weighting affects all metrics differently - may need threshold tuning per emotion
2. **Hamming Loss Interpretation**: May appear artificially low if model predicts mostly neutral class
3. **F1-Score Variance**: Macro F1 (which considers rare classes) will be much lower than Micro F1 initially
4. **Attention Layer Serialization**: Custom Attention layer requires `custom_objects` dict when loading
5. **Memory Constraints**: BiLSTM+Attention may be memory-intensive with large batch sizes

## Project Deliverables

1. **Technical Report** (10-15 pages, LaTeX snarticle format)
   - Include architecture diagrams, training curves, comparative results table
2. **Commented Source Code**
   - Clear variable names, explain key decisions (loss function choice, hyperparameters)
3. **Results Presentation** (15 minutes)
4. **Interactive Demonstration**
   - Optional but valued: Streamlit/Flask web interface or Jupyter notebook demo
   - Should allow real-time emotion prediction on user input text

## Key Hyperparameters to Track

- MAX_WORDS: 50,000 (vocabulary size)
- MAX_LEN: 70 (sequence length)
- Embedding dimensions: 100 (learned) or 300+ (pre-trained)
- LSTM units: varies by architecture (128 for simple, 64 bidirectional, etc.)
- Dropout: 0.2-0.5 for regularization
- Learning rate: typically 1e-3 (Adam optimizer)
- Batch size: 64 is common
- Epochs: typically 10-15 before convergence

## File Structure

```
GOEMOTION/
├── CLAUDE.md (this file)
├── ENONCE.txt (project specification in French)
├── GoEmotions_trial.ipynb (working notebook with implementations)
└── .vscode/settings.json (editor configuration)
```

## Next Steps

1. Implement Model 3 (CNN-BiLSTM with Attention) if not complete
2. Implement Model 4 (BERT-base fine-tuning)
3. Conduct ablation studies systematically
4. Create comprehensive results comparison table
5. Begin technical report writing
6. Develop interactive demonstration interface
