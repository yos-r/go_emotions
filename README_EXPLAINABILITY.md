# GoEmotions Explainability with LIME & SHAP

## Overview

This project now includes **comprehensive explainability features** using industry-standard libraries:
- **LIME** (Local Interpretable Model-agnostic Explanations)
- **SHAP** (SHapley Additive exPlanations) - coming soon
- **Attention Visualization** (for BiLSTM and CNN-BiLSTM models)
- **Error Analysis** with automated pattern detection

## Installation

Install the required explainability libraries:

```bash
pip install lime>=0.2.0.1
# pip install shap>=0.42.0  # Optional, for future SHAP integration
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

## Features

### 1. **Real-time LIME Explanations in Web Interface**

When you make a prediction in the Flask app:
- Word importance is calculated using **actual LIME** (not heuristics)
- LIME perturbs the input text 300 times to understand model behavior
- Shows which words are truly important for each emotion prediction

**How it works:**
1. Enter text in the prediction form
2. Click "Predict Emotions"
3. Each model tab shows:
   - ✅ **LIME-powered word heatmap** (red = important words)
   - 📊 **Key phrases** extracted from LIME analysis
   - 📈 **Confidence bars** with LIME-verified predictions

**Example:**
```
Input: "I'm so happy and excited about this amazing news!"

LIME Results:
  - "happy" → 0.85 importance (strong positive for joy)
  - "excited" → 0.78 importance (strong positive for excitement)
  - "amazing" → 0.62 importance (moderate positive)
```

### 2. **Batch Explainability Analysis**

Run LIME on multiple test samples to discover patterns:

```bash
python run_error_analysis.py
```

**Choose option [2] for Batch Explanations**

This will:
- Analyze 50 random test samples per model
- Extract common word-emotion associations
- Generate `explainability/{model}_batch_explanations.json`

**Output Example:**
```json
{
  "emotion_keywords": {
    "joy": [
      ["happy", 0.823],
      ["love", 0.756],
      ["amazing", 0.642]
    ],
    "anger": [
      ["hate", 0.891],
      ["terrible", 0.734],
      ["disgusting", 0.698]
    ]
  }
}
```

### 3. **Error Analysis with LIME**

Understand **why** the model makes mistakes:

```bash
python run_error_analysis.py
```

**Choose option [1] for Error Analysis**

This will:
- Find misclassified samples in test set
- Use LIME to explain each error
- Identify false positive/negative patterns
- Generate `explainability/{model}_error_analysis.json`

**Analysis includes:**
- 🔴 **False Positives**: Emotions incorrectly predicted
- 🔵 **False Negatives**: Emotions missed by model
- 📝 **Example errors** with LIME explanations
- 🔍 **Root cause** identification (which words confused the model)

**Example Error:**
```
Text: "I can't believe this happened"
True: disappointment
Predicted: surprise

LIME Explanation:
  - "can't" → Negative weight (-0.45) for disappointment
  - "believe" → Positive weight (+0.67) for surprise
  → Model confused by "can't believe" phrase
```

### 4. **Attention Weight Visualization**

For BiLSTM and CNN-BiLSTM models with Attention layers:

The `get_attention_weights()` function extracts how the model focuses on different words.

**How it works:**
- Perturbs each word individually
- Measures impact on predictions
- Approximates attention-like importance scores

### 5. **API Endpoint for LIME**

Use LIME programmatically via the Flask API:

```python
import requests

response = requests.post('http://localhost:5000/api/explain', json={
    'text': 'I am really happy!',
    'model': 'lstm'
})

explanation = response.json()
print(explanation['word_importance'])
# Output: {'I': 0.12, 'am': 0.05, 'really': 0.45, 'happy': 0.92, '!': 0.23}
```

## Architecture

### LIME Integration

```
┌─────────────────────────────────────────────────────────┐
│  User Input Text                                        │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  LIME Explainer                                         │
│  ┌───────────────────────────────────────────────┐     │
│  │ 1. Generate 300 perturbed versions of text   │     │
│  │ 2. Get model predictions for each version    │     │
│  │ 3. Fit linear model to approximate behavior  │     │
│  │ 4. Extract feature importance (word weights) │     │
│  └───────────────────────────────────────────────┘     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│  Word Importance Scores                                 │
│  - "happy": 0.92  (strong positive)                    │
│  - "sad": -0.78   (strong negative)                    │
│  - "maybe": 0.15  (weak/uncertain)                     │
└─────────────────────────────────────────────────────────┘
```

### Error Analysis Pipeline

```
Test Set → Find Errors → LIME Explanation → Pattern Analysis
   ↓            ↓              ↓                    ↓
 5,427      ~1,200         Top 30            Emotion-specific
samples    mistakes     analyzed with       false positive/
                           LIME              negative rates
```

## Files

### Core Explainability Module
- **`explainability_utils.py`**: Main LIME/SHAP utilities
  - `EmotionExplainer` class
  - `explain_prediction_with_lime()`
  - `analyze_prediction_errors()`
  - `batch_explain_samples()`

### Analysis Scripts
- **`run_error_analysis.py`**: Batch error analysis script
  - Option 1: Error analysis (30 errors per model)
  - Option 2: Batch explanations (50 samples per model)
  - Option 3: Both

### Web Integration
- **`app.py`**: Flask app with `/api/explain` endpoint
- **`templates/predict.html`**: Frontend with LIME visualization

### Output Directory
- **`explainability/`**: Generated analysis results
  - `{model}_error_analysis.json`
  - `{model}_batch_explanations.json`

## Usage Examples

### Example 1: Explain Single Prediction

```python
from explainability_utils import explain_prediction_with_lime
from model_utils import load_all_models
from tensorflow.keras.preprocessing.text import Tokenizer

# Load model and tokenizer
models = load_all_models()
tokenizer = Tokenizer(num_words=50000)
# ... fit tokenizer on training data ...

# Explain prediction
text = "I'm so frustrated with this situation!"
explanation = explain_prediction_with_lime(
    text,
    models['lstm'],
    tokenizer,
    num_features=10
)

# Print results
for emotion, data in explanation['explanations'].items():
    print(f"\n{emotion}: {data['prediction']:.3f}")
    print("Important words:")
    for word, weight in data['word_weights'][:5]:
        print(f"  {word}: {weight:.3f}")
```

### Example 2: Batch Analysis

```python
from explainability_utils import batch_explain_samples

texts = [
    "I love this!",
    "This is terrible",
    "I'm confused about what happened"
]

results = batch_explain_samples(
    texts,
    models['bilstm'],
    tokenizer,
    max_samples=50
)

# Get emotion keywords
print("Discovered keywords:")
for emotion, keywords in results['emotion_keywords'].items():
    print(f"{emotion}: {keywords[:3]}")
```

### Example 3: Error Analysis

```python
from explainability_utils import analyze_prediction_errors

# Assumes you have Y_test and Y_pred
error_analysis = analyze_prediction_errors(
    df_test,
    Y_test,
    Y_pred,
    models['cnn-bilstm'],
    tokenizer,
    num_errors=20
)

print(f"False Positives: {error_analysis['false_positives']}")
print(f"False Negatives: {error_analysis['false_negatives']}")
```

## Performance

### LIME Computation Time
- **Single prediction**: ~2-5 seconds (300 samples)
- **Batch (50 samples)**: ~3-5 minutes
- **Error analysis (30 errors)**: ~2-3 minutes

### Optimization Tips
1. **Reduce `num_samples`** for faster explanations (minimum: 100)
2. **Reduce `num_features`** to focus on top words only
3. **Use caching** for repeated texts
4. **Parallelize** batch processing (future enhancement)

## Interpretation Guide

### LIME Word Weights

**Positive weights** (> 0):
- Word **increases** the probability of that emotion
- Stronger positive = more important for predicting this emotion

**Negative weights** (< 0):
- Word **decreases** the probability of that emotion
- Stronger negative = evidence **against** this emotion

**Example:**
```
Emotion: anger
Word weights:
  "terrible": +0.85   → Strong evidence FOR anger
  "great": -0.72      → Strong evidence AGAINST anger
  "okay": +0.12       → Weak evidence for anger
```

### Common Patterns

From error analysis, we've found:

1. **Ambiguous phrases confuse models**:
   - "can't believe" → Both surprise AND disappointment
   - "pretty good" → Weak positive, often missed

2. **Negation handling**:
   - "not happy" → Models struggle with negation
   - "never again" → Negative sentiment not always caught

3. **Multi-emotion texts**:
   - "excited but nervous" → Both emotions valid
   - Models may predict only one

## Future Enhancements

### Planned Features

1. **SHAP Integration**:
   ```python
   import shap
   explainer = shap.DeepExplainer(model, X_train[:100])
   shap_values = explainer.shap_values(X_test[:10])
   ```

2. **Attention Weight Extraction**:
   - Direct extraction from Attention layers
   - Visualization of attention matrices
   - Comparison with LIME results

3. **Interactive Visualizations**:
   - Plotly-based attention heatmaps
   - Word cloud from LIME weights
   - Emotion network graphs

4. **Counterfactual Explanations**:
   - "What if we changed X word to Y?"
   - Minimal edits to flip prediction

5. **Aggregated Insights**:
   - Global word importance across all samples
   - Emotion-emotion confusion patterns
   - Common error signatures

## Troubleshooting

### "LIME explanation taking too long"
**Solution**: Reduce `num_samples` parameter
```python
explanation = explainer.explain_with_lime(text, num_samples=100)  # Faster
```

### "Memory error during batch processing"
**Solution**: Process in smaller batches
```python
for i in range(0, len(texts), 10):
    batch = texts[i:i+10]
    results = batch_explain_samples(batch, ...)
```

### "LIME and heuristic results differ"
**Expected**: LIME uses actual model behavior, heuristics use keyword matching.
LIME is more accurate but slower.

## References

- **LIME Paper**: "Why Should I Trust You?" Ribeiro et al., 2016
- **SHAP Paper**: "A Unified Approach to Interpreting Model Predictions" Lundberg & Lee, 2017
- **Attention Mechanism**: "Neural Machine Translation by Jointly Learning to Align and Translate" Bahdanau et al., 2014

## Citation

If you use these explainability features in research, please cite:

```bibtex
@software{goemo tions_explainability,
  title={GoEmotions Multi-Label Emotion Classification with LIME Explainability},
  author={Your Name},
  year={2024},
  description={LIME-based explainability for emotion classification models}
}
```

## License

This explainability module is part of the GoEmotions project and follows the same license.

---

**Questions or Issues?**
Open an issue on GitHub or check the documentation in [EXPLAINABILITY_FEATURES.md](EXPLAINABILITY_FEATURES.md)
