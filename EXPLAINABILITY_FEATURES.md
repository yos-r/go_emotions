# Explainability & Interpretability Features

This document describes the explainability features added to the GoEmotions emotion classification web application.

## Overview

The prediction interface now includes comprehensive explainability tools to help understand:
- **Which words** influenced the model's predictions
- **How confident** the models are in their predictions
- **Where models agree/disagree** on emotion detection

## Features Implemented

### 1. Word Importance Heatmap

**Location:** Under each model's predictions tab

**What it does:**
- Visualizes which words in the input text had the most influence on predictions
- Uses color intensity to show importance:
  - **Dark red/pink**: High importance words (strongly influenced the prediction)
  - **Light pink**: Moderate importance
  - **White/neutral**: Low importance

**How it works:**
- Analyzes text based on emotion-related keywords
- Considers punctuation emphasis (!, ?)
- Accounts for capitalization (ALL CAPS = emphasis)
- Normalizes scores across all words for relative comparison

**Hover interaction:** Hover over any word to see its exact importance percentage

### 2. Key Influential Phrases

**Location:** Below the word importance heatmap

**What it does:**
- Extracts consecutive high-importance words as key phrases
- Displays them as blue badges for easy identification
- Shows which specific parts of the text drove the emotion detection

**Example:**
- Input: "I'm so happy and excited about this amazing news!"
- Key phrases: "happy and excited", "amazing news!"

### 3. Top Detected Emotions with Confidence Bars

**Location:** At the top of the explainability section for each model

**What it does:**
- Shows the top 3 emotions detected by each model
- Visual confidence bars using gradient colors:
  - Green end = high confidence
  - Yellow = moderate confidence
  - Red = low confidence
- Percentage values for exact confidence levels

### 4. Model Agreement Analysis

**Location:** Bottom of the results section, after model comparison

**What it shows:**

#### Agreement Summary Cards:
1. **Unanimous Emotions**: Emotions all models agree on (high confidence)
2. **Partial Agreement**: Emotions some models detected (moderate confidence)
3. **Single Model Only**: Emotions only one model detected (low confidence, potential false positives)
4. **Agreement Rate**: Overall percentage of unanimous predictions

#### Detailed Agreement Table:
- **Emotion**: Name of detected emotion
- **Agreement**: How many models detected it (e.g., "3/4 models")
- **Models**: Which specific models detected it (LSTM, BiLSTM, CNN-BiLSTM, BERT)
- **Avg Confidence**: Average confidence across models that detected it
- **Confidence Range**: Min-max range with variance warning if high disagreement

#### Reliability Insights:
- Automatically generated insights about prediction confidence
- Categorizes emotions by reliability level
- Warns about potential false positives

### 5. Interpretation Guide

**Location:** Below the agreement analysis

**Purpose:** Helps users understand what the metrics mean:
- High agreement → Strong confidence
- Partial agreement → Nuanced emotions, moderate confidence
- Low agreement → Ambiguous content or mixed emotions
- Color-coded confidence bars explanation

## Technical Implementation

### Word Importance Algorithm

```javascript
// Keyword-based scoring
const emotionKeywords = {
    'happy': ['happy', 'joy', 'great', 'amazing', ...],
    'sad': ['sad', 'disappointed', 'unhappy', ...],
    'angry': ['angry', 'mad', 'furious', 'hate', ...],
    // ... more emotion categories
};

// Scoring factors:
- Base score: 0.3
- Emotion keyword match: 0.9
- Punctuation boost: +0.2
- Capitalization boost: +0.15
- Normalized relative to max score
```

### Agreement Metrics

```javascript
// Agreement classification:
- Unanimous: count === total_models (all agree)
- Partial: 1 < count < total_models (some agree)
- Single: count === 1 (only one model)

// Confidence metrics:
- Average score across agreeing models
- Min/max range showing variance
- Standard deviation for high variance detection
```

## Use Cases

### 1. Understanding Predictions
**Scenario:** "Why did the model classify this as 'excited'?"
**Solution:** Check the word importance heatmap to see which words triggered the 'excited' classification

### 2. Validating Confidence
**Scenario:** "Can I trust this prediction?"
**Solution:** Look at model agreement:
- All 4 models agree → High trust
- Only 1 model → Low trust, investigate further

### 3. Debugging Misclassifications
**Scenario:** "The prediction seems wrong"
**Solution:**
- Check which words were highlighted as important
- Review if those words actually convey the detected emotion
- Look at model disagreement to identify ambiguity

### 4. Multi-Label Understanding
**Scenario:** "Why are multiple emotions detected?"
**Solution:** Key phrases show which parts of text correspond to which emotions

### 5. Model Comparison
**Scenario:** "Which model is most reliable?"
**Solution:** Review agreement analysis to see which models consistently agree vs. outliers

## Limitations & Future Enhancements

### Current Limitations:
1. **Heuristic-based word importance**: Uses keyword matching rather than true model attention weights
2. **No SHAP/LIME integration**: True gradient-based explanations not yet implemented
3. **Simple phrase extraction**: Doesn't consider semantic relationships

### Planned Enhancements:
1. **Attention Weight Visualization**:
   - Extract actual attention weights from BiLSTM/CNN-BiLSTM models
   - Visualize attention heatmaps from the Attention layers

2. **LIME Integration**:
   - Implement Local Interpretable Model-agnostic Explanations
   - Show which word perturbations change predictions

3. **SHAP Values**:
   - Calculate Shapley values for each word's contribution
   - More accurate than keyword heuristics

4. **Error Analysis Dashboard**:
   - Aggregate statistics on common misclassifications
   - Confusion matrices per model
   - Error pattern identification

5. **Interactive Exploration**:
   - Click on words to see their impact on specific emotions
   - Modify text and see real-time importance changes

## Example Interpretation

**Input Text:** "I'm so happy and excited about this amazing news!"

**Expected Output:**

### Word Importance:
- High importance: "happy", "excited", "amazing"
- Moderate: "so", "news"
- Low: "I'm", "and", "about", "this"

### Key Phrases:
- "so happy and excited"
- "amazing news!"

### Model Agreement:
- **Unanimous (100%)**: joy, excitement
- **Partial (75%)**: optimism (3/4 models)
- **Single (25%)**: surprise (1/4 models)

### Interpretation:
✅ **High confidence prediction**: All models agree on "joy" and "excitement"
⚠️ **Moderate confidence**: Most models detected "optimism"
❌ **Low confidence**: Only one model detected "surprise" - likely false positive

## Usage Instructions

1. **Enter text** in the prediction form
2. **Submit** for prediction
3. **Navigate tabs** to view each model's predictions
4. **Review word heatmap** to see what influenced the prediction
5. **Check key phrases** to understand specific triggers
6. **Examine agreement analysis** to assess confidence
7. **Use interpretation guide** for understanding metrics

## References

- Attention mechanism visualization techniques
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Multi-model ensemble analysis
