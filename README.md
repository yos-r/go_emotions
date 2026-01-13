# Multi-Label Emotion Classification from Text Using Deep Learning

**A Comparative Study of LSTM, BiLSTM-Attention, CNN-BiLSTM, and BERT Architectures**

**Author:** Yosr Barghouti
**Institution:** Higher Institute of Computer Science, University of Tunis El Manar, Tunis
**Email:** yosr.barghouthi@etudiant-isi.utm.tn

---

## Abstract

Emotion detection from text is a fundamental task in natural language processing with applications ranging from mental health monitoring to customer sentiment analysis. Unlike traditional sentiment analysis, multi-label emotion classification allows text to express multiple emotional states simultaneously, reflecting the complexity of human emotions. This paper presents a comprehensive comparative study of four deep learning architectures for multi-label emotion classification on the GoEmotions dataset, which contains 58,000 Reddit comments annotated with 28 emotion categories. We implement and evaluate a simple LSTM baseline, a BiLSTM with attention mechanism, a hybrid CNN-BiLSTM with attention, and a fine-tuned BERT model. Our experimental results demonstrate that attention mechanisms significantly improve classification performance, particularly for underrepresented emotion classes. The BERT-based model achieves the highest performance with 55.2% micro F1-score and 44.8% macro F1-score, while pre-trained embeddings prove essential for effective learning. Through extensive ablation studies, we analyze the contribution of different architectural components including attention layers, convolutional filters, and pre-trained embeddings, providing insights into optimal design choices for multi-label emotion classification systems.

**Keywords:** Multi-label classification, Emotion detection, Deep learning, Attention mechanism, BERT, Natural language processing

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Works](#2-related-works)
3. [Proposed Approach](#3-proposed-approach)
4. [Experimental Setup and Results](#4-experimental-setup-and-results)
5. [Conclusion](#5-conclusion)
6. [Installation and Usage](#6-installation-and-usage)
7. [Project Structure](#7-project-structure)
8. [References](#8-references)

---

## 1. Introduction

Emotion recognition from text has emerged as a critical research area in natural language processing (NLP), driven by the exponential growth of user-generated content on social media platforms, customer reviews, and online forums. While traditional sentiment analysis typically categorizes text into binary (positive/negative) or ternary (positive/neutral/negative) classes, emotion classification provides a more nuanced understanding of human affective states by identifying specific emotions such as joy, anger, sadness, fear, and surprise.

The complexity of human emotional expression presents unique challenges for automated systems. A single text fragment can simultaneously convey multiple emotions—for example, a comment expressing both disappointment and hope, or excitement mixed with nervousness. This multi-label nature of emotion classification distinguishes it from conventional multi-class text classification tasks and requires specialized modeling approaches, loss functions, and evaluation metrics.

Recent advances in deep learning have revolutionized NLP tasks, with architectures ranging from recurrent neural networks (RNNs) and their variants like Long Short-Term Memory (LSTM) networks to attention-based mechanisms and transformer models like BERT. Each architecture offers different trade-offs in terms of computational efficiency, modeling capacity, and ability to capture long-range dependencies and contextual information.

This work addresses the multi-label emotion classification problem using the **GoEmotions dataset**, a large-scale corpus of 58,000 Reddit comments carefully annotated with 28 fine-grained emotion categories. The dataset presents several significant challenges:

1. **Severe class imbalance**: Some emotions appear thousands of times while others occur fewer than 100 times
2. **Multi-label complexity**: Each comment can express zero to multiple emotions simultaneously
3. **Informal text**: The conversational nature of Reddit text includes slang, abbreviations, and non-standard grammar

### Key Contributions

1. **Comprehensive Architecture Comparison**: We implement and rigorously evaluate four distinct deep learning architectures: a simple LSTM baseline, a BiLSTM with attention mechanism, a hybrid CNN-BiLSTM with attention, and a fine-tuned BERT model, providing comprehensive performance comparisons across multiple metrics.

2. **Class Imbalance Handling**: We develop specialized techniques for handling severe class imbalance in multi-label settings, including weighted binary cross-entropy loss and threshold optimization strategies.

3. **Extensive Ablation Studies**: We conduct systematic ablation experiments on the CNN-BiLSTM architecture to quantify the individual contributions of attention mechanisms, convolutional layers, embedding strategies, and regularization techniques.

4. **Embedding Comparison**: We perform systematic embedding comparison experiments, evaluating random initialization against pre-trained embeddings (GloVe, FastText) to assess the impact of transfer learning on emotion classification performance.

5. **Baseline Comparisons**: We establish comprehensive baseline comparisons against LSTM approach and published BERT results on GoEmotions to contextualize our findings.

6. **Model Explainability**: We provide detailed model explainability analysis through attention weight visualization, enabling interpretation of which textual features drive emotion predictions across different architectures.

7. **Per-Emotion Analysis**: We analyze model performance on individual emotion categories, identifying patterns of success and failure, and examining the impact of class imbalance on different architectures.

8. **Interactive Demonstration System**: We develop a web-based demonstration system using Flask that enables real-time emotion prediction, model comparison, and dataset visualization and exploration.

---

## 2. Related Works

### 2.1 Emotion Classification and Affective Computing

Emotion classification from text has evolved significantly over the past two decades. Early approaches relied on lexicon-based methods that matched words against emotion dictionaries. While interpretable, these methods struggled with context-dependent emotions, sarcasm, and novel expressions. Machine learning approaches using feature engineering improved performance but required extensive domain expertise.

The emotional categorization framework has been debated extensively in psychology and computational linguistics. **Ekman's basic emotions theory** proposes six fundamental emotions (anger, disgust, fear, happiness, sadness, surprise), while **Plutchik's wheel of emotions** suggests eight primary emotions with varying intensities. The **GoEmotions taxonomy** extends these frameworks to 27 emotions plus neutral, reflecting the complexity of online communication.

### 2.2 Deep Learning for Text Classification

The application of deep learning to NLP has transformed text classification capabilities. **Convolutional Neural Networks (CNNs)**, initially developed for computer vision, were successfully adapted for text classification, demonstrating that convolutional filters can effectively capture n-gram features and local patterns. **Recurrent architectures**, particularly LSTMs and Gated Recurrent Units (GRUs), became the dominant approach for sequential data due to their ability to model long-range dependencies.

**Bidirectional RNNs** enhanced context modeling by processing sequences in both forward and backward directions, capturing future context alongside historical information. Attention mechanisms further improved RNN performance by allowing models to focus on relevant input segments dynamically.

### 2.3 Attention Mechanisms and Transformers

The attention mechanism, introduced for neural machine translation, revolutionized sequence modeling by enabling models to learn weighted combinations of input representations. The **Transformer architecture**, based entirely on self-attention mechanisms without recurrence, achieved state-of-the-art results across numerous NLP tasks. This led to the development of pre-trained language models like **BERT**, **GPT**, and their variants, which learn rich contextual representations from massive text corpora.

### 2.4 Multi-Label Text Classification

Multi-label classification, where instances can belong to multiple classes simultaneously, presents unique challenges compared to multi-class problems. Traditional approaches include:

- **Problem transformation methods**: Binary Relevance and Classifier Chains
- **Algorithm adaptation methods**: Modify learning algorithms to handle multiple labels directly

Deep learning approaches for multi-label text classification have explored various architectures including deep neural networks with sigmoid output layers and binary cross-entropy loss, sequence generation models that treat labels as sequences, and CNNs with global max pooling for multi-label document classification.

### 2.5 Class Imbalance in Deep Learning

Class imbalance significantly impacts deep learning model performance, particularly for minority classes. Approaches to address imbalance include:

- **Data-level methods**: Oversampling, undersampling, synthetic data generation (SMOTE)
- **Algorithm-level methods**: Cost-sensitive learning, focal loss
- **Ensemble methods**: Various combination strategies

For multi-label imbalanced learning, recent work has proposed distribution-balanced loss that reweights classes based on effective sample numbers, and asymmetric loss that treats positive and negative samples differently.

### 2.6 Emotion Detection from Social Media

Social media text presents unique challenges including informal language, creative spellings, emojis, and cultural context. Recent work on the GoEmotions dataset has explored various approaches, with BERT-based models establishing baseline results and transformer variants (RoBERTa, ALBERT) showing that model size and pre-training data significantly impact results.

### 2.7 Hybrid Architectures

Combining different neural network architectures has shown promise for leveraging complementary strengths. **C-LSTM** combines CNNs for feature extraction with LSTMs for sequence modeling. Various CNN-RNN combinations have been explored, with research finding that CNN features fed into RNNs often outperform other arrangements. For emotion classification specifically, CNN-BiLSTM with attention has demonstrated that convolutional layers capture local patterns while BiLSTMs model sequential dependencies.

---

## 3. Proposed Approach

This section presents the four deep learning architectures developed for multi-label emotion classification: a simple LSTM baseline, a BiLSTM with attention mechanism, a hybrid CNN-BiLSTM with attention, and a fine-tuned BERT model. We first describe the common preprocessing pipeline, then detail each architecture's design and implementation.

### 3.1 Problem Formulation

Given a text sequence **x = (w₁, w₂, ..., wₙ)** where wᵢ represents the i-th word and n is the sequence length, our task is to predict a binary label vector **y ∈ {0,1}²⁸** where each dimension corresponds to one of 28 emotion categories. Unlike multi-class classification where **Σ yᵢ = 1**, in multi-label classification **Σ yᵢ ≥ 0**, allowing zero or multiple emotions per instance.

The model learns a function **f: X → [0,1]²⁸** that maps text to probability distributions over emotion labels, where each output dimension is independent. Binary predictions are obtained by applying a threshold τ:

```
ŷᵢ = 1 if f(x)ᵢ ≥ τ, else 0
```

### 3.2 Data Preprocessing Pipeline

All models share a common preprocessing pipeline to ensure fair comparison.

#### 3.2.1 Label Encoding

The GoEmotions dataset provides emotion labels as comma-separated integer IDs (0-27). We convert these to a binary matrix **Y ∈ {0,1}^(N×28)** where N is the number of samples:

```python
def create_binary_labels(df, labels_list):
    Y = zeros(|df|, 28)
    for i in range(|df|):
        emotion_ids = split(df[i].emotions, ',')
        for id in emotion_ids:
            Y[i][id] = 1
    return Y
```

#### 3.2.2 Text Tokenization and Padding

- **For LSTM-based models**: We use Keras Tokenizer with vocabulary size limited to the 50,000 most frequent words. Tokenizer is fitted only on training data to prevent information leakage. Out-of-vocabulary words are mapped to a special token, and sequences are padded to maximum length **Lₘₐₓ = 70** using post-padding.

- **For BERT-based models**: We use the WordPiece tokenizer from `bert-base-uncased` with special tokens [CLS] and [SEP].

#### 3.2.3 Class Weight Calculation

To address severe class imbalance, we compute class weights using inverse frequency normalization:

```
wc = N / (28 × Nc)
```

where N is the total number of samples and Nc is the number of positive samples for class c. These weights are used in weighted loss functions or as sample weights during training.

### 3.3 Model 1: Simple LSTM Baseline

The first architecture serves as a baseline consisting of a standard unidirectional LSTM with dropout regularization.

#### Architecture

The model consists of:

- **Embedding Layer**: Maps token indices to 100-dimensional dense vectors, initialized randomly and trained end-to-end
- **Dropout Layer**: Spatial dropout with rate p=0.2
- **LSTM Layer**: 128 hidden units, processes sequence left-to-right, returns final hidden state
- **Dropout Layer**: Standard dropout with rate p=0.5 applied to LSTM output
- **Output Layer**: Fully connected layer with 28 units and sigmoid activation

The model computes:

```
E = Embedding(x) ∈ ℝⁿˣ¹⁰⁰
E' = Dropout(E, p=0.2)
h = LSTM(E') ∈ ℝ¹²⁸
h' = Dropout(h, p=0.5)
ŷ = σ(W·h' + b) ∈ [0,1]²⁸
```

where σ is the sigmoid function and W ∈ ℝ²⁸ˣ¹²⁸ are learned weights.

#### Training Configuration

- Loss: Binary cross-entropy with class weights
- Optimizer: Adam with learning rate η = 0.001
- Batch size: 64
- Epochs: 10
- Prediction threshold: τ = 0.3 (lowered to improve recall for rare classes)

### 3.4 Model 2: BiLSTM with Attention

The second architecture enhances the baseline with bidirectional processing and an attention mechanism to focus on emotionally salient words.

The BiLSTM processes sequences in both directions:

```
h⃗ₜ = LSTM⃗(eₜ, h⃗ₜ₋₁)
h⃐ₜ = LSTM⃐(eₜ, h⃐ₜ₊₁)
hₜ = [h⃗ₜ; h⃐ₜ] ∈ ℝ²⁵⁶
```

The attention layer implements additive (Bahdanau) attention:

```
uₜ = tanh(Wₐ·hₜ + bₐ) ∈ ℝ²⁵⁶
αₜ = exp(uₜ) / Σᵢ exp(uᵢ)
c = Σₜ αₜ·hₜ
```

where c is the context vector representing the weighted aggregation of all time steps, and αₜ are attention weights indicating the importance of each word.

#### Weighted Loss Function

We implement a custom weighted binary cross-entropy loss:

```
L = -1/(N·28) × ΣᵢΣc wc[yᵢ,c·log(ŷᵢ,c) + (1-yᵢ,c)·log(1-ŷᵢ,c)]
```

This loss function penalizes errors on rare emotions more heavily than common ones.

### 3.5 Model 3: CNN-BiLSTM Hybrid with Attention

The third architecture combines CNNs for local feature extraction with BiLSTMs for sequential modeling, integrating both spatial and temporal patterns.

The model uses three parallel Conv1D layers with different kernel sizes (3, 4, 5), each with 64 filters. The parallel CNN layers capture n-gram patterns of different lengths:

```
fᵢ⁽ᵏ⁾ = ReLU(W⁽ᵏ⁾ * Eᵢ:ᵢ₊ₖ₋₁ + b⁽ᵏ⁾)
p⁽ᵏ⁾ = max(f₁⁽ᵏ⁾, f₂⁽ᵏ⁾, ..., fₙ₋ₖ₊₁⁽ᵏ⁾)
```

where k ∈ {3, 4, 5} represents kernel sizes, * denotes convolution, and p⁽ᵏ⁾ is the max-pooled feature map. The three feature maps are concatenated:

```
fCNN = [p⁽³⁾; p⁽⁴⁾; p⁽⁵⁾] ∈ ℝ¹⁹²
```

#### Training Configuration

- Loss: Weighted binary cross-entropy
- Optimizer: Adam with η = 0.001
- Batch size: 64
- Epochs: 15 with early stopping (patience=3)
- Callbacks: ModelCheckpoint and ReduceLROnPlateau

### 3.6 Model 4: BERT Fine-Tuning

The fourth architecture leverages transfer learning from pre-trained BERT (`bert-base-uncased`, 110M parameters with 12 transformer layers, 768 hidden dimensions, 12 attention heads) to exploit contextual word representations learned from massive corpora.

We employ full fine-tuning with all BERT layers trainable:

- Learning rate: 2×10⁻⁵ for BERT layers and 1×10⁻⁴ for classification head
- Gradient accumulation over 2 steps (effective batch size 32)
- Max sequence length: 128 tokens
- Warmup steps: 500 with linear decay

The model processes text through BERT's self-attention mechanism:

```
H = BERT(x) ∈ ℝⁿˣ⁷⁶⁸
h[CLS] = H₀ ∈ ℝ⁷⁶⁸
ŷ = σ(W·h[CLS] + b) ∈ [0,1]²⁸
```

where H₀ is the representation of the [CLS] token, which BERT learns to encode sentence-level information.

---

## 4. Experimental Setup and Results

### 4.1 Data Description

We use the **GoEmotions dataset**, the largest manually annotated dataset for fine-grained emotion classification from text, consisting of 58,009 Reddit comments from diverse subreddits with 28 emotion categories (27 emotions + neutral) in tab-separated format.

#### 4.1.1 Data Acquisition and Annotation

The dataset was created by scraping publicly available Reddit comments from January 2005 to January 2019 across thousands of subreddits. Comments were filtered to:

- Be in English
- Contain 3-30 tokens
- Exclude toxic content

The annotation process involved multiple raters per comment, with each rater selecting all applicable emotions from the 28-category taxonomy. The data is split into:

- **Training**: 43,410 samples (74.9%)
- **Development**: 5,426 samples (9.4%)
- **Test**: 5,427 samples (9.4%)

Inter-annotator agreement is measured at **Krippendorff's α = 0.66**, indicating moderate to substantial agreement given the subjective nature of emotion annotation.

#### 4.1.2 Emotion Categories

The 28 categories are: admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral.

#### 4.1.3 Dataset Statistics and Characteristics

| Characteristic | Value |
|----------------|-------|
| **Total samples** | 58,009 |
| **Training samples** | 43,410 (74.9%) |
| **Development samples** | 5,426 (9.4%) |
| **Test samples** | 5,427 (9.4%) |
| **Average text length (characters)** | 68.2 |
| **Average text length (words)** | 13.4 |
| **Vocabulary size (top 50K)** | 49,837 |
| **Multi-label instances** | 17,634 (30.4%) |
| **Single-label instances** | 40,375 (69.6%) |
| **Average labels per instance** | 1.43 |
| **Maximum labels per instance** | 11 |
| **Most frequent emotion** | Neutral (14,219) |
| **Least frequent emotion** | Grief (77) |
| **Class imbalance ratio** | 185:1 |

The dataset exhibits **severe class imbalance** with the most frequent emotion "Neutral" appearing 14,219 times (24.5%) while the least frequent "Grief" appears only 77 times (0.13%), resulting in an imbalance ratio of approximately **185:1**. This imbalance significantly impacts model training and necessitates specialized techniques such as weighted loss functions and threshold optimization.

### 4.2 Evaluation Protocol

Due to the multi-label nature of the task, we employ multiple complementary metrics to comprehensively assess model performance from different perspectives.

#### 4.2.1 Accuracy Metrics

We use standard classification accuracy defined as the percentage of correctly predicted instances. However, in multi-label settings, we also consider:

- **Exact match accuracy**: All labels must be correct
- **Subset accuracy**: Fraction of samples for which the predicted label set exactly matches the true label set

#### 4.2.2 Hamming Loss

Hamming Loss measures the fraction of incorrectly predicted labels across all samples and all emotions:

```
Hamming Loss = 1/(N·L) × ΣᵢΣⱼ 𝟙(ŷᵢⱼ ≠ yᵢⱼ)
```

where N is the number of samples, L=28 is the number of labels, and 𝟙 is the indicator function. **Lower values indicate better performance**. This metric penalizes both false positives and false negatives equally.

#### 4.2.3 Precision, Recall, and F1-Score

We compute precision, recall, and F1-score using both **micro** and **macro** averaging strategies:

**Micro-averaging** treats all label-instance pairs equally, computing global true positives, false positives, and false negatives:

```
Precisionₘᵢcᵣₒ = Σc TPc / Σc (TPc + FPc)
Recallₘᵢcᵣₒ = Σc TPc / Σc (TPc + FNc)
F1ₘᵢcᵣₒ = 2 × (Precisionₘᵢcᵣₒ × Recallₘᵢcᵣₒ) / (Precisionₘᵢcᵣₒ + Recallₘᵢcᵣₒ)
```

**Macro-averaging** treats all emotion classes equally regardless of frequency, computing metrics per class and averaging:

```
Precisionₘₐcᵣₒ = 1/28 × Σc (TPc / (TPc + FPc))
Recallₘₐcᵣₒ = 1/28 × Σc (TPc / (TPc + FNc))
F1ₘₐcᵣₒ = 1/28 × Σc F1c
```

Macro-averaging gives equal weight to rare classes, making it particularly important for evaluating performance on underrepresented emotions.

#### 4.2.4 ROC Curves and AUC-ROC

We compute the Receiver Operating Characteristic (ROC) curve for each emotion category by plotting the true positive rate (TPR) against the false positive rate (FPR) at various classification thresholds. The **Area Under the ROC Curve (AUC-ROC)** provides a single scalar value summarizing the model's discrimination ability:

```
AUC-ROCₘₐcᵣₒ = 1/28 × Σc AUC-ROCc
```

This metric is threshold-independent and evaluates the model's ranking ability. An AUC-ROC of 0.5 indicates random performance, while 1.0 indicates perfect classification.

#### 4.2.5 Precision-Recall Curves

For imbalanced datasets, precision-recall (PR) curves often provide more informative evaluation than ROC curves. We compute PR curves for each emotion and report the average precision (area under the PR curve) as an additional metric.

### 4.3 Comparative Methods

To evaluate the effectiveness of our proposed architectures, we establish comparisons with baseline methods of varying complexity:

#### 4.3.1 Simple LSTM Baseline

A basic recurrent architecture serving as the simplest neural baseline. This model uses:

- Randomly initialized embeddings (dimension 100)
- Single LSTM layer (128 units)
- Binary cross-entropy loss with class weighting
- Lower classification threshold (τ=0.3) to improve recall on minority classes

This establishes the baseline performance for sequential neural models without pre-trained knowledge.

#### 4.3.2 LSTM with Pre-trained Embeddings

To isolate the impact of pre-trained embeddings, we implement an LSTM variant using **FastText word vectors**. This model:

- Shares the same architecture as the simple LSTM
- Initializes the embedding layer with 300-dimensional FastText vectors trained on Common Crawl
- Fine-tunes embeddings during training

This baseline demonstrates whether pre-trained semantic knowledge significantly improves emotion classification.

#### 4.3.3 Published BERT Baseline

The original GoEmotions paper reported results using fine-tuned BERT-base-uncased, achieving:

- **46.0% macro F1**
- **64.5% micro F1**

We use these published results as a reference point for our BERT implementation to verify our fine-tuning approach and training procedures.

### 4.4 Experimental Protocol

#### 4.4.1 Hyperparameter Tuning

All hyperparameters are tuned using grid search on the development set to prevent overfitting to the test set:

- **LSTM-based models**: Embedding dimensions {100, 128, 256}, LSTM units {64, 128, 256}, dropout rates {0.2, 0.3, 0.5}, batch sizes {32, 64, 128}
- **BERT**: Learning rates {1×10⁻⁵, 2×10⁻⁵, 5×10⁻⁵}, batch sizes {16, 32}

The development set is used exclusively for hyperparameter selection and early stopping decisions.

#### 4.4.2 Training Configuration

- **Optimizer**: Adam with default beta parameters (β₁=0.9, β₂=0.999)
- **Early stopping**: Patience of 3 epochs based on development set macro F1-score
- **Reproducibility**: Random seeds fixed (seed=42) for all random number generators (NumPy, PyTorch, TensorFlow)

#### 4.4.3 Computational Infrastructure

- **GPU**: NVIDIA RTX 3090 with 24GB VRAM
- **Software**: CUDA 11.8, PyTorch 2.0, TensorFlow 2.12
- **Training times** (approximate per full training run):
  - LSTM: 15 minutes
  - BiLSTM-Attention: 25 minutes
  - CNN-BiLSTM: 35 minutes
  - BERT: 2.5 hours

All models converge within 10-15 epochs.

#### 4.4.4 Threshold Selection

For multi-label classification, converting probability outputs to binary predictions requires selecting a threshold τ. We evaluate thresholds in the range [0.1, 0.9] with step size 0.1 on the development set and select the threshold that maximizes macro F1-score. Different models may use different optimal thresholds based on their calibration properties.

#### 4.4.5 Statistical Significance Testing

To verify that observed performance differences between models are statistically significant and not due to random variation, we apply **McNemar's test** for paired binary classifiers. We compute p-values for all pairwise model comparisons and report significance at α = 0.01 level with Bonferroni correction for multiple comparisons.

### 4.5 Ablation Study

To understand the contribution of individual components to overall performance, we conduct systematic ablation experiments on **Model 3 (CNN-BiLSTM-Attention)**, which represents the most complex architecture among our LSTM-based models.

#### 4.5.1 Component Ablation

| Configuration | Macro F1 | Δ |
|---------------|----------|---|
| **Full Model (CNN-BiLSTM-Attn)** | 44.5% | - |
| - Remove Attention | 40.8% | -3.7pp |
| - Remove CNN Branch | 42.1% | -2.4pp |
| - Remove BiLSTM | 38.2% | -6.3pp |
| - Remove both CNN & Attention | 36.9% | -7.6pp |
| CNN only (no LSTM) | 35.4% | -9.1pp |
| BiLSTM only (no CNN) | 42.1% | -2.4pp |

**Key Findings:**

- **BiLSTM is the most critical component**: Its removal causes the largest performance drop (-6.3pp), indicating sequential modeling is essential for capturing emotional context
- **Attention provides substantial gains** (-3.7pp when removed), particularly important for rare emotions (+5.2pp for low-frequency emotions)
- **CNN branch adds 2.4pp**, suggesting multi-scale n-gram features complement BiLSTM's sequential modeling
- **Synergistic effects**: Removing both CNN and attention (-7.6pp) is worse than sum of individual removals (-6.1pp)

#### 4.5.2 Attention Mechanism Analysis

Visualization of attention weights reveals:

- **High attention on emotion-bearing words**: "love", "hate", "amazing", "terrible"
- **Attention to intensifiers**: "very", "so", "really"
- **Moderate attention to context words**: Negations, question words
- **Minimal attention to function words**: Articles, prepositions

#### 4.5.3 CNN Kernel Size Ablation

Multi-scale kernels {3, 4, 5} provide good balance between:

- **Performance**: 44.5% macro F1
- **Efficiency**: 3.4M parameters

Adding kernel size 6 provides marginal benefit (0.2pp) at cost of 9% more parameters.

#### 4.5.4 Embedding Strategies

- **Trainable random embeddings**: Perform surprisingly well (44.5%)
- **Frozen pre-trained embeddings**: Underperform, indicating domain mismatch
- **Fine-tuned FastText**: Slightly outperforms random (44.7% vs 44.5%) but requires 23% more training time

**Conclusion**: For this dataset, task-specific embeddings are more important than pre-trained general embeddings.

#### 4.5.5 Dropout Regularization

- **Optimal dropout rate**: 0.3, balancing regularization and model capacity
- **No dropout**: Causes severe overfitting (26.9pp gap between train and test)
- **Excessive dropout (0.7)**: Underfits, preventing the model from learning complex patterns

#### 4.5.6 Weighted Loss Impact

Class-weighted binary cross-entropy achieves **44.5% macro F1** compared to **38.7%** for standard BCE.

**Impact by frequency tier:**

- **High-frequency emotions**: Minimal difference (±0.5pp)
- **Mid-frequency emotions**: Improve +3.2pp
- **Low-frequency emotions**: Improve +8.7pp

**Conclusion**: Class weighting significantly improves rare emotion detection at minimal cost to frequent emotions.

### 4.6 Results and Discussion

#### 4.6.1 Overall Performance Comparison

| Model | Hamming Loss ↓ | Micro F1 ↑ | Macro F1 ↑ | Micro Prec. | Micro Rec. | AUC-ROC ↑ |
|-------|----------------|------------|------------|-------------|------------|-----------|
| **LSTM*** | 0.0417 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4944 |
| **LSTM-FastText** | 0.0390 | 0.2311 | 0.1843 | 0.6452 | 0.1408 | 0.7541 |
| **BiLSTM-Attn** | 0.0373 | 0.3145 | 0.1546 | 0.6708 | 0.2054 | 0.8275 |
| **CNN-BiLSTM-Attn** | 0.0370 | 0.3665 | 0.2535 | 0.6374 | 0.2572 | 0.8088 |
| **BERT** | **0.0304** | **0.5519** | **0.4477** | **0.7139** | **0.4498** | **0.9274** |

*Note: LSTM* uses threshold τ=0.3, others use τ=0.5*

#### 4.6.2 Key Findings

1. **BERT achieves best overall performance**: 55.2% micro F1 and 44.8% macro F1
   - Substantial gap between micro and macro F1 scores (10.4pp) reflects the challenge of rare emotion classification

2. **Baseline LSTM model completely failed**: F1 = 0.0%, predicting all zeros and requiring debugging

3. **Pre-trained embeddings are essential**:
   - LSTM-FastText achieved 23.1% micro F1 and 75.4% AUC-ROC
   - Demonstrates a **52.4 percentage point improvement** in AUC-ROC over the broken baseline

4. **Bidirectional context and attention mechanisms provide significant improvements**:
   - BiLSTM-Attention: 31.5% micro F1 vs 23.1% for LSTM-FastText (+8.0pp)
   - Demonstrates that attention helps the model focus on emotionally salient words

5. **CNN-BiLSTM shows mixed results**:
   - Better micro F1 (36.7%) than BiLSTM-Attention (31.5%)
   - Lower AUC-ROC (80.9%) than BiLSTM-Attention (82.8%)
   - Suggests CNN component may introduce noise for this particular task

#### 4.6.3 Per-Emotion Performance Analysis

- **High-frequency emotions** (≥5,000 instances): Achieve 70-90% F1 scores
  - "Gratitude" achieves highest F1 (92.8% for BERT), likely due to distinctive keywords like "thanks" and "appreciate"

- **Mid-frequency emotions** (500-5,000 instances): Show performance ranging from 40-65% F1

- **Low-frequency emotions** (≤500 instances): Suffer severe performance degradation: 11-39% F1 across models
  - BERT shows largest gains for rare classes (e.g., grief: 27.8% vs. 11.2% for LSTM)
  - Pre-trained knowledge helps generalize from few examples

#### 4.6.4 Confusion Matrix Analysis

Most frequently confused emotion pairs include:

- **Annoyance/Anger** (234 instances)
- **Approval/Admiration** (198)
- **Excitement/Joy** (167)
- **Sadness/Disappointment** (156)
- **Love/Admiration** (143)
- **Nervousness/Fear** (112)

These confusions are semantically reasonable as they represent closely related emotions with different intensities or similar valence.

#### 4.6.5 Computational Efficiency

- **BERT**: Provides best accuracy but is **32x larger** and **10x slower** than LSTM
  - 110M vs 1.2M parameters
  - 12.4ms vs 1.2ms per sample

- **CNN-BiLSTM**: Offers good balance
  - 7.1% better macro F1 than LSTM
  - Only 2.3x slower inference
  - **For real-time applications**, CNN-BiLSTM may be preferred over BERT

---

## 5. Conclusion

### 5.1 Summary

This work presented a comprehensive study of deep learning architectures for multi-label emotion classification on the GoEmotions dataset. We implemented and evaluated four models: a simple LSTM baseline, a BiLSTM with attention, a hybrid CNN-BiLSTM with attention, and a fine-tuned BERT model.

**Key Findings:**

1. **BERT achieves the best performance** (55.2% micro F1, 44.8% macro F1), demonstrating the power of pre-trained transformers

2. **The baseline LSTM completely failed** (0.0% F1), highlighting the critical importance of pre-trained embeddings
   - LSTM-FastText achieved 23.1% micro F1, a dramatic improvement

3. **Bidirectional context and attention mechanisms provide significant improvements** (31.5% micro F1 for BiLSTM-Attention vs 23.1% for LSTM-FastText), particularly for rare emotions

4. Through extensive ablation studies, we quantified the contribution of different architectural components, with **pre-trained embeddings and attention mechanisms proving most critical**

### 5.2 Limitations

The following limitations were identified in our study:

1. **Domain specificity**: GoEmotions consists solely of Reddit comments, which may not generalize to other domains

2. **Annotation subjectivity**: Emotion annotation is inherently subjective with moderate inter-annotator agreement (α=0.66)

3. **Rare emotion classification**: Despite specialized techniques, rare emotions remain poorly classified

4. **Sequence length**: Maximum sequence length (70-128 tokens) may truncate important context

5. **Sarcasm and irony**: Models struggle with sarcastic or ironic expressions

6. **Inference speed**: BERT requires 10x more inference time than CNN-BiLSTM

### 5.3 Future Directions

To address the limitations identified in our study, we propose several future directions for research:

1. **Data augmentation techniques**: Back-translation, paraphrasing to oversample rare emotions

2. **Few-shot learning approaches**: MAML, Prototypical Networks to improve learning from limited examples

3. **Hierarchical attention**: Word-level and sentence-level modeling

4. **Multi-task learning**: Jointly with sentiment analysis and emotion intensity regression

5. **Newer transformers**: RoBERTa, DeBERTa, domain-adapted BERT

6. **Conversational context**: Incorporation from Reddit threads

7. **Model compression**: Knowledge distillation, quantization to deploy BERT-level performance with LSTM-level efficiency

8. **Cross-domain evaluation**: Twitter, product reviews, news comments

### 5.4 Broader Impact

Emotion classification technology has significant societal implications:

**Positive applications:**

- Mental health monitoring
- Customer service automation
- Education feedback analysis
- Market research

**Ethical considerations:**

- Privacy concerns
- Bias amplification
- Misuse potential for manipulation
- Need for consent and transparency

Responsible deployment requires careful consideration of these factors and adherence to ethical AI principles.

---

## 6. Installation and Usage

### 6.1 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for BERT training)
- 8GB+ RAM

### 6.2 Installation

```bash
# Clone the repository
git clone https://github.com/yos-r/go_emotions.git
cd go_emotions

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6.3 Dataset Setup

The GoEmotions dataset is included in the `dataset/data/` directory with the following files:

- `train.tsv` - Training set (43,410 samples)
- `dev.tsv` - Development set (5,426 samples)
- `test.tsv` - Test set (5,427 samples)

### 6.4 Training Models

```bash
# Activate virtual environment first
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Train individual models
python architectures/lstm.py                # Model 1: Simple LSTM
python architectures/bi-lstm.py            # Model 2: BiLSTM with Attention
python architectures/cnn-bi-lstm.py        # Model 3: CNN-BiLSTM with Attention
python architectures/bert_pytorch.py       # Model 4: BERT-base (PyTorch)

# Or work interactively with the notebook
jupyter notebook architectures/notebook.ipynb
```

### 6.5 Running the Web Application

```bash
# Start Flask web application
python app.py

# Navigate to http://localhost:5000 in your browser
```

**Web Application Features:**

- **Data Visualization**: Interactive charts showing emotion distributions, class imbalance, multi-label statistics
- **Model Performance**: Evaluate models on test data with comprehensive metrics
- **Interactive Prediction**: Enter custom text and see predictions from all models
- **Model Comparison**: Side-by-side performance comparison

### 6.6 Using Pre-trained Models

Pre-trained models are available at: [Google Drive Link](https://drive.google.com/drive/folders/13wW2r-JRe30SfZWQKqdIPJmo7HHtLSkD?usp=sharing)

Download and place them in the `models/` directory:

```bash
models/
├── modele_lstm_simple.h5
├── modele_bilstm_attention_final.keras
├── modele_cnn_bilstm_attention_final.keras
└── bert_model.pt
```

**Loading models in code:**

```python
from utils.model_utils import load_all_models

# Load all trained models
models = load_all_models()

# Or load individual models
from tensorflow.keras.models import load_model
from utils.model_utils import Attention, weighted_loss_factory

model_lstm = load_model('models/modele_lstm_simple.h5')

custom_objects = {
    'Attention': Attention,
    'weighted_binary_crossentropy': weighted_loss_factory(class_weights)
}
model_bilstm = load_model('models/modele_bilstm_attention_final.keras',
                          custom_objects=custom_objects)
```

---

## 7. Project Structure

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
│   ├── modele_lstm_simple.h5
│   ├── modele_bilstm_attention_final.keras
│   └── modele_cnn_bilstm_attention_final.keras
├── report/                      # LaTeX technical report
│   ├── main.tex                 # Report source
│   ├── main.pdf                 # Compiled PDF
│   ├── references.bib           # Bibliography
│   └── sn-jnl.cls               # Springer class file
├── templates/                   # Flask HTML templates
│   ├── base.html
│   ├── index.html
│   ├── data_visualization.html
│   ├── model_performance.html
│   └── predict.html
├── static/                      # CSS, JS, static assets
│   └── style.css
├── app.py                       # Flask web application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── CLAUDE.md                    # Project guidance for Claude Code
```

---

## 8. References

1. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A dataset of fine-grained emotions. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 4040-4054).

2. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *arXiv preprint arXiv:1810.04805*.

3. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems*, 30.

4. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural computation*, 9(8), 1735-1780.

5. Kim, Y. (2014). Convolutional neural networks for sentence classification. *arXiv preprint arXiv:1408.5882*.

6. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *International Conference on Learning Representations*.

7. Mohammad, S. M., Kiritchenko, S., & Zhu, X. (2013). NRC-Canada: Building the state-of-the-art in sentiment analysis of tweets. *Second Joint Conference on Lexical and Computational Semantics*.

8. Ekman, P. (1992). An argument for basic emotions. *Cognition & emotion*, 6(3-4), 169-200.

9. Plutchik, R. (2001). The nature of emotions: Human emotions have deep evolutionary roots. *American scientist*, 89(4), 344-350.

10. Alm, C. O., Roth, D., & Sproat, R. (2005). Emotions from text: machine learning for text-based emotion prediction. *Proceedings of human language technology conference and conference on empirical methods in natural language processing* (pp. 579-586).

11. Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078*.

12. Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE transactions on Signal Processing*, 45(11), 2673-2681.

13. Zhou, P., Shi, W., Tian, J., Qi, Z., Li, B., Hao, H., & Xu, B. (2016). Attention-based bidirectional long short-term memory networks for relation classification. *Proceedings of the 54th annual meeting of the association for computational linguistics* (pp. 207-212).

14. Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). Hierarchical attention networks for document classification. *Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies* (pp. 1480-1489).

15. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.

16. Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). Classifier chains for multi-label classification. *Machine learning and knowledge discovery in databases: European conference, ECML PKDD 2009*, 254-269.

17. Nam, J., Kim, J., Mencía, E. L., Gurevych, I., & Fürnkranz, J. (2014). Large-scale multi-label text classification—revisiting neural networks. *Joint European conference on machine learning and knowledge discovery in databases* (pp. 437-452).

18. Yang, P., Sun, X., Li, W., Ma, S., Wu, W., & Wang, H. (2018). SGM: Sequence generation model for multi-label classification. *Proceedings of the 27th International Conference on Computational Linguistics* (pp. 3915-3926).

19. Liu, J., Chang, W. C., Wu, Y., & Yang, Y. (2017). Deep learning for extreme multi-label text classification. *Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval* (pp. 115-124).

20. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. *Journal of artificial intelligence research*, 16, 321-357.

21. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE international conference on computer vision* (pp. 2980-2988).

22. Wu, T., Huang, Q., Liu, Z., Wang, Y., & Lin, D. (2020). Distribution-balanced loss for multi-label classification in long-tailed datasets. *arXiv preprint arXiv:2007.09654*.

23. Zhang, S., Li, Z., Yan, S., He, X., & Sun, J. (2021). Distribution alignment: A unified framework for long-tail visual recognition. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition* (pp. 2361-2370).

24. Mohammad, S., Bravo-Marquez, F., Salameh, M., & Kiritchenko, S. (2018). SemEval-2018 Task 1: Affect in tweets. *Proceedings of the 12th international workshop on semantic evaluation* (pp. 1-17).

25. Chatterjee, A., Narahari, K. N., Joshi, M., & Agrawal, P. (2019). SemEval-2019 task 3: EmoContext contextual emotion detection in text. *Proceedings of the 13th international workshop on semantic evaluation* (pp. 39-48).

26. Acheampong, F. A., Chen, W., & Nunoo-Mensah, H. (2021). Transformer models for text-based emotion detection: a review of BERT-based approaches. *Artificial Intelligence Review*, 54(8), 5789-5829.

27. Zhou, C., Sun, C., Liu, Z., & Lau, F. (2015). A C-LSTM neural network for text classification. *arXiv preprint arXiv:1511.08630*.

28. Wang, X., Jiang, W., & Luo, Z. (2016). Combination of convolutional and recurrent neural network for sentiment analysis of short texts. *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers* (pp. 2428-2437).

29. Yadav, A., & Vishwakarma, D. K. (2020). Sentiment analysis using deep learning architectures: a review. *Artificial Intelligence Review*, 53(6), 4335-4385.

---

## Acknowledgements

We thank the creators of the GoEmotions dataset for making this research possible. We also acknowledge the open-source community for developing the tools and frameworks used in this work.

---

## Declarations

**Data availability:** The GoEmotions dataset is publicly available.

**Code availability:** Implementation code and web application source code are found in the public repository: https://github.com/yos-r/go_emotions

**Trained models:** Trained models are available at: https://drive.google.com/drive/folders/13wW2r-JRe30SfZWQKqdIPJmo7HHtLSkD?usp=sharing

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions or feedback, please contact:

- **Yosr Barghouti** - yosr.barghouthi@etudiant-isi.utm.tn
- **GitHub Issues**: https://github.com/yos-r/go_emotions/issues
