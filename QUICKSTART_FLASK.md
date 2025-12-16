# Quick Start Guide - GoEmotions Flask App

## 🚀 Get Started in 3 Steps

### Step 1: Install Flask Dependencies
```bash
# Activate your virtual environment
.\venv\Scripts\activate

# Install Flask and Plotly
pip install flask plotly
```

### Step 2: Start the Application
```bash
python app.py
```

You should see:
```
Loading models...
✅ Loaded LSTM model from models/modele_lstm_simple.h5
✅ Loaded BiLSTM+Attention model from models/modele_bilstm_attention_final.keras
✅ Loaded CNN-BiLSTM+Attention model from models/modele_cnn_bilstm_attention_final.keras
Loading tokenizer...
Computing dataset statistics...
App initialized successfully!
 * Running on http://0.0.0.0:5000
```

### Step 3: Open Your Browser
Navigate to: **http://localhost:5000**

---

## 📋 What You Can Do

### 1. Explore Dataset (Data Visualization Page)
- View 58,000 Reddit comments statistics
- See emotion distribution across 28 categories
- Understand class imbalance (grief: 77 samples vs neutral: 14,219)
- Analyze multi-label patterns

### 2. Predict Emotions (Predict Page)
**Try it now:**
1. Click "Predict" in the navigation
2. Type: *"I'm so happy and excited about this amazing news!"*
3. Click "Predict Emotions"
4. See results from all 3 models (LSTM, BiLSTM+Attention, CNN-BiLSTM+Attention)

**You'll see:**
- Emotions detected above threshold
- Top 5 emotions by probability
- Full probability distribution chart
- Comparison table across models

### 3. Evaluate Models (Model Performance Page)
1. Click "Model Performance"
2. Click "Evaluate All Models" (takes ~2-3 minutes)
3. View comprehensive metrics:
   - Hamming Loss
   - AUC-ROC (macro)
   - F1-Score (micro & macro)
   - Precision & Recall
   - Per-emotion performance

### 4. Compare Models (Compare Page)
1. Click "Compare Models"
2. Click "Run Comparison" (tests on 100 samples)
3. See rankings and performance charts
4. Review architecture comparison
5. Read recommendations for deployment

---

## 🎯 Example Use Cases

### Use Case 1: Test Custom Text
**Scenario:** You want to see how your models classify different emotions

**Steps:**
1. Go to "Predict" page
2. Enter your text
3. Compare predictions across models

**Example texts to try:**
- *"I'm really confused and not sure what to think about this."*
- *"Thank you so much for your help! I really appreciate it."*
- *"This is absolutely disgusting and makes me angry."*

### Use Case 2: Find Best Model
**Scenario:** You need to choose which model to deploy

**Steps:**
1. Go to "Model Performance"
2. Click "Evaluate All Models"
3. Compare metrics (AUC-ROC is best for multi-label)
4. Check per-emotion performance for rare classes
5. Go to "Compare Models" for recommendations

**Expected Results:**
- CNN-BiLSTM+Attention typically has best performance
- BiLSTM+Attention has good balance of speed/accuracy
- Simple LSTM is fastest but lower accuracy

### Use Case 3: Understand Dataset
**Scenario:** You're writing your technical report

**Steps:**
1. Go to "Data Visualization"
2. Take screenshots of:
   - Emotion distribution chart
   - Class imbalance visualization
   - Dataset split pie chart
3. Note statistics:
   - Average labels per sample: ~2.7
   - Imbalance ratio: ~184:1
   - Most common: neutral, approval, admiration
   - Least common: grief, relief, pride

---

## ⚡ Keyboard Shortcuts & Tips

### Navigation Tips
- **Home (/)** - Overview of project
- **Ctrl+Click** links - Open in new tab
- Use browser back button to return to previous page

### Prediction Tips
- Click "Load Example" for random sample text
- Results update in tabs - switch between models easily
- Scroll down for comparison table

### Performance Tips
- First load takes 30-60 seconds (models loading into memory)
- Subsequent predictions are instant
- "Compare Models" with 100 samples is faster than full evaluation

---

## 🐛 Troubleshooting

### Problem: "No models found!"
**Solution:** Check that model files exist in `models/` directory

### Problem: Dataset not found
**Solution:** Verify `dataset/data/train.tsv` exists

### Problem: Page loads but no charts
**Solution:** Check browser console for JavaScript errors, ensure internet connection (for CDN resources)

### Problem: Slow performance
**Solution:** This is normal for first load. Models are large (~80MB each for BiLSTM/CNN models)

---

## 📊 Understanding the Metrics

### Hamming Loss (Lower is Better)
- Fraction of labels incorrectly predicted
- Expect: 0.02-0.05
- **Good**: < 0.04

### AUC-ROC Macro (Higher is Better)
- Best metric for imbalanced multi-label tasks
- Expect: 0.80-0.90
- **Good**: > 0.85

### F1-Score Micro vs Macro
- **Micro**: Favors common emotions (weighted by frequency)
- **Macro**: Treats all emotions equally (sensitive to rare classes)
- **Good Micro**: > 0.60
- **Good Macro**: > 0.45

---

## 🎨 Emotion Colors Guide

The app uses consistent colors for emotions:
- **Positive**: Green shades (joy, gratitude, love, excitement)
- **Negative**: Red shades (anger, disgust, sadness, fear)
- **Neutral/Mixed**: Blue/Gray shades (confusion, curiosity, neutral)

---

## 📝 Next Steps

After exploring the Flask app:

1. **Take Screenshots** for your report
2. **Document Results** - Note which model performs best
3. **Test Edge Cases** - Try very short texts, long texts, ambiguous emotions
4. **Compare Thresholds** - Note LSTM uses 0.3, others use 0.5
5. **Analyze Errors** - Check per-emotion metrics for problematic categories

---

## 💡 Pro Tips

1. **Keep the app running** while working on your report - easier to grab screenshots
2. **Use the comparison table** in prediction results - great for report figures
3. **Export browser charts** - Right-click Plotly charts to save as PNG
4. **Test with different text lengths** - Models perform differently on short vs long texts
5. **Check rare emotions** - grief, relief, pride are hardest to predict

---

## 🔗 Quick Links

- Full documentation: [FLASK_README.md](FLASK_README.md)
- Model details: [CLAUDE.md](CLAUDE.md)
- Project specification: [ENONCE.txt](ENONCE.txt)

---

Enjoy exploring your GoEmotions models! 🎉
