# GoEmotions Flask Web Application

A comprehensive web interface for exploring and using the GoEmotions multi-label emotion classification models.

## Features

### 🎨 Data Visualization
- Dataset statistics (train/dev/test splits)
- Emotion distribution charts
- Class imbalance visualization (log scale)
- Multi-label statistics
- Text length analysis

### 📊 Model Performance
- Evaluate individual models on test data
- Comprehensive metrics:
  - Hamming Loss
  - AUC-ROC (macro)
  - F1-Score (micro & macro)
  - Precision & Recall (micro & macro)
- Per-emotion performance breakdown
- Side-by-side model comparison

### 🧠 Interactive Prediction
- Enter custom text for emotion prediction
- See predictions from all 4 models simultaneously
- Visual comparison of emotion probabilities
- Top-5 emotions for each model
- Threshold-based emotion detection

### ⚖️ Model Comparison
- Quick comparison on 100 test samples
- Performance metrics visualization
- Architecture comparison table
- Strengths & weaknesses analysis
- Deployment recommendations

## Installation

### 1. Activate Virtual Environment

```bash
# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install flask plotly
# Or install all dependencies:
pip install -r requirements.txt
```

## Running the Application

### Start the Flask Server

```bash
python app.py
```

The application will start on `http://localhost:5000`

### Accessing the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

## Application Structure

```
GOEMOTION/
├── app.py                      # Main Flask application
├── model_utils.py              # Model loading and prediction utilities
├── data_loader.py              # Dataset loading and statistics
├── templates/                  # HTML templates
│   ├── base.html              # Base template with navigation
│   ├── index.html             # Home page
│   ├── data_visualization.html # Data exploration
│   ├── model_performance.html  # Model evaluation
│   ├── predict.html           # Interactive prediction
│   └── compare.html           # Model comparison
├── models/                     # Trained model files
│   ├── modele_lstm_simple.h5
│   ├── modele_bilstm_attention_final.keras
│   └── modele_cnn_bilstm_attention_final.keras
└── dataset/data/              # TSV data files
    ├── train.tsv
    ├── dev.tsv
    └── test.tsv
```

## API Endpoints

### GET Routes
- `/` - Home page
- `/data-visualization` - Data visualization dashboard
- `/model-performance` - Model performance evaluation
- `/predict` - Interactive prediction interface
- `/compare` - Model comparison page

### POST Routes
- `/api/evaluate-model/<model_name>` - Evaluate a specific model on test data
- `/api/predict` - Predict emotions for input text using all models
- `/api/compare-models` - Compare all models on sample data

## Available Models

### 1. LSTM (Simple)
- **File**: `models/modele_lstm_simple.h5`
- **Architecture**: Embedding → LSTM → Dense
- **Threshold**: 0.3

### 2. BiLSTM + Attention
- **File**: `models/modele_bilstm_attention_final.keras`
- **Architecture**: Embedding → BiLSTM → Attention → Dense
- **Threshold**: 0.5
- **Custom Components**: Attention layer, Weighted loss

### 3. CNN-BiLSTM + Attention
- **File**: `models/modele_cnn_bilstm_attention_final.keras`
- **Architecture**: Embedding → CNN → BiLSTM → Attention → Dense
- **Threshold**: 0.5
- **Custom Components**: Attention layer, Weighted loss

### 4. BERT
- **File**: `models/best_bert_model_pytorch.pt`
- **Framework**: PyTorch + HuggingFace Transformers
- **Architecture**: bert-base-uncased with fine-tuning
- **Threshold**: 0.5
- **Status**: ✅ Fully integrated (1.3 GB model file)

## Usage Examples

### 1. View Dataset Statistics
1. Navigate to "Data Visualization"
2. Explore emotion distributions, class imbalance, and text statistics
3. Interactive Plotly charts for detailed analysis

### 2. Evaluate Models
1. Go to "Model Performance"
2. Click on a model button to evaluate
3. Or click "Evaluate All Models" for comprehensive comparison
4. View overall metrics and per-emotion performance

### 3. Predict Emotions
1. Navigate to "Predict"
2. Enter your text (or click "Load Example")
3. Click "Predict Emotions"
4. See results from all models with:
   - Predicted emotions (above threshold)
   - Top 5 emotions by probability
   - Full probability distribution
   - Model-by-model comparison

### 4. Compare Models
1. Go to "Compare Models"
2. Click "Run Comparison"
3. View:
   - Performance metrics chart
   - Detailed metrics table with rankings
   - Architecture comparison
   - Strengths & weaknesses
   - Deployment recommendations

## Troubleshooting

### Models Not Loading
**Error**: "No models found! Please train models first."

**Solution**: Ensure model files exist in the `models/` directory:
```bash
ls models/
# Should show:
# - modele_lstm_simple.h5
# - modele_bilstm_attention_final.keras
# - modele_cnn_bilstm_attention_final.keras
```

### Dataset Not Found
**Error**: File not found error for TSV files

**Solution**: Verify dataset path in `data_loader.py` and ensure files exist:
```bash
ls dataset/data/
# Should show: train.tsv, dev.tsv, test.tsv
```

### Custom Layer/Loss Loading Issues
**Error**: "Unknown layer" or "Unknown loss function"

**Solution**: The app automatically handles custom objects (Attention layer, weighted loss). If issues persist, check that `model_utils.py` contains the Attention class definition.

### Port Already in Use
**Error**: "Address already in use"

**Solution**: Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Change 5000 to 5001
```

### Slow Predictions
**Issue**: Model evaluation takes a long time

**Explanation**: This is normal for the first prediction as models are loaded into memory. Subsequent predictions will be faster. The "Evaluate Model" feature processes the entire test set (5,427 samples) which takes time.

## Performance Tips

1. **First Load**: The first time you access any model-related page, it will take 30-60 seconds to load all models into memory.

2. **Evaluation**: Full test set evaluation can take 1-3 minutes per model. Use the comparison feature with 100 samples for faster results.

3. **Memory**: All 3 Keras models are loaded simultaneously, requiring ~500MB RAM. Close other applications if you experience memory issues.

4. **GPU**: If you have a CUDA-capable GPU and TensorFlow-GPU installed, predictions will be significantly faster.

## Development Mode

The app runs in debug mode by default:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

**Debug mode features**:
- Auto-reload on code changes
- Detailed error messages
- Interactive debugger

**For production**, set `debug=False`:
```python
app.run(debug=False, host='0.0.0.0', port=5000)
```

## Technology Stack

- **Backend**: Flask 2.3+
- **Frontend**: Bootstrap 5.3, jQuery 3.6
- **Visualizations**: Plotly.js
- **ML Framework**: TensorFlow/Keras 2.13+
- **Data Processing**: Pandas, NumPy
- **Metrics**: scikit-learn

## Features Roadmap

- [ ] BERT model integration
- [ ] Export predictions to CSV
- [ ] Batch prediction upload
- [ ] Real-time attention visualization
- [ ] Model architecture diagrams
- [ ] Training curve visualization
- [ ] Confusion matrix per emotion
- [ ] ROC curves

## Contributing

To add new features:

1. **New Route**: Add route in `app.py`
2. **New Template**: Create HTML file in `templates/`
3. **New Utility**: Add functions to `model_utils.py` or `data_loader.py`
4. **Update Navigation**: Modify `base.html` navbar

## License

This is an academic project for the GoEmotions emotion classification challenge.

## Contact

For issues or questions about the Flask application, please refer to the main project documentation.
