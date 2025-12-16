# BERT Model Troubleshooting Guide

## Quick Test

Run this to verify BERT is working:

```bash
python test_bert_loading.py
```

Expected output:
```
Testing BERT model loading...
======================================================================

1. Checking if BERT model file exists...
   ✅ File exists: models/best_bert_model_pytorch.pt
   Size: 1309.5 MB

2. Checking PyTorch installation...
   ✅ PyTorch version: 2.x.x
   CUDA available: True/False

3. Checking Transformers library...
   ✅ Transformers version: 4.x.x

4. Loading BERT model...
   Loading model checkpoint...
   ✅ BERT model loaded successfully!
   Device: cuda:0 (or cpu)

5. Testing prediction...
   Input: 'I am so happy and excited!'
   ✅ Prediction shape: (1, 28)
   Top 3 emotions:
      - joy: 0.8234
      - excitement: 0.7654
      - optimism: 0.6543

======================================================================
✅ ALL TESTS PASSED! BERT model is working correctly.
======================================================================
```

## Common Issues and Solutions

### Issue 1: BERT Model Not Loading in Flask App

**Symptom**: Flask app starts but BERT tab doesn't appear

**Check Flask console output**:
```
Loading models...
✅ Loaded LSTM model
✅ Loaded BiLSTM+Attention model
✅ Loaded CNN-BiLSTM+Attention model
❌ Error loading BERT: <error message>
```

**Solutions**:

1. **Run the test script first**:
   ```bash
   python test_bert_loading.py
   ```

2. **Check dependencies**:
   ```bash
   pip list | grep -i torch
   pip list | grep -i transformers
   ```

3. **Install missing dependencies**:
   ```bash
   pip install torch transformers
   ```

### Issue 2: "CUDA out of memory"

**Symptom**: Error when evaluating BERT on full test set

**Solution 1 - Reduce batch size** (in `model_utils.py`):
```python
# In BertModelWrapper.predict(), change:
def predict(self, texts, verbose=0, batch_size=16):  # Changed from 32 to 16
```

**Solution 2 - Force CPU**:
```python
# In model_utils.py, load_bert_model():
def load_bert_model(model_path, device=None):
    device = torch.device('cpu')  # Force CPU
    ...
```

**Solution 3 - Disable GPU in Flask app**:
```python
# At top of app.py:
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
```

### Issue 3: BERT Evaluation is Very Slow

**Expected times**:
- **With GPU**: 2-3 minutes for 5,427 test samples
- **Without GPU**: 10-15 minutes for 5,427 test samples

**Current optimizations**:
- ✅ Batch processing (32 samples at a time)
- ✅ Progress logging every 10 batches

**To monitor progress**, check Flask console:
```
======================================================================
Evaluating BERT model on test data...
======================================================================
Loaded test data: 5427 samples
Preprocessing data (for_bert=True)...
Preprocessed X_test shape: 5427
Running predictions with bert...
Processed 320/5427 samples...
Processed 640/5427 samples...
...
Predictions complete! Shape: (5427, 28)

Evaluation complete for bert!
  Hamming Loss: 0.0289
  AUC-ROC: 0.9012
  F1-Micro: 0.6543
  F1-Macro: 0.5234
======================================================================
```

**If still too slow**, use comparison mode instead (100 samples only):
- Go to http://localhost:5000/compare
- Click "Run Comparison"
- Much faster: ~30 seconds

### Issue 4: "Model not found" Error

**Symptom**: Clicking BERT tab shows "Model not found"

**Check**:
1. Verify model loaded at startup (check Flask console)
2. Check browser console for JavaScript errors (F12)
3. Verify BERT is in models list:
   ```python
   # In app.py, check:
   print(list(models.keys()))
   # Should show: ['lstm', 'bilstm_attention', 'cnn_bilstm_attention', 'bert']
   ```

### Issue 5: First Time Downloading BERT Pretrained Model

**Symptom**: App hangs on first load, then shows:
```
Downloading: 100%|██████████| 440M/440M [XX:XX<00:00, XMB/s]
```

**Explanation**: This is **normal** on first run. HuggingFace downloads:
- `bert-base-uncased` model (~440 MB)
- Tokenizer files (~1 MB)

**One-time download**. Cached at:
- Windows: `C:\Users\<username>\.cache\huggingface\`
- Linux/Mac: `~/.cache/huggingface/`

**To pre-download manually**:
```python
from transformers import BertModel, BertTokenizer
BertModel.from_pretrained('bert-base-uncased')
BertTokenizer.from_pretrained('bert-base-uncased')
```

### Issue 6: ImportError for torch or transformers

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
```bash
# Activate virtual environment
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch transformers

# Or install all
pip install -r requirements.txt
```

**Verify installation**:
```python
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
```

### Issue 7: Different Results from Training Script

**Expected**: BERT results in Flask may differ slightly from training script

**Reasons**:
1. **Dropout**: Disabled in eval mode (correct)
2. **Batch processing**: May cause minor numerical differences
3. **Device**: CPU vs GPU can have tiny differences

**These differences are normal** (<0.01 difference in metrics)

## Performance Benchmarks

### Prediction Speed (single text)

| Environment | Time per Prediction |
|-------------|-------------------|
| CPU | ~0.5-1 second |
| GPU (CUDA) | ~0.05-0.1 second |

### Batch Prediction Speed (5,427 samples)

| Environment | Total Time | Throughput |
|-------------|-----------|------------|
| CPU | 10-15 min | ~6-9 samples/sec |
| GPU (CUDA) | 2-3 min | ~30-45 samples/sec |

### Memory Usage

| Component | RAM | VRAM (GPU) |
|-----------|-----|------------|
| BERT Model | ~1.5 GB | ~2 GB |
| + Keras Models | +0.5 GB | - |
| **Total** | ~2 GB | ~2 GB |

## Debug Mode

To enable verbose logging in BERT predictions:

```python
# In app.py, when calling predict:
Y_pred_proba = model.predict(X_test, verbose=1)  # Enable logging
```

This will show progress:
```
Processed 320/5427 samples...
Processed 640/5427 samples...
...
```

## Still Having Issues?

1. **Check Flask console** for error messages
2. **Check browser console** (F12) for JavaScript errors
3. **Run test script**: `python test_bert_loading.py`
4. **Check model file**: `ls -lh models/best_bert_model_pytorch.pt`
5. **Verify dependencies**: `pip list | grep -E "(torch|transformers)"`

## Support Checklist

Before asking for help, gather this info:

```bash
# 1. Python version
python --version

# 2. PyTorch version
python -c "import torch; print(torch.__version__)"

# 3. Transformers version
python -c "import transformers; print(transformers.__version__)"

# 4. CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 5. Model file size
ls -lh models/best_bert_model_pytorch.pt

# 6. Run test
python test_bert_loading.py

# 7. Flask console output (when starting app)
python app.py 2>&1 | tee flask_output.txt
```

---

**Most common issue**: PyTorch or Transformers not installed

**Quick fix**:
```bash
pip install torch transformers
python app.py
```

Then navigate to http://localhost:5000 and try BERT predictions!
