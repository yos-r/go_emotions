"""
Quick test script to verify BERT model loading
"""

import os
import sys

print("Testing BERT model loading...")
print("="*70)

# Test 1: Check file exists
print("\n1. Checking if BERT model file exists...")
bert_path = 'models/best_bert_model_pytorch.pt'
if os.path.exists(bert_path):
    size_mb = os.path.getsize(bert_path) / (1024 * 1024)
    print(f"   ✅ File exists: {bert_path}")
    print(f"   Size: {size_mb:.1f} MB")
else:
    print(f"   ❌ File not found: {bert_path}")
    sys.exit(1)

# Test 2: Check PyTorch
print("\n2. Checking PyTorch installation...")
try:
    import torch
    print(f"   ✅ PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"   ❌ PyTorch not installed: {e}")
    sys.exit(1)

# Test 3: Check Transformers
print("\n3. Checking Transformers library...")
try:
    import transformers
    print(f"   ✅ Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"   ❌ Transformers not installed: {e}")
    sys.exit(1)

# Test 4: Load BERT model
print("\n4. Loading BERT model...")
try:
    from model_utils import load_bert_model

    print("   Loading model checkpoint...")
    bert_model = load_bert_model(bert_path)
    print(f"   ✅ BERT model loaded successfully!")
    print(f"   Device: {bert_model.device}")

    # Test 5: Test prediction
    print("\n5. Testing prediction...")
    test_text = "I am so happy and excited!"
    print(f"   Input: '{test_text}'")

    predictions = bert_model.predict(test_text)
    print(f"   ✅ Prediction shape: {predictions.shape}")
    print(f"   Top 3 emotions:")

    from model_utils import EMOTION_LABELS
    top_indices = predictions[0].argsort()[-3:][::-1]
    for idx in top_indices:
        print(f"      - {EMOTION_LABELS[idx]}: {predictions[0][idx]:.4f}")

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED! BERT model is working correctly.")
    print("="*70)

except Exception as e:
    import traceback
    print(f"\n   ❌ Error loading BERT model:")
    print(traceback.format_exc())
    sys.exit(1)
