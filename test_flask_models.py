"""
Quick test to see which models are loaded in Flask app
"""

print("Testing Flask model loading...")
print("="*70)

# Import model_utils to load models
from model_utils import load_all_models

print("\nLoading models...")
models = load_all_models()

print(f"\n{'='*70}")
print(f"MODELS LOADED: {len(models)}")
print(f"{'='*70}")

for model_name in models.keys():
    print(f"  [OK] {model_name}")

print(f"{'='*70}")

# Check if BERT is in there
if 'bert' in models:
    print("\n[OK] BERT MODEL IS LOADED!")
    print(f"   Type: {type(models['bert'])}")
    print(f"   Has predict method: {hasattr(models['bert'], 'predict')}")
else:
    print("\n[ERROR] BERT MODEL IS NOT LOADED!")
    print("   Available models:", list(models.keys()))

print(f"\n{'='*70}")
print("Test complete!")
print(f"{'='*70}")
