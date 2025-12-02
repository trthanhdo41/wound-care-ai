"""
Quick test to verify model loads correctly with new implementation
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from ai.wound_analyzer import WoundAnalyzer
from config import settings

def test_model_load():
    print("=" * 60)
    print("Testing Model Load with Client's Implementation")
    print("=" * 60)
    
    model_path = settings.MODEL_PATH
    print(f"\n1. Model path: {model_path}")
    print(f"   Exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"   ❌ Model file not found!")
        return False
    
    try:
        print("\n2. Initializing WoundAnalyzer...")
        analyzer = WoundAnalyzer(model_path=model_path, device='cpu')
        print("   ✅ Model loaded successfully!")
        
        print("\n3. Model details:")
        print(f"   - Device: {analyzer.device}")
        print(f"   - Image size: {analyzer.img_size}")
        print(f"   - K-means clusters: {analyzer.universal_k}")
        print(f"   - Model type: {type(analyzer.model).__name__}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_load()
    sys.exit(0 if success else 1)
