import sys
import os
import pprint

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai.wound_analyzer import WoundAnalyzer
from config import settings as app_settings

def test_wound_analysis_pipeline(image_path):
    """
    Runs a test of the full AI wound analysis pipeline on a single image.
    """
    print("╔════════════════════════════════════════════════════════════╗")
    print("║           WOUND CARE AI - PIPELINE TEST SCRIPT             ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # --- 1. Configuration ---
    print("\n[STEP 1] Loading configuration...")
    model_path = app_settings.MODEL_PATH
    # Device is handled by the analyzer, defaults to 'cpu'
    print(f"- Model Path: {model_path}")

    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model file not found at '{model_path}'.")
        print("Please ensure the model exists and the path in '.env' or 'config.py' is correct.")
        return

    if not os.path.exists(image_path):
        print(f"\n❌ ERROR: Test image not found at '{image_path}'.")
        return

    # --- 2. Initialize Analyzer ---
    print("\n[STEP 2] Initializing Wound Analyzer...")
    try:
        analyzer = WoundAnalyzer(model_path=model_path)
        print("✅ Analyzer initialized successfully.")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize analyzer: {e}")
        return

    # --- 3. Run Full Analysis ---
    print(f"\n[STEP 3] Running full analysis on image: {image_path}")
    output_dir = os.path.join(os.path.dirname(image_path), 'test_output')
    
    try:
        results = analyzer.analyze_full(image_path, output_dir)
        print("✅ Analysis complete.")
    except Exception as e:
        print(f"\n❌ ERROR: Analysis pipeline failed: {e}")
        return

    # --- 4. Display Results ---
    print("\n[STEP 4] Displaying Analysis Results:")
    print("-" * 40)
    pprint.pprint(results)
    print("-" * 40)

    # --- 5. Summary ---
    print("\n[STEP 5] Summary of Results:")
    risk_level = results.get('risk_assessment', {}).get('risk_level', 'N/A')
    risk_score = results.get('risk_assessment', {}).get('risk_score', 'N/A')
    area = results.get('size_metrics', {}).get('area_cm2', 'N/A')
    color = results.get('color_analysis', {}).get('dominant_color', 'N/A')

    print(f"  - Wound Area: {area} cm²")
    print(f"  - Dominant Color: {color}")
    print(f"  - Risk Score: {risk_score}")
    print(f"  - Risk Level: {risk_level.upper()}")

    print(f"\n[INFO] Output images saved in: {output_dir}")
    print(f"  - Segmented Mask: {results.get('segmented_image')}")
    print(f"  - Visualization: {results.get('visualization_image')}")

    print("\n\n🎉 PIPELINE TEST COMPLETED SUCCESSFULLY! 🎉")

if __name__ == '__main__':
    # Use a default test image if no argument is provided
    # IMPORTANT: Change this path to a valid image on your system
    default_image = '../../Model/test_image.jpg' # You might need to add a test image here
    
    if len(sys.argv) > 1:
        image_to_test = sys.argv[1]
    else:
        print(f"\n[INFO] No image path provided. Using default: {default_image}")
        image_to_test = default_image

    test_wound_analysis_pipeline(image_to_test)

