"""
Test full analysis pipeline with real image
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ai.wound_analyzer import WoundAnalyzer
from config import settings
import json

def test_full_analysis():
    print("=" * 60)
    print("Testing Full Analysis Pipeline")
    print("=" * 60)
    
    # Use test image from Model folder
    test_image = "../../Model/test_image.jpg"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    print(f"\n1. Test image: {test_image}")
    
    try:
        print("\n2. Initializing analyzer...")
        analyzer = WoundAnalyzer(model_path=settings.MODEL_PATH, device='cpu')
        print("   ✅ Analyzer ready")
        
        print("\n3. Running full analysis...")
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        results = analyzer.analyze_full(test_image, output_dir)
        
        print("\n4. Analysis Results:")
        print("   " + "=" * 56)
        
        # Size metrics
        if results['size_metrics']:
            print("\n   📏 SIZE METRICS:")
            for key, value in results['size_metrics'].items():
                print(f"      - {key}: {value}")
        
        # Color analysis
        if results['color_analysis']:
            print("\n   🎨 COLOR ANALYSIS (K-means):")
            color = results['color_analysis']
            if 'cluster_percentages' in color:
                print("      Cluster Percentages:")
                for key, value in color['cluster_percentages'].items():
                    print(f"         - {key}: {value}%")
            if 'tissue_types' in color:
                print("      Tissue Types:")
                for key, value in color['tissue_types'].items():
                    print(f"         - {key}: {value}")
        
        # Roughness analysis
        if results['roughness_analysis']:
            print("\n   📊 ROUGHNESS ANALYSIS (GLCM):")
            for key, value in results['roughness_analysis'].items():
                print(f"      - {key}: {value}")
        
        # Risk assessment
        if results['risk_assessment']:
            print("\n   ⚠️  RISK ASSESSMENT:")
            risk = results['risk_assessment']
            print(f"      - Risk Level: {risk['risk_level']}")
            print(f"      - Risk Score: {risk['risk_score']}/100")
            if 'risk_factors' in risk and risk['risk_factors']:
                print("      - Risk Factors:")
                for factor in risk['risk_factors']:
                    print(f"         • {factor}")
            if 'recommendation' in risk:
                print(f"      - Recommendation: {risk['recommendation']}")
        
        print("\n   " + "=" * 56)
        print(f"\n5. Output files:")
        print(f"   - Segmented: {results['segmented_image']}")
        print(f"   - Visualization: {results['visualization_image']}")
        
        # Save full results to JSON
        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"   - Full results: {json_path}")
        
        print("\n✅ Full analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_analysis()
    sys.exit(0 if success else 1)
