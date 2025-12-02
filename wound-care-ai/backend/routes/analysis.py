"""
Analysis routes for wound image processing
"""
from flask import Blueprint, request, jsonify, send_file
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
from models import db, User, Patient, AnalysisResult
from ai.wound_analyzer import WoundAnalyzer
from config import settings
import os
from datetime import datetime

analysis_bp = Blueprint('analysis', __name__, url_prefix='/api/analysis')

# Define upload folder and allowed extensions directly
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize AI analyzer with risk assessment and color datasets
analyzer = WoundAnalyzer(
    model_path=settings.MODEL_PATH,
    dataset_path=settings.DATASET_PATH,
    color_dataset_path=settings.COLOR_DATASET_PATH
)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@analysis_bp.route('/upload', methods=['POST'])
@jwt_required()
def upload_and_analyze():
    """Upload wound image and perform AI analysis"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user or user.role != 'patient':
            return jsonify({'error': 'Only patients can upload images'}), 403
        
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
        
        # Save original image
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{user_id}_{timestamp}_{filename}"

        # Ensure 'original' subdirectory exists
        original_dir = os.path.join(UPLOAD_FOLDER, 'original')
        os.makedirs(original_dir, exist_ok=True)
        original_path = os.path.join(original_dir, unique_filename)
        file.save(original_path)

        # Create output directory for this analysis
        output_dir = os.path.join(UPLOAD_FOLDER, f'analysis_{user_id}_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform AI analysis
        results = analyzer.analyze_full(original_path, output_dir)
        
        # Get patient record
        patient = user.patient
        
        # Save to database
        analysis = AnalysisResult(
            patient_id=patient.id,
            original_image_url=original_path,
            segmented_image_url=results['segmented_image'],
            visualization_image_url=results['visualization_image'],
            wound_area_cm2=results['size_metrics']['area_cm2'] if results['size_metrics'] else None,
            wound_perimeter_cm=results['size_metrics']['perimeter_cm'] if results['size_metrics'] else None,
            wound_width_cm=results['size_metrics']['width_cm'] if results['size_metrics'] else None,
            wound_height_cm=results['size_metrics']['height_cm'] if results['size_metrics'] else None,
            color_analysis=results['color_analysis'],
            roughness_score=results['roughness_analysis']['roughness_score'] if results['roughness_analysis'] else None,
            risk_level=results['risk_assessment']['risk_level'],
            risk_score=results['risk_assessment']['risk_score'],
            ai_notes=f"Automated analysis completed at {datetime.now().isoformat()}",
            created_at=datetime.now()
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        # Convert file paths to API URLs (use relative path from uploads folder)
        wound_zoom_path = results.get('wound_zoom_image')
        gradcam_path = results.get('gradcam_image')
        
        # Extract just the filename from full path
        wound_zoom_url = f"/api/analysis/file/{os.path.basename(os.path.dirname(wound_zoom_path))}/{os.path.basename(wound_zoom_path)}" if wound_zoom_path else None
        gradcam_url = f"/api/analysis/file/{os.path.basename(os.path.dirname(gradcam_path))}/{os.path.basename(gradcam_path)}" if gradcam_path else None
        
        # Return image URLs
        return jsonify({
            'message': 'Analysis completed successfully',
            'analysis_id': analysis.id,
            'results': {
                'size_metrics': results['size_metrics'],
                'color_analysis': results['color_analysis'],
                'roughness_analysis': results['roughness_analysis'],
                'risk_assessment': results['risk_assessment']
            },
            'images': {
                'original': f'/api/analysis/image/{analysis.id}/original',
                'segmented': f'/api/analysis/image/{analysis.id}/segmented',
                'visualization': f'/api/analysis/image/{analysis.id}/visualization',
                'wound_zoom': wound_zoom_url,
                'gradcam': gradcam_url
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error in upload_and_analyze: {str(e)}")
        print(error_trace)
        return jsonify({'error': str(e), 'trace': error_trace}), 500

@analysis_bp.route('/history', methods=['GET'])
@jwt_required()
def get_analysis_history():
    """Get analysis history for current patient"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user or user.role != 'patient':
            return jsonify({'error': 'Only patients can view history'}), 403
        
        patient = user.patient
        analyses = AnalysisResult.query.filter_by(patient_id=patient.id)\
                                       .order_by(AnalysisResult.created_at.desc())\
                                       .all()
        
        return jsonify({
            'total': len(analyses),
            'analyses': [analysis.to_dict() for analysis in analyses]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/<int:analysis_id>', methods=['GET'])
@jwt_required()
def get_analysis(analysis_id):
    """Get specific analysis details"""
    try:
        analysis = AnalysisResult.query.get(analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        return jsonify(analysis.to_dict()), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/image/<int:analysis_id>/<image_type>', methods=['GET'])
def get_analysis_image(analysis_id, image_type):
    """Serve analysis images (no auth required for images)"""
    try:
        analysis = AnalysisResult.query.get(analysis_id)
        
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Get image path based on type
        if image_type == 'original':
            image_path = analysis.original_image_url
        elif image_type == 'segmented':
            image_path = analysis.segmented_image_url
        elif image_type == 'visualization':
            image_path = analysis.visualization_image_url
        else:
            return jsonify({'error': 'Invalid image type'}), 400
        
        if not image_path or not os.path.exists(image_path):
            return jsonify({'error': 'Image not found'}), 404
        
        return send_file(image_path, mimetype='image/png')
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/file/<folder>/<filename>', methods=['GET'])
def serve_file(folder, filename):
    """Serve file from uploads directory"""
    try:
        # Construct full path
        file_path = os.path.join(UPLOAD_FOLDER, folder, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, mimetype='image/png')
    except Exception as e:
        import traceback
        print(f"❌ Serve file error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

