from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, User, Patient, AnalysisResult, Recommendation
from datetime import datetime

doctor_ext_bp = Blueprint('doctor_ext', __name__, url_prefix='/api/doctor')

@doctor_ext_bp.route('/patients', methods=['GET'])
@jwt_required()
def get_patients():
    """Get all patients (with optional search)"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if user.role != 'doctor':
            return jsonify({'error': 'Unauthorized'}), 403
        
        search = request.args.get('search', '')
        
        # Query patients
        query = User.query.filter_by(role='patient')
        
        if search:
            query = query.filter(
                (User.full_name.ilike(f'%{search}%')) |
                (User.email.ilike(f'%{search}%')) |
                (User.id == int(search) if search.isdigit() else False)
            )
        
        patients = query.all()
        
        result = []
        for patient in patients:
            patient_info = patient.patient
            result.append({
                'id': patient.id,
                'full_name': patient.full_name,
                'email': patient.email,
                'age': patient_info.age if patient_info else None,
                'gender': patient_info.gender if patient_info else None,
                'diabetes_type': patient_info.diabetes_type if patient_info else None,
                'created_at': patient.created_at.isoformat()
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@doctor_ext_bp.route('/patient/<int:patient_id>/analyses', methods=['GET'])
@jwt_required()
def get_patient_analyses(patient_id):
    """Get all analyses for a specific patient"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if user.role != 'doctor':
            return jsonify({'error': 'Unauthorized'}), 403
        
        # Get patient
        patient = User.query.get(patient_id)
        if not patient or patient.role != 'patient':
            return jsonify({'error': 'Patient not found'}), 404
        
        # Get analyses
        analyses = AnalysisResult.query.filter_by(patient_id=patient_id).order_by(
            AnalysisResult.created_at.desc()
        ).all()
        
        result = []
        for analysis in analyses:
            risk = analysis.results.get('risk_assessment', {})
            result.append({
                'id': analysis.id,
                'created_at': analysis.created_at.isoformat(),
                'risk_level': risk.get('risk_level', 'N/A'),
                'risk_score': risk.get('risk_score', 0),
                'image_path': analysis.image_path
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@doctor_ext_bp.route('/recommendations/add', methods=['POST'])
@jwt_required()
def add_recommendation():
    """Add recommendation for a patient's analysis"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if user.role != 'doctor':
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        
        analysis_id = data.get('analysis_id')
        recommendation_text = data.get('recommendation_text')
        treatment_plan = data.get('treatment_plan')
        follow_up_date = data.get('follow_up_date')
        
        if not analysis_id or not recommendation_text:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if analysis exists
        analysis = AnalysisResult.query.get(analysis_id)
        if not analysis:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Create recommendation
        recommendation = Recommendation(
            analysis_id=analysis_id,
            doctor_id=user.doctor.id,
            recommendation_text=recommendation_text,
            treatment_plan=treatment_plan,
            follow_up_date=datetime.fromisoformat(follow_up_date) if follow_up_date else None
        )
        
        db.session.add(recommendation)
        db.session.commit()
        
        return jsonify({
            'message': 'Recommendation added',
            'data': {
                'id': recommendation.id,
                'recommendation_text': recommendation.recommendation_text,
                'treatment_plan': recommendation.treatment_plan,
                'follow_up_date': recommendation.follow_up_date.isoformat() if recommendation.follow_up_date else None
            }
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
