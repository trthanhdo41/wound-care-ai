"""
Patient routes
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, User, Patient, AnalysisResult

patients_bp = Blueprint('patients', __name__, url_prefix='/api/patients')

@patients_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get patient profile"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user or user.role != 'patient':
            return jsonify({'error': 'Not a patient'}), 403
        
        patient = user.patient
        
        return jsonify({
            'user': user.to_dict(),
            'patient': patient.to_dict() if patient else None
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@patients_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update patient profile"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user or user.role != 'patient':
            return jsonify({'error': 'Not a patient'}), 403
        
        data = request.get_json()
        patient = user.patient
        
        # Update patient fields
        if 'medical_history' in data:
            patient.medical_history = data['medical_history']
        if 'diabetes_type' in data:
            patient.diabetes_type = data['diabetes_type']
        if 'blood_sugar_level' in data:
            patient.blood_sugar_level = data['blood_sugar_level']
        if 'hba1c_level' in data:
            patient.hba1c_level = data['hba1c_level']
        
        db.session.commit()
        
        return jsonify({
            'message': 'Profile updated',
            'patient': patient.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@patients_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_stats():
    """Get patient statistics"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user or user.role != 'patient':
            return jsonify({'error': 'Not a patient'}), 403
        
        patient = user.patient
        analyses = AnalysisResult.query.filter_by(patient_id=patient.id).all()
        
        # Calculate stats
        total_analyses = len(analyses)
        risk_distribution = {
            'low': sum(1 for a in analyses if a.risk_level == 'low'),
            'medium': sum(1 for a in analyses if a.risk_level == 'medium'),
            'high': sum(1 for a in analyses if a.risk_level == 'high'),
            'critical': sum(1 for a in analyses if a.risk_level == 'critical')
        }
        
        avg_wound_size = sum(float(a.wound_area_cm2) for a in analyses if a.wound_area_cm2) / total_analyses if total_analyses > 0 else 0
        
        return jsonify({
            'total_analyses': total_analyses,
            'risk_distribution': risk_distribution,
            'average_wound_size_cm2': round(avg_wound_size, 2)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@patients_bp.route('/doctors', methods=['GET'])
@jwt_required()
def get_doctors():
    """Get list of all doctors (for patients to start chat)"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user or user.role != 'patient':
            return jsonify({'error': 'Not a patient'}), 403
        
        # Get all doctors
        doctors = User.query.filter_by(role='doctor').all()
        
        doctors_data = []
        for doctor in doctors:
            doctors_data.append({
                'id': doctor.id,
                'full_name': doctor.full_name,
                'email': doctor.email,
                'phone': doctor.phone if hasattr(doctor, 'phone') else None
            })
        
        return jsonify(doctors_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

