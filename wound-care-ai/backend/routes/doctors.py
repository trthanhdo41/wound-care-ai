"""
Doctor routes
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, User, Doctor, Patient, AnalysisResult, Recommendation

doctors_bp = Blueprint('doctors', __name__, url_prefix='/api/doctors')

@doctors_bp.route('/patients', methods=['GET'])
@jwt_required()
def get_patients():
    """Get all patients (for doctors)"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user or user.role != 'doctor':
            return jsonify({'error': 'Not a doctor'}), 403
        
        patients = Patient.query.all()
        
        patients_data = []
        for p in patients:
            try:
                # Get user info
                patient_user = User.query.get(p.user_id)
                
                # Count analyses safely
                try:
                    analysis_count = AnalysisResult.query.filter_by(patient_id=p.id).count()
                    latest_analysis = AnalysisResult.query.filter_by(patient_id=p.id)\
                                                          .order_by(AnalysisResult.created_at.desc())\
                                                          .first()
                except:
                    analysis_count = 0
                    latest_analysis = None
                
                patients_data.append({
                    'id': p.id,
                    'user_id': p.user_id,
                    'full_name': patient_user.full_name if patient_user else 'Unknown',
                    'email': patient_user.email if patient_user else 'N/A',
                    'phone': patient_user.phone if patient_user else None,
                    'diabetes_type': getattr(p, 'diabetes_type', None),
                    'gender': getattr(p, 'gender', None),
                    'date_of_birth': str(p.date_of_birth) if p.date_of_birth else None,
                    'analysis_count': analysis_count,
                    'latest_risk_level': latest_analysis.risk_level if latest_analysis and hasattr(latest_analysis, 'risk_level') else None,
                    'last_analysis_date': latest_analysis.created_at.isoformat() if latest_analysis else None
                })
            except Exception as e:
                print(f"Error processing patient {p.id}: {e}")
                continue
        
        return jsonify({
            'total': len(patients_data),
            'patients': patients_data
        }), 200
        
    except Exception as e:
        print(f"Error in get_patients: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@doctors_bp.route('/patients/<int:patient_id>/analyses', methods=['GET'])
@jwt_required()
def get_patient_analyses(patient_id):
    """Get all analyses for a specific patient"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user or user.role != 'doctor':
            return jsonify({'error': 'Not a doctor'}), 403
        
        analyses = AnalysisResult.query.filter_by(patient_id=patient_id)\
                                       .order_by(AnalysisResult.created_at.desc())\
                                       .all()
        
        return jsonify({
            'patient_id': patient_id,
            'total': len(analyses),
            'analyses': [a.to_dict() for a in analyses]
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@doctors_bp.route('/recommendations', methods=['POST'])
@jwt_required()
def add_recommendation():
    """Add recommendation to an analysis"""
    try:
        user_id = int(get_jwt_identity())
        user = User.query.get(user_id)
        
        if not user or user.role != 'doctor':
            return jsonify({'error': 'Not a doctor'}), 403
        
        data = request.get_json()
        
        if not data.get('analysis_id') or not data.get('recommendation_text'):
            return jsonify({'error': 'Analysis ID and recommendation text required'}), 400
        
        doctor = user.doctor
        
        recommendation = Recommendation(
            analysis_id=data['analysis_id'],
            doctor_id=doctor.id,
            recommendation_text=data['recommendation_text'],
            treatment_plan=data.get('treatment_plan'),
            follow_up_date=data.get('follow_up_date')
        )
        
        db.session.add(recommendation)
        db.session.commit()
        
        return jsonify({
            'message': 'Recommendation added',
            'recommendation': recommendation.to_dict()
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

