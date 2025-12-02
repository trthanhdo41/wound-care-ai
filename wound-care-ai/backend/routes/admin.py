from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models import db, User, Patient, Doctor, Appointment, AnalysisResult
from werkzeug.security import generate_password_hash
from datetime import datetime
from sqlalchemy import func

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

def check_admin():
    """Check if current user is admin"""
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    return user and user.role == 'admin'

@admin_bp.route('/users', methods=['GET'])
@jwt_required()
def get_users():
    """Get all users"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        role = request.args.get('role')  # 'doctor' or 'patient'
        
        query = User.query
        if role:
            query = query.filter_by(role=role)
        
        users = query.all()
        
        result = []
        for user in users:
            result.append({
                'id': user.id,
                'email': user.email,
                'full_name': user.full_name,
                'role': user.role,
                'is_active': user.is_active,
                'created_at': user.created_at.isoformat()
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users', methods=['POST'])
@jwt_required()
def create_user():
    """Create new user"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name')
        role = data.get('role')
        
        if not all([email, password, full_name, role]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if role not in ['patient', 'doctor']:
            return jsonify({'error': 'Invalid role'}), 400
        
        # Check if email exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already exists'}), 400
        
        # Create user
        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            full_name=full_name,
            role=role
        )
        db.session.add(user)
        db.session.flush()
        
        # Create role-specific record
        if role == 'patient':
            patient = Patient(
                user_id=user.id,
                age=data.get('age'),
                gender=data.get('gender'),
                diabetes_type=data.get('diabetes_type')
            )
            db.session.add(patient)
        elif role == 'doctor':
            doctor = Doctor(
                user_id=user.id,
                specialization=data.get('specialization', 'General'),
                license_number=data.get('license_number')
            )
            db.session.add(doctor)
        
        db.session.commit()
        
        return jsonify({
            'message': 'User created',
            'user_id': user.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users/<int:user_id>', methods=['PUT'])
@jwt_required()
def update_user(user_id):
    """Update user"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        if 'full_name' in data:
            user.full_name = data['full_name']
        if 'email' in data:
            user.email = data['email']
        
        db.session.commit()
        
        return jsonify({'message': 'User updated'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users/<int:user_id>', methods=['DELETE'])
@jwt_required()
def delete_user(user_id):
    """Delete user"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        db.session.delete(user)
        db.session.commit()
        
        return jsonify({'message': 'User deleted'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users/<int:user_id>/lock', methods=['POST'])
@jwt_required()
def toggle_user_lock(user_id):
    """Lock/Unlock user account"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        user.is_active = not user.is_active
        db.session.commit()
        
        status = 'unlocked' if user.is_active else 'locked'
        return jsonify({'message': f'User {status}', 'is_active': user.is_active}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/users/<int:user_id>/reset-password', methods=['POST'])
@jwt_required()
def reset_password(user_id):
    """Reset user password"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        new_password = data.get('new_password')
        
        if not new_password:
            return jsonify({'error': 'New password required'}), 400
        
        user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        return jsonify({'message': 'Password reset successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/appointments', methods=['GET'])
@jwt_required()
def get_appointments():
    """Get all appointments"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        appointments = Appointment.query.order_by(Appointment.appointment_date.desc()).all()
        
        result = []
        for apt in appointments:
            patient = User.query.get(apt.patient_id)
            doctor = User.query.get(apt.doctor_id)
            
            result.append({
                'id': apt.id,
                'patient': {'id': patient.id, 'name': patient.full_name},
                'doctor': {'id': doctor.id, 'name': doctor.full_name},
                'appointment_date': apt.appointment_date.isoformat(),
                'reason': apt.reason,
                'status': apt.status,
                'notes': apt.notes
            })
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/appointments', methods=['POST'])
@jwt_required()
def create_appointment():
    """Create new appointment"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        data = request.get_json()
        
        patient_id = data.get('patient_id')
        doctor_id = data.get('doctor_id')
        appointment_date = data.get('appointment_date')
        reason = data.get('reason')
        notes = data.get('notes')
        
        if not all([patient_id, doctor_id, appointment_date]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        appointment = Appointment(
            patient_id=patient_id,
            doctor_id=doctor_id,
            appointment_date=datetime.fromisoformat(appointment_date),
            reason=reason,
            notes=notes,
            status='scheduled'
        )
        
        db.session.add(appointment)
        db.session.commit()
        
        return jsonify({
            'message': 'Appointment created',
            'appointment_id': appointment.id
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/appointments/<int:appointment_id>', methods=['PUT'])
@jwt_required()
def update_appointment(appointment_id):
    """Update appointment"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        appointment = Appointment.query.get(appointment_id)
        if not appointment:
            return jsonify({'error': 'Appointment not found'}), 404
        
        data = request.get_json()
        
        if 'appointment_date' in data:
            appointment.appointment_date = datetime.fromisoformat(data['appointment_date'])
        if 'reason' in data:
            appointment.reason = data['reason']
        if 'notes' in data:
            appointment.notes = data['notes']
        if 'status' in data:
            appointment.status = data['status']
        
        db.session.commit()
        
        return jsonify({'message': 'Appointment updated'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/appointments/<int:appointment_id>', methods=['DELETE'])
@jwt_required()
def delete_appointment(appointment_id):
    """Delete appointment"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        appointment = Appointment.query.get(appointment_id)
        if not appointment:
            return jsonify({'error': 'Appointment not found'}), 404
        
        db.session.delete(appointment)
        db.session.commit()
        
        return jsonify({'message': 'Appointment deleted'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/statistics', methods=['GET'])
@jwt_required()
def get_statistics():
    """Get system statistics"""
    if not check_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        stats = {
            'total_patients': User.query.filter_by(role='patient').count(),
            'total_doctors': User.query.filter_by(role='doctor').count(),
            'total_analyses': AnalysisResult.query.count(),
            'total_appointments': Appointment.query.count(),
            'risk_distribution': {
                'low': 0,
                'medium': 0,
                'high': 0
            }
        }
        
        # Count risk levels (simplified - would need JSON query in production)
        analyses = AnalysisResult.query.all()
        for analysis in analyses:
            risk_level = analysis.results.get('risk_assessment', {}).get('risk_level', 'medium')
            if risk_level in stats['risk_distribution']:
                stats['risk_distribution'][risk_level] += 1
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
