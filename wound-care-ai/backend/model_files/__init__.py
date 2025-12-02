"""
Database models for Wound Care AI
"""
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import bcrypt

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Enum('patient', 'doctor', 'admin'), nullable=False, index=True)
    phone = db.Column(db.String(50))
    avatar_url = db.Column(db.String(500))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    patient = db.relationship('Patient', backref='user', uselist=False, cascade='all, delete-orphan')
    doctor = db.relationship('Doctor', backref='user', uselist=False, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        """Check if password matches"""
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'full_name': self.full_name,
            'role': self.role,
            'phone': self.phone,
            'avatar_url': self.avatar_url,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Patient(db.Model):
    __tablename__ = 'patients'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False)
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.Enum('male', 'female', 'other'))
    address = db.Column(db.Text)
    medical_history = db.Column(db.Text)
    diabetes_type = db.Column(db.Enum('type1', 'type2', 'gestational', 'other'))
    diabetes_duration_years = db.Column(db.Integer)
    blood_sugar_level = db.Column(db.Numeric(5, 2))
    hba1c_level = db.Column(db.Numeric(4, 2))
    
    # Relationships
    analysis_results = db.relationship('AnalysisResult', backref='patient', cascade='all, delete-orphan')
    appointments = db.relationship('Appointment', foreign_keys='Appointment.patient_id', backref='patient', cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'gender': self.gender,
            'address': self.address,
            'medical_history': self.medical_history,
            'diabetes_type': self.diabetes_type,
            'diabetes_duration_years': self.diabetes_duration_years,
            'blood_sugar_level': float(self.blood_sugar_level) if self.blood_sugar_level else None,
            'hba1c_level': float(self.hba1c_level) if self.hba1c_level else None
        }

class Doctor(db.Model):
    __tablename__ = 'doctors'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False)
    specialization = db.Column(db.String(255))
    license_number = db.Column(db.String(100), unique=True)
    years_of_experience = db.Column(db.Integer)
    hospital_affiliation = db.Column(db.String(255))
    bio = db.Column(db.Text)
    
    # Relationships
    recommendations = db.relationship('Recommendation', backref='doctor', cascade='all, delete-orphan')
    appointments = db.relationship('Appointment', foreign_keys='Appointment.doctor_id', backref='doctor', cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'specialization': self.specialization,
            'license_number': self.license_number,
            'years_of_experience': self.years_of_experience,
            'hospital_affiliation': self.hospital_affiliation,
            'bio': self.bio
        }

class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False, index=True)
    original_image_url = db.Column(db.String(500), nullable=False)
    segmented_image_url = db.Column(db.String(500))
    visualization_image_url = db.Column(db.String(500))
    wound_area_cm2 = db.Column(db.Numeric(10, 2))
    wound_perimeter_cm = db.Column(db.Numeric(10, 2))
    wound_width_cm = db.Column(db.Numeric(10, 2))
    wound_height_cm = db.Column(db.Numeric(10, 2))
    color_analysis = db.Column(db.JSON)
    roughness_score = db.Column(db.Numeric(5, 2))
    risk_level = db.Column(db.Enum('low', 'medium', 'high', 'critical'), index=True)
    risk_score = db.Column(db.Numeric(5, 2))
    ai_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Relationships
    recommendations = db.relationship('Recommendation', backref='analysis', cascade='all, delete-orphan')

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'original_image_url': self.original_image_url,
            'segmented_image_url': self.segmented_image_url,
            'visualization_image_url': self.visualization_image_url,
            'wound_area_cm2': float(self.wound_area_cm2) if self.wound_area_cm2 else None,
            'wound_perimeter_cm': float(self.wound_perimeter_cm) if self.wound_perimeter_cm else None,
            'wound_width_cm': float(self.wound_width_cm) if self.wound_width_cm else None,
            'wound_height_cm': float(self.wound_height_cm) if self.wound_height_cm else None,
            'color_analysis': self.color_analysis,
            'roughness_score': float(self.roughness_score) if self.roughness_score else None,
            'risk_level': self.risk_level,
            'risk_score': float(self.risk_score) if self.risk_score else None,
            'ai_notes': self.ai_notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Recommendation(db.Model):
    __tablename__ = 'recommendations'

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey('analysis_results.id', ondelete='CASCADE'), nullable=False, index=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id', ondelete='SET NULL'), index=True)
    recommendation_text = db.Column(db.Text, nullable=False)
    treatment_plan = db.Column(db.Text)
    follow_up_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'doctor_id': self.doctor_id,
            'recommendation_text': self.recommendation_text,
            'treatment_plan': self.treatment_plan,
            'follow_up_date': self.follow_up_date.isoformat() if self.follow_up_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Appointment(db.Model):
    __tablename__ = 'appointments'

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id', ondelete='CASCADE'), nullable=False, index=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctors.id', ondelete='CASCADE'), nullable=False, index=True)
    appointment_date = db.Column(db.DateTime, nullable=False, index=True)
    status = db.Column(db.Enum('scheduled', 'completed', 'cancelled', 'no_show'), default='scheduled', index=True)
    reason = db.Column(db.Text)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'doctor_id': self.doctor_id,
            'appointment_date': self.appointment_date.isoformat() if self.appointment_date else None,
            'status': self.status,
            'reason': self.reason,
            'notes': self.notes,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'

    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    receiver_id = db.Column(db.Integer, db.ForeignKey('users.id', ondelete='CASCADE'), index=True)
    message_type = db.Column(db.Enum('user_to_user', 'user_to_ai'), nullable=False, index=True)
    message_text = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type,
            'message_text': self.message_text,
            'is_read': self.is_read,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

