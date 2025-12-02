from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, index=True)
    password_hash = db.Column(db.String(255))
    full_name = db.Column(db.String(255))
    role = db.Column(db.String(20))
    phone = db.Column(db.String(50))
    avatar_url = db.Column(db.String(500))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    patients = db.relationship("Patient", back_populates="user")
    patient = db.relationship("Patient", uselist=False, back_populates="user", overlaps="patients")
    doctor = db.relationship("Doctor", uselist=False, backref="user")
    sent_messages = db.relationship("ChatMessage", foreign_keys="ChatMessage.sender_id", backref="sender")
    received_messages = db.relationship("ChatMessage", foreign_keys="ChatMessage.receiver_id", backref="receiver")
    
    def set_password(self, password):
        """Hash and set password"""
        import bcrypt
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        """Check if password matches"""
        import bcrypt
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
    __tablename__ = "patients"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    date_of_birth = db.Column(db.Date)
    gender = db.Column(db.String(10))
    address = db.Column(db.Text)
    medical_history = db.Column(db.Text)
    diabetes_type = db.Column(db.String(20))
    diabetes_duration_years = db.Column(db.Integer)
    blood_sugar_level = db.Column(db.Numeric(5, 2))
    hba1c_level = db.Column(db.Numeric(4, 2))
    
    user = db.relationship("User", back_populates="patients")
    analysis_results = db.relationship("AnalysisResult", back_populates="patient")
    
    def to_dict(self):
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
    __tablename__ = "doctors"
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    specialization = db.Column(db.String(255))
    license_number = db.Column(db.String(100))
    years_of_experience = db.Column(db.Integer)
    hospital_affiliation = db.Column(db.String(255))
    bio = db.Column(db.Text)
    
    def to_dict(self):
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
    __tablename__ = "analysis_results"
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"))
    original_image_url = db.Column(db.String(500))
    segmented_image_url = db.Column(db.String(500))
    visualization_image_url = db.Column(db.String(500))
    wound_area_cm2 = db.Column(db.Numeric(10, 2))
    wound_perimeter_cm = db.Column(db.Numeric(10, 2))
    wound_width_cm = db.Column(db.Numeric(10, 2))
    wound_height_cm = db.Column(db.Numeric(10, 2))
    color_analysis = db.Column(db.JSON)
    roughness_score = db.Column(db.Numeric(10, 3))
    risk_level = db.Column(db.String(20))
    risk_score = db.Column(db.Numeric(5, 2))
    ai_notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime)
    
    patient = db.relationship("Patient", back_populates="analysis_results")
    

    
    def to_dict(self):
        """Convert model to dictionary for JSON serialization"""
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
            'risk_level': self.risk_level.lower() if self.risk_level else None,
            'risk_score': float(self.risk_score) if self.risk_score else None,
            'ai_notes': self.ai_notes,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class ChatMessage(db.Model):
    __tablename__ = "chat_messages"
    
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    receiver_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    message = db.Column(db.Text)
    is_ai = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Recommendation(db.Model):
    __tablename__ = "recommendations"
    
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("analysis_results.id"))
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctors.id"))
    recommendation_text = db.Column(db.Text)
    treatment_plan = db.Column(db.Text)
    follow_up_date = db.Column(db.Date)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Conversation(db.Model):
    __tablename__ = "conversations"
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    doctor_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = db.relationship('Message', backref='conversation', cascade='all, delete-orphan')

class Message(db.Model):
    __tablename__ = "messages"
    
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversations.id"), nullable=False)
    sender_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    message_text = db.Column(db.Text, nullable=True)
    image_path = db.Column(db.String(500), nullable=True)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Appointment(db.Model):
    __tablename__ = "appointments"
    
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"))
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctors.id"))
    appointment_date = db.Column(db.DateTime)
    status = db.Column(db.String(20))
    reason = db.Column(db.Text)
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
