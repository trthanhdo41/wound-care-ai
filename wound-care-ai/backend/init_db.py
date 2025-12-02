#!/usr/bin/env python3
"""
Database Initialization Script for Render
Run this after deployment to create tables and seed data
"""

import os
import sys
from sqlalchemy import create_engine, text
from werkzeug.security import generate_password_hash

def init_database():
    """Initialize database with schema and seed data"""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("❌ ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)
    
    # Convert postgres:// to postgresql:// if needed (Render uses postgres://)
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    
    print(f"🔗 Connecting to database...")
    
    try:
        engine = create_engine(database_url)
        
        with engine.connect() as conn:
            print("✅ Connected to database successfully!")
            
            # Drop existing tables
            print("\n🗑️  Dropping existing tables...")
            drop_tables = [
                "DROP TABLE IF EXISTS chat_messages",
                "DROP TABLE IF EXISTS recommendations",
                "DROP TABLE IF EXISTS appointments",
                "DROP TABLE IF EXISTS analysis_results",
                "DROP TABLE IF EXISTS doctors",
                "DROP TABLE IF EXISTS patients",
                "DROP TABLE IF EXISTS users"
            ]
            
            for drop_sql in drop_tables:
                conn.execute(text(drop_sql))
                conn.commit()
            
            print("✅ Dropped existing tables")
            
            # Create tables
            print("\n📋 Creating tables...")
            
            # Users table
            conn.execute(text("""
                CREATE TABLE users (
                    id SERIAL PRIMARY KEY,
                    email VARCHAR(255) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL CHECK (role IN ('patient', 'doctor', 'admin')),
                    phone VARCHAR(50),
                    avatar_url VARCHAR(500),
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()
            print("✅ Created users table")
            
            # Patients table
            conn.execute(text("""
                CREATE TABLE patients (
                    id SERIAL PRIMARY KEY,
                    user_id INT UNIQUE NOT NULL,
                    date_of_birth DATE,
                    gender VARCHAR(20) CHECK (gender IN ('male', 'female', 'other')),
                    address TEXT,
                    medical_history TEXT,
                    diabetes_type VARCHAR(20) CHECK (diabetes_type IN ('type1', 'type2', 'gestational', 'other')),
                    diabetes_duration_years INT,
                    blood_sugar_level DECIMAL(5,2),
                    hba1c_level DECIMAL(4,2),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """))
            conn.commit()
            print("✅ Created patients table")
            
            # Doctors table
            conn.execute(text("""
                CREATE TABLE doctors (
                    id SERIAL PRIMARY KEY,
                    user_id INT UNIQUE NOT NULL,
                    specialization VARCHAR(255),
                    license_number VARCHAR(100) UNIQUE,
                    years_of_experience INT,
                    hospital_affiliation VARCHAR(255),
                    bio TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """))
            conn.commit()
            print("✅ Created doctors table")
            
            # Analysis Results table
            conn.execute(text("""
                CREATE TABLE analysis_results (
                    id SERIAL PRIMARY KEY,
                    patient_id INT NOT NULL,
                    original_image_url VARCHAR(500) NOT NULL,
                    segmented_image_url VARCHAR(500),
                    visualization_image_url VARCHAR(500),
                    wound_area_cm2 DECIMAL(10,2),
                    wound_perimeter_cm DECIMAL(10,2),
                    wound_width_cm DECIMAL(10,2),
                    wound_height_cm DECIMAL(10,2),
                    color_analysis JSON,
                    roughness_score DECIMAL(10,3),
                    risk_level VARCHAR(20) CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
                    risk_score DECIMAL(5,2),
                    ai_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
                )
            """))
            conn.commit()
            print("✅ Created analysis_results table")
            
            # Recommendations table
            conn.execute(text("""
                CREATE TABLE recommendations (
                    id SERIAL PRIMARY KEY,
                    analysis_id INT NOT NULL,
                    doctor_id INT,
                    recommendation_text TEXT NOT NULL,
                    treatment_plan TEXT,
                    follow_up_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_results(id) ON DELETE CASCADE,
                    FOREIGN KEY (doctor_id) REFERENCES doctors(id) ON DELETE SET NULL
                )
            """))
            conn.commit()
            print("✅ Created recommendations table")
            
            # Appointments table
            conn.execute(text("""
                CREATE TABLE appointments (
                    id SERIAL PRIMARY KEY,
                    patient_id INT NOT NULL,
                    doctor_id INT NOT NULL,
                    appointment_date TIMESTAMP NOT NULL,
                    status VARCHAR(20) DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'completed', 'cancelled', 'no_show')),
                    reason TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
                    FOREIGN KEY (doctor_id) REFERENCES doctors(id) ON DELETE CASCADE
                )
            """))
            conn.commit()
            print("✅ Created appointments table")
            
            # Chat Messages table
            conn.execute(text("""
                CREATE TABLE chat_messages (
                    id SERIAL PRIMARY KEY,
                    sender_id INT NOT NULL,
                    receiver_id INT,
                    message_type VARCHAR(20) NOT NULL CHECK (message_type IN ('user_to_user', 'user_to_ai')),
                    message_text TEXT NOT NULL,
                    is_read BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (sender_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (receiver_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """))
            conn.commit()
            print("✅ Created chat_messages table")
            
            # Create indexes
            print("\n📊 Creating indexes...")
            indexes = [
                "CREATE INDEX idx_users_email ON users(email)",
                "CREATE INDEX idx_users_role ON users(role)",
                "CREATE INDEX idx_patients_user_id ON patients(user_id)",
                "CREATE INDEX idx_doctors_user_id ON doctors(user_id)",
                "CREATE INDEX idx_analysis_patient_id ON analysis_results(patient_id)",
                "CREATE INDEX idx_analysis_risk_level ON analysis_results(risk_level)",
                "CREATE INDEX idx_analysis_created_at ON analysis_results(created_at)",
                "CREATE INDEX idx_recommendations_analysis_id ON recommendations(analysis_id)",
                "CREATE INDEX idx_recommendations_doctor_id ON recommendations(doctor_id)",
                "CREATE INDEX idx_appointments_patient_id ON appointments(patient_id)",
                "CREATE INDEX idx_appointments_doctor_id ON appointments(doctor_id)",
                "CREATE INDEX idx_appointments_date ON appointments(appointment_date)",
                "CREATE INDEX idx_appointments_status ON appointments(status)",
                "CREATE INDEX idx_chat_sender_id ON chat_messages(sender_id)",
                "CREATE INDEX idx_chat_receiver_id ON chat_messages(receiver_id)",
                "CREATE INDEX idx_chat_created_at ON chat_messages(created_at)",
                "CREATE INDEX idx_chat_message_type ON chat_messages(message_type)"
            ]
            
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    conn.commit()
                except Exception as e:
                    print(f"⚠️  Warning: {e}")
            
            print("✅ Created indexes")
            
            # Insert seed data
            print("\n🌱 Inserting seed data...")
            
            # Admin user (password: admin123)
            admin_password = generate_password_hash('admin123')
            conn.execute(text("""
                INSERT INTO users (email, password_hash, full_name, role, is_active)
                VALUES (:email, :password_hash, :full_name, :role, :is_active)
            """), {
                'email': 'admin@woundcare.ai',
                'password_hash': admin_password,
                'full_name': 'System Admin',
                'role': 'admin',
                'is_active': True
            })
            conn.commit()
            print("✅ Created admin user (admin@woundcare.ai / admin123)")
            
            # Test doctor (password: doctor123)
            doctor_password = generate_password_hash('doctor123')
            conn.execute(text("""
                INSERT INTO users (email, password_hash, full_name, role, phone, is_active)
                VALUES (:email, :password_hash, :full_name, :role, :phone, :is_active)
                RETURNING id
            """), {
                'email': 'doctor@woundcare.ai',
                'password_hash': doctor_password,
                'full_name': 'Dr. John Smith',
                'role': 'doctor',
                'phone': '+1234567890',
                'is_active': True
            })
            doctor_user_id = conn.execute(text("SELECT id FROM users WHERE email = 'doctor@woundcare.ai'")).fetchone()[0]
            
            conn.execute(text("""
                INSERT INTO doctors (user_id, specialization, license_number, years_of_experience, hospital_affiliation, bio)
                VALUES (:user_id, :specialization, :license_number, :years_of_experience, :hospital_affiliation, :bio)
            """), {
                'user_id': doctor_user_id,
                'specialization': 'Wound Care Specialist',
                'license_number': 'MD12345',
                'years_of_experience': 10,
                'hospital_affiliation': 'General Hospital',
                'bio': 'Experienced wound care specialist with focus on diabetic foot ulcers'
            })
            conn.commit()
            print("✅ Created test doctor (doctor@woundcare.ai / doctor123)")
            
            # Test patient (password: patient123)
            patient_password = generate_password_hash('patient123')
            conn.execute(text("""
                INSERT INTO users (email, password_hash, full_name, role, phone, is_active)
                VALUES (:email, :password_hash, :full_name, :role, :phone, :is_active)
                RETURNING id
            """), {
                'email': 'patient@woundcare.ai',
                'password_hash': patient_password,
                'full_name': 'Jane Doe',
                'role': 'patient',
                'phone': '+0987654321',
                'is_active': True
            })
            patient_user_id = conn.execute(text("SELECT id FROM users WHERE email = 'patient@woundcare.ai'")).fetchone()[0]
            
            conn.execute(text("""
                INSERT INTO patients (user_id, date_of_birth, gender, diabetes_type, diabetes_duration_years, blood_sugar_level, hba1c_level)
                VALUES (:user_id, :date_of_birth, :gender, :diabetes_type, :diabetes_duration_years, :blood_sugar_level, :hba1c_level)
            """), {
                'user_id': patient_user_id,
                'date_of_birth': '1980-01-15',
                'gender': 'female',
                'diabetes_type': 'type2',
                'diabetes_duration_years': 5,
                'blood_sugar_level': 140.5,
                'hba1c_level': 7.2
            })
            conn.commit()
            print("✅ Created test patient (patient@woundcare.ai / patient123)")
            
            print("\n✅ Database initialization completed successfully!")
            print("\n📝 Test Accounts:")
            print("   Admin:   admin@woundcare.ai / admin123")
            print("   Doctor:  doctor@woundcare.ai / doctor123")
            print("   Patient: patient@woundcare.ai / patient123")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    init_database()
