from flask import Blueprint, request, jsonify
from datetime import timedelta
from database import get_db
from models import User
from auth import get_password_hash, verify_password, create_access_token
from config import settings

auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    db = next(get_db())
    
    try:
        # Check if username exists
        db_user = db.query(User).filter(User.username == data['username']).first()
        if db_user:
            return jsonify({"detail": "Username already registered"}), 400
        
        # Check if email exists
        db_user = db.query(User).filter(User.email == data['email']).first()
        if db_user:
            return jsonify({"detail": "Email already registered"}), 400
        
        # Create new user
        hashed_password = get_password_hash(data['password'])
        db_user = User(
            username=data['username'],
            email=data['email'],
            hashed_password=hashed_password,
            full_name=data.get('full_name'),
            role=data.get('role', 'patient')
        )
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        
        return jsonify({
            "id": db_user.id,
            "username": db_user.username,
            "email": db_user.email
        }), 201
    finally:
        db.close()

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    db = next(get_db())
    
    try:
        db_user = db.query(User).filter(User.username == data['username']).first()
        if not db_user or not verify_password(data['password'], db_user.hashed_password):
            return jsonify({"detail": "Invalid credentials"}), 401
        
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": db_user.username}, expires_delta=access_token_expires
        )
        
        return jsonify({
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": db_user.id,
                "username": db_user.username,
                "email": db_user.email,
                "role": db_user.role
            }
        })
    finally:
        db.close()
