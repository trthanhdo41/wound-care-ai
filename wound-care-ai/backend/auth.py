from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from flask import request, jsonify
from functools import wraps
from config import settings
from sqlalchemy.orm import Session
from database import get_db
from models import User

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def get_current_user(f):
    """Decorator to get current user from JWT token"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        
        if auth_header:
            try:
                token = auth_header.split(" ")[1]  # Bearer <token>
            except IndexError:
                return jsonify({"detail": "Invalid token format"}), 401
        
        if not token:
            return jsonify({"detail": "Token is missing"}), 401
        
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            username = payload.get("sub")
            if username is None:
                return jsonify({"detail": "Invalid token"}), 401
            
            db = next(get_db())
            user = db.query(User).filter(User.username == username).first()
            db.close()
            
            if user is None:
                return jsonify({"detail": "User not found"}), 401
            
            return f(user, *args, **kwargs)
        except JWTError:
            return jsonify({"detail": "Could not validate credentials"}), 401
    
    return decorated_function

