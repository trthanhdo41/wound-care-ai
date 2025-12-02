from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str
    role: str  # patient, doctor, admin

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class PatientCreate(BaseModel):
    age: int
    gender: str
    medical_history: Optional[str] = None

class AnalysisResultResponse(BaseModel):
    id: int
    patient_id: int
    size_mm2: float
    color_analysis: str
    roughness_score: float
    risk_level: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class ChatMessageCreate(BaseModel):
    receiver_id: Optional[int] = None
    message: str
    is_ai: bool = False

class ChatMessageResponse(BaseModel):
    id: int
    sender_id: int
    message: str
    is_ai: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

