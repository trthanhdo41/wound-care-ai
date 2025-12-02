from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
import shutil
import os
from database import get_db
from models import User, Patient, AnalysisResult
from auth import get_current_user
from ai_pipeline import WoundAnalysisPipeline
from config import settings
import json
from datetime import datetime

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
pipeline = WoundAnalysisPipeline(settings.MODEL_PATH)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload image and perform wound analysis"""
    
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can upload images")
    
    patient = db.query(Patient).filter(Patient.user_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    
    # Save uploaded file
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Analyze wound
    try:
        analysis_result = pipeline.analyze_wound(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    # Save to database
    db_result = AnalysisResult(
        patient_id=patient.id,
        image_path=file_path,
        size_mm2=analysis_result["size_mm2"],
        color_analysis=json.dumps(analysis_result["color_analysis"]),
        roughness_score=analysis_result["roughness_score"],
        risk_level=analysis_result["risk_level"]
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    
    return {
        "id": db_result.id,
        "size_mm2": db_result.size_mm2,
        "color_analysis": json.loads(db_result.color_analysis),
        "roughness_score": db_result.roughness_score,
        "risk_level": db_result.risk_level,
        "created_at": db_result.created_at
    }

@router.get("/history")
def get_analysis_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get patient's analysis history"""
    
    if current_user.role != "patient":
        raise HTTPException(status_code=403, detail="Only patients can view history")
    
    patient = db.query(Patient).filter(Patient.user_id == current_user.id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found")
    
    results = db.query(AnalysisResult).filter(
        AnalysisResult.patient_id == patient.id
    ).order_by(AnalysisResult.created_at.desc()).all()
    
    return [
        {
            "id": r.id,
            "size_mm2": r.size_mm2,
            "color_analysis": json.loads(r.color_analysis),
            "roughness_score": r.roughness_score,
            "risk_level": r.risk_level,
            "created_at": r.created_at
        }
        for r in results
    ]

@router.get("/{analysis_id}")
def get_analysis_detail(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed analysis result"""
    
    result = db.query(AnalysisResult).filter(AnalysisResult.id == analysis_id).first()
    if not result:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    patient = db.query(Patient).filter(Patient.id == result.patient_id).first()
    if patient.user_id != current_user.id and current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "id": result.id,
        "size_mm2": result.size_mm2,
        "color_analysis": json.loads(result.color_analysis),
        "roughness_score": result.roughness_score,
        "risk_level": result.risk_level,
        "notes": result.notes,
        "created_at": result.created_at
    }

