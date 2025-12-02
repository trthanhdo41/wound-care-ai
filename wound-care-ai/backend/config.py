from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    DATABASE_URL: str = "mysql+mysqlconnector://root@localhost:3306/wound_care_ai"
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    MODEL_PATH: str = "model_files/segformer_wound.pth"
    DATASET_PATH: str = "../../Model/wound_features_with_risk.csv"
    COLOR_DATASET_PATH: str = "../../Model/color_features_ulcer_red_yellow_dark.csv"
    OPENAI_API_KEY: Optional[str] = None
    
    # Google OAuth
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")
    GOOGLE_CLIENT_SECRET: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    FE_URL: str = "http://localhost:3000"
    BE_URL: str = "http://localhost:5001"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Export for easy access
CLIENT_ID = settings.GOOGLE_CLIENT_ID
CLIENT_SECRET = settings.GOOGLE_CLIENT_SECRET
FE_URL = settings.FE_URL
BE_URL = settings.BE_URL

