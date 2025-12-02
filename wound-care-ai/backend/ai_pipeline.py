import torch
import cv2
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from sklearn.cluster import KMeans
from typing import Dict, Tuple
import os

class WoundAnalysisPipeline:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load SegFormerB3 model"""
        try:
            self.processor = SegformerImageProcessor.from_pretrained(
                "nvidia/segformer-b3-finetuned-ade-512-512"
            )
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b3-finetuned-ade-512-512"
            )
            
            # Load custom weights if available
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def segment_wound(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Segment wound from image"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        predicted_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()
        
        return np.array(image), predicted_mask
    
    def calculate_size(self, mask: np.ndarray, pixel_to_mm: float = 0.1) -> float:
        """Calculate wound size in mm²"""
        wound_pixels = np.sum(mask > 0)
        size_mm2 = wound_pixels * (pixel_to_mm ** 2)
        return size_mm2
    
    def analyze_color(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """Analyze wound color using K-means"""
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        wound_pixels = image_hsv[mask > 0]
        
        if len(wound_pixels) == 0:
            return {"error": "No wound detected"}
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(wound_pixels)
        
        colors = kmeans.cluster_centers_
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        percentages = counts / len(kmeans.labels_) * 100
        
        return {
            "dominant_colors": colors.tolist(),
            "color_percentages": percentages.tolist(),
            "color_distribution": dict(zip(labels.tolist(), percentages.tolist()))
        }
    
    def analyze_roughness(self, mask: np.ndarray) -> float:
        """Analyze wound surface roughness"""
        edges = cv2.Canny(mask.astype(np.uint8) * 255, 100, 200)
        roughness = np.sum(edges) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
        return float(roughness)
    
    def assess_risk_level(self, size: float, roughness: float, color_dist: Dict) -> str:
        """Assess wound risk level"""
        risk_score = 0
        
        if size > 100:
            risk_score += 3
        elif size > 50:
            risk_score += 2
        else:
            risk_score += 1
        
        if roughness > 0.5:
            risk_score += 2
        elif roughness > 0.3:
            risk_score += 1
        
        if risk_score >= 5:
            return "high"
        elif risk_score >= 3:
            return "medium"
        else:
            return "low"
    
    def analyze_wound(self, image_path: str) -> Dict:
        """Full wound analysis pipeline"""
        image, mask = self.segment_wound(image_path)
        size = self.calculate_size(mask)
        color_analysis = self.analyze_color(image, mask)
        roughness = self.analyze_roughness(mask)
        risk_level = self.assess_risk_level(size, roughness, color_analysis)
        
        return {
            "size_mm2": size,
            "color_analysis": color_analysis,
            "roughness_score": roughness,
            "risk_level": risk_level,
            "mask": mask
        }

