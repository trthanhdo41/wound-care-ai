"""
AI Wound Analysis Pipeline
Integrates: Segmentation -> Size -> Color (K-means) -> Roughness (GLCM) -> Risk Assessment
Based on client's Jupyter notebook implementation
"""
import torch
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class WoundAnalyzer:
    def __init__(self, model_path, device='cpu', universal_k=3, dataset_path=None, color_dataset_path=None):
        """Initialize the wound analyzer with SegFormer model"""
        self.device = torch.device(device)
        self.img_size = 512
        self.universal_k = universal_k  # K for K-means clustering
        self.model = self._load_model(model_path)
        self.transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        
        # Initialize K-means models
        self.kmeans_model = None  # For color clustering (per-image)
        self.color_kmeans_model = None  # For color classification (from dataset)
        self.color_scaler = None  # For color feature scaling
        self.risk_kmeans_model = None  # For risk assessment
        self.risk_scaler = None  # For feature scaling
        
        # Load risk assessment model from dataset
        if dataset_path and os.path.exists(dataset_path):
            self._load_risk_model(dataset_path)
        
        # Load color classification model from dataset
        if color_dataset_path and os.path.exists(color_dataset_path):
            self._load_color_model(color_dataset_path)

    def _load_model(self, model_path):
        """Load SegFormer model using segmentation_models_pytorch"""
        try:
            # Initialize model architecture (same as training)
            model = smp.Unet(
                encoder_name="mit_b3",  # SegFormer-B3 backbone
                encoder_weights=None,    # Will load from .pth file
                in_channels=3,
                classes=1,               # Binary segmentation
            )

            # Load custom weights
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"✅ Loaded model from {model_path}")
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def segment_wound(self, image_path):
        """Segment wound from image"""
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]  # (H, W)

        # Transform and predict
        transformed = self.transform(image=image)
        input_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.sigmoid(logits)
            pred_mask = (probabilities > 0.5).float()

        # Get mask as numpy array
        pred_mask = pred_mask.squeeze().cpu().numpy()

        # Resize mask back to original size if needed
        if pred_mask.shape != original_size:
            pred_mask = cv2.resize(pred_mask, (original_size[1], original_size[0]), 
                                   interpolation=cv2.INTER_NEAREST)

        return pred_mask, image

    def calculate_size(self, mask, pixel_to_cm_ratio=0.1):
        """Calculate wound size metrics"""
        # Label connected components
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        if not regions:
            return None

        # Get largest region (main wound)
        largest_region = max(regions, key=lambda r: r.area)

        # Calculate metrics
        area_pixels = largest_region.area
        area_cm2 = area_pixels * (pixel_to_cm_ratio ** 2)

        perimeter_pixels = largest_region.perimeter
        perimeter_cm = perimeter_pixels * pixel_to_cm_ratio

        bbox = largest_region.bbox
        height_pixels = bbox[2] - bbox[0]
        width_pixels = bbox[3] - bbox[1]
        height_cm = height_pixels * pixel_to_cm_ratio
        width_cm = width_pixels * pixel_to_cm_ratio

        return {
            'area_cm2': round(area_cm2, 2),
            'perimeter_cm': round(perimeter_cm, 2),
            'width_cm': round(width_cm, 2),
            'height_cm': round(height_cm, 2)
        }

    def analyze_color(self, image, mask):
        """
        Analyze wound color distribution using K-means clustering
        Following client's implementation: RGB -> HSV normalized [0,1] -> K-means
        """
        # Extract wound pixels
        wound_pixels = image[mask > 0]

        if len(wound_pixels) == 0:
            return None

        # Convert RGB to HSV (OpenCV: H∈[0,179], S,V∈[0,255])
        wound_pixels_reshaped = wound_pixels.reshape(-1, 1, 3).astype(np.uint8)
        wound_pixels_hsv = cv2.cvtColor(wound_pixels_reshaped, cv2.COLOR_RGB2HSV).reshape(-1, 3)
        
        # Normalize to [0,1] (exactly like client's code)
        h = wound_pixels_hsv[:, 0].astype(np.float64) / 179.0
        s = wound_pixels_hsv[:, 1].astype(np.float64) / 255.0
        v = wound_pixels_hsv[:, 2].astype(np.float64) / 255.0
        X_hsv = np.stack([h, s, v], axis=1)

        # Use pre-trained global cluster centers if available
        if hasattr(self, 'global_color_centers') and self.global_color_centers is not None:
            # Use pre-trained centers (like client's approach)
            from sklearn.metrics import pairwise_distances
            
            # Assign each pixel to nearest cluster center
            distances = pairwise_distances(X_hsv, self.global_color_centers, metric='euclidean')
            labels = np.argmin(distances, axis=1)
            
            # Create a dummy kmeans object for compatibility
            self.kmeans_model = type('obj', (object,), {
                'cluster_centers_': self.global_color_centers,
                'predict': lambda x: np.argmin(pairwise_distances(x, self.global_color_centers, metric='euclidean'), axis=1)
            })()
        else:
            # Fallback: train fresh K-means on this image (will differ from demo)
            self.kmeans_model = KMeans(
                n_clusters=self.universal_k,
                n_init=10,
                random_state=42,
                verbose=0
            )
            self.kmeans_model.fit(X_hsv)
            labels = self.kmeans_model.predict(X_hsv)
        
        # Calculate percentage for each cluster
        total_pixels = len(labels)
        cluster_percentages = {}
        cluster_colors = {}
        
        for i in range(self.universal_k):
            count = np.sum(labels == i)
            percentage = (count / total_pixels) * 100
            cluster_percentages[f'cluster_{i}_pct'] = round(percentage, 2)
            
            # Get representative color (cluster center)
            center_color = self.kmeans_model.cluster_centers_[i].astype(np.uint8)
            cluster_colors[f'cluster_{i}_rgb'] = center_color.tolist()
        
        # Use pre-trained cluster mapping if available
        color_counts = {'Red': 0, 'Yellow': 0, 'Dark': 0}
        
        if hasattr(self, 'cluster_to_color'):
            # Use pre-trained cluster mapping
            for i in range(self.universal_k):
                cluster_size = np.sum(labels == i)
                color_name = self.cluster_to_color.get(i, 'Red')
                color_counts[color_name] += cluster_size
        else:
            # Fallback: classify based on HSV characteristics
            for i in range(self.universal_k):
                cluster_size = np.sum(labels == i)
                
                if cluster_size == 0:
                    continue
                
                h_norm, s_norm, v_norm = self.kmeans_model.cluster_centers_[i]
                h = h_norm * 179
                s = s_norm * 255
                v = v_norm * 255
                
                if v < 100:
                    color_counts['Dark'] += cluster_size
                elif 15 <= h <= 50 and s > 50:
                    color_counts['Yellow'] += cluster_size
                elif (h <= 15 or h >= 160) and s > 50:
                    color_counts['Red'] += cluster_size
                else:
                    if v > 200:
                        color_counts['Yellow'] += cluster_size
                    else:
                        color_counts['Red'] += cluster_size
        
        # Calculate percentages
        total = len(wound_pixels)
        red_pct = (color_counts['Red'] / total) * 100
        yellow_pct = (color_counts['Yellow'] / total) * 100
        dark_pct = (color_counts['Dark'] / total) * 100
        
        print(f"Color classification: Red={red_pct:.1f}%, Yellow={yellow_pct:.1f}%, Dark={dark_pct:.1f}%")
        
        # Map clusters to tissue types based on their dominant color
        tissue_types = {}
        for i in range(self.universal_k):
            r, g, b = self.kmeans_model.cluster_centers_[i]
            rgb_pixel = np.uint8([[[r, g, b]]])
            hsv_pixel = cv2.cvtColor(rgb_pixel, cv2.COLOR_RGB2HSV)[0][0]
            h, s, v = hsv_pixel
            
            if v < 80:
                tissue_types[f'cluster_{i}'] = 'Dark'
            elif 20 <= h <= 40 and s > 30:
                tissue_types[f'cluster_{i}'] = 'Yellow'
            else:
                tissue_types[f'cluster_{i}'] = 'Red'

        return {
            'cluster_percentages': cluster_percentages,
            'cluster_colors': cluster_colors,
            'tissue_types': tissue_types,
            'total_wound_pixels': total_pixels,
            # Add pixel-based percentages (more accurate than cluster-based)
            'pixel_based_percentages': {
                'Red': round(red_pct, 2),
                'Yellow': round(yellow_pct, 2),
                'Dark': round(dark_pct, 2)
            }
        }

    def calculate_roughness(self, image, mask):
        """
        Calculate wound surface roughness using GLCM (Gray-Level Co-occurrence Matrix)
        Following client's implementation with texture analysis
        """
        if np.sum(mask) == 0:
            return None
        
        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        try:
            # Find bounding box to crop ROI (for performance)
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                return None
                
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Crop ROI
            roi_gray = gray_image[rmin:rmax+1, cmin:cmax+1]
            roi_mask = mask[rmin:rmax+1, cmin:cmax+1]
            
            # Set background to 0
            roi_gray_masked = roi_gray.copy()
            roi_gray_masked[roi_mask == 0] = 0
            
            # Downsample for performance (if too large)
            if roi_gray_masked.shape[0] > 256 or roi_gray_masked.shape[1] > 256:
                scale = 256 / max(roi_gray_masked.shape)
                new_size = (int(roi_gray_masked.shape[1] * scale), 
                           int(roi_gray_masked.shape[0] * scale))
                roi_gray_masked = cv2.resize(roi_gray_masked, new_size, 
                                            interpolation=cv2.INTER_AREA)
            
            # Normalize to 0-255 range
            if roi_gray_masked.max() > 0:
                roi_gray_masked = ((roi_gray_masked - roi_gray_masked.min()) / 
                                  (roi_gray_masked.max() - roi_gray_masked.min()) * 255).astype(np.uint8)
            
            # Calculate GLCM
            # distances: [1] - adjacent pixels
            # angles: [0, π/4, π/2, 3π/4] - 4 directions
            distances = [1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            glcm = graycomatrix(
                roi_gray_masked,
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normed=True
            )
            
            # Calculate texture properties
            contrast = graycoprops(glcm, 'contrast').mean()
            dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
            homogeneity = graycoprops(glcm, 'homogeneity').mean()
            energy = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            
            # Calculate roughness score (higher = rougher surface)
            # Contrast and dissimilarity indicate roughness
            # Homogeneity indicates smoothness (inverse of roughness)
            roughness_score = (contrast + dissimilarity) / (homogeneity + 1e-6)
            
            return {
                'roughness_score': round(roughness_score, 3),
                'contrast': round(contrast, 3),
                'dissimilarity': round(dissimilarity, 3),
                'homogeneity': round(homogeneity, 3),
                'energy': round(energy, 3),
                'correlation': round(correlation, 3)
            }
            
        except Exception as e:
            print(f"Error calculating roughness: {e}")
            return None

    def _load_color_model(self, color_dataset_path):
        """
        Load pre-trained K-means cluster centers for color classification
        Client trained K-means on full dataset and saved cluster centers
        We need to use the SAME cluster centers for consistent results
        """
        try:
            # Try to load pre-trained cluster centers if available
            import os
            centers_file = os.path.join(os.path.dirname(color_dataset_path), 'kmeans_color_cluster_centers.csv')
            
            if os.path.exists(centers_file):
                import pandas as pd
                df_centers = pd.read_csv(centers_file)
                self.global_color_centers = df_centers[['H_norm', 'S_norm', 'V_norm']].values
                
                # Analyze cluster centers to determine mapping
                # Cluster 0: H=0.986 (~176°, wraps around to red), S=0.58, V=0.61
                # Cluster 1: H=0.043 (~8°, red-orange), S=0.45, V=0.78 (brightest)
                # Cluster 2: H=0.044 (~8°, red-orange), S=0.53, V=0.39 (darkest)
                
                # Based on medical interpretation:
                # - Brightest (V=0.78) = Yellow-like (healing/epithelial tissue)
                # - Medium (V=0.61) = Red-like (granulation tissue)
                # - Darkest (V=0.39) = Dark-like (necrotic tissue)
                self.cluster_to_color = {1: 'Yellow', 0: 'Red', 2: 'Dark'}
                
                print(f"✅ Loaded pre-trained color cluster centers from {centers_file}")
                print(f"   Centers shape: {self.global_color_centers.shape}")
                print(f"   Cluster mapping: {self.cluster_to_color}")
            else:
                print(f"⚠️ Pre-trained cluster centers not found at {centers_file}")
                print(f"   Using estimated typical wound color centers")
                
                # Fallback estimated centers
                self.global_color_centers = np.array([
                    [0.028, 0.500, 0.700],  # Red-like
                    [0.167, 0.400, 0.850],  # Yellow-like
                    [0.100, 0.200, 0.350]   # Dark-like
                ])
                self.cluster_to_color = {0: 'Red', 1: 'Yellow', 2: 'Dark'}
                print(f"   Using estimated centers (may differ from demo)")
                
            self.color_kmeans_model = 'global'  # Flag to use global centers
                
        except Exception as e:
            print(f"⚠️ Could not load color model: {e}")
            self.color_kmeans_model = None
            self.global_color_centers = None

    def _load_risk_model(self, dataset_path):
        """Load and train K-means model for risk assessment from dataset"""
        try:
            import pandas as pd
            
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            # Select features for risk assessment (using new column names)
            feature_cols = [
                'pct_red_like', 'pct_yellow_like', 'pct_dark_like',
                'area_pixels', 'perimeter', 'circularity',
                'texture_contrast', 'texture_homogeneity', 'texture_energy', 'texture_correlation'
            ]
            
            # Check if all features exist
            if all(col in df.columns for col in feature_cols):
                X = df[feature_cols].values
                
                # Standardize features
                self.risk_scaler = StandardScaler()
                X_scaled = self.risk_scaler.fit_transform(X)
                
                # Train K-means with K=3 (Low, Medium, High risk)
                self.risk_kmeans_model = KMeans(n_clusters=3, random_state=42, n_init=10)
                cluster_labels = self.risk_kmeans_model.fit_predict(X_scaled)
                
                # Map clusters to risk levels based on actual risk_level in dataset
                if 'risk_level' in df.columns:
                    # Use actual risk labels from dataset
                    cluster_risk_mapping = {}
                    for i in range(3):
                        cluster_data = df[cluster_labels == i]
                        # Get most common risk level in this cluster
                        most_common_risk = cluster_data['risk_level'].mode()[0] if len(cluster_data) > 0 else 'medium'
                        cluster_risk_mapping[i] = most_common_risk.lower()
                    self.cluster_to_risk = cluster_risk_mapping
                else:
                    # Fallback: calculate severity score
                    cluster_means = []
                    for i in range(3):
                        cluster_data = df[cluster_labels == i]
                        mean_area = cluster_data['area_pixels'].mean()
                        mean_dark = cluster_data['pct_dark_like'].mean()
                        mean_contrast = cluster_data['texture_contrast'].mean()
                        # Higher area, dark %, and contrast = higher risk
                        severity_score = mean_area * 0.3 + mean_dark * 10 + mean_contrast * 0.1
                        cluster_means.append((i, severity_score))
                    
                    # Sort by severity score
                    cluster_means.sort(key=lambda x: x[1])
                    
                    # Map: lowest severity = Low, middle = Medium, highest = High
                    self.cluster_to_risk = {
                        cluster_means[0][0]: 'low',
                        cluster_means[1][0]: 'medium',
                        cluster_means[2][0]: 'high'
                    }
                
                print(f"✅ Loaded risk assessment model from {dataset_path}")
                print(f"   Cluster mapping: {self.cluster_to_risk}")
            else:
                missing_cols = [col for col in feature_cols if col not in df.columns]
                print(f"⚠️ Missing features in dataset: {missing_cols}")
                print(f"   Using rule-based risk assessment")
                self.risk_kmeans_model = None
                
        except Exception as e:
            print(f"⚠️ Could not load risk model: {e}")
            self.risk_kmeans_model = None

    def assess_risk(self, size_metrics, color_analysis, roughness_analysis):
        """
        Assess overall wound risk level using K-means clustering
        If K-means model is available, use it; otherwise fall back to rule-based
        """
        # Try K-means based risk assessment first
        if self.risk_kmeans_model is not None and self.risk_scaler is not None:
            try:
                # Extract features
                features = []
                
                # Tissue percentages (use pixel_based_percentages)
                if color_analysis and 'pixel_based_percentages' in color_analysis:
                    red_pct = color_analysis['pixel_based_percentages'].get('Red', 0)
                    yellow_pct = color_analysis['pixel_based_percentages'].get('Yellow', 0)
                    dark_pct = color_analysis['pixel_based_percentages'].get('Dark', 0)
                    features.extend([red_pct, yellow_pct, dark_pct])
                else:
                    features.extend([0, 0, 0])
                
                # Size metrics (convert cm² to pixels approximately)
                if size_metrics:
                    area_pixels = size_metrics['area_cm2'] * 100  # Rough conversion
                    perimeter = size_metrics['perimeter_cm'] * 10
                    # Calculate circularity
                    if perimeter > 0:
                        circularity = (4 * np.pi * area_pixels) / (perimeter ** 2)
                    else:
                        circularity = 0
                    features.extend([area_pixels, perimeter, circularity])
                else:
                    features.extend([0, 0, 0])
                
                # Texture metrics
                if roughness_analysis:
                    features.extend([
                        roughness_analysis.get('contrast', 0),
                        roughness_analysis.get('homogeneity', 0),
                        roughness_analysis.get('energy', 0),
                        roughness_analysis.get('correlation', 0)
                    ])
                else:
                    features.extend([0, 0, 0, 0])
                
                # Scale features
                features_array = np.array(features).reshape(1, -1)
                features_scaled = self.risk_scaler.transform(features_array)
                
                # Predict cluster
                cluster_id = self.risk_kmeans_model.predict(features_scaled)[0]
                risk_level = self.cluster_to_risk.get(cluster_id, 'medium')
                
                # Use risk_level directly from K-means prediction (no adjustment)
                # Map risk_level to score for display
                risk_score_map = {'low': 25, 'medium': 55, 'high': 85}
                risk_score = risk_score_map.get(risk_level, 50)
                
                # Generate risk factors
                risk_factors = []
                if dark_pct > 10:
                    risk_factors.append(f'Necrotic tissue present ({dark_pct:.1f}%)')
                if yellow_pct > 60:
                    risk_factors.append(f'Good healing tissue ({yellow_pct:.1f}%)')
                if red_pct > 70:
                    risk_factors.append(f'High granulation tissue ({red_pct:.1f}%)')
                if size_metrics and size_metrics['area_cm2'] > 5:
                    risk_factors.append(f"Large ulcer area ({size_metrics['area_cm2']:.1f} cm²)")
                if roughness_analysis and roughness_analysis.get('contrast', 0) > 500:
                    risk_factors.append('Irregular ulcer surface texture')
                
                # Recommendation
                if risk_level == 'high':
                    recommendation = 'Urgent medical consultation recommended. Close monitoring needed.'
                elif risk_level == 'medium':
                    recommendation = 'Regular medical follow-up advised. Monitor for changes.'
                else:
                    recommendation = 'Continue current care routine. Routine check-ups recommended.'
                
                # Generate care guidelines based on risk and tissue composition
                care_guidelines = self._generate_care_guidelines(risk_level, dark_pct, yellow_pct, red_pct, size_metrics)
                
                return {
                    'risk_score': risk_score,
                    'risk_level': risk_level,
                    'risk_factors': risk_factors if risk_factors else ['Wound assessment completed'],
                    'recommendation': recommendation,
                    'care_guidelines': care_guidelines
                }
                
            except Exception as e:
                print(f"❌ K-means risk assessment failed: {e}")
                raise Exception(f"Risk assessment failed: {e}")
        
        # If K-means model not loaded, raise error (no fallback)
        raise Exception("Risk assessment model not loaded. Cannot perform risk assessment without K-means model.")
    


    def _generate_care_guidelines(self, risk_level, dark_pct, yellow_pct, red_pct, size_metrics):
        """Generate personalized care guidelines based on wound characteristics"""
        guidelines = []
        
        # Basic wound care (always include)
        guidelines.append({
            'icon': 'check',
            'text': 'Keep the ulcer clean and dry with sterile dressings'
        })
        
        # Guidelines based on tissue composition
        if dark_pct > 10:
            # Necrotic tissue present
            guidelines.append({
                'icon': 'alert',
                'text': 'Seek immediate medical attention for necrotic tissue removal'
            })
            guidelines.append({
                'icon': 'eye',
                'text': 'Monitor closely for signs of infection (increased pain, odor, discharge)'
            })
        elif yellow_pct > 60:
            # Good healing progress
            guidelines.append({
                'icon': 'check',
                'text': 'Continue current treatment - ulcer is healing well'
            })
            guidelines.append({
                'icon': 'activity',
                'text': 'Maintain healthy nutrition to support tissue regeneration'
            })
        else:
            # Moderate healing
            guidelines.append({
                'icon': 'clock',
                'text': 'Follow prescribed medication and dressing change schedule'
            })
            guidelines.append({
                'icon': 'eye',
                'text': 'Monitor for signs of infection or delayed healing'
            })
        
        # Guidelines based on wound size
        if size_metrics and size_metrics['area_cm2'] > 10:
            guidelines.append({
                'icon': 'alert',
                'text': f"Large ulcer area ({size_metrics['area_cm2']:.1f} cm²) - requires professional care"
            })
        
        # Guidelines based on risk level
        if risk_level == 'high':
            guidelines.append({
                'icon': 'alert',
                'text': 'Schedule urgent follow-up appointment within 24-48 hours'
            })
        elif risk_level == 'medium':
            guidelines.append({
                'icon': 'clock',
                'text': 'Schedule follow-up appointment within 1-2 weeks'
            })
        else:
            guidelines.append({
                'icon': 'check',
                'text': 'Schedule routine follow-up appointment as recommended'
            })
        
        # Blood sugar management (for diabetic wounds)
        guidelines.append({
            'icon': 'activity',
            'text': 'Maintain healthy blood sugar levels to promote healing'
        })
        
        # Limit to 5-6 most relevant guidelines
        return guidelines[:6]
    
    def create_visualization(self, image, mask, output_path):
        """Create visualization with mask overlay"""
        # Create colored mask overlay
        overlay = image.copy()
        overlay[mask > 0] = [255, 0, 0]  # Red for wound

        # Blend original and overlay
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

        # Draw contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        # Save
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        return output_path
    
    def create_wound_zoom(self, image, mask, output_path):
        """Create zoomed view of wound area with red overlay"""
        # Find bounding box of wound
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # If no wound detected, return original
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            return output_path
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding (20%)
        padding_h = int((rmax - rmin) * 0.2)
        padding_w = int((cmax - cmin) * 0.2)
        
        rmin = max(0, rmin - padding_h)
        rmax = min(image.shape[0], rmax + padding_h)
        cmin = max(0, cmin - padding_w)
        cmax = min(image.shape[1], cmax + padding_w)
        
        # Crop image and mask
        cropped_image = image[rmin:rmax, cmin:cmax].copy()
        cropped_mask = mask[rmin:rmax, cmin:cmax]
        
        # Create red overlay
        overlay = cropped_image.copy()
        overlay[cropped_mask > 0] = [255, 100, 100]  # Light red
        
        # Blend
        result = cv2.addWeighted(cropped_image, 0.5, overlay, 0.5, 0)
        
        # Save
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        return output_path
    
    def create_gradcam_heatmap(self, image, mask, output_path):
        """Create Grad-CAM: Zoom into wound + overlay heatmap on grayscale background"""
        # Find bounding box of wound to zoom in
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # If no wound, just save grayscale image
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(output_path, cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR))
            return output_path
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding (20%)
        padding_h = int((rmax - rmin) * 0.2)
        padding_w = int((cmax - cmin) * 0.2)
        
        rmin = max(0, rmin - padding_h)
        rmax = min(image.shape[0], rmax + padding_h)
        cmin = max(0, cmin - padding_w)
        cmax = min(image.shape[1], cmax + padding_w)
        
        # Crop image and mask to wound region
        cropped_image = image[rmin:rmax, cmin:cmax].copy()
        cropped_mask = mask[rmin:rmax, cmin:cmax]
        
        # Convert cropped image to grayscale
        gray_cropped = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2GRAY)
        gray_cropped_rgb = cv2.cvtColor(gray_cropped, cv2.COLOR_GRAY2RGB)
        
        # Create attention map using distance transform
        attention_map = np.zeros(cropped_mask.shape, dtype=np.float32)
        
        if np.sum(cropped_mask) > 0:
            # Distance transform - gradient from center to edges
            dist_transform = cv2.distanceTransform(cropped_mask.astype(np.uint8), cv2.DIST_L2, 5)
            
            if dist_transform.max() > 0:
                attention_map = dist_transform / dist_transform.max()
                # Smooth gradient
                attention_map = np.power(attention_map, 0.5)
        
        # Normalize to 0-255
        attention_map = (attention_map * 255).astype(np.uint8)
        
        # Apply JET colormap (blue -> green -> yellow -> red)
        heatmap = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Create smooth mask for blending
        mask_smooth = cv2.GaussianBlur(cropped_mask.astype(np.float32), (15, 15), 0)
        mask_smooth = np.expand_dims(mask_smooth, axis=2)
        
        # Blend: grayscale background + heatmap overlay (strong 0.8 opacity)
        result = gray_cropped_rgb.astype(np.float32)
        result = result * (1 - mask_smooth * 0.8) + heatmap.astype(np.float32) * (mask_smooth * 0.8)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Save
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        return output_path

    def analyze_full(self, image_path, output_dir):
        """Full analysis pipeline"""
        try:
            # 1. Segmentation
            mask, image = self.segment_wound(image_path)

            # 2. Size analysis
            size_metrics = self.calculate_size(mask)

            # 3. Color analysis
            color_analysis = self.analyze_color(image, mask)

            # 4. Roughness (GLCM texture analysis)
            roughness_analysis = self.calculate_roughness(image, mask)

            # 5. Risk assessment
            risk_assessment = self.assess_risk(size_metrics, color_analysis, roughness_analysis)

            # 6. Create visualizations
            os.makedirs(output_dir, exist_ok=True)

            # Save segmented mask
            segmented_path = os.path.join(output_dir, 'segmented.png')
            cv2.imwrite(segmented_path, (mask * 255).astype(np.uint8))

            # Save visualization (original overlay)
            viz_path = os.path.join(output_dir, 'visualization.png')
            self.create_visualization(image, mask, viz_path)
            
            # Save wound zoom
            wound_zoom_path = os.path.join(output_dir, 'wound_zoom.png')
            self.create_wound_zoom(image, mask, wound_zoom_path)
            
            # Save Grad-CAM heatmap
            gradcam_path = os.path.join(output_dir, 'gradcam.png')
            self.create_gradcam_heatmap(image, mask, gradcam_path)

            # Compile results
            results = {
                'size_metrics': size_metrics,
                'color_analysis': color_analysis,
                'roughness_analysis': roughness_analysis,
                'risk_assessment': risk_assessment,
                'segmented_image': segmented_path,
                'visualization_image': viz_path,
                'wound_zoom_image': wound_zoom_path,
                'gradcam_image': gradcam_path
            }

            return results

        except Exception as e:
            print(f"❌ Error in analysis: {e}")
            raise

