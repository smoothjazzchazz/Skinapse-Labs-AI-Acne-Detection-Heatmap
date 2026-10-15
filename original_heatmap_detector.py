"""
Original Heatmap Detector - EXACT original version from conversation start.
This is the untouched original AcneDetectorML class with COLORMAP_JET.
"""
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from scipy import ndimage
import joblib
import os

class AcneDetectorML:
    """
    Acne detection using machine learning (scikit-learn).
    Uses ensemble methods with feature extraction for robust acne detection.
    """
    
    def __init__(self):
        self.patch_size = 32
        self.model = None
        self.scaler = StandardScaler()
        self.trained = False
    
    def extract_features(self, patch):
        """
        Extract multi-scale features from image patch.
        Combines color, texture, and shape information.
        """
        features = []
        
        # 1. Color features (mean RGB in different color spaces)
        rgb_mean = patch.mean(axis=(0, 1))
        features.extend(rgb_mean)
        
        # Convert to HSV
        patch_hsv = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv_mean = patch_hsv.mean(axis=(0, 1))
        features.extend(hsv_mean)
        
        # 2. Texture features - Histogram of Oriented Gradients
        gray = cv2.cvtColor(patch.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        try:
            hog_features = hog(gray, orientations=8, pixels_per_cell=(4, 4),
                             cells_per_block=(2, 2), feature_vector=True)
            features.extend(hog_features[:50])  # Limit to top 50 HOG features
        except:
            features.extend(np.zeros(50))
        
        # 3. Local Binary Pattern (texture descriptor)
        try:
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=59, range=(0, 59))
            features.extend(lbp_hist[:20])  # Top 20 LBP features
        except:
            features.extend(np.zeros(20))
        
        # 4. Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (gray.size)
        features.append(edge_density)
        
        # 5. Statistical features
        features.append(gray.std())
        features.append(gray.var())
        
        # 6. Red/Saturation prominence (key for acne detection)
        h_channel = patch_hsv[:, :, 0]
        s_channel = patch_hsv[:, :, 1]
        features.append(np.mean(s_channel))  # High saturation = inflamed acne
        features.append(np.std(s_channel))
        
        return np.array(features, dtype=np.float32)
    
    def sliding_window_detection(self, image_path, output_path='acne_heatmap_ml.png', 
                                 stride=16, threshold=0.5):
        """
        Detect acne using sliding window with ML classifier.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image from {image_path}")
            return None, None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Create probability map
        prob_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        print("Scanning image for acne with ML model...")
        patch_count = 0
        
        for y in range(0, max(height - self.patch_size, 1), stride):
            for x in range(0, max(width - self.patch_size, 1), stride):
                y_end = min(y + self.patch_size, height)
                x_end = min(x + self.patch_size, width)
                
                patch = img_rgb[y:y_end, x:x_end]
                
                # Skip patches that are too small
                if patch.shape[0] < self.patch_size // 2 or patch.shape[1] < self.patch_size // 2:
                    continue
                
                # Pad patch if needed
                if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
                    patch = cv2.resize(patch, (self.patch_size, self.patch_size))
                
                # Extract features
                features = self.extract_features(patch)
                features = features.reshape(1, -1)
                
                # Predict
                if self.trained:
                    prob = self.model.predict_proba(features)[0, 1]
                else:
                    # Heuristic for untrained model: detect red, saturated areas
                    patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
                    h = patch_hsv[:, :, 0]
                    s = patch_hsv[:, :, 1]
                    # Red hue with high saturation
                    red_mask = ((h < 15) | (h > 165)) & (s > 100)
                    prob = red_mask.mean()
                
                # Add to probability map
                prob_map[y:y_end, x:x_end] += prob
                count_map[y:y_end, x:x_end] += 1
                
                patch_count += 1
        
        # Normalize
        prob_map = np.divide(prob_map, count_map, where=count_map > 0, 
                            out=np.zeros_like(prob_map))
        
        # Apply morphological operations for smoothing
        prob_map_8bit = (prob_map * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        prob_map_8bit = cv2.morphologyEx(prob_map_8bit, cv2.MORPH_CLOSE, kernel)
        prob_map_8bit = cv2.morphologyEx(prob_map_8bit, cv2.MORPH_OPEN, kernel)
        
        # Create acne mask
        acne_mask = (prob_map_8bit > threshold * 255).astype(np.uint8) * 255
        
        # Generate heatmap
        heatmap = cv2.applyColorMap(prob_map_8bit, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(img_rgb, 0.6, heatmap_rgb, 0.4, 0)
        
        # Save
        output_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)
        
        print(f"Processed {patch_count} patches")
        print(f"Heatmap saved to {output_path}")
        
        return overlay, acne_mask
    
    def train(self, training_data, training_labels):
        """
        Train the ML model on labeled data.
        
        Args:
            training_data: List of image patches
            training_labels: List of labels (1 = acne, 0 = no acne)
        """
        print("Extracting features from training data...")
        features_list = []
        
        for patch in training_data:
            features = self.extract_features(patch)
            features_list.append(features)
        
        X = np.array(features_list)
        y = np.array(training_labels)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        print("Training ensemble model...")
        # Use ensemble of models for robustness
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_scaled, y)
        self.trained = True
        
        print("Model trained successfully!")
    
    def save_model(self, filepath):
        """Save trained model and scaler."""
        if self.trained:
            joblib.dump(self.model, filepath + '.model')
            joblib.dump(self.scaler, filepath + '.scaler')
            print(f"Model saved to {filepath}")
        else:
            print("Model not trained yet!")
    
    def load_model(self, filepath):
        """Load trained model and scaler."""
        try:
            self.model = joblib.load(filepath + '.model')
            self.scaler = joblib.load(filepath + '.scaler')
            self.trained = True
            print(f"Model loaded from {filepath}")
        except:
            print(f"Failed to load model from {filepath}")


def main():
    """
    Original main - uses heuristic detection (no trained model).
    """
    from datetime import datetime
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use current directory structure
    input_image = os.path.join(script_dir, 'input', 'testImage.png')
    output_dir = os.path.join(script_dir, 'output', 'original_heatmap')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image = os.path.join(output_dir, f'original_{timestamp}.png')
    
    print("Initializing ML-based Acne Detector...")
    detector = AcneDetectorML()
    
    print(f"\nLooking for test image at: {input_image}")
    
    if not os.path.exists(input_image):
        print(f"Error: Test image not found at {input_image}")
        print("Please place a 'testImage.png' file in the input folder.")
        return
    
    print("Detecting acne on test image...")
    overlay, mask = detector.sliding_window_detection(
        input_image,
        output_path=output_image,
        stride=16,
        threshold=0.3
    )
    
    if overlay is not None:
        print("Detection complete!")
    
    print("\nNote: For best results, train the model on labeled acne dataset.")
    print("Use: detector.train(training_patches, training_labels)")

if __name__ == "__main__":
    main()
