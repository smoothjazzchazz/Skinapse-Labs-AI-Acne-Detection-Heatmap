"""
Simple Heatmap Detector - Original basic version
Creates a thermal-style heatmap (blue -> green -> yellow -> red) 
using sliding window detection without complex severity analysis.

Directory Structure:
    input/          - Place input images here
    output/simple_heatmap/  - Output heatmaps saved here
    models/simple_heatmap/  - Trained model files
"""

import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
import joblib
import os
import json
import random
from datetime import datetime


class SimpleHeatmapDetector:
    """
    Simple acne detection using sliding window and ML/heuristic detection.
    Outputs a standard thermal heatmap (COLORMAP_JET).
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
        s_channel = patch_hsv[:, :, 1]
        features.append(np.mean(s_channel))  # High saturation = inflamed acne
        features.append(np.std(s_channel))
        
        return np.array(features, dtype=np.float32)
    
    def detect(self, image_path, output_path='heatmap.png', stride=5, threshold=0.3, use_heuristic=False):
        """
        Detect acne using sliding window and create thermal heatmap.
        
        Args:
            image_path: Path to input image
            output_path: Path for output heatmap
            stride: Sliding window stride (smaller = more detail, slower)
            threshold: Detection threshold for mask (0-1)
            use_heuristic: Force heuristic mode (ignore trained model)
        
        Returns:
            overlay: Heatmap overlaid on original image
            acne_mask: Binary mask of detected areas
        """
        # Temporarily disable trained model if heuristic mode requested
        original_trained = self.trained
        if use_heuristic:
            self.trained = False
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image from {image_path}")
            return None, None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        print(f"Processing: {image_path}")
        print(f"Image size: {width}x{height}")
        
        # Create probability map
        prob_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        print("Scanning image for acne...")
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
                
                # Predict probability
                if self.trained:
                    features = self.extract_features(patch)
                    features = features.reshape(1, -1)
                    features_scaled = self.scaler.transform(features)
                    prob = self.model.predict_proba(features_scaled)[0, 1]
                else:
                    # Heuristic detection: red, saturated areas
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
        
        print(f"Processed {patch_count} patches")
        
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
        
        # Generate thermal heatmap (blue -> green -> yellow -> red)
        heatmap = cv2.applyColorMap(prob_map_8bit, cv2.COLORMAP_JET)
        
        # Overlay heatmap on original
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Save
        cv2.imwrite(output_path, overlay)
        
        print(f"Heatmap saved to: {output_path}")
        
        # Restore trained status if we temporarily disabled it
        if use_heuristic:
            self.trained = original_trained
        
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
        
        print("Training model...")
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
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


def load_coco_dataset(dataset_path, patch_size=32, negative_ratio=1.0):
    """
    Load training data from COCO format dataset (Roboflow export).
    
    Args:
        dataset_path: Path to the downloaded dataset folder
        patch_size: Size of patches to extract
        negative_ratio: Ratio of negative samples to positive samples
    
    Returns:
        training_data: List of image patches
        training_labels: List of labels (1 = acne, 0 = no acne)
    """
    training_data = []
    training_labels = []
    
    # Look for annotations file in common locations
    possible_annotation_paths = [
        os.path.join(dataset_path, '_annotations.coco.json'),
        os.path.join(dataset_path, 'train', '_annotations.coco.json'),
        os.path.join(dataset_path, 'annotations.json'),
    ]
    
    annotations_file = None
    images_dir = None
    
    for ann_path in possible_annotation_paths:
        if os.path.exists(ann_path):
            annotations_file = ann_path
            images_dir = os.path.dirname(ann_path)
            break
    
    if annotations_file is None:
        print(f"Error: Could not find COCO annotations file in {dataset_path}")
        return [], []
    
    print(f"Loading annotations from: {annotations_file}")
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build image id to filename mapping
    image_map = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"Found {len(coco_data['images'])} images with {len(coco_data['annotations'])} annotations")
    
    positive_count = 0
    negative_count = 0
    
    # Limit to 50 images for speed
    image_list = list(image_map.items())[:50]
    
    for idx, (img_id, filename) in enumerate(image_list):
        if idx % 10 == 0:
            print(f"  Processing {idx + 1}/{len(image_list)}...")
        
        img_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(img_path):
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # Get annotations for this image
        annotations = annotations_by_image.get(img_id, [])
        
        # Extract positive patches (acne regions)
        acne_regions = []
        for ann in annotations:
            bbox = ann['bbox']  # COCO format: [x, y, width, height]
            x, y, w, h = [int(v) for v in bbox]
            
            center_x = x + w // 2
            center_y = y + h // 2
            
            x1 = max(0, center_x - patch_size // 2)
            y1 = max(0, center_y - patch_size // 2)
            x2 = min(width, x1 + patch_size)
            y2 = min(height, y1 + patch_size)
            
            patch = img_rgb[y1:y2, x1:x2]
            
            if patch.shape[0] >= patch_size // 2 and patch.shape[1] >= patch_size // 2:
                patch = cv2.resize(patch, (patch_size, patch_size))
                training_data.append(patch)
                training_labels.append(1)  # Acne
                positive_count += 1
                acne_regions.append((x1, y1, x2, y2))
        
        # Extract negative patches (non-acne regions)
        attempts = 0
        max_attempts = 50
        
        while negative_count < positive_count * negative_ratio and attempts < max_attempts:
            attempts += 1
            
            rx = random.randint(0, max(0, width - patch_size))
            ry = random.randint(0, max(0, height - patch_size))
            
            # Check if overlaps with any acne region
            overlaps = False
            for (ax1, ay1, ax2, ay2) in acne_regions:
                if not (rx + patch_size < ax1 or rx > ax2 or ry + patch_size < ay1 or ry > ay2):
                    overlaps = True
                    break
            
            if not overlaps:
                patch = img_rgb[ry:ry+patch_size, rx:rx+patch_size]
                if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                    training_data.append(patch)
                    training_labels.append(0)  # No acne
                    negative_count += 1
    
    print(f"Extracted {positive_count} positive patches (acne)")
    print(f"Extracted {negative_count} negative patches (no acne)")
    
    return training_data, training_labels


def main():
    import sys
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Directory structure (matches project ecosystem)
    input_dir = os.path.join(script_dir, 'input')
    output_dir = os.path.join(script_dir, 'output', 'simple_heatmap')
    models_dir = os.path.join(script_dir, 'models', 'simple_heatmap')
    dataset_dir = os.path.join(script_dir, 'acne_dataset')
    
    # Create directories if needed
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Check for --heuristic flag
    use_heuristic = '--heuristic' in sys.argv or '-h' in sys.argv
    
    # Timestamped output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = '_heuristic' if use_heuristic else ''
    
    input_image = os.path.join(input_dir, 'testImage.png')
    output_image = os.path.join(output_dir, f'simple_heatmap{mode_suffix}_{timestamp}.png')
    model_path = os.path.join(models_dir, 'simple_heatmap_model')
    
    print("="*50)
    print("SIMPLE THERMAL HEATMAP DETECTOR")
    if use_heuristic:
        print(">>> HEURISTIC MODE (red/saturated detection)")
    print("="*50)
    
    detector = SimpleHeatmapDetector()
    
    # Try to load existing model, or train from dataset (skip if heuristic mode)
    if not use_heuristic:
        if os.path.exists(model_path + '.model'):
            print("\nLoading trained model...")
            detector.load_model(model_path)
        elif os.path.exists(dataset_dir):
            print("\nNo trained model found. Training from dataset...")
            training_data, training_labels = load_coco_dataset(dataset_dir, patch_size=32)
            
            if len(training_data) > 0:
                print(f"\nTraining on {len(training_data)} samples...")
                detector.train(training_data, training_labels)
                detector.save_model(model_path)
            else:
                print("No training data loaded. Using heuristic detection.")
        else:
            print("\nNo trained model or dataset found.")
            print("Using heuristic detection (red/saturated area detection)")
    else:
        print("\nUsing heuristic detection (red/saturated area detection)")
    
    # Check for input image
    if not os.path.exists(input_image):
        print(f"\nError: Input image not found at {input_image}")
        print("Please place 'testImage.png' in the 'input' folder.")
        return
    
    # Run detection
    print("\n" + "-"*50)
    overlay, mask = detector.detect(
        input_image,
        output_path=output_image,
        stride=16,              # Original stride
        threshold=0.3,          # Detection threshold
        use_heuristic=use_heuristic
    )
    
    if overlay is not None:
        print("\n" + "="*50)
        print("Detection complete!")
        print("="*50)
        print("\nUsage:")
        print("  python simple_heatmap_detector.py            # ML mode")
        print("  python simple_heatmap_detector.py --heuristic  # Heuristic mode")


if __name__ == "__main__":
    main()
