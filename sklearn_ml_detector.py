import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from scipy import ndimage
import joblib
import os
import json
import random

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
        print("Looked for: _annotations.coco.json or annotations.json")
        return [], []
    
    print(f"Loading annotations from: {annotations_file}")
    print(f"Images directory: {images_dir}")
    
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build image id to filename mapping
    image_map = {img['id']: img['file_name'] for img in coco_data['images']}
    image_sizes = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
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
    
    for img_id, filename in image_map.items():
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
            
            # Expand bbox to patch_size if smaller
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Extract patch centered on annotation
            x1 = max(0, center_x - patch_size // 2)
            y1 = max(0, center_y - patch_size // 2)
            x2 = min(width, x1 + patch_size)
            y2 = min(height, y1 + patch_size)
            
            patch = img_rgb[y1:y2, x1:x2]
            
            if patch.shape[0] >= patch_size // 2 and patch.shape[1] >= patch_size // 2:
                # Resize to exact patch_size
                patch = cv2.resize(patch, (patch_size, patch_size))
                training_data.append(patch)
                training_labels.append(1)  # Acne
                positive_count += 1
                acne_regions.append((x1, y1, x2, y2))
        
        # Extract negative patches (non-acne regions)
        num_negatives = int(len(annotations) * negative_ratio) if annotations else 2
        attempts = 0
        max_attempts = num_negatives * 10
        
        while negative_count < positive_count * negative_ratio and attempts < max_attempts:
            attempts += 1
            
            # Random position
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


def download_roboflow_dataset(api_key, workspace, project_name, version_num, output_dir):
    """
    Download dataset from Roboflow.
    
    Returns the path to the downloaded dataset.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Installing roboflow package...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'roboflow'])
        from roboflow import Roboflow
    
    print("Connecting to Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    
    print(f"Downloading dataset to {output_dir}...")
    dataset = version.download("coco", location=output_dir)
    
    return dataset.location


def main():
    from datetime import datetime
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Organized directory structure
    input_dir = os.path.join(script_dir, 'input')
    output_dir = os.path.join(script_dir, 'output', 'sklearn_ml')
    models_dir = os.path.join(script_dir, 'models', 'sklearn_ml')
    dataset_dir = os.path.join(script_dir, 'acne_dataset')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate unique timestamped filename for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Paths
    input_image = os.path.join(input_dir, 'testImage.png')
    output_image = os.path.join(output_dir, f'sklearn_heatmap_{timestamp}.png')
    model_path = os.path.join(models_dir, 'sklearn_acne_model')
    
    print("="*50)
    print("Acne Detection with ML")
    print("="*50)
    
    detector = AcneDetectorML()
    
    # Check if we have a trained model
    if os.path.exists(model_path + '.model'):
        print("\nFound existing trained model, loading...")
        detector.load_model(model_path)
    else:
        print("\nNo trained model found. Training from Roboflow dataset...")
        
        # Download dataset if not already present
        if not os.path.exists(dataset_dir):
            print("\nDownloading Roboflow dataset...")
            # Your Roboflow credentials
            API_KEY = "kY77GnSm55rBAWgLCFuO"
            WORKSPACE = "drone-detection-qa3ye"
            PROJECT = "acne-kbm0q-ejfe5"
            VERSION = 3
            
            try:
                download_roboflow_dataset(API_KEY, WORKSPACE, PROJECT, VERSION, dataset_dir)
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                print("Please check your API key and internet connection.")
                return
        
        # Load and train
        print("\nLoading COCO dataset...")
        training_data, training_labels = load_coco_dataset(dataset_dir, patch_size=32)
        
        if len(training_data) > 0:
            print(f"\nTraining on {len(training_data)} samples...")
            detector.train(training_data, training_labels)
            
            # Save the trained model
            print("\nSaving trained model...")
            detector.save_model(model_path)
        else:
            print("No training data loaded. Running with heuristic detection.")
    
    # Run detection on test image
    print("\n" + "="*50)
    print("Running Acne Detection")
    print("="*50)
    
    print(f"\nLooking for test image at: {input_image}")
    
    if not os.path.exists(input_image):
        print(f"Error: Test image not found at {input_image}")
        print("Please place a 'testImage.png' file in the same folder as this script.")
        return
    
    print("Detecting acne on test image...")
    overlay, mask = detector.sliding_window_detection(
        input_image,
        output_path=output_image,
        stride=16,
        threshold=0.3
    )
    
    if overlay is not None:
        print("\n" + "="*50)
        print("Detection complete!")
        print(f"Heatmap saved to: {output_image}")
        print("="*50)

if __name__ == "__main__":
    main()
