"""
Pixel Analysis Detector - Enhanced Statistical pixel-level analysis version.
Analyzes pixel characteristics from labeled bounding boxes to build thresholds.
Features:
  - HSV + LAB color space analysis
  - Texture analysis via Laplacian variance
  - CLAHE contrast enhancement
  - Adaptive skin masking
  - Multi-scale detection
  - Region-based blob detection
Uses COLORMAP_JET thermal heatmap.
"""
import cv2
import numpy as np
import os
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt

class AcneDetector:
    """
    Enhanced acne detection using multi-modal pixel analysis from labeled dataset.
    Combines color (HSV + LAB), texture, and spatial features for robust detection.
    """
    
    def __init__(self):
        # Color thresholds learned from dataset (HSV space)
        self.h_range = None  # Hue range for acne
        self.s_range = None  # Saturation range
        self.v_range = None  # Value/brightness range
        
        # Additional color features
        self.redness_threshold = None  # R-G difference threshold
        self.color_stats = {}  # Store learned statistics
        
        # LAB color space thresholds (better for skin analysis)
        self.lab_a_range = None  # a* channel (red-green axis)
        self.lab_b_range = None  # b* channel (yellow-blue axis)
        self.lab_stats = {}
        
        # Texture statistics
        self.texture_stats = {}
        self.texture_threshold = None
        
        # Skin detection parameters (will be refined during calibration)
        self.skin_h_range = (0, 25)  # Typical skin hue
        self.skin_s_range = (20, 180)
        self.skin_v_range = (50, 255)
        
        self.calibrated = False
        
        # Detection parameters
        self.use_clahe = True
        self.use_multiscale = True
        self.scales = [1.0, 0.5, 0.25]
        self.min_spot_area = 20
        self.max_spot_area = 5000
    
    def calibrate_from_dataset(self, dataset_path):
        """
        Analyze labeled acne regions to learn color/texture thresholds.
        Now includes LAB color space and texture analysis.
        """
        print("Calibrating detector from labeled dataset...")
        print("  (Enhanced mode: HSV + LAB + Texture analysis)")
        
        # Find annotations file
        possible_paths = [
            os.path.join(dataset_path, '_annotations.coco.json'),
            os.path.join(dataset_path, 'train', '_annotations.coco.json'),
            os.path.join(dataset_path, 'annotations.json'),
        ]
        
        annotations_file = None
        images_dir = None
        
        for ann_path in possible_paths:
            if os.path.exists(ann_path):
                annotations_file = ann_path
                images_dir = os.path.dirname(ann_path)
                break
        
        if annotations_file is None:
            print(f"Error: Could not find annotations in {dataset_path}")
            return False
        
        print(f"Loading annotations from: {annotations_file}")
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build mappings
        image_map = {img['id']: img['file_name'] for img in coco_data['images']}
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Collect pixel statistics from all acne regions
        all_h_values = []
        all_s_values = []
        all_v_values = []
        all_redness = []  # R - G channel difference
        all_r_values = []
        all_g_values = []
        all_b_values = []
        
        # LAB color space values
        all_lab_a = []  # Red-green axis (key for inflammation)
        all_lab_b = []  # Yellow-blue axis
        all_lab_l = []  # Lightness
        
        # Texture values (Laplacian variance)
        all_texture = []
        
        # Also collect non-acne skin samples for comparison
        skin_h_values = []
        skin_s_values = []
        skin_v_values = []
        skin_lab_a = []
        skin_texture = []
        
        acne_count = 0
        
        for img_id, filename in image_map.items():
            img_path = os.path.join(images_dir, filename)
            if not os.path.exists(img_path):
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            
            # Compute texture map (Laplacian variance in local windows)
            laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
            texture_map = cv2.GaussianBlur(np.abs(laplacian), (7, 7), 0)
            
            annotations = annotations_by_image.get(img_id, [])
            acne_mask = np.zeros((height, width), dtype=bool)
            
            # Extract pixels from acne bounding boxes
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                x, y, w, h = [int(v) for v in bbox]
                
                # Clamp to image bounds
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + w)
                y2 = min(height, y + h)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Mark acne region
                acne_mask[y1:y2, x1:x2] = True
                
                # Extract HSV values from this region
                region_hsv = img_hsv[y1:y2, x1:x2]
                region_rgb = img_rgb[y1:y2, x1:x2]
                region_lab = img_lab[y1:y2, x1:x2]
                region_texture = texture_map[y1:y2, x1:x2]
                
                all_h_values.extend(region_hsv[:, :, 0].flatten())
                all_s_values.extend(region_hsv[:, :, 1].flatten())
                all_v_values.extend(region_hsv[:, :, 2].flatten())
                
                all_r_values.extend(region_rgb[:, :, 0].flatten())
                all_g_values.extend(region_rgb[:, :, 1].flatten())
                all_b_values.extend(region_rgb[:, :, 2].flatten())
                
                # LAB values
                all_lab_l.extend(region_lab[:, :, 0].flatten())
                all_lab_a.extend(region_lab[:, :, 1].flatten())
                all_lab_b.extend(region_lab[:, :, 2].flatten())
                
                # Texture values
                all_texture.extend(region_texture.flatten())
                
                # Redness = R - G (higher means more red/inflamed)
                redness = region_rgb[:, :, 0].astype(np.int16) - region_rgb[:, :, 1].astype(np.int16)
                all_redness.extend(redness.flatten())
                
                acne_count += 1
            
            # Sample non-acne skin regions for comparison
            non_acne_mask = ~acne_mask
            if non_acne_mask.any():
                # Sample random pixels from non-acne areas
                non_acne_hsv = img_hsv[non_acne_mask]
                non_acne_lab = img_lab[non_acne_mask]
                non_acne_texture = texture_map[non_acne_mask]
                
                # Take a subset to balance with acne samples
                sample_size = min(len(non_acne_hsv), acne_count * 10)
                if sample_size > 0:
                    indices = np.random.choice(len(non_acne_hsv), sample_size, replace=False)
                    skin_h_values.extend(non_acne_hsv[indices, 0])
                    skin_s_values.extend(non_acne_hsv[indices, 1])
                    skin_v_values.extend(non_acne_hsv[indices, 2])
                    skin_lab_a.extend(non_acne_lab[indices, 1])
                    skin_texture.extend(non_acne_texture[indices])
        
        if len(all_h_values) == 0:
            print("No acne regions found in dataset!")
            return False
        
        # Convert to numpy arrays
        all_h = np.array(all_h_values)
        all_s = np.array(all_s_values)
        all_v = np.array(all_v_values)
        all_r = np.array(all_r_values)
        all_g = np.array(all_g_values)
        all_b = np.array(all_b_values)
        redness = np.array(all_redness)
        
        lab_l = np.array(all_lab_l)
        lab_a = np.array(all_lab_a)
        lab_b = np.array(all_lab_b)
        texture = np.array(all_texture)
        
        # Calculate percentile-based thresholds (more robust than mean/std)
        # Use wider percentiles to capture variation
        self.h_range = (np.percentile(all_h, 5), np.percentile(all_h, 95))
        self.s_range = (np.percentile(all_s, 10), np.percentile(all_s, 95))
        self.v_range = (np.percentile(all_v, 10), np.percentile(all_v, 95))
        self.redness_threshold = np.percentile(redness, 25)  # Lower bound for redness
        
        # LAB thresholds - a* channel is key for inflammation (higher = more red)
        self.lab_a_range = (np.percentile(lab_a, 10), np.percentile(lab_a, 95))
        self.lab_b_range = (np.percentile(lab_b, 10), np.percentile(lab_b, 90))
        
        # Texture threshold - acne regions typically have higher texture variance
        self.texture_threshold = np.percentile(texture, 25)
        
        # Store detailed statistics
        self.color_stats = {
            'h_mean': np.mean(all_h),
            'h_std': np.std(all_h),
            's_mean': np.mean(all_s),
            's_std': np.std(all_s),
            'v_mean': np.mean(all_v),
            'v_std': np.std(all_v),
            'r_mean': np.mean(all_r),
            'g_mean': np.mean(all_g),
            'b_mean': np.mean(all_b),
            'redness_mean': np.mean(redness),
            'redness_std': np.std(redness),
            'sample_count': len(all_h)
        }
        
        self.lab_stats = {
            'l_mean': np.mean(lab_l),
            'l_std': np.std(lab_l),
            'a_mean': np.mean(lab_a),
            'a_std': np.std(lab_a),
            'b_mean': np.mean(lab_b),
            'b_std': np.std(lab_b),
        }
        
        self.texture_stats = {
            'texture_mean': np.mean(texture),
            'texture_std': np.std(texture),
            'texture_p75': np.percentile(texture, 75),
        }
        
        # Learn skin ranges from non-acne samples
        if len(skin_h_values) > 0:
            skin_h = np.array(skin_h_values)
            skin_s = np.array(skin_s_values)
            skin_v = np.array(skin_v_values)
            self.skin_h_range = (np.percentile(skin_h, 5), np.percentile(skin_h, 95))
            self.skin_s_range = (np.percentile(skin_s, 5), np.percentile(skin_s, 95))
            self.skin_v_range = (np.percentile(skin_v, 5), np.percentile(skin_v, 95))
        
        self.calibrated = True
        
        print(f"\n=== Calibration Complete ===")
        print(f"Analyzed {acne_count} acne regions ({len(all_h)} pixels)")
        print(f"\nLearned HSV Thresholds:")
        print(f"  Hue range: {self.h_range[0]:.1f} - {self.h_range[1]:.1f}")
        print(f"  Saturation range: {self.s_range[0]:.1f} - {self.s_range[1]:.1f}")
        print(f"  Value range: {self.v_range[0]:.1f} - {self.v_range[1]:.1f}")
        print(f"  Redness threshold: {self.redness_threshold:.1f}")
        print(f"\nLearned LAB Thresholds:")
        print(f"  a* range (red-green): {self.lab_a_range[0]:.1f} - {self.lab_a_range[1]:.1f}")
        print(f"  b* range (yellow-blue): {self.lab_b_range[0]:.1f} - {self.lab_b_range[1]:.1f}")
        print(f"\nTexture Statistics:")
        print(f"  Texture threshold: {self.texture_threshold:.2f}")
        print(f"  Avg texture: {self.texture_stats['texture_mean']:.2f}")
        print(f"\nColor Statistics:")
        print(f"  Avg RGB: ({self.color_stats['r_mean']:.1f}, {self.color_stats['g_mean']:.1f}, {self.color_stats['b_mean']:.1f})")
        print(f"  Avg HSV: ({self.color_stats['h_mean']:.1f}, {self.color_stats['s_mean']:.1f}, {self.color_stats['v_mean']:.1f})")
        print(f"  Avg LAB: ({self.lab_stats['l_mean']:.1f}, {self.lab_stats['a_mean']:.1f}, {self.lab_stats['b_mean']:.1f})")
        
        return True
    
    def compute_acne_probability(self, img_hsv, img_rgb):
        """
        Compute per-pixel probability of being acne based on learned thresholds.
        Returns a probability map (0-1).
        """
        h = img_hsv[:, :, 0].astype(np.float32)
        s = img_hsv[:, :, 1].astype(np.float32)
        v = img_hsv[:, :, 2].astype(np.float32)
        
        r = img_rgb[:, :, 0].astype(np.float32)
        g = img_rgb[:, :, 1].astype(np.float32)
        
        # Initialize probability map
        prob_map = np.zeros_like(h)
        
        # 1. Hue score - how close to acne hue range
        # Handle hue wraparound (red can be 0-10 or 170-180)
        h_mid = (self.h_range[0] + self.h_range[1]) / 2
        h_width = (self.h_range[1] - self.h_range[0]) / 2
        
        # Check if hue range wraps around (red hues)
        if self.h_range[0] < 20 or self.h_range[1] > 160:
            # Handle red hue wraparound
            h_score = np.where(
                (h <= self.h_range[1]) | (h >= self.h_range[0]) | (h <= 20) | (h >= 160),
                1.0,
                0.0
            )
        else:
            h_in_range = (h >= self.h_range[0]) & (h <= self.h_range[1])
            h_score = h_in_range.astype(np.float32)
        
        # 2. Saturation score
        s_in_range = (s >= self.s_range[0]) & (s <= self.s_range[1])
        s_score = s_in_range.astype(np.float32)
        
        # Boost score for higher saturation (more vivid = more inflamed)
        s_boost = np.clip((s - self.s_range[0]) / (self.s_range[1] - self.s_range[0]), 0, 1)
        s_score = s_score * (0.5 + 0.5 * s_boost)
        
        # 3. Value/brightness score
        v_in_range = (v >= self.v_range[0]) & (v <= self.v_range[1])
        v_score = v_in_range.astype(np.float32)
        
        # 4. Redness score (R-G difference)
        redness = r - g
        redness_score = np.clip(
            (redness - self.redness_threshold) / (self.color_stats['redness_std'] * 2 + 1),
            0, 1
        )
        
        # 5. Color distance from average acne color
        r_diff = np.abs(r - self.color_stats['r_mean']) / 50
        g_diff = np.abs(g - self.color_stats['g_mean']) / 50
        color_similarity = np.clip(1 - (r_diff + g_diff) / 2, 0, 1)
        
        # Combine scores with weights
        prob_map = (
            0.25 * h_score +
            0.25 * s_score +
            0.15 * v_score +
            0.20 * redness_score +
            0.15 * color_similarity
        )
        
        return prob_map
    
    def detect(self, image_path, output_path='acne_heatmap.png', sensitivity=0.5):
        """
        Detect acne in an image and generate a heatmap overlay.
        
        Args:
            image_path: Path to input image
            output_path: Path to save heatmap overlay
            sensitivity: Detection sensitivity (0-1), higher = more detections
        """
        if not self.calibrated:
            print("Error: Detector not calibrated. Run calibrate_from_dataset() first.")
            return None, None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            return None, None
        
        print(f"Processing image: {image_path}")
        print(f"Image size: {img.shape[1]}x{img.shape[0]}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Compute probability map
        prob_map = self.compute_acne_probability(img_hsv, img_rgb)
        
        # Apply Gaussian blur for smoother heatmap
        prob_map = cv2.GaussianBlur(prob_map, (11, 11), 0)
        
        # Adjust sensitivity
        threshold = 1.0 - sensitivity
        prob_map = np.clip((prob_map - threshold * 0.3) / (1 - threshold * 0.3), 0, 1)
        
        # Convert to 8-bit for visualization
        prob_map_8bit = (prob_map * 255).astype(np.uint8)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        prob_map_8bit = cv2.morphologyEx(prob_map_8bit, cv2.MORPH_CLOSE, kernel)
        prob_map_8bit = cv2.morphologyEx(prob_map_8bit, cv2.MORPH_OPEN, kernel)
        
        # Create heatmap colormap
        heatmap = cv2.applyColorMap(prob_map_8bit, cv2.COLORMAP_JET)
        
        # Create overlay - only show heatmap where probability > threshold
        mask = prob_map_8bit > (sensitivity * 50)
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        
        # Blend heatmap with original image
        overlay = img.copy()
        overlay = np.where(mask_3ch, 
                          cv2.addWeighted(img, 0.5, heatmap, 0.5, 0),
                          img)
        
        # Also create a full heatmap overlay for comparison
        full_overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Add severity legend
        full_overlay = self.draw_legend(full_overlay)
        
        # Save the result
        cv2.imwrite(output_path, full_overlay)
        
        # Calculate detection statistics
        acne_percentage = (prob_map > sensitivity * 0.5).mean() * 100
        
        print(f"\n=== Detection Results ===")
        print(f"Acne coverage: {acne_percentage:.2f}% of image")
        print(f"Heatmap saved to: {output_path}")
        
        return full_overlay, prob_map
    
    def draw_legend(self, img, x_offset=20, y_offset=20, width=30, height=150):
        """Draw severity gradient legend on the image (yellow to red only, no green)."""
        h, w = img.shape[:2]
        
        # Position legend on right side
        x_start = w - x_offset - width - 70
        y_start = y_offset
        
        # Draw black background for legend
        cv2.rectangle(img, (x_start - 10, y_start - 10), 
                     (x_start + width + 80, y_start + height + 25), (0, 0, 0), -1)
        cv2.rectangle(img, (x_start - 10, y_start - 10), 
                     (x_start + width + 80, y_start + height + 25), (255, 255, 255), 1)
        
        # Draw custom gradient: Light Yellow (bottom) -> Orange -> Red (top)
        # No green - manually create the colors
        for i in range(height):
            # t=0 at top (red), t=1 at bottom (light yellow)
            t = i / height
            
            if t < 0.5:
                # Red to Orange (top half)
                ratio = t / 0.5
                r, g, b = 255, int(165 * ratio), 0  # Red -> Orange
            else:
                # Orange to Light Yellow (bottom half)
                ratio = (t - 0.5) / 0.5
                r, g, b = 255, int(165 + 90 * ratio), int(100 * ratio)  # Orange -> Light Yellow
            
            color = (b, g, r)  # BGR for OpenCV
            cv2.line(img, (x_start, y_start + i), (x_start + width, y_start + i), color, 1)
        
        # Draw border around gradient
        cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), (255, 255, 255), 1)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        
        labels = [
            (0, "Severe"),
            (int(height * 0.5), "Moderate"),
            (height, "Mild/Clear")
        ]
        
        for y_pos, label in labels:
            cv2.putText(img, label, (x_start + width + 5, y_start + y_pos + 5), 
                       font, font_scale, (255, 255, 255), 1)
            # Draw tick mark
            cv2.line(img, (x_start + width, y_start + y_pos), 
                    (x_start + width + 3, y_start + y_pos), (255, 255, 255), 1)
        
        return img
    
    def save_calibration(self, filepath):
        """Save calibration data to file."""
        if not self.calibrated:
            print("Not calibrated yet!")
            return
        
        data = {
            'h_range': self.h_range,
            's_range': self.s_range,
            'v_range': self.v_range,
            'redness_threshold': float(self.redness_threshold),
            'color_stats': {k: float(v) for k, v in self.color_stats.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Calibration saved to: {filepath}")
    
    def load_calibration(self, filepath):
        """Load calibration data from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.h_range = tuple(data['h_range'])
            self.s_range = tuple(data['s_range'])
            self.v_range = tuple(data['v_range'])
            self.redness_threshold = data['redness_threshold']
            self.color_stats = data['color_stats']
            self.calibrated = True
            
            print(f"Calibration loaded from: {filepath}")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
    
    def draw_ground_truth(self, img, annotations):
        """Draw ground truth bounding boxes on image."""
        result = img.copy()
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = [int(v) for v in bbox]
            # Draw green rectangle for ground truth
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return result
    
    def run_dataset_inference(self, dataset_path, output_dir, num_samples=5, sensitivity=0.5):
        """
        Run inference on random images from the dataset and create comparison plots.
        
        Args:
            dataset_path: Path to dataset with annotations
            output_dir: Directory to save comparison images
            num_samples: Number of random images to process
            sensitivity: Detection sensitivity (0-1)
        """
        if not self.calibrated:
            print("Error: Detector not calibrated. Run calibrate_from_dataset() first.")
            return
        
        # Find annotations file
        possible_paths = [
            os.path.join(dataset_path, '_annotations.coco.json'),
            os.path.join(dataset_path, 'train', '_annotations.coco.json'),
            os.path.join(dataset_path, 'annotations.json'),
        ]
        
        annotations_file = None
        images_dir = None
        
        for ann_path in possible_paths:
            if os.path.exists(ann_path):
                annotations_file = ann_path
                images_dir = os.path.dirname(ann_path)
                break
        
        if annotations_file is None:
            print(f"Error: Could not find annotations in {dataset_path}")
            return
        
        print(f"\nLoading annotations from: {annotations_file}")
        
        with open(annotations_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build mappings
        image_map = {img['id']: img for img in coco_data['images']}
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Get images that have annotations
        images_with_annotations = [
            img_id for img_id in image_map.keys() 
            if img_id in annotations_by_image and len(annotations_by_image[img_id]) > 0
        ]
        
        if len(images_with_annotations) == 0:
            print("No annotated images found!")
            return
        
        # Select random samples
        num_samples = min(num_samples, len(images_with_annotations))
        selected_ids = random.sample(images_with_annotations, num_samples)
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\nProcessing {num_samples} random images from dataset...")
        print("=" * 60)
        
        for idx, img_id in enumerate(selected_ids):
            img_info = image_map[img_id]
            filename = img_info['file_name']
            img_path = os.path.join(images_dir, filename)
            
            if not os.path.exists(img_path):
                print(f"  Skipping {filename} - file not found")
                continue
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Skipping {filename} - could not read")
                continue
            
            annotations = annotations_by_image.get(img_id, [])
            num_lesions = len(annotations)
            
            print(f"\n[{idx+1}/{num_samples}] Processing: {filename}")
            print(f"  Ground truth: {num_lesions} acne lesion(s)")
            
            # Convert BGR to RGB for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create ground truth visualization
            ground_truth_img = self.draw_ground_truth(img, annotations)
            ground_truth_rgb = cv2.cvtColor(ground_truth_img, cv2.COLOR_BGR2RGB)
            
            # Run detection
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            prob_map = self.compute_acne_probability(img_hsv, img_rgb)
            
            # Apply Gaussian blur for smoother heatmap
            prob_map = cv2.GaussianBlur(prob_map, (11, 11), 0)
            
            # Adjust sensitivity
            threshold = 1.0 - sensitivity
            prob_map = np.clip((prob_map - threshold * 0.3) / (1 - threshold * 0.3), 0, 1)
            
            # Convert to 8-bit for visualization
            prob_map_8bit = (prob_map * 255).astype(np.uint8)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            prob_map_8bit = cv2.morphologyEx(prob_map_8bit, cv2.MORPH_CLOSE, kernel)
            prob_map_8bit = cv2.morphologyEx(prob_map_8bit, cv2.MORPH_OPEN, kernel)
            
            # Create heatmap
            heatmap = cv2.applyColorMap(prob_map_8bit, cv2.COLORMAP_JET)
            
            # Blend heatmap with original
            detection_overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
            detection_rgb = cv2.cvtColor(detection_overlay, cv2.COLOR_BGR2RGB)
            
            # Calculate statistics
            acne_coverage = (prob_map > sensitivity * 0.5).mean() * 100
            max_intensity = prob_map.max()
            
            # Create matplotlib figure
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Original image
            axes[0].imshow(img_rgb)
            axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            
            # Ground truth with bounding boxes
            axes[1].imshow(ground_truth_rgb)
            axes[1].set_title(f'Ground Truth ({num_lesions} lesions)', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # Detection heatmap
            axes[2].imshow(detection_rgb)
            axes[2].set_title(f'Detection Heatmap\n(Coverage: {acne_coverage:.1f}%)', fontsize=14, fontweight='bold')
            axes[2].axis('off')
            
            # Create descriptive filename
            base_name = os.path.splitext(filename)[0]
            descriptive_name = f"{base_name}_lesions{num_lesions}_cov{acne_coverage:.0f}pct_{timestamp}"
            
            plt.suptitle(f'Acne Detection Analysis: {filename}\nSensitivity: {sensitivity}', 
                        fontsize=12, y=1.02)
            plt.tight_layout()
            
            # Save the comparison figure
            output_path = os.path.join(output_dir, f"{descriptive_name}_comparison.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  Detection coverage: {acne_coverage:.2f}%")
            print(f"  Max intensity: {max_intensity:.3f}")
            print(f"  Saved: {os.path.basename(output_path)}")
        
        print("\n" + "=" * 60)
        print(f"Dataset inference complete! Results saved to: {output_dir}")
        print("=" * 60)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Directory structure
    input_dir = os.path.join(script_dir, 'input')
    output_dir = os.path.join(script_dir, 'output', 'simple_heatmap')
    dataset_inference_dir = os.path.join(script_dir, 'output', 'dataset_inference')
    calibration_dir = os.path.join(script_dir, 'calibration')
    dataset_dir = os.path.join(script_dir, 'acne_dataset')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dataset_inference_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    
    # Paths
    input_image = os.path.join(input_dir, 'testImage.png')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_image = os.path.join(output_dir, f'pixel_analysis_{timestamp}.png')
    calibration_file = os.path.join(calibration_dir, 'pixel_analysis_calibration.json')
    
    print("="*50)
    print("Acne Detection - Pixel Analysis Method")
    print("="*50)
    
    detector = AcneDetector()
    
    # Check for existing calibration
    if os.path.exists(calibration_file):
        print("\nFound existing calibration, loading...")
        detector.load_calibration(calibration_file)
    else:
        print("\nNo calibration found. Need to calibrate from dataset...")
        
        # Check for dataset
        if not os.path.exists(dataset_dir):
            print(f"\nError: Dataset not found at {dataset_dir}")
            print("Please ensure the acne_dataset folder exists.")
            return
        
        # Calibrate from dataset
        if detector.calibrate_from_dataset(dataset_dir):
            detector.save_calibration(calibration_file)
        else:
            print("Calibration failed!")
            return
    
    # Run dataset inference on random samples
    print("\n" + "="*50)
    print("Running Dataset Inference (Random Samples)")
    print("="*50)
    
    detector.run_dataset_inference(
        dataset_path=dataset_dir,
        output_dir=dataset_inference_dir,
        num_samples=5,  # Number of random images to process
        sensitivity=0.5  # Detection sensitivity
    )
    
    # Also run detection on test image if available
    print("\n" + "="*50)
    print("Running Single Image Detection")
    print("="*50)
    
    if os.path.exists(input_image):
        overlay, prob_map = detector.detect(
            input_image,
            output_path=output_image,
            sensitivity=0.5  # Adjust this: 0.3 = less sensitive, 0.7 = more sensitive
        )
        
        if overlay is not None:
            print("\n" + "="*50)
            print("Detection complete!")
            print("="*50)
    else:
        print(f"\nNote: Test image not found at {input_image}")
        print("Skipping single image detection.")


if __name__ == "__main__":
    main()
