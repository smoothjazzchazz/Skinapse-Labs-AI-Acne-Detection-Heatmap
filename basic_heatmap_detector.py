import cv2
import numpy as np
import os
import json


class AcneDetector:
    """
    Acne detection using local contrast and redness analysis.
    Creates a severity heatmap with color-coded visualization.
    """
    
    def __init__(self):
        self.color_stats = {}
        self.redness_threshold = None
        self.calibrated = False
    
    def calibrate_from_dataset(self, dataset_path):
        """Analyze labeled acne regions to learn color thresholds."""
        print("Calibrating detector from labeled dataset...")
        
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
        
        image_map = {img['id']: img['file_name'] for img in coco_data['images']}
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Limit to 50 images for speed
        image_list = list(image_map.items())[:50]
        
        # Collect statistics from CENTER of acne regions
        all_redness = []
        all_saturation = []
        all_r = []
        all_g = []
        all_b = []
        
        acne_count = 0
        
        print(f"Analyzing {len(image_list)} images...")
        
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
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            height, width = img.shape[:2]
            
            annotations = annotations_by_image.get(img_id, [])
            
            for ann in annotations:
                bbox = ann['bbox']
                x, y, w, h = [int(v) for v in bbox]
                
                # Focus on center
                cx = x + w // 2
                cy = y + h // 2
                margin_x = max(1, w // 4)
                margin_y = max(1, h // 4)
                
                x1 = max(0, cx - margin_x)
                y1 = max(0, cy - margin_y)
                x2 = min(width, cx + margin_x)
                y2 = min(height, cy + margin_y)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                region_rgb = img_rgb[y1:y2, x1:x2]
                region_hsv = img_hsv[y1:y2, x1:x2]
                
                redness = region_rgb[:, :, 0].astype(np.float32) - region_rgb[:, :, 1].astype(np.float32)
                all_redness.extend(redness.flatten())
                all_saturation.extend(region_hsv[:, :, 1].flatten())
                all_r.extend(region_rgb[:, :, 0].flatten())
                all_g.extend(region_rgb[:, :, 1].flatten())
                all_b.extend(region_rgb[:, :, 2].flatten())
                
                acne_count += 1
        
        if len(all_redness) == 0:
            print("No acne regions found!")
            return False
        
        redness = np.array(all_redness)
        saturation = np.array(all_saturation)
        
        self.color_stats = {
            'redness_mean': float(np.mean(redness)),
            'redness_std': float(np.std(redness)),
            'redness_p25': float(np.percentile(redness, 25)),
            'redness_p75': float(np.percentile(redness, 75)),
            'saturation_mean': float(np.mean(saturation)),
            'saturation_std': float(np.std(saturation)),
            'r_mean': float(np.mean(all_r)),
            'g_mean': float(np.mean(all_g)),
            'b_mean': float(np.mean(all_b)),
            'sample_count': len(redness)
        }
        
        self.redness_threshold = max(15, self.color_stats['redness_p25'])
        self.calibrated = True
        
        print(f"\n=== Calibration Complete ===")
        print(f"Analyzed {acne_count} acne regions")
        print(f"Redness threshold: {self.redness_threshold:.1f}")
        
        return True
    
    def compute_severity_map(self, img_rgb, img_hsv, img_gray, sensitivity=0.5):
        """
        Compute per-pixel severity score (0-1).
        Distinguishes acne (localized inflamed spots) from natural redness (smooth areas like nose).
        """
        r = img_rgb[:, :, 0].astype(np.float32)
        g = img_rgb[:, :, 1].astype(np.float32)
        s = img_hsv[:, :, 1].astype(np.float32)
        gray_float = img_gray.astype(np.float32)
        
        # === 1. REDNESS ===
        redness = r - g
        redness_norm = np.clip(
            (redness - self.redness_threshold) / (self.color_stats['redness_std'] + 1),
            0, 1
        )
        
        # === 2. LOCAL REDNESS CONTRAST (key for distinguishing spots from uniform redness) ===
        local_redness_mean = cv2.blur(redness, (21, 21))
        local_redness_contrast = redness - local_redness_mean
        local_redness_score = np.clip(local_redness_contrast / 20, 0, 1)
        
        # === 3. TEXTURE DETECTION (acne has texture, smooth nose doesn't) ===
        # Use Laplacian to detect texture/edges
        laplacian = cv2.Laplacian(img_gray, cv2.CV_32F)
        texture = np.abs(laplacian)
        # Local texture - compare to surroundings
        local_texture_mean = cv2.blur(texture, (15, 15))
        texture_anomaly = texture - local_texture_mean
        texture_score = np.clip(texture_anomaly / 15, 0, 1)
        
        # === 4. SMOOTHNESS PENALTY (penalize uniformly red areas like nose) ===
        # Calculate local variance in redness - low variance = smooth = likely not acne
        redness_squared = redness ** 2
        local_mean_sq = cv2.blur(redness_squared, (15, 15))
        local_variance = local_mean_sq - local_redness_mean ** 2
        local_variance = np.maximum(local_variance, 0)
        local_std = np.sqrt(local_variance)
        
        # Smooth areas (low local std) get penalized
        smoothness_penalty = np.clip(1 - local_std / 15, 0, 1)
        
        # === 5. BLOB DETECTION (acne spots are blob-like) ===
        blob_response = np.zeros_like(gray_float)
        for sigma in [2, 3, 4]:
            ksize = int(sigma * 4) | 1
            blurred = cv2.GaussianBlur(img_gray, (ksize, ksize), sigma)
            log = cv2.Laplacian(blurred, cv2.CV_32F)
            blob_response = np.maximum(blob_response, -log)
        blob_score = np.clip(blob_response / 40, 0, 1)
        
        # === 6. SATURATION ANOMALY ===
        local_sat_mean = cv2.blur(s, (21, 21))
        sat_anomaly = s - local_sat_mean
        sat_score = np.clip(sat_anomaly / 30, 0, 1)
        
        # === DETECTION SCORE (is this acne at all?) ===
        # Must have: local redness contrast OR texture OR blob-like structure
        # This filters out uniform red areas like the nose
        is_spot_like = np.maximum(local_redness_score, np.maximum(texture_score * 0.7, blob_score * 0.7))
        
        # === SEVERITY SCORE (how severe is the acne?) ===
        # Based primarily on redness intensity
        severity_base = (
            0.50 * redness_norm +           # How red (inflammation level)
            0.30 * local_redness_score +    # Stands out from surroundings
            0.20 * sat_score                # Saturation
        )
        
        # === COMBINE: Detection gates severity ===
        # Must pass the "is this a spot" test to get a severity score
        # Smooth uniform areas get reduced even if red
        raw_severity = severity_base * is_spot_like * (1 - smoothness_penalty * 0.5)
        
        # Apply sensitivity
        threshold_adjust = (1 - sensitivity) * 0.15
        severity = np.clip((raw_severity - threshold_adjust) / (1 - threshold_adjust + 0.01), 0, 1)
        
        # Light smoothing
        severity = cv2.GaussianBlur(severity, (3, 3), 0)
        
        return severity
    
    def create_severity_colormap(self):
        """Create colormap: Green (mild) -> Yellow -> Orange -> Red (severe)"""
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        
        for i in range(256):
            t = i / 255.0
            
            if t < 0.33:
                ratio = t / 0.33
                colormap[i, 0] = [0, 255, int(255 * ratio)]  # Green -> Yellow
            elif t < 0.66:
                ratio = (t - 0.33) / 0.33
                colormap[i, 0] = [0, int(255 * (1 - ratio * 0.35)), 255]  # Yellow -> Orange
            else:
                ratio = (t - 0.66) / 0.34
                colormap[i, 0] = [0, int(165 * (1 - ratio)), 255]  # Orange -> Red
        
        return colormap
    
    def apply_severity_colormap(self, severity_map):
        """Apply the custom severity colormap."""
        severity_8bit = (np.clip(severity_map, 0, 1) * 255).astype(np.uint8)
        colormap = self.create_severity_colormap()
        
        h, w = severity_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(256):
            mask = severity_8bit == i
            colored[mask] = colormap[i, 0]
        
        return colored
    
    def draw_severity_legend(self, img, x_start, y_start, width=30, height=200):
        """Draw severity gradient legend."""
        colormap = self.create_severity_colormap()
        
        for i in range(height):
            severity = 1.0 - (i / height)
            severity_8bit = int(severity * 255)
            color = tuple(int(c) for c in colormap[severity_8bit, 0])
            cv2.line(img, (x_start, y_start + i), (x_start + width, y_start + i), color, 1)
        
        cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), (255, 255, 255), 1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = [(0.0, "Severe"), (0.33, "Moderate"), (0.66, "Mild"), (1.0, "Clear")]
        
        for pos, label in labels:
            y_pos = y_start + int(pos * height)
            cv2.putText(img, label, (x_start + width + 5, y_pos + 5), font, 0.45, (255, 255, 255), 1)
            cv2.line(img, (x_start + width, y_pos), (x_start + width + 3, y_pos), (255, 255, 255), 1)
        
        return img
    
    def detect(self, image_path, output_path='acne_heatmap.png', sensitivity=0.7,
               overlay_opacity=0.5, detection_threshold=0.15):
        """
        Detect acne and create severity heatmap.
        
        Args:
            sensitivity: Detection sensitivity (0.5-0.9 recommended)
            overlay_opacity: Heatmap opacity (0-1)
            detection_threshold: Min severity to show (higher = fewer false positives)
        """
        if not self.calibrated:
            print("Error: Detector not calibrated!")
            return None, None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading: {image_path}")
            return None, None
        
        print(f"Processing: {image_path}")
        print(f"Size: {img.shape[1]}x{img.shape[0]}")
        print(f"Sensitivity: {sensitivity}, Threshold: {detection_threshold}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute severity map
        severity_map = self.compute_severity_map(img_rgb, img_hsv, img_gray, sensitivity)
        
        # Create colored heatmap
        heatmap_colored = self.apply_severity_colormap(severity_map)
        
        # Create mask for areas above threshold (reduces false positives)
        mask = severity_map > detection_threshold
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        
        # Blend heatmap with original (only where detected)
        output = img.copy()
        blended = cv2.addWeighted(img, 1 - overlay_opacity, heatmap_colored, overlay_opacity, 0)
        output = np.where(mask_3ch, blended, img)
        
        # Add legend
        h, w = output.shape[:2]
        legend_x = w - 120
        legend_y = 20
        
        cv2.rectangle(output, (legend_x - 10, legend_y - 10), (w - 10, legend_y + 220), (0, 0, 0), -1)
        cv2.rectangle(output, (legend_x - 10, legend_y - 10), (w - 10, legend_y + 220), (255, 255, 255), 1)
        output = self.draw_severity_legend(output, legend_x, legend_y, width=25, height=200)
        
        # Calculate statistics
        stats = self.calculate_statistics(severity_map, detection_threshold)
        
        # Add stats text
        self.draw_stats(output, stats, 10, 30)
        
        # Save
        cv2.imwrite(output_path, output)
        
        self.print_results(stats, output_path)
        
        return output, stats
    
    def calculate_statistics(self, severity_map, threshold=0.1):
        """Calculate coverage statistics by severity level."""
        total_pixels = severity_map.size
        
        detected = severity_map > threshold
        mild = (severity_map > threshold) & (severity_map <= 0.4)
        moderate = (severity_map > 0.4) & (severity_map <= 0.7)
        severe = severity_map > 0.7
        
        stats = {
            'total_coverage_percent': float(np.sum(detected) / total_pixels * 100),
            'mild_coverage_percent': float(np.sum(mild) / total_pixels * 100),
            'moderate_coverage_percent': float(np.sum(moderate) / total_pixels * 100),
            'severe_coverage_percent': float(np.sum(severe) / total_pixels * 100),
            'avg_severity': float(np.mean(severity_map[detected])) if np.any(detected) else 0,
            'max_severity': float(np.max(severity_map)),
        }
        
        return stats
    
    def draw_stats(self, img, stats, x, y):
        """Draw statistics on image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.rectangle(img, (x - 5, y - 20), (x + 200, y + 110), (0, 0, 0), -1)
        cv2.rectangle(img, (x - 5, y - 20), (x + 200, y + 110), (255, 255, 255), 1)
        
        lines = [
            ("ACNE ANALYSIS", (255, 255, 255)),
            (f"Coverage: {stats['total_coverage_percent']:.1f}%", (255, 255, 255)),
            (f"  Mild: {stats['mild_coverage_percent']:.1f}%", (0, 255, 0)),
            (f"  Moderate: {stats['moderate_coverage_percent']:.1f}%", (0, 165, 255)),
            (f"  Severe: {stats['severe_coverage_percent']:.1f}%", (0, 0, 255)),
            (f"Avg Severity: {stats['avg_severity']:.2f}", (255, 255, 255)),
        ]
        
        for i, (text, color) in enumerate(lines):
            cv2.putText(img, text, (x, y + i * 22), font, 0.5, color, 1)
        
        return img
    
    def print_results(self, stats, output_path):
        """Print results to console."""
        print("\n" + "="*50)
        print("ACNE SEVERITY ANALYSIS")
        print("="*50)
        print(f"\nTotal Affected: {stats['total_coverage_percent']:.2f}%")
        print(f"\nBy Severity:")
        print(f"  🟢 Mild:     {stats['mild_coverage_percent']:.2f}%")
        print(f"  🟠 Moderate: {stats['moderate_coverage_percent']:.2f}%")
        print(f"  🔴 Severe:   {stats['severe_coverage_percent']:.2f}%")
        print(f"\nAvg Severity: {stats['avg_severity']:.3f}")
        
        total = stats['total_coverage_percent']
        severe = stats['severe_coverage_percent']
        
        print("\n" + "-"*50)
        if severe > 5 or total > 30:
            print("⚠️  SIGNIFICANT acne presence")
        elif severe > 1 or total > 15:
            print("⚡ MODERATE acne presence")
        elif total > 5:
            print("📍 MILD acne presence")
        else:
            print("✓  MINIMAL acne detected")
        print("-"*50)
        
        print(f"\nSaved: {output_path}")
        print("="*50)
    
    def save_calibration(self, filepath):
        """Save calibration."""
        if not self.calibrated:
            return
        data = {'redness_threshold': self.redness_threshold, 'color_stats': self.color_stats}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {filepath}")
    
    def load_calibration(self, filepath):
        """Load calibration."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.redness_threshold = data['redness_threshold']
            self.color_stats = data['color_stats']
            self.calibrated = True
            print(f"Loaded: {filepath}")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            return False


def main():
    from datetime import datetime
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # New organized directory structure
    input_dir = os.path.join(script_dir, 'input')
    output_dir = os.path.join(script_dir, 'output', 'basic_heatmap')
    calibration_dir = os.path.join(script_dir, 'calibration', 'basic_heatmap')
    dataset_dir = os.path.join(script_dir, 'acne_dataset')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    
    # Generate unique timestamped filename for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    input_image = os.path.join(input_dir, 'testImage.png')
    output_image = os.path.join(output_dir, f'heatmap_{timestamp}.png')
    calibration_file = os.path.join(calibration_dir, 'acne_calibration.json')
    
    print("="*50)
    print("ACNE SEVERITY HEATMAP")
    print("="*50)
    
    detector = AcneDetector()
    
    if os.path.exists(calibration_file):
        print("\nLoading calibration...")
        detector.load_calibration(calibration_file)
    else:
        print("\nCalibrating...")
        
        if not os.path.exists(dataset_dir):
            API_KEY = "kY77GnSm55rBAWgLCFuO"
            WORKSPACE = "drone-detection-qa3ye"
            PROJECT = "acne-kbm0q-ejfe5"
            VERSION = 3
            
            try:
                from roboflow import Roboflow
                rf = Roboflow(api_key=API_KEY)
                project = rf.workspace(WORKSPACE).project(PROJECT)
                version = project.version(VERSION)
                version.download("coco", location=dataset_dir)
            except Exception as e:
                print(f"Error: {e}")
                return
        
        if detector.calibrate_from_dataset(dataset_dir):
            detector.save_calibration(calibration_file)
        else:
            return
    
    if not os.path.exists(input_image):
        print(f"\nError: {input_image} not found")
        return
    
    # === ADJUST THESE PARAMETERS ===
    output_img, stats = detector.detect(
        input_image,
        output_path=output_image,
        sensitivity=0.9,           # Higher = more detection (0.5-0.9)
        overlay_opacity=0.5,       # Heatmap visibility (0.3-0.7)
        detection_threshold=0.05   # Higher = fewer false positives (0.1-0.3)
    )


if __name__ == "__main__":
    main()
