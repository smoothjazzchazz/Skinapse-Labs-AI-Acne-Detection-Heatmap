import cv2
import numpy as np
import os
import json
from scipy import ndimage

# Severity levels
SEVERITY_LEVELS = {
    'mild': {'color': (0, 255, 0), 'label': 'Mild', 'min_score': 0.0, 'max_score': 0.4},
    'moderate': {'color': (0, 165, 255), 'label': 'Moderate', 'min_score': 0.4, 'max_score': 0.7},
    'severe': {'color': (0, 0, 255), 'label': 'Severe', 'min_score': 0.7, 'max_score': 1.0}
}


class AcneRegion:
    """Represents a detected acne region with severity classification."""
    def __init__(self, contour, bbox, area, severity_score, severity_label, 
                 avg_intensity, avg_redness, center):
        self.contour = contour
        self.bbox = bbox
        self.area = area
        self.severity_score = severity_score
        self.severity_label = severity_label
        self.avg_intensity = avg_intensity
        self.avg_redness = avg_redness
        self.center = center


class AcneDetector:
    """
    Acne detection using local contrast and redness analysis.
    Focuses on finding individual spots rather than large regions.
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
        
        # Collect statistics focusing on CENTER pixels of acne regions
        all_redness = []
        all_saturation = []
        all_local_contrast = []
        all_r = []
        all_g = []
        all_b = []
        
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
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            
            # Compute local mean for contrast
            local_mean = cv2.blur(img_gray, (15, 15))
            
            annotations = annotations_by_image.get(img_id, [])
            
            for ann in annotations:
                bbox = ann['bbox']
                x, y, w, h = [int(v) for v in bbox]
                
                # Focus on center of bounding box (more accurate)
                cx = x + w // 2
                cy = y + h // 2
                
                # Sample from center region only (inner 50%)
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
                region_gray = img_gray[y1:y2, x1:x2]
                region_local_mean = local_mean[y1:y2, x1:x2]
                
                # Redness (R - G)
                redness = region_rgb[:, :, 0].astype(np.float32) - region_rgb[:, :, 1].astype(np.float32)
                all_redness.extend(redness.flatten())
                
                # Saturation
                all_saturation.extend(region_hsv[:, :, 1].flatten())
                
                # Local contrast (how much brighter/darker than surroundings)
                contrast = np.abs(region_gray.astype(np.float32) - region_local_mean.astype(np.float32))
                all_local_contrast.extend(contrast.flatten())
                
                all_r.extend(region_rgb[:, :, 0].flatten())
                all_g.extend(region_rgb[:, :, 1].flatten())
                all_b.extend(region_rgb[:, :, 2].flatten())
                
                acne_count += 1
        
        if len(all_redness) == 0:
            print("No acne regions found!")
            return False
        
        # Calculate statistics
        redness = np.array(all_redness)
        saturation = np.array(all_saturation)
        contrast = np.array(all_local_contrast)
        
        self.color_stats = {
            'redness_mean': float(np.mean(redness)),
            'redness_std': float(np.std(redness)),
            'redness_p25': float(np.percentile(redness, 25)),
            'redness_p75': float(np.percentile(redness, 75)),
            'saturation_mean': float(np.mean(saturation)),
            'saturation_std': float(np.std(saturation)),
            'saturation_p25': float(np.percentile(saturation, 25)),
            'contrast_mean': float(np.mean(contrast)),
            'contrast_std': float(np.std(contrast)),
            'r_mean': float(np.mean(all_r)),
            'g_mean': float(np.mean(all_g)),
            'b_mean': float(np.mean(all_b)),
            'sample_count': len(redness)
        }
        
        # Set thresholds (more aggressive)
        self.redness_threshold = max(15, self.color_stats['redness_p25'])
        
        self.calibrated = True
        
        print(f"\n=== Calibration Complete ===")
        print(f"Analyzed {acne_count} acne regions")
        print(f"Redness: mean={self.color_stats['redness_mean']:.1f}, threshold={self.redness_threshold:.1f}")
        print(f"Saturation: mean={self.color_stats['saturation_mean']:.1f}")
        print(f"Local contrast: mean={self.color_stats['contrast_mean']:.1f}")
        
        return True
    
    def detect_spots(self, img_rgb, img_hsv, img_gray, sensitivity=0.5):
        """
        Detect acne spots using multiple techniques:
        1. Local redness peaks
        2. Saturation anomalies  
        3. Local contrast (spots stand out from skin)
        4. Blob detection for circular spots
        """
        height, width = img_gray.shape
        
        r = img_rgb[:, :, 0].astype(np.float32)
        g = img_rgb[:, :, 1].astype(np.float32)
        b = img_rgb[:, :, 2].astype(np.float32)
        s = img_hsv[:, :, 1].astype(np.float32)
        
        # === 1. REDNESS MAP ===
        # Redness = R - G (inflamed acne is redder)
        redness = r - g
        
        # Normalize redness
        redness_norm = np.clip(
            (redness - self.redness_threshold) / (self.color_stats['redness_std'] + 1),
            0, 1
        )
        
        # === 2. LOCAL REDNESS CONTRAST ===
        # Find spots that are redder than their surroundings
        local_redness_mean = cv2.blur(redness, (25, 25))
        local_redness_contrast = redness - local_redness_mean
        local_redness_score = np.clip(local_redness_contrast / 30, 0, 1)
        
        # === 3. SATURATION ANOMALY ===
        # Acne often has higher saturation than surrounding skin
        local_sat_mean = cv2.blur(s, (25, 25))
        sat_anomaly = s - local_sat_mean
        sat_score = np.clip(sat_anomaly / 40, 0, 1)
        
        # === 4. INTENSITY ANOMALY (LOCAL CONTRAST) ===
        # Spots stand out from surrounding skin (darker or lighter)
        gray_float = img_gray.astype(np.float32)
        local_mean = cv2.blur(gray_float, (25, 25))
        local_contrast = np.abs(gray_float - local_mean)
        contrast_score = np.clip(local_contrast / 30, 0, 1)
        
        # === 5. BLOB DETECTION (Laplacian of Gaussian) ===
        # Detect circular spot-like structures
        gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # Multi-scale LoG
        blob_response = np.zeros_like(gray_float)
        for sigma in [2, 3, 5]:
            ksize = int(sigma * 4) | 1  # Make odd
            log = cv2.Laplacian(cv2.GaussianBlur(gray_blur, (ksize, ksize), sigma), cv2.CV_32F)
            blob_response = np.maximum(blob_response, -log)  # Negative because spots are usually darker
        
        blob_score = np.clip(blob_response / 50, 0, 1)
        
        # === 6. COLOR DISTANCE FROM TYPICAL ACNE ===
        r_diff = np.abs(r - self.color_stats['r_mean']) / 40
        g_diff = np.abs(g - self.color_stats['g_mean']) / 40
        color_match = np.clip(1 - np.sqrt(r_diff**2 + g_diff**2), 0, 1)
        
        # === COMBINE SCORES ===
        # Weight local features more heavily (they find actual spots)
        combined = (
            0.30 * local_redness_score +   # Local redness (key indicator)
            0.20 * redness_norm +           # Overall redness
            0.15 * sat_score +              # Saturation anomaly
            0.15 * contrast_score +         # Local contrast
            0.10 * blob_score +             # Blob-like structure
            0.10 * color_match              # Color similarity
        )
        
        # Apply sensitivity adjustment
        threshold_adjust = (1 - sensitivity) * 0.3
        combined = np.clip((combined - threshold_adjust) / (1 - threshold_adjust), 0, 1)
        
        return combined, redness
    
    def find_acne_regions(self, prob_map, redness_map, img_gray, 
                          min_area=15, max_area=2000, threshold=0.25):
        """
        Find individual acne spots using watershed segmentation.
        """
        # Threshold
        binary = (prob_map > threshold).astype(np.uint8) * 255
        
        # Small morphological cleanup - preserve small spots
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
        
        # Distance transform for watershed
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find local maxima as markers for watershed
        # This separates touching spots
        local_max = ndimage.maximum_filter(dist_transform, size=7)
        markers = (dist_transform == local_max) & (dist_transform > 1)
        markers = ndimage.label(markers)[0]
        
        # Watershed
        if markers.max() > 0:
            markers = markers.astype(np.int32)
            # Expand binary to 3 channels for watershed
            binary_3ch = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.watershed(binary_3ch, markers)
            
            # Create labeled regions
            labeled = np.where(markers > 0, markers, 0).astype(np.uint8)
        else:
            # Fallback to connected components
            _, labeled = cv2.connectedComponents(binary)
        
        regions = []
        
        for label_id in range(1, labeled.max() + 1):
            mask = (labeled == label_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                continue
            
            contour = contours[0]
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < min_area or area > max_area:
                continue
            
            # Check circularity (acne tends to be round)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.2:  # Too elongated, probably not acne
                    continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Aspect ratio check
            aspect_ratio = max(w, h) / (min(w, h) + 1)
            if aspect_ratio > 3:  # Too elongated
                continue
            
            # Get values for this region
            region_prob = prob_map[mask > 0]
            region_redness = redness_map[mask > 0]
            
            if len(region_prob) == 0:
                continue
            
            # Classify severity
            severity_score, severity_label, avg_intensity, avg_redness = \
                self.classify_severity(region_prob, region_redness, area)
            
            # Calculate center
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            
            regions.append(AcneRegion(
                contour=contour,
                bbox=(x, y, w, h),
                area=area,
                severity_score=severity_score,
                severity_label=severity_label,
                avg_intensity=avg_intensity,
                avg_redness=avg_redness,
                center=(cx, cy)
            ))
        
        return regions
    
    def classify_severity(self, region_prob, region_redness, area):
        """Classify severity based on intensity and redness."""
        avg_intensity = np.mean(region_prob)
        avg_redness = np.mean(region_redness)
        
        # Normalize redness
        redness_norm = np.clip(
            (avg_redness - self.redness_threshold) / (self.color_stats['redness_std'] * 2 + 1),
            0, 1
        )
        
        # Size factor (larger = potentially more severe)
        size_factor = np.clip(area / 200, 0, 1)
        
        severity_score = (
            0.35 * avg_intensity +
            0.45 * redness_norm +
            0.20 * size_factor
        )
        
        if severity_score < 0.4:
            severity_label = 'mild'
        elif severity_score < 0.7:
            severity_label = 'moderate'
        else:
            severity_label = 'severe'
        
        return severity_score, severity_label, avg_intensity, avg_redness
    
    def detect(self, image_path, output_path='acne_heatmap.png', sensitivity=0.5,
               show_labels=True, min_area=15, max_area=2000):
        """
        Detect acne spots and classify severity.
        """
        if not self.calibrated:
            print("Error: Detector not calibrated!")
            return None, None, None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            return None, None, None
        
        print(f"Processing: {image_path}")
        print(f"Size: {img.shape[1]}x{img.shape[0]}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect spots
        prob_map, redness_map = self.detect_spots(img_rgb, img_hsv, img_gray, sensitivity)
        
        # Find regions
        threshold = 0.15 + (1 - sensitivity) * 0.25
        regions = self.find_acne_regions(prob_map, redness_map, img_gray,
                                         min_area=min_area, max_area=max_area, 
                                         threshold=threshold)
        
        # Create visualization
        output_img = self.create_visualization(img, prob_map, regions, show_labels)
        
        # Save
        cv2.imwrite(output_path, output_img)
        
        # Generate summary
        summary = self.generate_summary(regions)
        self.print_results(regions, summary, output_path)
        
        return output_img, regions, summary
    
    def create_visualization(self, img, prob_map, regions, show_labels=True):
        """Create output visualization with detected spots."""
        output = img.copy()
        
        # Light heatmap overlay only on detected areas
        prob_8bit = (np.clip(prob_map, 0, 1) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(prob_8bit, cv2.COLORMAP_JET)
        
        # Only show heatmap where there's detection
        mask = prob_8bit > 30
        mask_3ch = np.stack([mask, mask, mask], axis=-1)
        output = np.where(mask_3ch, 
                         cv2.addWeighted(img, 0.7, heatmap, 0.3, 0),
                         img)
        
        # Draw each region
        for region in regions:
            color = SEVERITY_LEVELS[region.severity_label]['color']
            
            # Draw contour
            cv2.drawContours(output, [region.contour], -1, color, 2)
            
            # Draw center dot
            cv2.circle(output, region.center, 3, color, -1)
            
            # Draw label
            if show_labels:
                x, y, w, h = region.bbox
                label = SEVERITY_LEVELS[region.severity_label]['label'][0]  # First letter
                
                # Position label
                label_x = region.center[0] - 5
                label_y = max(region.center[1] - 8, 15)
                
                cv2.putText(output, label, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw legend
        output = self.draw_legend(output, regions)
        
        return output
    
    def draw_legend(self, img, regions):
        """Draw legend with counts."""
        h, w = img.shape[:2]
        
        counts = {'mild': 0, 'moderate': 0, 'severe': 0}
        for region in regions:
            counts[region.severity_label] += 1
        
        legend_x = w - 140
        legend_y = 10
        
        cv2.rectangle(img, (legend_x - 5, legend_y - 5), 
                     (w - 5, legend_y + 80), (0, 0, 0), -1)
        cv2.rectangle(img, (legend_x - 5, legend_y - 5), 
                     (w - 5, legend_y + 80), (255, 255, 255), 1)
        
        y_offset = legend_y + 15
        for severity in ['mild', 'moderate', 'severe']:
            color = SEVERITY_LEVELS[severity]['color']
            label = SEVERITY_LEVELS[severity]['label']
            count = counts[severity]
            
            cv2.rectangle(img, (legend_x, y_offset - 10), 
                         (legend_x + 15, y_offset + 2), color, -1)
            cv2.putText(img, f"{label}: {count}", (legend_x + 20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y_offset += 22
        
        return img
    
    def generate_summary(self, regions):
        """Generate summary statistics."""
        summary = {
            'total_count': len(regions),
            'mild_count': 0,
            'moderate_count': 0,
            'severe_count': 0,
            'avg_severity_score': 0,
            'total_affected_area': 0
        }
        
        if len(regions) == 0:
            return summary
        
        scores = []
        for region in regions:
            summary[f'{region.severity_label}_count'] += 1
            summary['total_affected_area'] += region.area
            scores.append(region.severity_score)
        
        summary['avg_severity_score'] = np.mean(scores)
        return summary
    
    def print_results(self, regions, summary, output_path):
        """Print results."""
        print("\n" + "="*50)
        print("ACNE DETECTION RESULTS")
        print("="*50)
        print(f"\nTotal Spots Detected: {summary['total_count']}")
        print(f"\nBy Severity:")
        print(f"  🟢 Mild:     {summary['mild_count']}")
        print(f"  🟠 Moderate: {summary['moderate_count']}")
        print(f"  🔴 Severe:   {summary['severe_count']}")
        
        if summary['total_count'] > 0:
            print(f"\nAvg Severity: {summary['avg_severity_score']:.2f}")
        
        print(f"\nSaved to: {output_path}")
        print("="*50)
    
    def save_calibration(self, filepath):
        """Save calibration."""
        if not self.calibrated:
            return
        
        data = {
            'redness_threshold': self.redness_threshold,
            'color_stats': self.color_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Calibration saved: {filepath}")
    
    def load_calibration(self, filepath):
        """Load calibration."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.redness_threshold = data['redness_threshold']
            self.color_stats = data['color_stats']
            self.calibrated = True
            print(f"Calibration loaded: {filepath}")
            return True
        except Exception as e:
            print(f"Failed to load calibration: {e}")
            return False
    
    def export_results_json(self, regions, summary, filepath):
        """Export results to JSON."""
        data = {
            'summary': summary,
            'regions': [{
                'id': i + 1,
                'severity': r.severity_label,
                'score': float(r.severity_score),
                'center': {'x': int(r.center[0]), 'y': int(r.center[1])},
                'area': float(r.area)
            } for i, r in enumerate(regions)]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def download_roboflow_dataset(api_key, workspace, project_name, version_num, output_dir):
    """Download dataset from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Please install roboflow: pip install roboflow")
        return None
    
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download("coco", location=output_dir)
    return dataset.location


def main():
    from datetime import datetime
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Organized directory structure
    input_dir = os.path.join(script_dir, 'input')
    output_dir = os.path.join(script_dir, 'output', 'individual_spots')
    calibration_dir = os.path.join(script_dir, 'calibration', 'individual_spots')
    dataset_dir = os.path.join(script_dir, 'acne_dataset')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    
    # Generate unique timestamped filename for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    input_image = os.path.join(input_dir, 'testImage.png')
    output_image = os.path.join(output_dir, f'spots_{timestamp}.png')
    output_json = os.path.join(output_dir, f'results_{timestamp}.json')
    calibration_file = os.path.join(calibration_dir, 'acne_calibration.json')
    
    print("="*50)
    print("Acne Spot Detection with Severity")
    print("="*50)
    
    detector = AcneDetector()
    
    # Load or create calibration
    if os.path.exists(calibration_file):
        print("\nLoading calibration...")
        detector.load_calibration(calibration_file)
    else:
        print("\nCalibrating from dataset...")
        
        if not os.path.exists(dataset_dir):
            print("\nDownloading dataset...")
            API_KEY = "kY77GnSm55rBAWgLCFuO"
            WORKSPACE = "drone-detection-qa3ye"
            PROJECT = "acne-kbm0q-ejfe5"
            VERSION = 3
            
            try:
                download_roboflow_dataset(API_KEY, WORKSPACE, PROJECT, VERSION, dataset_dir)
            except Exception as e:
                print(f"Error: {e}")
                return
        
        if detector.calibrate_from_dataset(dataset_dir):
            detector.save_calibration(calibration_file)
        else:
            return
    
    # Detect
    print("\n" + "="*50)
    print("Detecting Acne Spots...")
    print("="*50)
    
    if not os.path.exists(input_image):
        print(f"\nError: {input_image} not found")
        return
    
    output_img, regions, summary = detector.detect(
        input_image,
        output_path=output_image,
        sensitivity=0.7,    # 0.3=less, 0.7=more sensitive
        show_labels=True,
        min_area=15,        # Min spot size in pixels
        max_area=2000       # Max spot size (filters large false positives)
    )
    
    if regions:
        detector.export_results_json(regions, summary, output_json)


if __name__ == "__main__":
    main()
