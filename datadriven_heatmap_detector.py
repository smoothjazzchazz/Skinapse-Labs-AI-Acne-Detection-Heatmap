import cv2
import numpy as np
import os
import json
from scipy import ndimage


class AcneDetector:
    """
    Acne detection using local contrast and redness analysis.
    Creates a severity heatmap with color-coded visualization.
    """
    
    def __init__(self):
        self.color_stats = {}
        self.redness_threshold = None
        self.calibrated = False
        # Data-driven severity thresholds (will be learned from dataset)
        self.severity_mild_threshold = 0.4  # Default fallback
        self.severity_moderate_threshold = 0.7  # Default fallback
    
    def calibrate_from_dataset(self, dataset_path):
        """Analyze labeled acne regions AND normal skin to learn thresholds."""
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
        
        # Collect statistics from ENTIRE acne regions (not just center)
        acne_redness = []
        acne_saturation = []
        acne_local_contrast = []  # NEW: texture measure
        acne_r = []
        acne_g = []
        acne_b = []
        
        # Also collect NORMAL SKIN statistics (non-acne regions)
        skin_redness = []
        skin_saturation = []
        skin_local_contrast = []
        
        acne_count = 0
        
        print("  Analyzing acne regions and normal skin...")
        
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
            
            # Create mask of acne regions
            acne_mask = np.zeros((height, width), dtype=bool)
            
            annotations = annotations_by_image.get(img_id, [])
            
            # Compute local contrast (texture) for the whole image
            gray_float = img_gray.astype(np.float32)
            local_mean = cv2.blur(gray_float, (7, 7))
            local_contrast_map = np.abs(gray_float - local_mean)
            
            for ann in annotations:
                bbox = ann['bbox']
                x, y, w, h = [int(v) for v in bbox]
                
                # Use WHOLE bounding box (the entire annotation is acne)
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + w)
                y2 = min(height, y + h)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Mark this region as acne
                acne_mask[y1:y2, x1:x2] = True
                
                region_rgb = img_rgb[y1:y2, x1:x2]
                region_hsv = img_hsv[y1:y2, x1:x2]
                region_contrast = local_contrast_map[y1:y2, x1:x2]
                
                redness = region_rgb[:, :, 0].astype(np.float32) - region_rgb[:, :, 1].astype(np.float32)
                acne_redness.extend(redness.flatten())
                acne_saturation.extend(region_hsv[:, :, 1].flatten())
                acne_local_contrast.extend(region_contrast.flatten())
                acne_r.extend(region_rgb[:, :, 0].flatten())
                acne_g.extend(region_rgb[:, :, 1].flatten())
                acne_b.extend(region_rgb[:, :, 2].flatten())
                
                acne_count += 1
            
            # Sample NORMAL SKIN (non-acne regions) - avoid edges
            margin = 20
            skin_mask = ~acne_mask
            skin_mask[:margin, :] = False
            skin_mask[-margin:, :] = False
            skin_mask[:, :margin] = False
            skin_mask[:, -margin:] = False
            
            # Also avoid very dark (hair/shadow) and very bright regions
            skin_mask &= (img_gray > 60) & (img_gray < 230)
            
            if np.sum(skin_mask) > 1000:  # Need enough skin pixels
                r = img_rgb[:, :, 0].astype(np.float32)
                g = img_rgb[:, :, 1].astype(np.float32)
                skin_red = (r - g)[skin_mask]
                skin_redness.extend(skin_red.flatten()[:5000])  # Limit samples per image
                skin_saturation.extend(img_hsv[:, :, 1][skin_mask].flatten()[:5000])
                skin_local_contrast.extend(local_contrast_map[skin_mask].flatten()[:5000])
        
        if len(acne_redness) == 0:
            print("No acne regions found!")
            return False
        
        acne_redness = np.array(acne_redness)
        acne_saturation = np.array(acne_saturation)
        acne_local_contrast = np.array(acne_local_contrast)
        skin_redness = np.array(skin_redness) if len(skin_redness) > 0 else np.array([0])
        skin_saturation = np.array(skin_saturation) if len(skin_saturation) > 0 else np.array([0])
        skin_local_contrast = np.array(skin_local_contrast) if len(skin_local_contrast) > 0 else np.array([0])
        
        self.color_stats = {
            # Acne statistics
            'acne_redness_mean': float(np.mean(acne_redness)),
            'acne_redness_std': float(np.std(acne_redness)),
            'acne_redness_p25': float(np.percentile(acne_redness, 25)),
            'acne_redness_p75': float(np.percentile(acne_redness, 75)),
            'acne_saturation_mean': float(np.mean(acne_saturation)),
            'acne_contrast_mean': float(np.mean(acne_local_contrast)),
            'acne_contrast_std': float(np.std(acne_local_contrast)),
            'r_mean': float(np.mean(acne_r)),
            'g_mean': float(np.mean(acne_g)),
            'b_mean': float(np.mean(acne_b)),
            'acne_sample_count': len(acne_redness),
            
            # Normal skin statistics (for comparison)
            'skin_redness_mean': float(np.mean(skin_redness)),
            'skin_redness_std': float(np.std(skin_redness)),
            'skin_redness_p75': float(np.percentile(skin_redness, 75)),
            'skin_redness_p90': float(np.percentile(skin_redness, 90)),
            'skin_saturation_mean': float(np.mean(skin_saturation)),
            'skin_contrast_mean': float(np.mean(skin_local_contrast)),
            'skin_sample_count': len(skin_redness)
        }
        
        # Set redness threshold between skin and acne
        # Acne should be redder than normal skin
        self.redness_threshold = max(
            self.color_stats['skin_redness_p90'],  # Above 90% of normal skin
            self.color_stats['acne_redness_p25'] * 0.8  # But not too far from acne
        )
        
        # Texture threshold - acne has more local contrast than smooth skin
        self.texture_threshold = self.color_stats['skin_contrast_mean'] + self.color_stats['acne_contrast_std'] * 0.5
        
        self.calibrated = True
        
        print(f"\n  Acne stats: redness={self.color_stats['acne_redness_mean']:.1f}, contrast={self.color_stats['acne_contrast_mean']:.1f}")
        print(f"  Skin stats: redness={self.color_stats['skin_redness_mean']:.1f}, contrast={self.color_stats['skin_contrast_mean']:.1f}")
        print(f"  Redness threshold: {self.redness_threshold:.1f}")
        print(f"  Texture threshold: {self.texture_threshold:.1f}")
        
        # === LEARN SEVERITY THRESHOLDS FROM DATASET ===
        print("\nComputing severity thresholds from labeled acne regions...")
        all_acne_severities = []
        
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
            
            # Compute severity map for this image
            severity_map = self.compute_severity_map(img_rgb, img_hsv, img_gray, sensitivity=0.7)
            
            annotations = annotations_by_image.get(img_id, [])
            
            for ann in annotations:
                bbox = ann['bbox']
                x, y, w, h = [int(v) for v in bbox]
                
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(width, x + w)
                y2 = min(height, y + h)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Get severity values within the labeled acne region
                region_severity = severity_map[y1:y2, x1:x2]
                # Use the mean severity of each acne spot
                all_acne_severities.append(float(np.mean(region_severity)))
        
        if len(all_acne_severities) > 0:
            acne_severities = np.array(all_acne_severities)
            
            # Use percentiles of actual acne severity scores
            self.severity_mild_threshold = float(np.percentile(acne_severities, 25))
            self.severity_moderate_threshold = float(np.percentile(acne_severities, 60))
            
            self.color_stats['severity_distribution'] = {
                'min': float(np.min(acne_severities)),
                'max': float(np.max(acne_severities)),
                'mean': float(np.mean(acne_severities)),
                'std': float(np.std(acne_severities)),
                'p10': float(np.percentile(acne_severities, 10)),
                'p25': float(np.percentile(acne_severities, 25)),
                'p50': float(np.percentile(acne_severities, 50)),
                'p75': float(np.percentile(acne_severities, 75)),
                'p90': float(np.percentile(acne_severities, 90)),
                'acne_count': len(acne_severities)
            }
            
            print(f"  Analyzed {len(acne_severities)} acne regions")
            print(f"  Severity range: {np.min(acne_severities):.3f} - {np.max(acne_severities):.3f}")
            print(f"  Mean severity: {np.mean(acne_severities):.3f}")
            print(f"  Mild threshold (p25): {self.severity_mild_threshold:.3f}")
            print(f"  Moderate threshold (p60): {self.severity_moderate_threshold:.3f}")
        
        print(f"\n=== Calibration Complete ===")
        print(f"Analyzed {acne_count} acne regions")
        
        return True
    
    def compute_severity_map(self, img_rgb, img_hsv, img_gray, sensitivity=0.5):
        """
        Compute per-pixel severity score (0-1).
        Higher = more severe acne.
        Includes texture analysis to filter out uniformly red areas (lips/nose).
        """
        height, width = img_gray.shape
        
        r = img_rgb[:, :, 0].astype(np.float32)
        g = img_rgb[:, :, 1].astype(np.float32)
        b = img_rgb[:, :, 2].astype(np.float32)
        s = img_hsv[:, :, 1].astype(np.float32)
        gray_float = img_gray.astype(np.float32)
        
        # Get stats (use defaults if missing for backwards compatibility)
        redness_std = self.color_stats.get('acne_redness_std', self.color_stats.get('redness_std', 20))
        
        # === 1. REDNESS (must be redder than threshold) ===
        redness = r - g
        redness_norm = np.clip(
            (redness - self.redness_threshold) / (redness_std + 1),
            0, 1
        )
        
        # === 2. LOCAL REDNESS CONTRAST (acne is redder than surroundings) ===
        local_redness_mean = cv2.blur(redness, (31, 31))
        local_redness_contrast = redness - local_redness_mean
        local_redness_score = np.clip(local_redness_contrast / 25, 0, 1)
        
        # === 3. SATURATION ANOMALY ===
        local_sat_mean = cv2.blur(s, (31, 31))
        sat_anomaly = s - local_sat_mean
        sat_score = np.clip(sat_anomaly / 40, 0, 1)
        
        # === 4. LOCAL TEXTURE (acne has texture, smooth skin doesn't) ===
        # Use Laplacian to detect texture/edges
        laplacian = np.abs(cv2.Laplacian(gray_float, cv2.CV_32F))
        local_texture = cv2.blur(laplacian, (9, 9))
        texture_threshold = getattr(self, 'texture_threshold', 5.0)
        texture_score = np.clip((local_texture - texture_threshold * 0.5) / (texture_threshold + 1), 0, 1)
        
        # === 5. UNIFORMITY PENALTY (penalize uniformly red areas like lips/nose) ===
        # Calculate local variance of redness - low variance = uniform = not acne
        redness_blur = cv2.blur(redness, (15, 15))
        redness_sq_blur = cv2.blur(redness ** 2, (15, 15))
        redness_variance = np.maximum(0, redness_sq_blur - redness_blur ** 2)
        redness_std_local = np.sqrt(redness_variance)
        
        # High variance = spotty/textured = more likely acne
        # Low variance = uniform = more likely lips/nose
        uniformity_penalty = np.clip(redness_std_local / 15, 0, 1)
        
        # === 6. BLOB DETECTION (acne forms isolated spots) ===
        gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        blob_response = np.zeros_like(gray_float)
        for sigma in [2, 3, 5, 7]:
            ksize = int(sigma * 4) | 1
            log = cv2.Laplacian(cv2.GaussianBlur(gray_blur, (ksize, ksize), sigma), cv2.CV_32F)
            blob_response = np.maximum(blob_response, -log)
        blob_score = np.clip(blob_response / 40, 0, 1)
        
        # === 7. COLOR MATCH (matches typical acne color) ===
        r_diff = np.abs(r - self.color_stats['r_mean']) / 50
        g_diff = np.abs(g - self.color_stats['g_mean']) / 50
        color_match = np.clip(1 - np.sqrt(r_diff**2 + g_diff**2), 0, 1)
        
        # === 8. SIZE/ISOLATION FILTER ===
        # Large area redness vs local redness - acne is isolated, not broad
        very_large_redness = cv2.blur(redness, (61, 61))
        isolation_score = np.clip((local_redness_contrast - (redness - very_large_redness) * 0.3) / 20, 0, 1)
        
        # === COMBINE (with uniformity penalty as multiplier) ===
        base_severity = (
            0.30 * local_redness_score +
            0.20 * redness_norm +
            0.15 * texture_score +      # Texture matters
            0.15 * blob_score +
            0.10 * sat_score +
            0.10 * isolation_score      # Isolated spots
        )
        
        # Apply uniformity penalty - reduce score for uniformly red areas
        # But not too aggressively - use a softer penalty
        severity = base_severity * (0.5 + 0.5 * uniformity_penalty)
        
        # Apply sensitivity
        threshold_adjust = (1 - sensitivity) * 0.3
        severity = np.clip((severity - threshold_adjust) / (1 - threshold_adjust), 0, 1)
        
        # Light smoothing
        severity = cv2.GaussianBlur(severity, (5, 5), 0)
        
        return severity
    
    def create_severity_colormap(self):
        """
        Create custom colormap: Green (mild) -> Yellow -> Orange -> Red (severe)
        Uses data-driven thresholds for color transitions.
        """
        # Create 256-level colormap
        colormap = np.zeros((256, 1, 3), dtype=np.uint8)
        
        # Use the learned thresholds for color transitions
        mild_t = self.severity_mild_threshold
        mod_t = self.severity_moderate_threshold
        
        for i in range(256):
            t = i / 255.0
            
            if t < mild_t:
                # Green to Yellow (mild)
                ratio = t / mild_t if mild_t > 0 else 0
                colormap[i, 0] = [0, 255, int(255 * ratio)]  # BGR: Green -> Yellow
            elif t < mod_t:
                # Yellow to Orange (moderate)
                ratio = (t - mild_t) / (mod_t - mild_t) if (mod_t - mild_t) > 0 else 0
                colormap[i, 0] = [0, int(255 * (1 - ratio * 0.35)), 255]  # BGR: Yellow -> Orange
            else:
                # Orange to Red (severe)
                ratio = min(1.0, (t - mod_t) / (1.0 - mod_t)) if (1.0 - mod_t) > 0 else 1.0
                colormap[i, 0] = [0, int(165 * (1 - ratio)), 255]  # BGR: Orange -> Red
        
        return colormap
    
    def apply_severity_colormap(self, severity_map):
        """Apply the custom severity colormap to the severity map."""
        # Convert to 8-bit
        severity_8bit = (np.clip(severity_map, 0, 1) * 255).astype(np.uint8)
        
        # Get custom colormap
        colormap = self.create_severity_colormap()
        
        # Apply colormap manually
        h, w = severity_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for i in range(256):
            mask = severity_8bit == i
            colored[mask] = colormap[i, 0]
        
        return colored
    
    def draw_severity_legend(self, img, x_start, y_start, width=30, height=200):
        """Draw a vertical severity gradient legend with labels."""
        # Draw gradient bar
        for i in range(height):
            # Map position to severity (bottom=0, top=1)
            severity = 1.0 - (i / height)
            severity_8bit = int(severity * 255)
            
            colormap = self.create_severity_colormap()
            color = tuple(int(c) for c in colormap[severity_8bit, 0])
            
            cv2.line(img, (x_start, y_start + i), (x_start + width, y_start + i), color, 1)
        
        # Draw border
        cv2.rectangle(img, (x_start, y_start), (x_start + width, y_start + height), (255, 255, 255), 1)
        
        # Draw labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_color = (255, 255, 255)
        
        # Use data-driven thresholds for legend labels
        labels = [
            (0.0, "Severe"),
            (1.0 - self.severity_moderate_threshold, "Moderate"),
            (1.0 - self.severity_mild_threshold, "Mild"),
            (1.0, "Clear")
        ]
        
        for pos, label in labels:
            y_pos = y_start + int(pos * height)
            cv2.putText(img, label, (x_start + width + 5, y_pos + 5), 
                       font, font_scale, font_color, 1)
            # Tick mark
            cv2.line(img, (x_start + width, y_pos), (x_start + width + 3, y_pos), (255, 255, 255), 1)
        
        return img
    
    def detect(self, image_path, output_path='acne_heatmap.png', sensitivity=0.7,
               overlay_opacity=0.5, detection_threshold=0.1):
        """
        Detect acne and create severity heatmap visualization.
        
        Args:
            image_path: Input image path
            output_path: Output path
            sensitivity: Detection sensitivity (0-1)
            overlay_opacity: Heatmap opacity (0-1)
            detection_threshold: Minimum severity to show color (0-1)
        """
        if not self.calibrated:
            print("Error: Detector not calibrated!")
            return None, None
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            return None, None
        
        print(f"Processing: {image_path}")
        print(f"Size: {img.shape[1]}x{img.shape[0]}")
        print(f"Sensitivity: {sensitivity}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Compute severity map
        severity_map = self.compute_severity_map(img_rgb, img_hsv, img_gray, sensitivity)
        
        # Create colored heatmap
        heatmap_colored = self.apply_severity_colormap(severity_map)
        
        # Create mask for areas above threshold
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
        
        # Legend background
        cv2.rectangle(output, (legend_x - 10, legend_y - 10), 
                     (w - 10, legend_y + 220), (0, 0, 0), -1)
        cv2.rectangle(output, (legend_x - 10, legend_y - 10), 
                     (w - 10, legend_y + 220), (255, 255, 255), 1)
        
        output = self.draw_severity_legend(output, legend_x, legend_y, width=25, height=200)
        
        # Calculate statistics
        stats = self.calculate_statistics(severity_map, detection_threshold)
        
        # Add stats text
        self.draw_stats(output, stats, 10, 30)
        
        # Save
        cv2.imwrite(output_path, output)
        
        # Print results
        self.print_results(stats, output_path)
        
        return output, stats
    
    def calculate_statistics(self, severity_map, threshold=0.1):
        """Calculate coverage statistics by severity level."""
        total_pixels = severity_map.size
        
        # Define severity ranges using data-driven thresholds
        detected = severity_map > threshold
        mild = (severity_map > threshold) & (severity_map <= self.severity_mild_threshold)
        moderate = (severity_map > self.severity_mild_threshold) & (severity_map <= self.severity_moderate_threshold)
        severe = severity_map > self.severity_moderate_threshold
        
        stats = {
            'total_coverage_percent': float(np.sum(detected) / total_pixels * 100),
            'mild_coverage_percent': float(np.sum(mild) / total_pixels * 100),
            'moderate_coverage_percent': float(np.sum(moderate) / total_pixels * 100),
            'severe_coverage_percent': float(np.sum(severe) / total_pixels * 100),
            'avg_severity': float(np.mean(severity_map[detected])) if np.any(detected) else 0,
            'max_severity': float(np.max(severity_map)),
            'affected_pixels': int(np.sum(detected)),
            'mild_pixels': int(np.sum(mild)),
            'moderate_pixels': int(np.sum(moderate)),
            'severe_pixels': int(np.sum(severe))
        }
        
        return stats
    
    def draw_stats(self, img, stats, x, y):
        """Draw statistics on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 22
        
        # Background
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
            cv2.putText(img, text, (x, y + i * line_height), 
                       font, font_scale, color, 1)
        
        return img
    
    def print_results(self, stats, output_path):
        """Print results to console."""
        print("\n" + "="*50)
        print("ACNE SEVERITY ANALYSIS")
        print("="*50)
        
        print(f"\nTotal Affected Area: {stats['total_coverage_percent']:.2f}%")
        print(f"\nBreakdown by Severity:")
        print(f"  [MILD]     {stats['mild_coverage_percent']:.2f}%")
        print(f"  [MODERATE] {stats['moderate_coverage_percent']:.2f}%")
        print(f"  [SEVERE]   {stats['severe_coverage_percent']:.2f}%")
        print(f"\nAverage Severity Score: {stats['avg_severity']:.3f}")
        print(f"Maximum Severity: {stats['max_severity']:.3f}")
        
        # Assessment
        total = stats['total_coverage_percent']
        severe = stats['severe_coverage_percent']
        
        print("\n" + "-"*50)
        if severe > 5 or total > 30:
            print(">>> Assessment: SIGNIFICANT acne presence")
        elif severe > 1 or total > 15:
            print(">>  Assessment: MODERATE acne presence")
        elif total > 5:
            print(">   Assessment: MILD acne presence")
        else:
            print("    Assessment: MINIMAL acne detected")
        print("-"*50)
        
        print(f"\nSaved to: {output_path}")
        print("="*50)
    
    def save_calibration(self, filepath):
        """Save calibration."""
        if not self.calibrated:
            return
        
        data = {
            'redness_threshold': self.redness_threshold,
            'texture_threshold': getattr(self, 'texture_threshold', 5.0),
            'color_stats': self.color_stats,
            'severity_mild_threshold': self.severity_mild_threshold,
            'severity_moderate_threshold': self.severity_moderate_threshold
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
            self.texture_threshold = data.get('texture_threshold', 5.0)
            self.color_stats = data['color_stats']
            
            # Load data-driven severity thresholds (with fallback to defaults)
            self.severity_mild_threshold = data.get('severity_mild_threshold', 0.4)
            self.severity_moderate_threshold = data.get('severity_moderate_threshold', 0.7)
            
            self.calibrated = True
            print(f"Calibration loaded: {filepath}")
            print(f"  Redness threshold: {self.redness_threshold:.1f}")
            print(f"  Texture threshold: {self.texture_threshold:.1f}")
            print(f"  Severity thresholds - Mild: {self.severity_mild_threshold:.3f}, Moderate: {self.severity_moderate_threshold:.3f}")
            return True
        except Exception as e:
            print(f"Failed to load: {e}")
            return False
    
    def export_results(self, stats, filepath):
        """Export results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Results exported: {filepath}")


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
    
    # New organized directory structure
    input_dir = os.path.join(script_dir, 'input')
    output_dir = os.path.join(script_dir, 'output', 'datadriven_heatmap')
    calibration_dir = os.path.join(script_dir, 'calibration', 'datadriven_heatmap')
    dataset_dir = os.path.join(script_dir, 'acne_dataset')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    
    # Generate unique timestamped filename for each run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    input_image = os.path.join(input_dir, 'testImage.png')
    output_image = os.path.join(output_dir, f'heatmap_{timestamp}.png')
    output_json = os.path.join(output_dir, f'results_{timestamp}.json')
    calibration_file = os.path.join(calibration_dir, 'acne_calibration.json')
    
    print("="*50)
    print("ACNE SEVERITY HEATMAP ANALYZER")
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
    print("Analyzing Image...")
    print("="*50)
    
    if not os.path.exists(input_image):
        print(f"\nError: {input_image} not found")
        return
    
    output_img, stats = detector.detect(
        input_image,
        output_path=output_image,
        sensitivity=0.85,          # Detection sensitivity (0.3-0.9)
        overlay_opacity=0.5,       # Heatmap opacity (0-1)
        detection_threshold=0.015  # Lowered to catch more acne with new algorithm
    )
    
    if stats:
        detector.export_results(stats, output_json)


if __name__ == "__main__":
    main()
