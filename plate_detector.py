import cv2
import numpy as np
from typing import List, Tuple

class LicensePlateDetector:
    def __init__(self):
   # ↓ allow square-ish plates too

        self.min_aspect_ratio = 2.0
        self.max_aspect_ratio = 5.0
        self.min_area = 400
        self.max_area = 50000
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale and apply preprocessing"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        return blurred
    
    def detect_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detection"""
        # Apply Canny edge detection with adaptive thresholds
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        
        # Apply morphological operations to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        return edges
    
    def find_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        """Find contours from edge image"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
    def filter_contours(self, contours: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """Filter contours based on license plate characteristics"""
        potential_plates = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate area and aspect ratio
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            # Filter based on size and aspect ratio
            if (self.min_area <= area <= self.max_area and
                self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                
                # Additional filtering based on contour properties
                contour_area = cv2.contourArea(contour)
                rect_area = w * h
                
                # Check if contour fills a reasonable portion of bounding rectangle
                if contour_area / rect_area > 0.3:
                    potential_plates.append((x, y, x + w, y + h))
        
        return potential_plates
    
    def refine_detections(self, image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Refine detections using additional image analysis"""
        refined_boxes = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        for x1, y1, x2, y2 in boxes:
            # Extract region of interest
            roi = gray[y1:y2, x1:x2]
            
            if roi.size == 0:
                continue
            
            # Check for high contrast (license plates typically have high contrast)
            std_dev = np.std(roi)
            
            # Check for horizontal edges (text on license plates)
            sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            horizontal_edges = np.sum(np.abs(sobel_x))
            
            # Filter based on contrast and edge strength
            if std_dev > 30 and horizontal_edges > 1000:
            # if std_dev > 10 and horizontal_edges > 300:
                refined_boxes.append((x1, y1, x2, y2))
        
        return refined_boxes
    
    def non_max_suppression(self, boxes: List[Tuple[int, int, int, int]], 
                           overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Apply non-maximum suppression to remove overlapping boxes"""
        if not boxes:
            return []
        
        # Convert to numpy array for easier processing
        boxes_array = np.array(boxes)
        
        # Calculate areas
        areas = (boxes_array[:, 2] - boxes_array[:, 0]) * (boxes_array[:, 3] - boxes_array[:, 1])
        
        # Sort by bottom-right y coordinate
        indices = np.argsort(boxes_array[:, 3])
        
        keep = []
        while len(indices) > 0:
            # Pick the last index
            last = len(indices) - 1
            i = indices[last]
            keep.append(i)
            
            # Find the largest coordinates for intersection rectangle
            xx1 = np.maximum(boxes_array[i, 0], boxes_array[indices[:last], 0])
            yy1 = np.maximum(boxes_array[i, 1], boxes_array[indices[:last], 1])
            xx2 = np.minimum(boxes_array[i, 2], boxes_array[indices[:last], 2])
            yy2 = np.minimum(boxes_array[i, 3], boxes_array[indices[:last], 3])
            
            # Compute width and height of intersection rectangle
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            # Compute intersection over union
            intersection = w * h
            union = areas[i] + areas[indices[:last]] - intersection
            overlap = intersection / union
            
            # Delete all indices from the index list that have IoU greater than threshold
            indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
        
        return [boxes[i] for i in keep]
    
    def detect_license_plates(self, image_path: str, output_path: str) -> None:
        """Main function to detect license plates in an image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Preprocess image
        gray = self.preprocess_image(image)
        
        # Detect edges
        edges = self.detect_edges(gray)
        
        # Find contours
        contours = self.find_contours(edges)
        
        # Filter contours based on license plate characteristics
        potential_plates = self.filter_contours(contours)
        
        # Refine detections
        refined_plates = self.refine_detections(image, potential_plates)
        
        # Apply non-maximum suppression
        final_plates = self.non_max_suppression(refined_plates)
        
        # Draw bounding boxes on the original image
        result_image = image.copy()
        for x1, y1, x2, y2 in final_plates:
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Add label
            cv2.putText(result_image, 'License Plate', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save the output image
        cv2.imwrite(output_path, result_image)
        
        print(f"Detected {len(final_plates)} license plate(s)")
        print(f"Output saved as {output_path}")
        
        return final_plates

def main():
    """Main function to run the license plate detector"""
    detector = LicensePlateDetector()
    
    try:
        # Detect license plates
        plates = detector.detect_license_plates('input.jpg', 'output.jpg')
        
        # Print detected regions
        for i, (x1, y1, x2, y2) in enumerate(plates):
            print(f"Plate {i+1}: ({x1}, {y1}) to ({x2}, {y2})")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()