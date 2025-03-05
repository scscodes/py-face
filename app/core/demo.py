"""
Demo script for testing face recognition and object detection.
"""

import cv2
from pathlib import Path
from typing import Tuple
import numpy as np

from .face_recognition import FaceDetector
from .object_detection import YOLODetector

class DetectionDemo:
    """Demo class combining face recognition and object detection."""
    
    def __init__(self,
                 faces_dir: str = "data/images",
                 yolo_path: str = "data/yolo",
                 window_name: str = "Detection Demo"):
        """
        Initialize the demo with face and object detection.
        
        Args:
            faces_dir: Directory containing known face images
            yolo_path: Directory containing YOLO model files
            window_name: Name of the display window
        """
        self.window_name = window_name
        
        # Initialize face detector
        print("Initializing face detector...")
        self.face_detector = FaceDetector(faces_dir)
        
        # Initialize YOLO detector
        print("Initializing YOLO detector...")
        yolo_dir = Path(yolo_path)
        self.object_detector = YOLODetector(
            weights_path=yolo_dir / "yolov3.weights",
            config_path=yolo_dir / "yolov3.cfg",
            names_path=yolo_dir / "coco.names"
        )
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Process a frame with both detectors.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Tuple of (processed frame, number of faces, number of objects)
        """
        # Detect faces
        face_detections = self.face_detector.detect(frame)
        frame = self.face_detector.draw_detections(frame, face_detections)
        
        # Detect objects
        object_detections = self.object_detector.detect(frame)
        frame = self.object_detector.draw_detections(frame, object_detections)
        
        return frame, len(face_detections), len(object_detections)
    
    def run_webcam(self):
        """Run the demo using webcam feed."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        
        print("Starting webcam feed... Press 'q' to quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, num_faces, num_objects = self.process_frame(frame)
                
                # Add stats to frame
                stats = f"Faces: {num_faces} | Objects: {num_objects}"
                cv2.putText(
                    processed_frame,
                    stats,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
                
                # Display result
                cv2.imshow(self.window_name, processed_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def process_image(self, image_path: str) -> np.ndarray:
        """
        Process a single image file.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Processed image with detections
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        processed_image, num_faces, num_objects = self.process_frame(image)
        print(f"Detected {num_faces} faces and {num_objects} objects")
        
        return processed_image

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run face and object detection demo")
    parser.add_argument("--image", help="Path to image file to process")
    parser.add_argument("--faces-dir", default="data/images", help="Directory with known faces")
    parser.add_argument("--yolo-path", default="data/yolo", help="Directory with YOLO files")
    args = parser.parse_args()
    
    demo = DetectionDemo(
        faces_dir=args.faces_dir,
        yolo_path=args.yolo_path
    )
    
    if args.image:
        # Process single image
        result = demo.process_image(args.image)
        
        # Show result
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Run webcam demo
        demo.run_webcam()

if __name__ == "__main__":
    main() 