"""
Face detection and recognition implementation using face_recognition library.
"""

import cv2
import numpy as np
import face_recognition
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from .base import DetectorBase, BoundingBox

class FaceDetector(DetectorBase):
    """Face detection and recognition implementation."""
    
    def __init__(self, known_faces_dir: Optional[Union[str, Path]] = None):
        """
        Initialize face detector with optional known faces directory.
        
        Args:
            known_faces_dir: Directory containing known face images
        """
        self.known_faces: Dict[str, np.ndarray] = {}
        self.detection_model = "hog"  # Can be 'hog' (CPU) or 'cnn' (GPU)
        
        if known_faces_dir:
            self.load_known_faces(known_faces_dir)
    
    def load_known_faces(self, faces_dir: Union[str, Path]) -> None:
        """
        Load and encode faces from the specified directory.
        
        Args:
            faces_dir: Directory containing face images
        """
        faces_path = Path(faces_dir)
        if not faces_path.exists():
            raise FileNotFoundError(f"Directory {faces_dir} not found")

        # Load all supported image files
        supported_formats = ('.jpg', '.jpeg', '.png')
        for image_path in faces_path.glob("*"):
            if image_path.suffix.lower() not in supported_formats:
                continue

            try:
                # Load and encode face
                image = face_recognition.load_image_file(str(image_path))
                encodings = face_recognition.face_encodings(image)
                
                if not encodings:
                    print(f"Warning: No face found in {image_path.name}")
                    continue

                # Use first face found in the image
                name = image_path.stem.replace('_', ' ').title()
                self.known_faces[name] = encodings[0]
                print(f"Loaded face: {name}")

            except Exception as e:
                print(f"Error loading {image_path.name}: {str(e)}")

        print(f"Loaded {len(self.known_faces)} faces")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB for face_recognition library."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detect and recognize faces in the image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detected faces with their bounding boxes and names
        """
        rgb_image = self.preprocess_image(image)
        
        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_image, model=self.detection_model)
        
        if not face_locations:
            return []
        
        # Get face encodings for recognition
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        detections = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            confidence = 0.0
            
            if self.known_faces:
                # Compare with known faces
                matches = []
                distances = []
                for known_name, known_encoding in self.known_faces.items():
                    distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                    match = distance < 0.6  # Threshold for face matching
                    matches.append(match)
                    distances.append(distance)
                
                if any(matches):
                    best_match_idx = np.argmin(distances)
                    if matches[best_match_idx]:
                        name = list(self.known_faces.keys())[best_match_idx]
                        confidence = 1 - distances[best_match_idx]
            
            detections.append(BoundingBox(
                x=left,
                y=top,
                width=right - left,
                height=bottom - top,
                confidence=confidence,
                label=name
            ))
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[BoundingBox]) -> np.ndarray:
        """
        Draw face detection boxes and labels on the image.
        
        Args:
            image: Input image in BGR format
            detections: List of face detections
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for detection in detections:
            # Choose color based on recognition status
            color = (0, 0, 255) if detection.label == "Unknown" else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(
                result,
                detection.top_left,
                detection.bottom_right,
                color,
                2
            )
            
            # Draw name label
            label = f"{detection.label} ({detection.confidence:.2f})"
            cv2.putText(
                result,
                label,
                (detection.x, detection.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return result 