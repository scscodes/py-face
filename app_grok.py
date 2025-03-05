import cv2
import numpy as np
import face_recognition
from pathlib import Path
import os
from typing import List, Tuple, Dict

class DetectionApp:
    """
    A class that combines object detection using YOLO and face recognition capabilities.
    Supports loading multiple known faces from a directory.
    """
    
    def __init__(self, faces_dir: str = "data/images"):
        """
        Initialize the DetectionApp with YOLO model and face recognition.
        
        Args:
            faces_dir (str): Directory containing known face images
        """
        # Initialize YOLO for object detection
        model_path = Path("data/yolo")
        if not model_path.exists() or not all(f.exists() for f in [model_path / "yolov3.weights", model_path / "yolov3.cfg", model_path / "coco.names"]):
            raise FileNotFoundError(
                "YOLO model files not found. Please run download_yolo_files.py first to download required files."
            )
            
        self.net = cv2.dnn.readNet("data/yolo/yolov3.weights", "data/yolo/yolov3.cfg")
        self.classes = []
        with open("app/configdata/yolo/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # Known faces database
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.load_known_faces(faces_dir)

    def load_known_faces(self, faces_dir: str) -> None:
        """
        Load all face images from the specified directory.
        
        Args:
            faces_dir (str): Directory containing face images
        """
        faces_path = Path(faces_dir)
        if not faces_path.exists():
            print(f"Warning: Directory {faces_dir} not found")
            return

        # Load all supported image files
        supported_formats = ('.jpg', '.jpeg', '.png')
        for image_path in faces_path.glob("*"):
            if image_path.suffix.lower() not in supported_formats:
                continue

            try:
                # Load and encode face
                image = face_recognition.load_image_file(str(image_path))
                face_encodings = face_recognition.face_encodings(image)
                
                if not face_encodings:
                    print(f"Warning: No face found in {image_path.name}")
                    continue

                # Use first face found in the image
                self.known_face_encodings.append(face_encodings[0])
                
                # Use filename without extension as person's name
                name = image_path.stem.replace('_', ' ').title()
                self.known_face_names.append(name)
                print(f"Loaded face: {name}")

            except Exception as e:
                print(f"Error loading {image_path.name}: {str(e)}")

        print(f"Loaded {len(self.known_face_names)} faces")

    def detect_objects(self, frame: np.ndarray) -> Tuple[List[List[int]], List[float], List[int], np.ndarray]:
        """
        Detect objects in the frame using YOLO.
        
        Args:
            frame (np.ndarray): Input frame to process
            
        Returns:
            Tuple containing:
            - boxes: List of bounding boxes [x, y, w, h]
            - confidences: List of confidence scores
            - class_ids: List of class identifiers
            - indexes: Array of valid detection indexes after NMS
        """
        height, width = frame.shape[:2]
        
        # Prepare image for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Process detections
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-max suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        return boxes, confidences, class_ids, indexes

    def recognize_faces(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
        """
        Recognize faces in the frame by comparing with known faces.
        
        Args:
            frame (np.ndarray): Input frame to process
            
        Returns:
            Tuple containing:
            - face_locations: List of face locations (top, right, bottom, left)
            - face_names: List of recognized face names
        """
        # Convert BGR to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if self.known_face_encodings:
                # Find best match
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]
            
            face_names.append(name)

        return face_locations, face_names

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a frame with both object detection and face recognition.
        
        Args:
            frame (np.ndarray): Input frame to process
            
        Returns:
            np.ndarray: Processed frame with detection boxes and labels
        """
        # Object detection
        boxes, confidences, class_ids, indexes = self.detect_objects(frame)
        
        # Draw object detection boxes
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = (0, 255, 0)  # Green for objects
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Face recognition
        face_locations, face_names = self.recognize_faces(frame)
        
        # Draw face recognition boxes
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            color = (0, 0, 255) if name == "Unknown" else (255, 0, 0)  # Red for unknown, Blue for known faces
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

def main():
    """
    Main function to run the detection app using webcam feed.
    """
    app = DetectionApp()
    
    if not app.known_face_encodings:
        print("Warning: No face encodings loaded. Face recognition will mark all faces as 'Unknown'")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = app.process_frame(frame)
        
        # Display the result
        cv2.imshow('Detection App', processed_frame)
        
        # Quit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()