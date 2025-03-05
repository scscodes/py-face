"""
Object detection implementation using YOLO.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from .base import DetectorBase, BoundingBox

class YOLODetector(DetectorBase):
    """YOLO-based object detection implementation."""
    
    def __init__(self, 
                 weights_path: Path,
                 config_path: Path,
                 names_path: Path,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        """
        Initialize YOLO detector with model files.
        
        Args:
            weights_path: Path to weights file
            config_path: Path to config file
            names_path: Path to class names file
            confidence_threshold: Confidence threshold for detections
            nms_threshold: Non-maximum suppression threshold
        """
        # Verify all files exist
        for path in [weights_path, config_path, names_path]:
            if not Path(path).exists():
                raise FileNotFoundError(f"File not found: {path}")
        
        # Load YOLO network
        self.net = cv2.dnn.readNet(str(weights_path), str(config_path))
        
        # Load class names
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Get output layer names
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # Detection parameters
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Detection colors (one per class)
        np.random.seed(42)  # For consistent colors
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO model.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed image blob
        """
        return cv2.dnn.blobFromImage(
            image,
            scalefactor=0.00392,
            size=(416, 416),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False
        )
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Detect objects in the image using YOLO.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detected objects with their bounding boxes
        """
        height, width = image.shape[:2]
        
        # Prepare image for YOLO
        blob = self.preprocess_image(image)
        self.net.setInput(blob)
        
        # Get detections
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
                
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold
        )
        
        # Create detection results
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                detections.append(BoundingBox(
                    x=max(0, x),  # Ensure coordinates are within image
                    y=max(0, y),
                    width=min(w, width - x),
                    height=min(h, height - y),
                    confidence=confidences[i],
                    label=self.classes[class_ids[i]]
                ))
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[BoundingBox]) -> np.ndarray:
        """
        Draw detection boxes and labels on the image.
        
        Args:
            image: Input image in BGR format
            detections: List of detections to draw
            
        Returns:
            Image with drawn detections
        """
        result = image.copy()
        
        for detection in detections:
            # Get color for this class
            color = tuple(map(int, self.colors[self.classes.index(detection.label)]))
            
            # Draw bounding box
            cv2.rectangle(
                result,
                detection.top_left,
                detection.bottom_right,
                color,
                2
            )
            
            # Draw label
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