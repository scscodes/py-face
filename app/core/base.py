"""
Base classes and types for detection functionality.
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class BoundingBox:
    """Represents a detection bounding box with confidence score."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    label: str

    @property
    def coordinates(self) -> Tuple[int, int, int, int]:
        """Returns (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)

    @property
    def top_left(self) -> Tuple[int, int]:
        """Returns (x, y) of top left corner."""
        return (self.x, self.y)

    @property
    def bottom_right(self) -> Tuple[int, int]:
        """Returns (x, y) of bottom right corner."""
        return (self.x + self.width, self.y + self.height)

class DetectorBase:
    """Base class for all detection implementations."""
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path {model_path} does not exist")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image before detection.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Preprocessed image
        """
        raise NotImplementedError("Subclasses must implement preprocess_image")
    
    def detect(self, image: np.ndarray) -> List[BoundingBox]:
        """
        Perform detection on an image.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of detected objects with their bounding boxes
        """
        raise NotImplementedError("Subclasses must implement detect")
    
    def draw_detections(self, image: np.ndarray, detections: List[BoundingBox]) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image in BGR format
            detections: List of detections to draw
            
        Returns:
            Image with drawn detections
        """
        raise NotImplementedError("Subclasses must implement draw_detections") 