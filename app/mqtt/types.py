"""
Shared message types for MQTT-based YOLO offload.
"""

from dataclasses import dataclass, asdict
from typing import List
import json
from datetime import datetime


@dataclass
class FrameMessage:
    """Published by Pi; consumed by processor."""
    frame_id: int
    timestamp: float
    width: int
    height: int
    format: str  # 'h264' or 'jpeg'
    size_bytes: int
    data: str  # base64-encoded frame

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @staticmethod
    def from_json(payload: str) -> "FrameMessage":
        data = json.loads(payload)
        return FrameMessage(**data)


@dataclass
class Detection:
    """Single object detection result."""
    class_name: str
    confidence: float
    x: int  # top-left x
    y: int  # top-left y
    width: int
    height: int

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        return Detection(**d)


@dataclass
class ResultMessage:
    """Published by processor; consumed by Pi."""
    frame_id: int
    timestamp: float
    inference_time_ms: int
    detections: List[Detection]

    def to_json(self) -> str:
        return json.dumps({
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "inference_time_ms": self.inference_time_ms,
            "detections": [d.to_dict() for d in self.detections],
        })

    @staticmethod
    def from_json(payload: str) -> "ResultMessage":
        data = json.loads(payload)
        detections = [Detection(**d) for d in data.pop("detections", [])]
        return ResultMessage(detections=detections, **data)
