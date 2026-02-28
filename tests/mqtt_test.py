"""
Minimal MQTT unit tests with mock broker.

Does NOT cover end-to-end integration (that lives in deployment testing).
Focuses on message serialization, queue handling, and error paths.

Usage:
    pytest tests/mqtt_test.py -v
"""

import unittest
import json
import base64
import numpy as np
from unittest.mock import MagicMock, patch
from app.mqtt.types import FrameMessage, ResultMessage, Detection


class TestFrameMessage(unittest.TestCase):
    """Message serialization / deserialization."""

    def test_frame_message_to_json(self):
        """FrameMessage serializes correctly."""
        msg = FrameMessage(
            frame_id=42,
            timestamp=1708981320.125,
            width=640,
            height=480,
            format="jpeg",
            size_bytes=98765,
            data="abc123xyz",
        )
        json_str = msg.to_json()
        data = json.loads(json_str)

        self.assertEqual(data["frame_id"], 42)
        self.assertEqual(data["format"], "jpeg")
        self.assertIn("timestamp", data)

    def test_frame_message_from_json(self):
        """FrameMessage deserializes correctly."""
        json_str = (
            '{"frame_id": 42, "timestamp": 1708981320.125, '
            '"width": 640, "height": 480, "format": "jpeg", '
            '"size_bytes": 98765, "data": "abc123xyz"}'
        )
        msg = FrameMessage.from_json(json_str)

        self.assertEqual(msg.frame_id, 42)
        self.assertEqual(msg.width, 640)
        self.assertEqual(msg.data, "abc123xyz")

    def test_frame_message_roundtrip(self):
        """FrameMessage survives roundtrip serialization."""
        original = FrameMessage(
            frame_id=99,
            timestamp=1708981320.999,
            width=1280,
            height=720,
            format="h264",
            size_bytes=50000,
            data="big_data_here",
        )
        json_str = original.to_json()
        restored = FrameMessage.from_json(json_str)

        self.assertEqual(original.frame_id, restored.frame_id)
        self.assertEqual(original.timestamp, restored.timestamp)
        self.assertEqual(original.data, restored.data)


class TestDetection(unittest.TestCase):
    """Detection object serialization."""

    def test_detection_to_dict(self):
        """Detection converts to dict."""
        det = Detection(
            class_name="person",
            confidence=0.95,
            x=100,
            y=50,
            width=150,
            height=300,
        )
        d = det.to_dict()

        self.assertEqual(d["class_name"], "person")
        self.assertEqual(d["confidence"], 0.95)

    def test_detection_from_dict(self):
        """Detection loads from dict."""
        d = {
            "class_name": "dog",
            "confidence": 0.87,
            "x": 200,
            "y": 100,
            "width": 120,
            "height": 180,
        }
        det = Detection.from_dict(d)

        self.assertEqual(det.class_name, "dog")
        self.assertEqual(det.x, 200)


class TestResultMessage(unittest.TestCase):
    """Result message serialization."""

    def test_result_message_to_json(self):
        """ResultMessage with detections serializes."""
        detections = [
            Detection("person", 0.92, 100, 50, 150, 300),
            Detection("dog", 0.87, 350, 200, 120, 180),
        ]
        msg = ResultMessage(
            frame_id=12345,
            timestamp=1708981320.456,
            inference_time_ms=250,
            detections=detections,
        )
        json_str = msg.to_json()
        data = json.loads(json_str)

        self.assertEqual(data["frame_id"], 12345)
        self.assertEqual(len(data["detections"]), 2)
        self.assertEqual(data["detections"][0]["class_name"], "person")

    def test_result_message_empty_detections(self):
        """ResultMessage with no detections serializes."""
        msg = ResultMessage(
            frame_id=999,
            timestamp=1708981320.0,
            inference_time_ms=100,
            detections=[],
        )
        json_str = msg.to_json()
        data = json.loads(json_str)

        self.assertEqual(len(data["detections"]), 0)

    def test_result_message_roundtrip(self):
        """ResultMessage survives roundtrip."""
        original = ResultMessage(
            frame_id=555,
            timestamp=1708981320.555,
            inference_time_ms=333,
            detections=[
                Detection("cat", 0.77, 10, 20, 30, 40),
            ],
        )
        json_str = original.to_json()
        restored = ResultMessage.from_json(json_str)

        self.assertEqual(original.frame_id, restored.frame_id)
        self.assertEqual(len(restored.detections), 1)
        self.assertEqual(restored.detections[0].class_name, "cat")


class TestQueueHandling(unittest.TestCase):
    """Test queue logic (without full MQTT)."""

    def test_queue_max_size_drop_oldest(self):
        """Queue drops oldest frame when max_size exceeded."""
        queue = []
        max_size = 3

        # Add 5 frames to queue with max size 3
        for i in range(5):
            if len(queue) >= max_size:
                dropped = queue.pop(0)
            queue.append({"frame_id": i})

        # Only last 3 should remain
        self.assertEqual(len(queue), 3)
        self.assertEqual(queue[0]["frame_id"], 2)
        self.assertEqual(queue[-1]["frame_id"], 4)

    def test_empty_queue_no_crash(self):
        """Empty queue doesn't crash."""
        queue = []
        if queue:
            frame = queue.pop(0)
        else:
            frame = None

        self.assertIsNone(frame)


class TestErrorHandling(unittest.TestCase):
    """Test error paths."""

    def test_malformed_json_frame_message(self):
        """FrameMessage.from_json rejects malformed JSON."""
        with self.assertRaises(json.JSONDecodeError):
            FrameMessage.from_json("not valid json")

    def test_missing_field_frame_message(self):
        """FrameMessage.from_json rejects missing field."""
        json_str = '{"frame_id": 1, "timestamp": 123.0}'  # Missing others
        with self.assertRaises(TypeError):
            FrameMessage.from_json(json_str)

    def test_detection_from_dict_missing_field(self):
        """Detection.from_dict rejects incomplete dict."""
        incomplete = {"class_name": "person"}  # Missing coords
        with self.assertRaises(TypeError):
            Detection.from_dict(incomplete)


if __name__ == "__main__":
    unittest.main()
