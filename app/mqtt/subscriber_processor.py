"""
MQTT subscriber for offload processor: listens for frames, runs YOLO inference, publishes results.

Usage:
    from app.core.object_detection import YOLODetector
    from pathlib import Path

    detector = YOLODetector(
        weights_path=Path("data/yolo/yolov3.weights"),
        config_path=Path("data/yolo/yolov3.cfg"),
        names_path=Path("data/yolo/coco.names"),
    )

    processor = InferenceProcessor(
        mqtt_host="192.168.1.100",
        detector=detector,
        input_topic="frames/pi",
        output_topic="results/pi",
    )
    processor.start()
    try:
        processor.wait()
    except KeyboardInterrupt:
        processor.stop()
"""

import cv2
import paho.mqtt.client as mqtt
import threading
import time
import base64
from typing import Optional, Callable
from .types import FrameMessage, ResultMessage, Detection
from ..core.base import DetectorBase


class InferenceProcessor:
    """Subscribes to frames, runs YOLO inference, publishes results."""

    def __init__(
        self,
        mqtt_host: str,
        detector: DetectorBase,
        input_topic: str = "frames/pi",
        output_topic: str = "results/pi",
        inference_timeout_sec: float = 5.0,
        max_queue_size: int = 10,
    ):
        """
        Args:
            mqtt_host: MQTT broker address
            detector: YOLODetector instance (from app.core.object_detection)
            input_topic: Topic to subscribe for frames
            output_topic: Topic to publish inference results
            inference_timeout_sec: Max time to wait for inference before dropping frame
            max_queue_size: Max pending frames before dropping oldest
        """
        self.mqtt_host = mqtt_host
        self.detector = detector
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.inference_timeout_sec = inference_timeout_sec
        self.max_queue_size = max_queue_size

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.frame_queue: list = []
        self.queue_lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self.error_callback: Optional[Callable[[str], None]] = None

    def _on_connect(self, client, userdata, flags, rc):
        """Called when client connects."""
        if rc == 0:
            self._connected = True
            client.subscribe(self.input_topic, qos=1)
            if self.error_callback:
                self.error_callback(f"Connected and subscribed to {self.input_topic}")
        else:
            if self.error_callback:
                self.error_callback(f"MQTT connection failed: code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Called when client disconnects."""
        self._connected = False
        if rc != 0:
            if self.error_callback:
                self.error_callback(f"Unexpected disconnect: code {rc}")

    def _on_message(self, client, userdata, msg):
        """Called when a frame message arrives."""
        try:
            frame_msg = FrameMessage.from_json(msg.payload.decode("utf-8"))

            with self.queue_lock:
                if len(self.frame_queue) >= self.max_queue_size:
                    dropped = self.frame_queue.pop(0)
                    if self.error_callback:
                        self.error_callback(f"Dropped frame {dropped.frame_id} (queue full)")

                self.frame_queue.append(frame_msg)
        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Failed to parse frame message: {e}")

    def _process_frames(self):
        """Main loop: consume frames, run inference, publish results."""
        try:
            while self._running:
                frame_msg = None

                with self.queue_lock:
                    if self.frame_queue:
                        frame_msg = self.frame_queue.pop(0)

                if frame_msg is None:
                    time.sleep(0.01)
                    continue

                try:
                    # Decode frame
                    frame_data = base64.b64decode(frame_msg.data)
                    nparr = cv2.imdecode(
                        bytearray(frame_data), cv2.IMREAD_COLOR
                    )

                    if nparr is None:
                        if self.error_callback:
                            self.error_callback(
                                f"Failed to decode frame {frame_msg.frame_id}"
                            )
                        continue

                    # Run inference with timeout
                    start_time = time.time()
                    detections = self.detector.detect(nparr)
                    inference_time_ms = int((time.time() - start_time) * 1000)

                    if inference_time_ms > self.inference_timeout_sec * 1000:
                        if self.error_callback:
                            self.error_callback(
                                f"Inference timeout for frame {frame_msg.frame_id}"
                            )
                        continue

                    # Convert detections to message format
                    detection_list = [
                        Detection(
                            class_name=det.label,
                            confidence=det.confidence,
                            x=det.x,
                            y=det.y,
                            width=det.width,
                            height=det.height,
                        )
                        for det in detections
                    ]

                    result_msg = ResultMessage(
                        frame_id=frame_msg.frame_id,
                        timestamp=time.time(),
                        inference_time_ms=inference_time_ms,
                        detections=detection_list,
                    )

                    if self._connected:
                        self.client.publish(
                            self.output_topic, result_msg.to_json(), qos=1, retain=False
                        )

                except Exception as e:
                    if self.error_callback:
                        self.error_callback(
                            f"Inference error for frame {frame_msg.frame_id}: {e}"
                        )

        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Processing loop error: {e}")

    def start(self):
        """Start listening and processing frames."""
        self._running = True
        self.thread = threading.Thread(target=self._process_frames, daemon=False)
        self.thread.start()
        self.client.connect(self.mqtt_host, 1883, keepalive=60)
        self.client.loop_start()

    def stop(self):
        """Stop processing and clean up."""
        self._running = False
        self.client.loop_stop()
        self.client.disconnect()
        if self.thread:
            self.thread.join(timeout=5)

    def wait(self):
        """Block until stop() is called."""
        if self.thread:
            self.thread.join()
