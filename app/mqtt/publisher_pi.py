"""
MQTT publisher for Raspberry Pi: captures frames and publishes to MQTT broker.

Usage:
    publisher = FramePublisher(
        mqtt_host=os.getenv("MQTT_HOST", "localhost"),
        camera_index=0,
        width=640,
        height=480,
        fps=5,
    )
    publisher.start()
    try:
        publisher.wait()
    except KeyboardInterrupt:
        publisher.stop()
"""

import cv2
import paho.mqtt.client as mqtt
import threading
import time
import base64
import os
from typing import Optional, Callable
from .types import FrameMessage


class FramePublisher:
    """Captures frames from camera and publishes via MQTT."""

    def __init__(
        self,
        mqtt_host: str,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 5,
        quality: int = 80,
        topic: str = "frames/pi",
    ):
        """
        Args:
            mqtt_host: MQTT broker address (e.g., 'localhost' or '192.168.1.100')
            camera_index: OpenCV camera index (0 for /dev/video0)
            width: Frame width (recommended: 640)
            height: Frame height (recommended: 480)
            fps: Target frames per second for capture
            quality: H.264 compression quality (1-100)
            topic: MQTT topic for publishing frames
        """
        self.mqtt_host = mqtt_host
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.quality = quality
        self.topic = topic

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

        self.cap: Optional[cv2.VideoCapture] = None
        self.thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self.frame_id = 0
        self.error_callback: Optional[Callable[[str], None]] = None

    def _on_connect(self, client, userdata, flags, rc):
        """Called when client connects to broker."""
        if rc == 0:
            self._connected = True
            if self.error_callback:
                self.error_callback(f"Connected to MQTT broker at {self.mqtt_host}")
        else:
            if self.error_callback:
                self.error_callback(f"MQTT connection failed with code {rc}")

    def _on_disconnect(self, client, userdata, rc):
        """Called when client disconnects."""
        self._connected = False
        if rc != 0:
            if self.error_callback:
                self.error_callback(f"Unexpected disconnect: code {rc}")

    def _on_publish(self, client, userdata, mid):
        """Called when frame publish completes."""
        pass

    def _capture_and_publish(self):
        """Main loop: capture frames and publish."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_index}")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            frame_delay = 1.0 / self.fps

            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    if self.error_callback:
                        self.error_callback("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue

                # Encode frame to H.264 (in practice, use ffmpeg or similar for stream encoding)
                # For this prototype, we use JPEG for simplicity (easier decode/encode)
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                frame_data = base64.b64encode(buffer).decode("utf-8")

                msg = FrameMessage(
                    frame_id=self.frame_id,
                    timestamp=time.time(),
                    width=self.width,
                    height=self.height,
                    format="jpeg",  # In production, use "h264"
                    size_bytes=len(buffer),
                    data=frame_data,
                )

                if self._connected:
                    self.client.publish(
                        self.topic, msg.to_json(), qos=1, retain=False
                    )

                self.frame_id += 1
                time.sleep(frame_delay)

        except Exception as e:
            if self.error_callback:
                self.error_callback(f"Capture error: {e}")
        finally:
            if self.cap:
                self.cap.release()

    def start(self):
        """Start capturing and publishing frames."""
        self._running = True
        self.thread = threading.Thread(target=self._capture_and_publish, daemon=False)
        self.thread.start()
        self.client.connect(self.mqtt_host, 1883, keepalive=60)
        self.client.loop_start()

    def stop(self):
        """Stop publishing and clean up."""
        self._running = False
        self.client.loop_stop()
        self.client.disconnect()
        if self.thread:
            self.thread.join(timeout=5)

    def wait(self):
        """Block until stop() is called."""
        if self.thread:
            self.thread.join()
