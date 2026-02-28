# MQTT Integration Notes

## Architecture Placement

The MQTT offload modules are optional and sit alongside the existing local inference stack. Activate only on the Pi (publisher) and processor (subscriber); local YOLO paths remain unchanged.

```
pi-app.py (Pi, e.g., Raspberry Pi 4B)
├── app/mqtt/publisher_pi.py  ← NEW
│   - Capture frames, publish via MQTT
│   - Background thread, no blocking
└── [existing face recognition / local YOLO optional]

processor-app.py (Dedicated machine, e.g., NUC, Docker host)
├── app/mqtt/subscriber_processor.py  ← NEW
│   - Subscribe to frames
│   - Run YOLO inference
│   - Publish results back
├── app/core/object_detection.py  (YOLODetector)
└── [existing models path]
```

## Minimal Setup

### Prerequisites

- **MQTT Broker:** Mosquitto or equivalent running on stable IP (LAN, reachable from both Pi and processor)
  ```bash
  # Example: Ubuntu/Debian host
  apt install mosquitto mosquitto-clients
  systemctl enable mosquitto && systemctl start mosquitto
  ```

- **Python Dependencies:**
  ```bash
  # Add to requirements.txt
  paho-mqtt>=1.7.0
  ```

- **YOLO Models:** Processor must have weights, config, class names (e.g., `data/yolo/yolov3.*`)

### Pi Publisher Setup

1. Set MQTT broker address:
   ```bash
   export MQTT_HOST=192.168.1.100  # or hostname of MQTT broker
   ```

2. Start publisher in app:
   ```python
   from app.mqtt import FramePublisher
   import os

   publisher = FramePublisher(
       mqtt_host=os.getenv("MQTT_HOST", "localhost"),
       camera_index=0,
       width=640,
       height=480,
       fps=5,
       quality=80,
   )
   publisher.error_callback = print  # Log errors to stdout
   publisher.start()
   ```

3. Optionally subscribe to results:
   ```python
   # In Pi app, subscribe to MQTT results/pi topic
   # Apply detections to HA/alerts/logging
   ```

### Processor Setup

1. Set MQTT broker address:
   ```bash
   export MQTT_HOST=192.168.1.100
   ```

2. Start processor in app:
   ```python
   from app.core.object_detection import YOLODetector
   from app.mqtt import InferenceProcessor
   from pathlib import Path

   detector = YOLODetector(
       weights_path=Path("data/yolo/yolov3.weights"),
       config_path=Path("data/yolo/yolov3.cfg"),
       names_path=Path("data/yolo/coco.names"),
       confidence_threshold=0.5,
   )

   processor = InferenceProcessor(
       mqtt_host=os.getenv("MQTT_HOST", "localhost"),
       detector=detector,
       input_topic="frames/pi",
       output_topic="results/pi",
       inference_timeout_sec=5.0,
   )
   processor.error_callback = print  # Log errors to stdout
   processor.start()
   ```

## Gotchas & Workarounds

### 1. Network Outage / Frame Loss

**Symptom:** Results stop arriving after network hiccup.

**Cause:** MQTT reconnect is automatic, but frames published during outage are lost (no persistence).

**Mitigation:**
- Log frame_id sequence on Pi; detect gaps.
- Set `MQTT_HOST` to stable IP (not hostname, to avoid DNS issues).
- Use MQTT QoS 1 (at-least-once) for reliability; no higher QoS needed.
- Tolerate 100–500 ms of frame loss; this is expected.

### 2. Processor Backlog / High Latency

**Symptom:** Results lag behind capture; E2E latency grows over time.

**Cause:** Processor can't keep up with Pi frame rate (e.g., Pi @ 10 fps, processor @ 4 fps inference).

**Mitigation:**
- Reduce Pi FPS to match processor FPS (e.g., Pi @ 5 fps, processor @ 8 fps).
- Increase max_queue_size to buffer spikes, but watch memory on processor.
- Monitor CPU on processor; if >90%, offload not helping; upgrade hardware or reduce Pi FPS.

### 3. H.264 Decode Latency

**Symptom:** Inference time reported as <100 ms, but E2E latency is 300+ ms.

**Cause:** Frame encoding (Pi) + decode (processor) + network latency = ~100–200 ms overhead.

**Mitigation:**
- Accept this as baseline for offload cost.
- Measure E2E latency as (result_timestamp - frame_timestamp), not just inference_time_ms.
- For <100 ms E2E latency, local inference (GOOD tier) is required; offload unsuitable.

### 4. Pi Camera Failure / Permission

**Symptom:** Publisher exits silently or logs "Failed to open camera".

**Cause:** Camera permission (Pi user != camera group) or camera already in use.

**Mitigation:**
- Add Pi user to video group: `usermod -a -G video pi`
- Check no other app is using camera: `lsof /dev/video*`
- publisher auto-reconnects; set error_callback to alert on failures.

### 5. Processor Timeout / Hung YOLO

**Symptom:** Results stop publishing; queue grows; processor thread hangs.

**Cause:** YOLO inference stalls (bad model, OOM, corrupt frame).

**Mitigation:**
- Set inference_timeout_sec to 5 sec (default); frame is dropped if timeout.
- Monitor processor memory: `free -h`, `docker stats`.
- Log all errors via error_callback; fail loudly.
- Restart processor if wedged; no persistent state lost.

### 6. MQTT Broker Out of Memory

**Symptom:** Broker crashes; clients can't reconnect.

**Cause:** Retained messages or unbounded queues.

**Mitigation:**
- Set retain=False on all publishes (done in code).
- Use clean_session=True on client (implicit with paho-mqtt).
- Monitor broker: `mosquitto_sub -h localhost -t '#' -v` (see message flow).
- Restart broker if needed; no data loss (frames are ephemeral).

## Performance Tuning

| Parameter | Impact | Recommendation |
|-----------|--------|-----------------|
| `fps` (Pi) | Frame rate; affects bandwidth and latency | 5–10 for 640x480. Higher requires more processor. |
| `quality` (Pi) | H.264 compression; affects bandwidth | 70–85 is good tradeoff; >90 wastes bandwidth. |
| `max_queue_size` (Processor) | Buffer depth; affects memory and latency | 5–10 frames (~50–100 MB at 640x480 JPEG). Raise if spiky load. |
| `inference_timeout_sec` | Max inference time before frame dropped | 5 sec; raise to 10 if processor < 2 FPS. |

## Assumptions

- **Network:** Stable LAN (WiFi or wired) with <10 ms latency. WAN (internet) NOT supported (too much jitter).
- **Familiarity:** User familiar with Linux, environment variables, threading concepts.
- **Broker IP:** Reachable from both Pi and processor (no NAT, no firewall blocks on port 1883).
- **Models:** YOLO weights/config/names pre-downloaded on processor; no auto-fetch.

## Testing & Debugging

**Check broker connectivity (from Pi or processor):**
```bash
mosquitto_sub -h 192.168.1.100 -t 'frames/pi' -v
# Should see frame messages scroll by
```

**Monitor latency:**
```bash
mosquitto_sub -h 192.168.1.100 -t 'results/pi' | \
  jq '.inference_time_ms'
# Shows per-frame inference time; typical: 100–500 ms
```

**Check processor load:**
```bash
# On processor machine
watch -n 1 'ps aux | grep python'
top -u pi  # or other user running processor
```

---

**Version:** 1.0  
**Last Updated:** 2026  
**Status:** Prototype; tested on Raspberry Pi 4B + NUC i5.
