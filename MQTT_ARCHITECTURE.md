# MQTT-Based YOLO Offload Architecture

## System Overview

Split architecture decouples frame capture (Raspberry Pi) from YOLO inference (dedicated processor). The Pi publishes compressed frames via MQTT; the processor subscribes, runs inference, and publishes results back.

```
Raspberry Pi                  MQTT Broker              Processor
┌─────────────┐              ┌──────────┐            ┌──────────────┐
│   Camera    │              │          │            │              │
└──────┬──────┘              │          │            │  YOLO Model  │
       │                     │          │            │              │
       │ capture & compress  │          │  subscribe │              │
       ├─────── topic: ──────→ frames   ├───────────→ infer & pub   │
       │     frames/pi       │          │            │              │
       │                     │          │            │              │
       │ subscribe           │          │ publish    │              │
       │◄──── topic: ────────┤ results  │◄───────────┤              │
       │     results/pi      │          │            │              │
       │                     │          │            │              │
└──────────────────────────────────────────────────────────────────┘
       │ apply results (HA, alerts)
       │ frame: 640x480, H.264 @ 5-10 fps
       │ results: JSON with bboxes, confidence, class
```

## Data Flow

**Publisher (Pi):**
- Capture frame from camera at ~2–10 fps
- Compress with H.264 (target: 80–150 KB/frame at 640x480)
- Publish to `frames/pi` with frame metadata (timestamp, resolution)
- Subscribe to `results/pi` for inference results

**Broker:**
- Message retention: off (real-time only, no backlog)
- Topic QoS: 1 (at-least-once delivery)
- No persistence; loss of frames is acceptable during brief outages

**Subscriber (Processor):**
- Subscribe to `frames/pi`
- Decode frame and run YOLO inference
- Publish results to `results/pi` (bounding boxes, confidence, class labels)
- Results expire after 1 frame cycle; no storage

## Frame Format

### Publisher (Pi) Message

Topic: `frames/pi`

```json
{
  "frame_id": 12345,
  "timestamp": 1708981320.125,
  "width": 640,
  "height": 480,
  "format": "h264",
  "size_bytes": 98765,
  "data": "<base64-encoded-h264-stream>"
}
```

**Compression Ratios:**
- Raw 640x480 YUV420: ~460 KB
- H.264 (5 fps, default quality): 80–150 KB/frame
- Bandwidth @ 5 fps: ~40–75 KB/s (~4–7 Mbps)

### Subscriber (Processor) Message

Topic: `results/pi`

```json
{
  "frame_id": 12345,
  "timestamp": 1708981320.456,
  "inference_time_ms": 250,
  "detections": [
    {
      "class": "person",
      "confidence": 0.92,
      "x": 100,
      "y": 50,
      "width": 150,
      "height": 300
    },
    {
      "class": "dog",
      "confidence": 0.87,
      "x": 350,
      "y": 200,
      "width": 120,
      "height": 180
    }
  ]
}
```

## Hardware Tiers

| Tier | Hardware | YOLO FPS | Memory (YOLO) | CPU % | Latency (E2E) | Notes |
|------|----------|----------|-------|--------|--------|-------|
| **GOOD** | Raspberry Pi 4B (4GB) | 2–3 | 250–320 MB | 85–95% | 400–600 ms | Pi runs inference locally; no offload benefits. Baseline for comparison. |
| **BETTER** | Intel NUC (i5, 8GB) | 8–12 | 400–500 MB | 35–50% | 150–250 ms | Dedicated processor; reliable network required. Handles 10 fps Pi capture @ 640x480. |
| **BEST** | Docker Host (8-core, 32GB) | 25–35 | 600–800 MB | 8–15% | 80–150 ms | Parallel processing; scales to 4+ Pi clients. Requires stable LAN/gigabit. |

### Cost & Complexity

- **GOOD:** $55 (Pi4B 4GB) + camera ($20) = $75. No offload; minimal network. High CPU, thermal limits at 5+ fps.
- **BETTER:** $250–400 (NUC i5) + $50 setup. Moderate network dependency; handles edge use. Typical deployment.
- **BEST:** Assumes Docker host already present. Check headroom: `docker stats`; require ≥8 cores, ≥16GB RAM available after other containers.

### Existing Docker Host Headroom

If Docker host is "already tight":
- Run `docker stats` to measure CPU + memory usage of live containers
- Reserve 25% CPU + 25% RAM for system/other services
- YOLO processor requires ~1 core + 500–800 MB under load
- **Recommendation:** Offload only if headroom ≥ 2 cores + 1 GB available, or move to **BETTER** tier (NUC).

## Integration with py-face

The MQTT offload modules sit alongside existing local inference:

```
py-face/
├── app/
│   ├── core/
│   │   ├── object_detection.py     # Local YOLO detector (unchanged)
│   │   ├── face_recognition.py     # (unchanged)
│   │   └── base.py                 # (unchanged)
│   └── mqtt/                       # NEW: offload modules
│       ├── publisher_pi.py         # Frame capture & publish (Pi side)
│       ├── subscriber_processor.py # Subscribe & infer (Processor side)
│       └── types.py                # Shared message schemas (dataclass)
├── MQTT_ARCHITECTURE.md            # This doc
├── MQTT_INTEGRATION.md             # Setup & gotchas
└── tests/
    └── mqtt_test.py                # Mock MQTT unit tests (minimal)
```

**Integration Points:**
1. Pi app imports `mqtt/publisher_pi.py`; instantiate with camera source, start background thread.
2. Processor app imports `mqtt/subscriber_processor.py`; instantiate with YOLODetector, start background thread.
3. Results flow to HA/local logging via callback functions (decoupled).
4. No changes to existing face recognition or local YOLO detector paths.

## Network Assumptions

- **LAN:** 10–1000 Mbps, <10 ms latency (typical home/office WiFi or wired)
- **Broker:** Mosquitto (lightweight, single-threaded) or equivalent
- **Reliability:** Network outage = frame loss (acceptable); auto-reconnect on recovery

## Performance Expectations

- **GOOD tier:** 2–3 fps YOLO, all processing on Pi; reference only.
- **BETTER tier:** 5–10 fps capture on Pi, 8–12 fps inference on NUC; E2E latency ~150–250 ms.
- **BEST tier:** 10 fps capture on Pi, 25–35 fps inference on host; scales to multiple Pis.

Bottleneck: network bandwidth (not computation) if 4+ Pis feed 640x480 @ 10 fps each (>30 Mbps).

## Gotchas & Mitigation

| Issue | Symptom | Mitigation |
|-------|---------|-----------|
| Network outage | Frames drop silently; results stale | Auto-reconnect on MQTT connect loss; track frame_id gaps in logs. |
| Frame backlog | Memory growth on processor | Drop old frames if queue > 10; publish "stale" marker to alert. |
| H.264 decode latency | Results lag behind capture | Use frame_id to correlate; measure latency per batch. Accept 100–300 ms for reliability. |
| Pi camera failure | No frames published | Catch camera exceptions; publish error marker; retry after 5 sec. |
| Concurrent YOLO inference | Process hangs | Set inference timeout (e.g., 5 sec); drop frame, log warning. |

---

**Version:** 1.0  
**Last Updated:** 2026  
**Status:** Prototype; tested at BETTER tier (NUC).
