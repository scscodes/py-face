# py-face

A Python-based facial recognition and object detection application with modular design, combining face recognition and YOLO-based object detection capabilities.

## Features

- **Face Recognition**
  - Real-time face detection and recognition
  - Support for multiple known faces
  - Confidence scoring for matches
  - Visual distinction between known (blue) and unknown (red) faces

- **Object Detection**
  - YOLO v3 implementation for object detection
  - Support for 80+ object classes
  - Configurable confidence thresholds
  - Color-coded object categories

- **Core Features**
  - Real-time processing of webcam feed
  - Single image processing
  - Combined face and object detection
  - Performance optimized for real-time use
  - Comprehensive error handling
  - Type-annotated codebase

## Project Structure

```
py-face/
├── app/
│   ├── api/               # API endpoints (future integration)
│   ├── core/             # Core detection functionality
│   │   ├── base.py      # Base classes and types
│   │   ├── face_recognition.py  # Face detection implementation
│   │   ├── object_detection.py  # YOLO detection implementation
│   │   └── demo.py      # Demo application
│   └── database/         # Database models (future integration)
├── data/
│   ├── images/          # Known face images
│   │   ├── steve_1.jpg
│   │   └── ...
│   └── yolo/           # YOLO model files
│       ├── yolov3.weights
│       ├── yolov3.cfg
│       └── coco.names
├── .env                 # Environment variables
├── config.py           # Configuration initialization
├── app.py             # FastAPI application entry point
├── app_grok.py        # Grok-based, self-contained solution
└── requirements.txt    # Project dependencies
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLO model files:
```bash
python download_yolo_files.py
```
This will download the required YOLO files to the `data/yolo` directory:
- `yolov3.weights` (236MB)
- `yolov3.cfg`
- `coco.names`

4. Prepare known faces:
- Place face images in `data/images/`
- Use clear, front-facing photos
- Name format: `person_name.jpg` (will be displayed as "Person Name")

## Usage

### Running the Demo

1. Basic webcam demo:
```bash
python -m app.core.demo
```

2. Process a single image:
```bash
python -m app.core.demo --image path/to/image.jpg
```

3. Custom directories:
```bash
python -m app.core.demo --faces-dir custom/faces --yolo-path custom/yolo
```

### Controls
- Press 'q' to quit the demo
- Stats are displayed in the top-left corner:
  - Number of faces detected
  - Number of objects detected

### Detection Features
- Face Recognition:
  - Blue boxes: Known faces with confidence scores
  - Red boxes: Unknown faces
- Object Detection:
  - Unique colors per object class
  - Labels with confidence scores

## Development

### Core Components

1. `base.py`: Base classes and types
   - `BoundingBox`: Detection result data class
   - `DetectorBase`: Abstract base for detectors

2. `face_recognition.py`: Face detection
   - Uses face_recognition library
   - Supports known face database
   - Configurable matching threshold

3. `object_detection.py`: Object detection
   - YOLO v3 implementation
   - Configurable thresholds
   - Non-maximum suppression

4. `demo.py`: Demo application
   - Combines face and object detection
   - Real-time visualization
   - Command-line interface

### Future Extensions

- REST API integration via FastAPI
- Database storage for face encodings
- Additional detection models
- Performance optimizations
- Batch processing capabilities

## Requirements

- Python 3.8+
- OpenCV
- face_recognition
- NumPy
- FastAPI (for future API integration)

## License

MIT
