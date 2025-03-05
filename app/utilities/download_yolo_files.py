import urllib.request
import os
from pathlib import Path

def download_file(url: str, filename: str) -> None:
    """Download a file from a URL if it doesn't exist."""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

def main():
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # YOLO files URLs
    files = {
        "models/yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "models/yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "models/coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    
    print("Downloading YOLO files...")
    for filename, url in files.items():
        download_file(url, filename)
    
    print("\nAll files downloaded successfully!")
    print("\nNow update the paths in grok.py to point to the models directory:")
    print('self.net = cv2.dnn.readNet("models/yolov3.weights", "models/yolov3.cfg")')
    print('with open("models/coco.names", "r") as f:')

if __name__ == "__main__":
    main() 