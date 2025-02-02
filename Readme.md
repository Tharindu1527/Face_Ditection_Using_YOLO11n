# Face Detection using YOLOv8

## Project Overview
This project implements a face detection system using the YOLOv8 model. It can process both single images and entire directories of images to detect faces, draw bounding boxes, and extract face regions.

## Features
- Face detection using YOLOv8 nano model
- Single image processing
- Batch processing for directories
- Visualization of detection results
- Face region extraction
- Support for multiple image formats (jpg, jpeg, png)

## Requirements
```
ultralytics==8.3.70
opencv-python
numpy
matplotlib
torch>=1.8.0
torchvision>=0.9.0
```

## Installation
1. Clone the repository
```bash
git clone [repository-url]
cd face-detection-yolo
```

2. Install required packages
```bash
pip install ultralytics opencv-python numpy
```

3. Download the YOLOv8 nano model
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
```

## Usage

### Initialize Face Detector
```python
from face_detector import FaceDetector

# Initialize detector with path to images directory
detector = FaceDetector('/path/to/images')
```

### Process Single Image
```python
# Process a single image
image_path = 'path/to/image.jpg'
save_path = 'path/to/save/detected_image.jpg'

face_img, annotated_img = detector.process_single_image(image_path, save_path)
```

### Process Entire Directory
```python
# Process all images in a directory
detector.process_directory(output_dir='path/to/output')
```

### Visualize Results
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_image)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(annotated_image)
plt.title('Detected Face')

plt.subplot(1, 3, 3)
plt.imshow(face_image)
plt.title('Extracted Face')

plt.show()
```

## Project Structure
```
face-detection-yolo/
├── face_detector.py     # Main implementation
├── requirements.txt     # Dependencies
├── yolo11n.pt          # YOLOv8 model
└── README.md           # Documentation
```
