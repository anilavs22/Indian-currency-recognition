# Indian-currency-recognition
Real-time currency denomination detection using YOLOv8, OpenCV, and Text-to-Speech with confidence filtering and voice-assisted feedback


## Overview

The Indian Currency Recognition System is a real-time computer vision application developed using YOLO (You Only Look Once), OpenCV, and Python. The system detects and recognizes Indian currency denominations from a live webcam feed and provides voice-based feedback using Text-to-Speech technology.

The project is designed to improve accessibility and assist users in identifying currency notes quickly and accurately through AI-powered object detection.

---

## Features

* Real-time currency detection using YOLO
* Live webcam-based recognition
* Voice announcements using Text-to-Speech
* Confidence score visualization
* Stability-based detection filtering
* Reduced false positives through confidence thresholds
* User-friendly interface with live detection display
* Supports multiple Indian currency denominations

---

## Technologies Used

* Python
* YOLO (Ultralytics)
* OpenCV
* PyTorch
* pyttsx3 (Text-to-Speech)
* Computer Vision
* Deep Learning

---

## Project Workflow

1. Capture live video from the webcam.
2. Process each frame using a trained YOLO model.
3. Detect currency notes and confidence scores.
4. Apply confidence and stability filtering.
5. Display bounding boxes and labels.
6. Announce detected denomination using voice output.

---

## Detection Stability Mechanism

To improve reliability and reduce false detections:

* Confidence threshold of 75% is applied.
* Multiple consecutive frames are analyzed.
* Announcements occur only when the same denomination is consistently detected.
* Speech delay prevents repeated announcements.

This ensures more accurate and stable real-time predictions.

---

## Project Architecture

Webcam Input

↓

OpenCV Frame Capture

↓

YOLO Object Detection

↓

Confidence Filtering

↓

Stability Verification

↓

Bounding Box Visualization

↓

Text-to-Speech Output

---

## Supported Currency Classes

The model is trained to recognize Indian currency denominations such as:

* ₹10
* ₹20
* ₹50
* ₹100
* ₹200
* ₹500

(Modify this list according to your trained classes.)

---

## Installation

### Clone Repository

```bash
git clone https://github.com/anilavs22/Indian-currency-recognition.git
cd Indian-currency-recognition
```

### Create Virtual Environment

```bash
python -m venv yenv
```

### Activate Environment

Windows:

```bash
yenv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
python main.py
```



---

## Future Improvements

* Mobile application integration
* Multi-currency support
* Currency counting feature
* Distance estimation
* Edge device deployment
* Improved low-light performance

---

## Applications

* Assistive technology for visually impaired users
* Smart banking solutions
* Automated currency verification
* Retail and cash handling systems
* Educational AI applications

---


