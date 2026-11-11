# ADAS Multi-Hazard Driver Monitoring System

An end-to-end Edge AI Advanced Driver Assistance System (ADAS) engineered for real-time, in-cabin driver safety monitoring deployed on the **NVIDIA Jetson Orin Nano**. The platform integrates a dual-stream computer vision pipeline combining deep learning object detection and neural feature classification to continuously evaluate driver cognitive focus, detecting mobile phone distraction and ocular fatigue/drowsiness simultaneously under low-latency constraints.

---

## 👨‍💻 Project Overview & Resume Summary

**ADAS Driver Monitor**  
* Python, PyTorch, YOLOv8, MobileNetV2, OpenCV, NVIDIA Jetson Orin Nano | Advanced Driver Assistance Systems (ADAS)
* Built and deployed a real-time Edge AI driver monitoring pipeline on NVIDIA Jetson Orin Nano combining object detection and neural feature classification for in-cabin safety.
* Implemented dual-stream distraction and fatigue detection utilizing YOLOv8 for mobile phone detection alongside a lightweight MobileNetV2 network for eye-state and drowsiness classification.
* Optimized spatial face-and-eye localization pipeline using OpenCV Haar Cascades to crop region-of-interest patches dynamically, ensuring low latency and consistent real-time frame rates.

---

## 🏗 Architecture & Technical System Design

The system runs a modular, multi-stage processing pipeline designed to optimize compute resource allocation across embedded hardware:

1. **Frame Ingestion & Preprocessing:** Captures live webcam feeds at 640x480 resolution, executing real-time spatial flipping and colorspace normalization.
2. **Object Detection Stream (YOLOv8):** Evaluates full-frame imagery using custom-trained **YOLOv8** weights to track mobile phone interaction with bounding-box regression and confidence scoring (>= 0.40).
3. **Spatial Localization & ROI Extraction:** Utilizes an OpenCV Haar Cascade face-and-eye localization pipeline to dynamically extract Eye Region-of-Interest (ROI) patches, minimizing matrix dimensions passed downstream.
4. **Ocular Fatigue Classification (MobileNetV2):** Resizes cropped eye ROI patches to 64x64, normalizes ImageNet tensors, and runs forward passes through a fine-tuned **MobileNetV2** deep network to classify binary eye states (Open vs. Closed).
5. **Multi-Hazard Decision Engine & UI Overlay:** Aggregates temporal state outputs to determine driver risk levels, rendering dynamic visual alerts (`Driver Focus: Normal`, `WARNING: DROWSINESS DETECTED!`, `ALERT: PHONE DISTRACTION!`, `CRITICAL ALERT: PHONE + DROWSINESS!`) onto the local display output.

Architecture Diagram 
<img width="1024" height="572" alt="image" src="https://github.com/user-attachments/assets/83f41432-dfff-48b7-94a0-a8d6f7fe69de" />

---

## 🌟 Key Engineering Features

* **Dual-Stream Inference Engine:** Executes parallel object detection and neural image classification pipelines without introducing frame-rate bottlenecks.
* **Low-Latency ROI Crop Pipeline:** Reduces overall compute overhead by isolating facial landmarks prior to running deep feature classification.
* **Adaptive Compute Execution:** Built with seamless fallback paths targeting PyTorch CPU execution or native Jetson CUDA acceleration.
* **Multi-Hazard Alerting Logic:** Implements layered hazard scoring to flag simultaneous high-risk behaviors in real time.

---

## 🛠 Tech Stack & Specifications

* **Hardware:** NVIDIA Jetson Orin Nano / Linux Workstation
* **Languages & Runtime:** Python 3.10+
* **Deep Learning Frameworks:** PyTorch, Torchvision, Ultralytics YOLOv8
* **Computer Vision & Math:** OpenCV (`opencv-contrib-python`), Pillow, NumPy
* **Model Architectures:** YOLOv8 (Object Detection), MobileNetV2 (Neural Classification)

---

## 🚀 Quick Start

### 1. Prerequisites

Install core dependencies in your local Python environment:

```bash
pip install torch torchvision ultralytics opencv-contrib-python Pillow numpy
```

## Weight Placement
Ensure trained model weights are located in the models/ directory:

models/phone_detector.pt (YOLOv8 Phone Detection)

models/eye_classifier.pth (MobileNetV2 Eye Classifier)

(Note: Model binary files are excluded from version control via .gitignore).

## Execution
Run main.py directly on your Jetson desktop display interface:

```Bash
cd adas-driver-monitor
python3 main.py
```
Press q inside the display window to exit the application cleanly.

## 💡 Conclusion & Future Work

This project demonstrates the deployment of a real-time, multi-hazard driver safety application on resource-constrained Edge AI hardware. By decoupling heavy full-frame object detection from targeted regional neural classification, the system achieves stable operational performance without taxing system bandwidth.

Future developments focus on converting PyTorch models into TensorRT plan files (.engine) for FP16/INT8 hardware acceleration on the NVIDIA Jetson platform, integrating Head Pose Estimation via 6DRepNet to detect visual off-road loss of attention, and introducing CAN-bus telemetry hooks for direct vehicle integration.
