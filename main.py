import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
from ultralytics import YOLO
from src.face_tracker import FaceEyeTracker

# 1. Initialize Face & Eye Tracker
eye_tracker = FaceEyeTracker()

# 2. Load Phone Distraction Detector (YOLOv8)
phone_model_path = "models/phone_detector.pt"
if not os.path.exists(phone_model_path):
    # Fallback to default weights if custom weights aren't in models/
    phone_model_path = "runs/detect/runs/detect/phone_run/weights/best.pt"

print(f"Loading Phone Detector weights from: {phone_model_path}")
phone_model = YOLO(phone_model_path)

# 3. Load Eye Classifier (MobileNetV2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eye_model = models.mobilenet_v2(pretrained=False)
eye_model.classifier[1] = nn.Linear(eye_model.last_channel, 2)

eye_pth_path = "models/eye_classifier.pth"
if os.path.exists(eye_pth_path):
    eye_model.load_state_dict(torch.load(eye_pth_path, map_location=device))
    eye_model.to(device)
    eye_model.eval()
    print("Loaded Eye Classifier PyTorch weights successfully.")

# Image preprocessing for eye model
eye_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 4. Initialize Video Capture / Camera Stream
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("\n[WARNING] Webcam index 0 could not be opened!")
    print("Running in STATIC IMAGE TEST MODE to verify pipeline inference...")
    
    # Try to find a test image in phone_dataset/test/images
    test_img_dir = "data/phone_dataset/test/images"
    test_frame = None
    if os.path.exists(test_img_dir):
        images = [f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))]
        if images:
            sample_path = os.path.join(test_img_dir, images[0])
            test_frame = cv2.imread(sample_path)
            print(f"Loaded sample test image from: {sample_path}")

    if test_frame is None:
        # Create a blank dummy frame if no test image is available
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "NO WEBCAM / DUMMY FRAME", (120, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    frames_to_process = [test_frame]
    USE_WEBCAM = False
else:
    print("\nStarting ADAS Multi-Hazard Driver Monitoring System (Webcam Live Mode)...")
    USE_WEBCAM = True

# Main Execution Loop
while True:
    if USE_WEBCAM:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting...")
            break
        # Flip frame horizontally for natural mirror view
        frame = cv2.flip(frame, 1)
    else:
        frame = frames_to_process[0].copy()

    # --- A. PHONE DISTRACTION DETECTION ---
    phone_results = phone_model(frame, conf=0.40, verbose=False)
    phone_detected = False

    for r in phone_results:
        for box in r.boxes:
            phone_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = f"Phone: {conf * 100:.1f}%"

            # Draw Red Bounding Box & Label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- B. EYE DROWSINESS DETECTION ---
    left_eye_data, right_eye_data = eye_tracker.get_eye_crops(frame)
    drowsy_detected = False

    for eye_crop, bbox in [left_eye_data, right_eye_data]:
        if eye_crop is not None and bbox is not None and eye_crop.size > 0:
            ex1, ey1, ex2, ey2 = bbox
            cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (255, 255, 0), 1)

            # Preprocess crop for Eye Model Inference
            try:
                pil_img = Image.fromarray(cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB))
                tensor_img = eye_transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = eye_model(tensor_img)
                    _, pred = torch.max(output, 1)
                    # Class 0: Closed / Drowsy, Class 1: Open
                    if pred.item() == 0:
                        drowsy_detected = True
            except Exception:
                pass

    # --- C. SYSTEM STATUS OVERLAY ---
    if phone_detected and drowsy_detected:
        cv2.putText(frame, "CRITICAL ALERT: PHONE + DROWSINESS!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    elif phone_detected:
        cv2.putText(frame, "ALERT: PHONE DISTRACTION!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    elif drowsy_detected:
        cv2.putText(frame, "WARNING: DROWSINESS DETECTED!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(frame, "Driver Focus: Normal", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display Result Window
    cv2.imshow("ADAS Driver Monitor", frame)

    # Press 'q' to exit loop or press any key in Static Mode
    key = cv2.waitKey(1 if USE_WEBCAM else 0) & 0xFF
    if key == ord('q') or not USE_WEBCAM:
        break

cap.release()
cv2.destroyAllWindows()
print("ADAS Pipeline run complete.")