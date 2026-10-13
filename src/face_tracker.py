import os
import urllib.request
import cv2
import numpy as np

# Handle MediaPipe import safely across version differences
HAS_MEDIAPIPE = False
try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        mp_face_mesh = mp.solutions.face_mesh
        HAS_MEDIAPIPE = True
except Exception:
    HAS_MEDIAPIPE = False


class FaceEyeTracker:
    def __init__(self):
        self.use_mediapipe = HAS_MEDIAPIPE

        if self.use_mediapipe:
            try:
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception:
                self.use_mediapipe = False

        if not self.use_mediapipe:
            # Download XML locally to avoid OpenCV missing file error
            xml_path = "haarcascade_eye.xml"
            if not os.path.exists(xml_path):
                print("Downloading haarcascade_eye.xml fallback file...")
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml"
                urllib.request.urlretrieve(url, xml_path)
            self.eye_cascade = cv2.CascadeClassifier(xml_path)

        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

    def get_eye_crops(self, frame):
        h, w, _ = frame.shape

        if self.use_mediapipe:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return (None, None), (None, None)

            landmarks = results.multi_face_landmarks[0].landmark

            def extract_crop(indices):
                pts = np.array([
                    (int(landmarks[i].x * w), int(landmarks[i].y * h))
                    for i in indices
                ])
                x_min, y_min = np.min(pts, axis=0)
                x_max, y_max = np.max(pts, axis=0)

                pad_x = int((x_max - x_min) * 0.4)
                pad_y = int((y_max - y_min) * 0.4)

                x1, y1 = max(0, x_min - pad_x), max(0, y_min - pad_y)
                x2, y2 = min(w, x_max + pad_x), min(h, y_max + pad_y)

                crop = frame[y1:y2, x1:x2]
                return crop, (x1, y1, x2, y2)

            left_crop, left_bbox = extract_crop(self.LEFT_EYE_INDICES)
            right_crop, right_bbox = extract_crop(self.RIGHT_EYE_INDICES)

            return (left_crop, left_bbox), (right_crop, right_bbox)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5
            )

            if len(eyes) < 2:
                return (None, None), (None, None)

            eyes = sorted(eyes, key=lambda e: e[0])
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]

            left_crop = frame[ey1 : ey1 + eh1, ex1 : ex1 + ew1]
            right_crop = frame[ey2 : ey2 + eh2, ex2 : ex2 + ew2]

            return (left_crop, (ex1, ey1, ex1 + ew1, ey1 + eh1)), (
                right_crop,
                (ex2, ey2, ex2 + ew2, ey2 + eh2),
            )