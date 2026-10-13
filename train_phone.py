import os
import shutil
import torch
import yaml
from ultralytics import YOLO


def locate_and_fix_dataset_yaml():
    """Finds dataset.yaml across possible extraction locations

    and ensures absolute paths are configured correctly for YOLO.
    """
    possible_paths = [
        os.path.abspath("data/phone_dataset/dataset.yaml"),
        os.path.abspath("data/phone_dataset/data.yaml"),
        os.path.abspath("data/dataset.yaml"),
        os.path.abspath("data/data.yaml"),
    ]

    yaml_path = None
    for path in possible_paths:
        if os.path.exists(path):
            yaml_path = path
            break

    if not yaml_path:
        raise FileNotFoundError(
            "Could not locate dataset.yaml or data.yaml! "
            "Please check that your phone dataset is extracted inside data/phone_dataset/."
        )

    print(f"Found dataset configuration file at: {yaml_path}")

    # Fix relative/absolute path entries inside dataset.yaml for cross-platform compatibility
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    phone_dataset_dir = os.path.abspath("data/phone_dataset")
    config["path"] = phone_dataset_dir  # Set dataset root path

    # Rewrite updated yaml file
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return yaml_path


def train_phone_detector():
    # 1. Ensure dataset YAML is correctly located and configured
    yaml_path = locate_and_fix_dataset_yaml()

    # 2. Select compute device (GPU if available, CPU otherwise)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training YOLOv8 on device: {device}")

    # 3. Initialize YOLOv8 Nano pre-trained model
    model = YOLO("yolov8n.pt")

    # 4. Train the model
    print("Starting YOLOv8 fine-tuning for Phone Distraction Detection...")
    model.train(
        data=yaml_path,
        epochs=30,  # 30 epochs is optimal to prevent overfitting
        imgsz=640,  # Standard resolution for object detection
        batch=16,  # Reduce to 8 if you run into GPU Memory errors
        patience=10,  # Early stopping patience
        project="runs/detect",
        name="phone_run",
        device=device,
        exist_ok=True,
    )

    # 5. Export best weights to models/ directory
    os.makedirs("models", exist_ok=True)
    best_weights = os.path.join("runs", "detect", "phone_run", "weights", "best.pt")
    target_weights = os.path.join("models", "phone_detector.pt")

    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        print(f"\nTraining Complete! Best model exported to: {target_weights}")
    else:
        print(f"\nWarning: Could not find best weights at {best_weights}")


if __name__ == "__main__":
    train_phone_detector()