


import os
import shutil
import zipfile


def prepare_eye_dataset(zip_path="mrlEyeDataset.zip", output_base="data/eye_dataset"):
    temp_extract = "data/mrl_temp"

    # 1. Unzip the downloaded file
    print("Extracting downloaded MRL Eye Dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_extract)

    # 2. Map 'awake' -> 'open' and 'sleepy' -> 'closed'
    class_map = {"awake": "open", "sleepy": "closed"}

    print("Organizing dataset into data/eye_dataset...")

    for root, dirs, files in os.walk(temp_extract):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                path_parts = root.lower().split(os.sep)

                # Identify split (train / val / test) and class name
                split = None
                if "train" in path_parts:
                    split = "train"
                elif "val" in path_parts or "valid" in path_parts:
                    split = "val"
                elif "test" in path_parts:
                    split = "test"

                raw_class = None
                if "awake" in path_parts or "open" in path_parts:
                    raw_class = "open"
                elif "sleepy" in path_parts or "closed" in path_parts:
                    raw_class = "closed"

                if split and raw_class:
                    dest_dir = os.path.join(output_base, split, raw_class)
                    os.makedirs(dest_dir, exist_ok=True)
                    src_file = os.path.join(root, file)
                    shutil.copy(src_file, os.path.join(dest_dir, file))

    # Clean up temp folder
    shutil.rmtree(temp_extract, ignore_errors=True)
    print("MRL Eye Dataset ready in 'data/eye_dataset/'!")


if __name__ == "__main__":
    prepare_eye_dataset()