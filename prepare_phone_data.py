import os
import shutil
import zipfile


def extract_and_prepare_phone_data(
    zip_path="phone_dataset.zip", output_dir="data/phone_dataset"
):
    if not os.path.exists(zip_path):
        print(f"Error: Could not find '{zip_path}' in the project root directory!")
        print(
            "Please download the dataset zip file and place it in the root folder."
        )
        return

    print(f"Extracting {zip_path} into {output_dir}...")

    # Clear old data if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Unzip directly into data/phone_dataset
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print("Phone dataset extraction complete!")
    print(f"Files extracted into: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    extract_and_prepare_phone_data()