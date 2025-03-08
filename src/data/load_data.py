import tensorflow as tf
import os
import yaml
import numpy as np

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = "data/raw/plantvillage/color/"  # Point to the color folder

def load_data():
    # Verify the directory exists
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Dataset directory {DATA_DIR} not found. Please check your data structure.")

    # Get class names from folder names (ignore files like .gitkeep)
    class_names = [d for d in sorted(os.listdir(DATA_DIR)) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not class_names:
        raise ValueError(f"No class folders found in {DATA_DIR}")

    # Create dataset from directory
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        image_size=tuple(config["data"]["image_size"]),
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        label_mode="int"  # Integer labels (0, 1, 2, ...)
    )

    # Normalize images
    dataset = dataset.map(lambda x, y: (x / 255.0, y))

    # Dummy info object to mimic tfds
    class Info:
        features = {"label": type("Label", (), {"names": class_names, "num_classes": len(class_names)})}

    info = Info()
    return dataset, info

if __name__ == "__main__":
    train_ds, info = load_data()
    print(f"Class names: {info.features['label'].names}")
    print(f"Number of classes: {info.features['label'].num_classes}")