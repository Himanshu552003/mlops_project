import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tensorflow as tf
import numpy as np
import cv2
import yaml
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from src.data.load_data import load_data
from src.visualization.visualize import get_gradcam_heatmap

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def build_model(num_classes):
    base_model = MobileNetV2(weights="imagenet", include_top=False, 
                             input_shape=(*config["data"]["image_size"], 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation="relu")(x)
    output = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(config["model"]["learning_rate"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_model():
    train_ds, info = load_data()
    num_classes = info.features["label"].num_classes
    model = build_model(num_classes)
    model.fit(train_ds, epochs=config["model"]["epochs"])
    model.save(config["model"]["output_path"])
    return model, info

def load_trained_model():
    return tf.keras.models.load_model(config["model"]["output_path"])

def predict_crop(file_path, rainfall_value=0.0):
    model = load_trained_model()
    _, info = load_data()
    class_names = info.features["label"].names

    img = cv2.imread(file_path)
    if img is None:
        return {"error": "Unable to load image"}
    img_resized = cv2.resize(img, tuple(config["data"]["image_size"]))
    img_array = np.expand_dims(img_resized / 255.0, axis=0)

    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])
    label = class_names[class_idx]
    crop_type, disease = label.split("___") if "___" in label else (label, "Healthy")
    disease_status = "Healthy" if disease == "healthy" else "Diseased"  # Fixed logic
    unified_prediction = f"{crop_type} - {disease_status} - {disease}"

    heatmap = get_gradcam_heatmap(model, img_array)
    heatmap_path = file_path.replace(".jpg", "_heatmap.jpg").replace(".png", "_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)

    return {
        "crop_type": crop_type,
        "disease_status": disease_status,
        "disease_type": disease,
        "unified_prediction": unified_prediction,
        "confidence": confidence,
        "heatmap_path": heatmap_path,
        "original_img_path": file_path
    }

if __name__ == "__main__":
    train_model()