import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import pickle
import yaml
import logging

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a')
logger = logging.getLogger(__name__)

# Load config
with open(os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Define paths
YIELD_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", config["data"]["processed_output_path"])
YIELD_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", config["model"]["yield_output_path"])

def train_yield_model():
    if not os.path.exists(YIELD_DATA_PATH):
        logger.info(f"Yield data not found at {YIELD_DATA_PATH}. Generating synthetic data.")
        from src.data.generate_synthetic import generate_synthetic_data
        generate_synthetic_data()

    df = pd.read_csv(YIELD_DATA_PATH)
    logger.info(f"Loaded synthetic yield data from: {YIELD_DATA_PATH}")

    # Encode categorical variables
    le_crop = LabelEncoder()
    le_soil = LabelEncoder()
    le_disease = LabelEncoder()
    df["Crop Type"] = le_crop.fit_transform(df["Crop Type"])
    df["Soil Type"] = le_soil.fit_transform(df["Soil Type"])
    df["Disease Status"] = le_disease.fit_transform(df["Disease Status"])

    X = df.drop("Yield", axis=1)
    y = df["Yield"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config["model"]["yield_params"]["random_state"])

    model = XGBRegressor(
        n_estimators=config["model"]["yield_params"]["n_estimators"],
        max_depth=config["model"]["yield_params"]["max_depth"],
        learning_rate=config["model"]["yield_params"]["learning_rate"],
        random_state=config["model"]["yield_params"]["random_state"]
    )
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    logger.info(f"Yield model R^2 score: {score:.4f}")

    # Save model and encoders
    os.makedirs(os.path.dirname(YIELD_MODEL_PATH), exist_ok=True)
    with open(YIELD_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "le_crop": le_crop, "le_soil": le_soil, "le_disease": le_disease}, f)
    logger.info(f"Yield model saved to: {YIELD_MODEL_PATH}")

def load_yield_model():
    if not os.path.exists(YIELD_MODEL_PATH):
        logger.error(f"Yield model not found at {YIELD_MODEL_PATH}. Training new model.")
        train_yield_model()
    with open(YIELD_MODEL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["le_crop"], data["le_soil"], data["le_disease"]

def predict_yield(crop_type, disease_status, soil_type, rainfall, fertilization, month):
    model, le_crop, le_soil, le_disease = load_yield_model()
    
    # Map PlantVillage crop format
    crop_map = {
        "Cherry_(including_sour)": "Cherry",
        "Corn_(maize)": "Maize",
        "Pepper": "Pepper"
    }
    mapped_crop = crop_map.get(crop_type, crop_type.split("_")[0])
    
    try:
        crop_encoded = le_crop.transform([mapped_crop])[0]
        soil_encoded = le_soil.transform([soil_type])[0]
        disease_encoded = le_disease.transform([disease_status])[0]
    except ValueError as e:
        logger.error(f"Encoding error: {str(e)}. Check input values against synthetic data.")
        return {"error": f"Invalid input: {str(e)}. Ensure values match PlantVillage crops."}

    input_data = np.array([[crop_encoded, disease_encoded, soil_encoded, float(rainfall), float(fertilization), int(month)]])
    yield_pred = model.predict(input_data)[0]
    return {"yield": round(yield_pred, 2)}

if __name__ == "__main__":
    train_yield_model()