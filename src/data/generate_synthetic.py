import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import numpy as np
import yaml
import logging

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a')
logger = logging.getLogger(__name__)

# Load config
with open(os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Paths
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", config["data"]["processed_output_path"])

def generate_synthetic_data():
    np.random.seed(42)
    data = []

    for crop in config["data"]["plantvillage_crops"]:
        n_samples = config["data"]["synthetic_ranges"]["n_samples_per_crop"]
        crop_df = pd.DataFrame({
            "Crop Type": [crop] * n_samples,
            "Soil Type": np.random.choice(config["data"]["synthetic_ranges"]["soil_types"], n_samples),
            "Rainfall": np.random.uniform(*config["data"]["synthetic_ranges"]["rainfall"], n_samples),
            "Fertilization": np.random.uniform(*config["data"]["synthetic_ranges"]["fertilization"], n_samples),
            "Month": np.random.randint(*config["data"]["synthetic_ranges"]["month"], n_samples),
            "Disease Status": np.random.choice(config["data"]["synthetic_ranges"]["disease_status"], n_samples),
            "Yield": np.random.uniform(*config["data"]["synthetic_ranges"]["yield"][crop], n_samples)
        })
        crop_df["Yield"] = crop_df.apply(lambda row: row["Yield"] * 0.8 if row["Disease Status"] == "Diseased" else row["Yield"], axis=1)
        data.append(crop_df)

    df = pd.concat(data, ignore_index=True)
    logger.info(f"Generated synthetic data with {len(df)} rows for {len(config['data']['plantvillage_crops'])} crops.")

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Synthetic data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_synthetic_data()