"""Stage 2: normalize pixels to [0, 1] and split a validation set."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]
raw = np.load("data/raw/fashion_mnist.npz")

# Normalize pixel values to [0, 1]
x_train = raw["x_train"].astype("float32") / 255.0
x_test = raw["x_test"].astype("float32") / 255.0

x_tr, x_val, y_tr, y_val = train_test_split(
    x_train, raw["y_train"],
    test_size=params["val_size"],
    random_state=params["seed"],
    stratify=raw["y_train"],
)

os.makedirs("data/processed", exist_ok=True)
np.savez_compressed(
    "data/processed/data.npz",
    x_train=x_tr, y_train=y_tr,
    x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=raw["y_test"],
)
print(f"Saved processed data: train={x_tr.shape}, val={x_val.shape}, test={x_test.shape}")
print("Preprocessing finished successfully.")
