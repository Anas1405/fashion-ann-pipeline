"""Stage 2: normalize pixels and split a validation set."""
import os

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]
raw = np.load("data/raw/fashion_mnist.npz")

# Merged: normalization method is chosen in params.yaml (preprocess.norm)
if params["norm"] == "standard":
    # Teammate's approach: zero mean / unit variance using training-set stats
    mean = raw["x_train"].mean()
    std = raw["x_train"].std()
    x_train = (raw["x_train"].astype("float32") - mean) / std
    x_test = (raw["x_test"].astype("float32") - mean) / std
else:
    # Main's approach: scale pixel values to [-1, 1]
    x_train = raw["x_train"].astype("float32") / 127.5 - 1.0
    x_test = raw["x_test"].astype("float32") / 127.5 - 1.0

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