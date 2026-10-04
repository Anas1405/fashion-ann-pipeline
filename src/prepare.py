"""Stage 1: download Fashion-MNIST and save the raw arrays to data/raw/."""
import os

import numpy as np
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

os.makedirs("data/raw", exist_ok=True)
np.savez_compressed(
    "data/raw/fashion_mnist.npz",
    x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
)
print(f"Saved raw data: train={x_train.shape}, test={x_test.shape}")
