"""Stage 3: build and train the ANN using hyperparameters from params.yaml."""
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

params = yaml.safe_load(open("params.yaml"))["train"]
tf.keras.utils.set_random_seed(params["seed"])

data = np.load("data/processed/data.npz")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(params["dense_units"], activation="relu"),
    tf.keras.layers.Dropout(params["dropout_rate"]),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    data["x_train"], data["y_train"],
    validation_data=(data["x_val"], data["y_val"]),
    epochs=params["epochs"],
    batch_size=params["batch_size"],
)

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)
print("Saved models/model.h5 and models/history.csv")
