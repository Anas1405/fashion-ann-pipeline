"""Stage 4: evaluate the model on the test set and write metrics.json."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

data = np.load("data/processed/data.npz")
model = tf.keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(data["x_test"], data["y_test"], verbose=0)
preds = model.predict(data["x_test"], verbose=0).argmax(axis=1)

cm = confusion_matrix(data["y_test"], preds)
fig, ax = plt.subplots(figsize=(9, 9))
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.tight_layout()
plt.savefig("confusion_matrix.png")

with open("metrics.json", "w") as f:
    json.dump({"test_loss": round(float(loss), 4), "test_accuracy": round(float(acc), 4)}, f, indent=2)
print(f"Test loss={loss:.4f}  Test accuracy={acc:.4f}")
