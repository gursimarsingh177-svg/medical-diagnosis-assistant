from pathlib import Path

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


# ==========================================
# SETTINGS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = PROJECT_ROOT / "dataset" / "chest_xray" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "chest_xray_model.keras"


# ==========================================
# LOAD TEST DATA
# ==========================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = test_ds.class_names

print("Class names:", class_names)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. Please train the model first."
    )

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully.")


# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("\nRunning predictions on test set...")

predictions = model.predict(
    test_ds,
    verbose=1,
) 


# ==========================================
# GET TRUE LABELS
# ==========================================

y_true = np.concatenate(
    [
        labels.numpy().flatten()
        for _, labels in test_ds
    ]
)


# ==========================================
# CONVERT PROBABILITIES TO CLASS LABELS
# ==========================================

y_probability = predictions.flatten()

y_pred = (y_probability >= 0.5).astype(int)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

print("\nTest Accuracy:")
print(f"{accuracy:.4f} ({accuracy * 100:.2f}%)")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

print("\nConfusion Matrix:")
print(cm)