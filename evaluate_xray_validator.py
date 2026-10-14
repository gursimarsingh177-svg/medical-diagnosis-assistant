import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "dataset/xray_validator/test"
MODEL_PATH = "models/xray_validator.keras"


# =========================================================
# LOAD TEST DATA
# =========================================================

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

class_names = test_ds.class_names

print("Class names:", class_names)


# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(MODEL_PATH)

print("\nValidator model loaded successfully.")


# =========================================================
# PREDICT
# =========================================================

print("\nRunning validator predictions...")

predictions = model.predict(
    test_ds,
    verbose=1,
)


# =========================================================
# TRUE LABELS
# =========================================================

y_true = np.concatenate(
    [
        labels.numpy().flatten()
        for _, labels in test_ds
    ]
)

y_probability = predictions.flatten()

y_pred = (y_probability >= 0.5).astype(int)


# =========================================================
# ACCURACY
# =========================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

print("\nTest Accuracy:")
print(
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)


# =========================================================
# AUC
# =========================================================

auc = roc_auc_score(
    y_true,
    y_probability,
)

print("\nTest AUC:")
print(
    f"{auc:.4f}"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
)

print("\nConfusion Matrix:")
print(cm)