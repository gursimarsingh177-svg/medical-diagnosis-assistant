from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


# ==========================================
# 1. SETTINGS
# ==========================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

TRAIN_DIR = "dataset/chest_xray/train"
VAL_DIR = "dataset/chest_xray/val"


# ==========================================
# 2. LOAD TRAINING DATA
# ==========================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)


# ==========================================
# 3. LOAD VALIDATION DATA
# ==========================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary"
)


# ==========================================
# 4. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
])


# ==========================================
# 5. PRETRAINED MODEL
# ==========================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially
base_model.trainable = False


# ==========================================
# 6. BUILD OUR MODEL
# ==========================================

model = models.Sequential([
    data_augmentation,

    layers.Rescaling(1.0 / 255),

    base_model,

    layers.GlobalAveragePooling2D(),

    layers.Dropout(0.2),

    layers.Dense(1, activation="sigmoid")
])


# ==========================================
# 7. COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 8. DISPLAY MODEL
# ==========================================

model.summary()


# ==========================================
# 9. TRAIN
# ==========================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


# ==========================================
# 10. SAVE MODEL
# ==========================================

project_root = Path(__file__).resolve().parent
model_dir = project_root / "models"
model_dir.mkdir(exist_ok=True)

model_path = model_dir / "chest_xray_model.keras"
model.save(model_path)

print("\nModel training completed.")
print(f"Saved to: {model_path}")