import tensorflow as tf

from keras import layers, models
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input


# =========================================================
# SETTINGS
# =========================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

TRAIN_DIR = "dataset/chest_xray/train"
VAL_DIR = "dataset/chest_xray/val"

MODEL_PATH = "models/chest_xray_model.keras"


# =========================================================
# LOAD TRAINING DATA
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42,
)


# =========================================================
# LOAD VALIDATION DATA
# =========================================================

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)


# =========================================================
# PRINT CLASS NAMES
# =========================================================

print("\nClass names:")
print(train_ds.class_names)


# =========================================================
# DATA AUGMENTATION
# =========================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)


# =========================================================
# LOAD PRETRAINED MOBILENETV2
# =========================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)


# Freeze pretrained layers
base_model.trainable = False


# =========================================================
# BUILD MODEL
# =========================================================

inputs = layers.Input(
    shape=(224, 224, 3),
    name="image_input",
)

# Data augmentation
x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = preprocess_input(x)

# Feature extraction
x = base_model(
    x,
    training=False,
)

# Convert feature maps to a vector
x = layers.GlobalAveragePooling2D()(x)

# Reduce overfitting
x = layers.Dropout(0.2)(x)

# Binary classification
outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="pneumonia_output",
)(x)


model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="medical_xray_mobilenetv2",
)


# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# =========================================================
# DISPLAY MODEL
# =========================================================

print("\nModel Summary:\n")

model.summary()


# =========================================================
# TRAIN MODEL
# =========================================================

print("\nStarting training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)