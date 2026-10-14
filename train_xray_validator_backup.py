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

TRAIN_DIR = "dataset/xray_validator/train"
VAL_DIR = "dataset/xray_validator/val"

MODEL_PATH = "models/xray_validator.keras"


# =========================================================
# LOAD DATA
# =========================================================

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

print("\nClasses:")
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
    name="validator_augmentation",
)


# =========================================================
# MOBILENETV2
# =========================================================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze pretrained feature extractor
base_model.trainable = False


# =========================================================
# BUILD MODEL
# =========================================================

inputs = layers.Input(
    shape=(224, 224, 3),
    name="validator_input",
)

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(
    x,
    training=False,
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="xray_output",
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="xray_validator",
)


# =========================================================
# COMPILE
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
    ],
)


# =========================================================
# MODEL SUMMARY
# =========================================================

print("\nModel Summary:\n")

model.summary()


# =========================================================
# TRAIN
# =========================================================

print("\nStarting X-ray validator training...\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)


# =========================================================
# SAVE
# =========================================================

model.save(MODEL_PATH)

print("\n==========================================")
print("X-ray validator training completed.")
print(f"Model saved to: {MODEL_PATH}")
print("==========================================")