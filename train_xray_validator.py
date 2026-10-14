from pathlib import Path

import tensorflow as tf
from tensorflow import keras


# =========================================================
# PATHS AND SETTINGS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "dataset" / "xray_validator"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "xray_validator.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42

CLASS_NAMES = ["NOT_XRAY", "XRAY"]


# =========================================================
# DATASET LOADING
# =========================================================

def load_split(split_name, shuffle):
    folder = DATASET_DIR / split_name

    if not folder.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {folder}"
        )

    for class_name in CLASS_NAMES:
        class_folder = folder / class_name

        if not class_folder.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {class_folder}"
            )

    dataset = keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading X-ray validator datasets...")

    train_ds = load_split("train", shuffle=True)
    val_ds = load_split("val", shuffle=False)

    test_folder = DATASET_DIR / "test"

    test_ds = None
    if test_folder.is_dir():
        test_ds = load_split("test", shuffle=False)

    print("Class order:", CLASS_NAMES)
    print("Image size:", IMAGE_SIZE)

    # =====================================================
    # BUILD MODEL
    # =====================================================

    base_model = keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    base_model.trainable = False

    inputs = keras.Input(
        shape=(*IMAGE_SIZE, 3),
        name="image_pixels_0_to_255",
    )

    # Data augmentation
    x = keras.layers.RandomFlip("horizontal")(inputs)
    x = keras.layers.RandomRotation(0.03)(x)
    x = keras.layers.RandomZoom(0.05)(x)

    # Preprocessing is included inside the saved model.
    # Input pixels are expected in the range [0, 255].
    x = keras.layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenetv2_rescaling",
    )(x)

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)

    outputs = keras.layers.Dense(
        1,
        activation="sigmoid",
        name="xray_probability",
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=1,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # =====================================================
    # TRAIN
    # =====================================================

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # Save the restored best weights.
    model.save(MODEL_PATH)

    print("\nValidator saved successfully:")
    print(MODEL_PATH)
    print("Output: sigmoid probability of XRAY.")
    print("Class 0 = NOT_XRAY; class 1 = XRAY.")

    # Evaluate on the separate test dataset, if available.
    if test_ds is not None:
        print("\nEvaluating on the test dataset...")

        results = model.evaluate(
            test_ds,
            return_dict=True,
            verbose=1,
        )

        for metric_name, value in results.items():
            print(f"{metric_name}: {value:.4f}")


if __name__ == "__main__":
    main()