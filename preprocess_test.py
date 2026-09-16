from pathlib import Path

import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent / "dataset" / "chest_xray" / "test"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load_test_dataset() -> tf.data.Dataset:
    """Load and preprocess the test images without changing their order."""
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Test folder not found: {BASE_DIR}")

    dataset = tf.keras.utils.image_dataset_from_directory(
        BASE_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        color_mode="rgb",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)


if __name__ == "__main__":
    test_dataset = load_test_dataset()
    image_count = sum(images.shape[0] for images, _ in test_dataset)

    print("Test images:", image_count)
    print("Image size:", IMAGE_SIZE)
    print("Class names:", CLASS_NAMES)
    print("Channels:", next(iter(test_dataset))[0].shape[-1])
