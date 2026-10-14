from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
from tensorflow.keras.datasets import cifar10


# =========================================================
# SETTINGS
# =========================================================

BASE_DIR = Path("dataset/xray_validator")

RANDOM_SEED = 42

TRAIN_XRAY_COUNT = 1800
VAL_XRAY_COUNT = 300
TEST_XRAY_COUNT = 300

TRAIN_NON_XRAY_COUNT = 1800
VAL_NON_XRAY_COUNT = 300
TEST_NON_XRAY_COUNT = 300


random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# CREATE FOLDERS
# =========================================================

splits = ["train", "val", "test"]
classes = ["XRAY", "NOT_XRAY"]

for split in splits:
    for class_name in classes:
        folder = BASE_DIR / split / class_name
        folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# FIND CHEST X-RAY IMAGES
# =========================================================

original_base = Path("dataset/chest_xray")

train_xray_files = []

for class_name in ["NORMAL", "PNEUMONIA"]:

    folder = original_base / "train" / class_name

    train_xray_files.extend(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )


val_xray_files = []

for class_name in ["NORMAL", "PNEUMONIA"]:

    folder = original_base / "val" / class_name

    val_xray_files.extend(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )


test_xray_files = []

for class_name in ["NORMAL", "PNEUMONIA"]:

    folder = original_base / "test" / class_name

    test_xray_files.extend(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )


# Shuffle deterministically
random.shuffle(train_xray_files)
random.shuffle(val_xray_files)
random.shuffle(test_xray_files)


# =========================================================
# CHECK X-RAY COUNTS
# =========================================================

if len(train_xray_files) < TRAIN_XRAY_COUNT:
    raise ValueError("Not enough training X-ray images.")

if len(val_xray_files) < VAL_XRAY_COUNT:
    raise ValueError("Not enough validation X-ray images.")

if len(test_xray_files) < TEST_XRAY_COUNT:
    raise ValueError("Not enough test X-ray images.")


# =========================================================
# COPY X-RAY IMAGES
# =========================================================

def copy_xray_images(files, destination, prefix):

    for index, source in enumerate(files):

        new_name = f"{prefix}_{index:05d}{source.suffix.lower()}"

        destination_path = destination / new_name

        shutil.copy2(source, destination_path)


copy_xray_images(
    train_xray_files[:TRAIN_XRAY_COUNT],
    BASE_DIR / "train" / "XRAY",
    "xray_train",
)

copy_xray_images(
    val_xray_files[:VAL_XRAY_COUNT],
    BASE_DIR / "val" / "XRAY",
    "xray_val",
)

copy_xray_images(
    test_xray_files[:TEST_XRAY_COUNT],
    BASE_DIR / "test" / "XRAY",
    "xray_test",
)


# =========================================================
# LOAD CIFAR-10
# =========================================================

print("\nDownloading/loading CIFAR-10...")

(x_train, _), (x_test, _) = cifar10.load_data()


print("CIFAR-10 loaded.")


# =========================================================
# SELECT NON-XRAY IMAGES
# =========================================================

train_indices = np.random.choice(
    len(x_train),
    size=TRAIN_NON_XRAY_COUNT + VAL_NON_XRAY_COUNT,
    replace=False,
)

test_indices = np.random.choice(
    len(x_test),
    size=TEST_NON_XRAY_COUNT,
    replace=False,
)


train_indices = train_indices.tolist()
test_indices = test_indices.tolist()


validator_train_indices = train_indices[:TRAIN_NON_XRAY_COUNT]

validator_val_indices = train_indices[
    TRAIN_NON_XRAY_COUNT:
]


# =========================================================
# SAVE NON-XRAY IMAGES
# =========================================================

def save_cifar_images(images, indices, destination, prefix):

    for output_index, image_index in enumerate(indices):

        image_array = images[image_index]

        image = Image.fromarray(image_array)

        destination_path = (
            destination
            / f"{prefix}_{output_index:05d}.png"
        )

        image.save(destination_path)


save_cifar_images(
    x_train,
    validator_train_indices,
    BASE_DIR / "train" / "NOT_XRAY",
    "nonxray_train",
)

save_cifar_images(
    x_train,
    validator_val_indices,
    BASE_DIR / "val" / "NOT_XRAY",
    "nonxray_val",
)

save_cifar_images(
    x_test,
    test_indices,
    BASE_DIR / "test" / "NOT_XRAY",
    "nonxray_test",
)


# =========================================================
# FINAL COUNTS
# =========================================================

print("\n==========================================")
print("X-RAY VALIDATOR DATASET CREATED")
print("==========================================")

for split in splits:

    print(f"\n{split.upper()}")

    for class_name in classes:

        folder = BASE_DIR / split / class_name

        image_count = len(
            [
                file
                for file in folder.iterdir()
                if file.is_file()
            ]
        )

        print(
            f"{class_name}: {image_count} images"
        )