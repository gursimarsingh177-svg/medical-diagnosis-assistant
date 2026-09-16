from pathlib import Path
import random
import shutil


BASE_DIR = Path(__file__).parent / "dataset" / "chest_xray"
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALIDATION_RATIO = 0.2
RANDOM_SEED = 42


def create_validation_split() -> None:
    """Move a fixed percentage of training images into the validation split."""
    random_generator = random.Random(RANDOM_SEED)

    for class_name in CLASSES:
        train_dir = BASE_DIR / "train" / class_name
        validation_dir = BASE_DIR / "val" / class_name

        if not train_dir.exists():
            raise FileNotFoundError(f"Training folder not found: {train_dir}")

        validation_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(
            image
            for image in train_dir.iterdir()
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS
        )
        random_generator.shuffle(images)

        available_images = [
            image for image in images if not (validation_dir / image.name).exists()
        ]
        validation_count = int(len(available_images) * VALIDATION_RATIO)

        for image in available_images[:validation_count]:
            shutil.move(str(image), validation_dir / image.name)

        print(
            f"{class_name}: moved {validation_count} images to "
            f"{validation_dir.relative_to(BASE_DIR.parent.parent)}"
        )


if __name__ == "__main__":
    create_validation_split()
