from pathlib import Path

BASE_DIR = Path("dataset/chest_xray")

classes = ["NORMAL", "PNEUMONIA"]
splits = ["train", "val", "test"]

for split in splits:
    print(f"\n{split.upper()}")

    for class_name in classes:
        folder = BASE_DIR / split / class_name

        if not folder.exists():
            print(f"{class_name}: FOLDER NOT FOUND")
            continue

        images = [
            file
            for file in folder.iterdir()
            if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]

        print(f"{class_name}: {len(images)} images")