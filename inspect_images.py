import sys
from pathlib import Path

import matplotlib

if not sys.stdout.isatty():
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent / "dataset" / "chest_xray" / "train"

if not BASE_DIR.exists():
    raise FileNotFoundError(f"Training folder not found: {BASE_DIR}")

normal_images = sorted((BASE_DIR / "NORMAL").glob("*.*"))
pneumonia_images = sorted((BASE_DIR / "PNEUMONIA").glob("*.*"))

if not normal_images or not pneumonia_images:
    raise FileNotFoundError(
        f"No training images found under {BASE_DIR / 'NORMAL'} or {BASE_DIR / 'PNEUMONIA'}"
    )

normal_image_path = normal_images[0]
pneumonia_image_path = pneumonia_images[0]

normal_image = Image.open(normal_image_path)
pneumonia_image = Image.open(pneumonia_image_path)

print("NORMAL image:")
print("File:", normal_image_path)
print("Size:", normal_image.size)
print("Mode:", normal_image.mode)

print("\nPNEUMONIA image:")
print("File:", pneumonia_image_path)
print("Size:", pneumonia_image.size)
print("Mode:", pneumonia_image.mode)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(normal_image, cmap="gray")
plt.title("NORMAL")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pneumonia_image, cmap="gray")
plt.title("PNEUMONIA")
plt.axis("off")

plt.tight_layout()

if sys.stdout.isatty():
    plt.show()
else:
    output_path = Path(__file__).resolve().parent / "inspection_preview.png"
    plt.savefig(output_path, bbox_inches="tight")
    print(f"\nSaved preview image to: {output_path}")