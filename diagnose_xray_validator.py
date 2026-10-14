import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras

import src.image_predictor as image_predictor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TEST_DIR = image_predictor.PROJECT_ROOT / "dataset" / "xray_validator" / "test"


def prepare_image(image_path):
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = image.resize(image_predictor.IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)

    return np.expand_dims(image_array, axis=0)


def print_rescaling_layers(model):
    layers = [
        (layer.name, layer.scale, layer.offset)
        for layer in model.layers
        if isinstance(layer, keras.layers.Rescaling)
    ]
    print("  rescaling layers:", layers or "none at model top level")


def inspect_image(validator, image_path, expected_label):
    image_batch = prepare_image(image_path)
    raw_output = validator.predict(image_batch, verbose=0)
    xray_probability = image_predictor.get_xray_probability(raw_output)
    decision = (
        "XRAY"
        if xray_probability >= image_predictor.MIN_XRAY_PROBABILITY
        else "NOT_XRAY"
    )
    display_path = image_path
    try:
        display_path = image_path.resolve().relative_to(
            image_predictor.PROJECT_ROOT
        )
    except ValueError:
        display_path = image_path.resolve()

    print(
        f"{display_path} | raw_output={raw_output.tolist()} "
        f"| XRAY_probability={xray_probability:.6f} "
        f"| expected={expected_label} | threshold_decision={decision}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Inspect saved X-ray validator outputs on local images."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
        help="Number of sorted test images to inspect from each class folder.",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        help="Additional image path to inspect; may be supplied more than once.",
    )
    args = parser.parse_args()

    if args.samples_per_class < 1:
        parser.error("--samples-per-class must be at least 1")

    validator, disease_model = image_predictor.load_models()

    print("Predictor module:", Path(image_predictor.__file__).resolve())
    print("Validator model:", image_predictor.VALIDATOR_PATH.resolve())
    print("Disease model:", image_predictor.DISEASE_MODEL_PATH.resolve())
    print("Validator input/output:", validator.input_shape, validator.output_shape)
    print("Disease input/output:", disease_model.input_shape, disease_model.output_shape)
    print("Validator output mapping: sigmoid output is XRAY probability (class 1).")
    print("Validator preprocessing layers:")
    print_rescaling_layers(validator)
    print("Disease preprocessing layers:")
    print_rescaling_layers(disease_model)

    for expected_label in ("XRAY", "NOT_XRAY"):
        class_dir = TEST_DIR / expected_label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Test class folder not found: {class_dir}")

        images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )[: args.samples_per_class]
        if not images:
            raise FileNotFoundError(f"No supported images found in {class_dir}")

        print(f"\nExpected {expected_label}:")
        for image_path in images:
            inspect_image(validator, image_path, expected_label)

    for image_arg in args.image:
        image_path = Path(image_arg)
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        print("\nAdditional image:")
        inspect_image(validator, image_path, "unspecified")


if __name__ == "__main__":
    main()
