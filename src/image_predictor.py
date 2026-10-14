from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

VALIDATOR_PATH = MODEL_DIR / "xray_validator.keras"
DISEASE_MODEL_PATH = MODEL_DIR / "chest_xray_model.keras"

IMAGE_SIZE = (224, 224)

# Lowered from 0.85 to 0.50 for troubleshooting.
# This is not a clinically validated threshold.
MIN_XRAY_PROBABILITY = 0.50


# =========================================================
# LOAD MODELS
# =========================================================

@lru_cache(maxsize=1)
def load_models():
    """Load both saved models once."""

    missing_files = [
        str(path)
        for path in (VALIDATOR_PATH, DISEASE_MODEL_PATH)
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required model files are missing: "
            + ", ".join(missing_files)
        )

    try:
        validator = keras.models.load_model(
            VALIDATOR_PATH,
            compile=False
        )

        disease_model = keras.models.load_model(
            DISEASE_MODEL_PATH,
            compile=False
        )

    except Exception as error:
        raise RuntimeError(
            f"Unable to load the saved Keras models: {error}"
        ) from error

    return validator, disease_model


# =========================================================
# READ UPLOADED IMAGE
# =========================================================

def read_uploaded_image(uploaded_file):
    """Read an uploaded image and convert it to RGB."""

    if hasattr(uploaded_file, "getvalue"):
        image_bytes = uploaded_file.getvalue()

    elif isinstance(uploaded_file, (bytes, bytearray)):
        image_bytes = bytes(uploaded_file)

    elif hasattr(uploaded_file, "read"):
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        image_bytes = uploaded_file.read()

    else:
        raise TypeError(
            "Please upload a JPG, JPEG, or PNG image."
        )

    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()

        return image

    except Exception as error:
        raise ValueError(
            "The uploaded file is not a valid readable image."
        ) from error


# =========================================================
# VALIDATOR OUTPUT
# =========================================================

def get_xray_probability(model_output):
    """
    Return the predicted XRAY probability.

    Expected validator classes for two outputs:
    index 0 = NOT_XRAY
    index 1 = XRAY

    For a single sigmoid output, it must represent XRAY.
    """

    values = np.asarray(
        model_output,
        dtype=np.float32
    ).reshape(-1)

    if values.size == 1:
        return float(np.clip(values[0], 0.0, 1.0))

    if values.size == 2:
        return float(np.clip(values[1], 0.0, 1.0))

    raise ValueError(
        "Unexpected output shape from the X-ray validator."
    )


# =========================================================
# DISEASE MODEL OUTPUT
# =========================================================

def get_pneumonia_probability(model_output):
    """
    Return the predicted PNEUMONIA probability.

    Expected disease classes for two outputs:
    index 0 = NORMAL
    index 1 = PNEUMONIA

    For a single sigmoid output, it must represent PNEUMONIA.
    """

    values = np.asarray(
        model_output,
        dtype=np.float32
    ).reshape(-1)

    if values.size == 1:
        return float(np.clip(values[0], 0.0, 1.0))

    if values.size == 2:
        return float(np.clip(values[1], 0.0, 1.0))

    raise ValueError(
        "Unexpected output shape from the chest X-ray model."
    )


# =========================================================
# MAIN PREDICTION FUNCTION
# =========================================================

def predict_xray(uploaded_file):
    """
    Analyze an uploaded image.

    Returns:
        (label, score)

    Possible labels:
        NOT A CHEST X-RAY
        NORMAL
        PNEUMONIA

    The score is a model probability, not medical certainty.
    """

    # Step 1: Read the uploaded image.
    image = read_uploaded_image(uploaded_file)

    # Step 2: Load the validator and disease models.
    validator, disease_model = load_models()

    # Both saved models include their own input rescaling layers.
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_batch = np.expand_dims(
        image_array,
        axis=0
    )

    # Step 4: Validate whether the image resembles an X-ray.
    validator_output = validator.predict(
        image_batch,
        verbose=0
    )

    xray_probability = get_xray_probability(
        validator_output
    )

    # Reject only if the validator score is below the threshold.
    if xray_probability < MIN_XRAY_PROBABILITY:
        return "NOT A CHEST X-RAY", 0.0

    # Step 5: Predict the disease only after passing validation.
    disease_output = disease_model.predict(
        image_batch,
        verbose=0
    )

    pneumonia_probability = get_pneumonia_probability(
        disease_output
    )

    # Step 6: Return the prediction and its associated score.
    if pneumonia_probability >= 0.50:
        return "PNEUMONIA", pneumonia_probability

    return "NORMAL", 1.0 - pneumonia_probability