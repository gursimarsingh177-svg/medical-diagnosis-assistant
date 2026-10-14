from pathlib import Path

import joblib
import pandas as pd


# =========================================================
# MODEL PATH
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "symptom_model.pkl"


# =========================================================
# LOAD SAVED MODEL
# =========================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Symptom model not found: {MODEL_PATH}"
    )

model_bundle = joblib.load(MODEL_PATH)

model = model_bundle["model"]
label_encoder = model_bundle["label_encoder"]
feature_names = model_bundle["feature_names"]


# =========================================================
# SYMPTOM PREDICTION
# =========================================================

def predict_symptoms(selected_symptoms, top_k=3):
    """
    Predict possible disease labels from selected symptoms.

    Parameters
    ----------
    selected_symptoms:
        List of symptom names matching the training columns.

    top_k:
        Number of highest-scoring predictions to return.

    Returns
    -------
    Dictionary containing the top prediction and alternatives.

    This is an educational ML prediction, not a diagnosis.
    """

    if not isinstance(selected_symptoms, (list, tuple, set)):
        raise TypeError(
            "selected_symptoms must be a list, tuple, or set."
        )

    if not 1 <= top_k <= len(label_encoder.classes_):
        raise ValueError(
            "top_k must be between 1 and the number of disease classes."
        )

    # Remove duplicate selections
    selected_symptoms = set(selected_symptoms)

    # Verify symptom names
    unknown_symptoms = selected_symptoms.difference(feature_names)

    if unknown_symptoms:
        raise ValueError(
            "Unknown symptom names: "
            + ", ".join(sorted(unknown_symptoms))
        )

    if not selected_symptoms:
        raise ValueError(
            "Please select at least one symptom."
        )

    # Create a row with all 132 expected features
    symptom_vector = {
        feature: int(feature in selected_symptoms)
        for feature in feature_names
    }

    input_df = pd.DataFrame(
        [symptom_vector],
        columns=feature_names,
    )

    # Predict the leading disease class
    encoded_prediction = model.predict(input_df)[0]

    predicted_disease = label_encoder.inverse_transform(
        [int(encoded_prediction)]
    )[0]

    # Obtain the top model scores when supported
    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "The saved model does not support probability predictions."
        )

    probabilities = model.predict_proba(input_df)[0]

    top_indices = probabilities.argsort()[::-1][:top_k]

    class_ids = model.classes_[top_indices]

    top_labels = label_encoder.inverse_transform(
        class_ids.astype(int)
    )

    top_predictions = []

    for disease, index in zip(top_labels, top_indices):
        top_predictions.append(
            {
                "disease": str(disease),
                "probability": float(probabilities[index]),
            }
        )

    return {
        "prediction": str(predicted_disease),
        "top_predictions": top_predictions,
        "selected_symptoms": sorted(selected_symptoms),
    }