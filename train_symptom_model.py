from pathlib import Path
from io import StringIO

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


# =========================================================
# SETTINGS
# =========================================================

DATA_PATH = Path("dataset/symptoms/Training.csv")
MODEL_DIR = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.20


# =========================================================
# LOAD DATASET
# =========================================================

raw_text = DATA_PATH.read_text(
    encoding="utf-8-sig"
)

first_line = raw_text.splitlines()[0]

# Handle the tab-separated dataset
if "\t" in first_line:
    df = pd.read_csv(
        StringIO(raw_text),
        sep="\t"
    )

elif r"\t" in first_line:
    fixed_text = raw_text.replace(
        r"\t",
        "\t"
    )

    df = pd.read_csv(
        StringIO(fixed_text),
        sep="\t"
    )

else:
    df = pd.read_csv(
        StringIO(raw_text)
    )


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = [
    str(column).strip()
    for column in df.columns
]


# =========================================================
# VALIDATE DATA
# =========================================================

if "prognosis" not in df.columns:
    raise ValueError(
        "The 'prognosis' column was not found."
    )


if df.isnull().sum().sum() > 0:
    raise ValueError(
        "Dataset contains missing values."
    )


# =========================================================
# SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop(
    columns=["prognosis"]
)

y = df["prognosis"].astype(str).str.strip()


print("Dataset shape:", df.shape)
print("Number of symptom features:", X.shape[1])
print("Number of diseases:", y.nunique())


# =========================================================
# ENCODE DISEASE LABELS
# =========================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("Disease classes:", len(label_encoder.classes_))


# =========================================================
# STRATIFIED TRAIN/TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_encoded,
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# DEFINE MODELS
# =========================================================

models_to_test = {

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_STATE,
    ),
}


# =========================================================
# TRAIN AND COMPARE
# =========================================================

results = {}

trained_models = {}


for model_name, model in models_to_test.items():

    print(
        f"\nTraining {model_name}..."
    )

    model.fit(
        X_train,
        y_train,
    )

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    results[model_name] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }

    trained_models[model_name] = model

    print(
        f"{model_name} Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"{model_name} Macro F1: "
        f"{macro_f1 * 100:.2f}%"
    )


# =========================================================
# FIND BEST MODEL
# =========================================================

best_model_name = max(
    results,
    key=lambda name: (
        results[name]["macro_f1"],
        results[name]["accuracy"],
    ),
)

best_model = trained_models[
    best_model_name
]


# =========================================================
# PRINT COMPARISON
# =========================================================

print("\n==========================================")
print("MODEL COMPARISON")
print("==========================================")

for model_name, metrics in results.items():

    print(
        f"\n{model_name}"
    )

    print(
        f"Accuracy : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Macro F1 : "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )


print("\n==========================================")
print(
    f"BEST MODEL: {best_model_name}"
)
print("==========================================")


# =========================================================
# CREATE MODEL DIRECTORY
# =========================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# SAVE EVERYTHING NEEDED FOR PREDICTION
# =========================================================

model_bundle = {
    "model": best_model,
    "label_encoder": label_encoder,
    "feature_names": list(X.columns),
    "results": results,
}


MODEL_PATH = (
    MODEL_DIR / "symptom_model.pkl"
)


joblib.dump(
    model_bundle,
    MODEL_PATH,
)


print(
    f"\nSaved symptom model to:"
    f"\n{MODEL_PATH}"
)

print(
    "\nSymptom model training completed successfully."
)