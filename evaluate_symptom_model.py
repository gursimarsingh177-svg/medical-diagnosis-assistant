from pathlib import Path
from io import StringIO

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


DATA_PATH = Path("dataset/symptoms/Training.csv")
MODEL_PATH = Path("models/symptom_model.pkl")


# =========================================================
# LOAD DATASET
# =========================================================

raw_text = DATA_PATH.read_text(
    encoding="utf-8-sig"
)

first_line = raw_text.splitlines()[0]

if "\t" in first_line:
    df = pd.read_csv(
        StringIO(raw_text),
        sep="\t"
    )
elif r"\t" in first_line:
    df = pd.read_csv(
        StringIO(raw_text.replace(r"\t", "\t")),
        sep="\t"
    )
else:
    df = pd.read_csv(
        StringIO(raw_text)
    )

df.columns = [
    str(column).strip()
    for column in df.columns
]


# =========================================================
# SEPARATE FEATURES / TARGET
# =========================================================

X = df.drop(columns=["prognosis"])
y = df["prognosis"].astype(str).str.strip()


# =========================================================
# DUPLICATE CHECK
# =========================================================

duplicate_rows = df.duplicated().sum()

duplicate_features = X.duplicated().sum()

print("Total rows:", len(df))
print("Duplicate complete rows:", duplicate_rows)
print("Duplicate feature rows:", duplicate_features)


# =========================================================
# LOAD SAVED MODEL
# =========================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]

label_encoder = bundle["label_encoder"]

print("\nSaved model:", type(model).__name__)


# =========================================================
# ENCODE TARGET
# =========================================================

y_encoded = label_encoder.transform(y)


# =========================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
# =========================================================

print("\nRunning 5-fold stratified cross-validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_val_score(
    model,
    X,
    y_encoded,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)

print("\nCross-validation scores:")

for index, score in enumerate(scores, start=1):
    print(
        f"Fold {index}: "
        f"{score * 100:.2f}%"
    )

print(
    f"\nMean CV Accuracy: "
    f"{scores.mean() * 100:.2f}%"
)

print(
    f"CV Standard Deviation: "
    f"{scores.std() * 100:.2f}%"
)


# =========================================================
# BASIC TEST SPLIT REPRODUCTION
# =========================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded,
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# =========================================================
# TEST METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred,
)

print(
    f"\nHold-out Accuracy: "
    f"{accuracy * 100:.2f}%"
)


print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
    )
)