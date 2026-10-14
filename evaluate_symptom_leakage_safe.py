from pathlib import Path
from io import StringIO

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


# =========================================================
# SETTINGS
# =========================================================

DATA_PATH = Path("dataset/symptoms/Training.csv")
MODEL_PATH = Path("models/symptom_model.pkl")

TEST_SIZE = 0.20
RANDOM_STATE = 42


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


# =========================================================
# CLEAN COLUMNS
# =========================================================

df.columns = [
    str(column).strip()
    for column in df.columns
]

target_column = "prognosis"

if target_column not in df.columns:
    raise ValueError(
        "prognosis column not found."
    )


feature_columns = [
    column
    for column in df.columns
    if column != target_column
]


# =========================================================
# CHECK DUPLICATE PATTERNS
# =========================================================

feature_duplicate_count = df.duplicated(
    subset=feature_columns
).sum()

print("Total rows:", len(df))
print(
    "Duplicate symptom patterns:",
    feature_duplicate_count
)


# =========================================================
# CHECK WHETHER DUPLICATE PATTERNS HAVE
# CONFLICTING DISEASE LABELS
# =========================================================

pattern_label_counts = (
    df.groupby(feature_columns, dropna=False)[target_column]
    .nunique()
)

conflicting_patterns = (
    pattern_label_counts > 1
).sum()

print(
    "Conflicting symptom patterns:",
    conflicting_patterns
)


if conflicting_patterns > 0:

    print(
        "\nWARNING:"
        "\nSome identical symptom patterns have "
        "different disease labels."
    )

    raise ValueError(
        "Conflicting duplicate patterns found."
    )


# =========================================================
# CREATE ONE ROW PER UNIQUE SYMPTOM PATTERN
# =========================================================

unique_df = df.drop_duplicates(
    subset=feature_columns
).reset_index(drop=True)


print(
    "\nUnique symptom patterns:",
    len(unique_df)
)


# =========================================================
# ENCODE TARGET USING SAVED ENCODER
# =========================================================

bundle = joblib.load(
    MODEL_PATH
)

saved_model = bundle["model"]
label_encoder = bundle["label_encoder"]

unique_y = label_encoder.transform(
    unique_df[target_column].astype(str).str.strip()
)


# =========================================================
# SPLIT UNIQUE PATTERNS
# =========================================================

unique_train, unique_test = train_test_split(
    unique_df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=unique_y,
)


# =========================================================
# CREATE SETS OF UNIQUE PATTERN SIGNATURES
# =========================================================

def make_signature(row):
    return tuple(
        row[column]
        for column in feature_columns
    )


train_signatures = {
    make_signature(row)
    for _, row in unique_train.iterrows()
}

test_signatures = {
    make_signature(row)
    for _, row in unique_test.iterrows()
}


# =========================================================
# VERIFY NO OVERLAP
# =========================================================

overlap = train_signatures.intersection(
    test_signatures
)

print(
    "\nTrain/test pattern overlap:",
    len(overlap)
)

if len(overlap) != 0:
    raise ValueError(
        "Data leakage detected: "
        "train and test patterns overlap."
    )


# =========================================================
# BUILD FINAL TRAINING DATA
# =========================================================

df_signatures = df.apply(
    make_signature,
    axis=1
)

train_mask = df_signatures.isin(
    train_signatures
)

test_mask = df_signatures.isin(
    test_signatures
)

X_train = df.loc[
    train_mask,
    feature_columns
]

y_train = df.loc[
    train_mask,
    target_column
]

X_test = unique_test[
    feature_columns
]

y_test = unique_test[
    target_column
]


# =========================================================
# ENCODE TARGET
# =========================================================

y_train = label_encoder.transform(
    y_train.astype(str).str.strip()
)

y_test = label_encoder.transform(
    y_test.astype(str).str.strip()
)


print(
    "\nTraining rows:",
    len(X_train)
)

print(
    "Unseen test patterns:",
    len(X_test)
)


# =========================================================
# TRAIN A FRESH MODEL
# =========================================================

model = type(saved_model)(**saved_model.get_params())

print(
    "\nTraining leakage-safe "
    "Logistic Regression..."
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICT UNSEEN PATTERNS
# =========================================================

y_pred = model.predict(
    X_test
)


# =========================================================
# METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\n=========================================="
)

print(
    "LEAKAGE-SAFE RESULTS"
)

print(
    "=========================================="
)

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)