from pathlib import Path
from io import StringIO

import pandas as pd


TRAIN_PATH = Path("dataset/symptoms/Training.csv")
TEST_PATH = Path("dataset/symptoms/Testing.csv")


def load_symptom_file(file_path):
    """
    Load the symptom dataset.
    Handles both real tab characters and literal \\t characters.
    """

    # Read the raw file
    raw_text = file_path.read_text(
        encoding="utf-8-sig"
    )

    # Get first line/header
    first_line = raw_text.splitlines()[0]

    # Case 1: file contains real tab characters
    if "\t" in first_line:
        print(f"Detected TAB separator in {file_path.name}")
        return pd.read_csv(
            StringIO(raw_text),
            sep="\t"
        )

    # Case 2: file contains literal characters: \t
    if r"\t" in first_line:
        print(
            f"Detected literal TAB characters (\\t) "
            f"in {file_path.name}"
        )

        fixed_text = raw_text.replace(
            r"\t",
            "\t"
        )

        return pd.read_csv(
            StringIO(fixed_text),
            sep="\t"
        )

    # Case 3: normal comma-separated CSV
    print(f"Detected COMMA separator in {file_path.name}")

    return pd.read_csv(
        StringIO(raw_text)
    )


# =========================================================
# LOAD DATA
# =========================================================

train_df = load_symptom_file(TRAIN_PATH)
test_df = load_symptom_file(TEST_PATH)


# =========================================================
# TRAINING DATA
# =========================================================

print("\n========== TRAINING DATA ==========")

print("Shape:", train_df.shape)
print("Number of columns:", len(train_df.columns))

print("\nFirst 5 rows:")
print(train_df.head())


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

train_df.columns = [
    str(column).strip()
    for column in train_df.columns
]

test_df.columns = [
    str(column).strip()
    for column in test_df.columns
]


# =========================================================
# SYMPTOM FEATURES
# =========================================================

if "prognosis" not in train_df.columns:

    print("\nERROR: 'prognosis' column was not found.")

    print("\nColumns detected:")
    for column in train_df.columns:
        print(repr(column))

    raise ValueError(
        "The dataset still could not be parsed correctly."
    )


symptom_columns = [
    column
    for column in train_df.columns
    if column != "prognosis"
]


print(
    "\nNumber of symptom features:",
    len(symptom_columns)
)

print(
    "Number of diseases:",
    train_df["prognosis"].nunique()
)


# =========================================================
# DISEASE DISTRIBUTION
# =========================================================

print("\nDisease distribution:")

print(
    train_df["prognosis"].value_counts()
)


# =========================================================
# MISSING VALUES
# =========================================================

print("\nTotal missing values:")

print(
    train_df.isnull().sum().sum()
)


# =========================================================
# TESTING DATA
# =========================================================

print("\n========== TESTING DATA ==========")

print("Shape:", test_df.shape)

print(
    "Number of columns:",
    len(test_df.columns)
)

if "prognosis" in test_df.columns:

    print(
        "Number of diseases:",
        test_df["prognosis"].nunique()
    )

else:

    print(
        "WARNING: prognosis column not found "
        "in Testing.csv"
    )


# =========================================================
# DUPLICATE COLUMN CHECK
# =========================================================

duplicate_columns = (
    train_df.columns[
        train_df.columns.duplicated()
    ]
    .tolist()
)

print("\nDuplicate column names:")

if duplicate_columns:
    print(duplicate_columns)
else:
    print("None")


print("\nDataset inspection completed successfully.")