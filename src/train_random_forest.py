
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit


# --------------------------------------------------
# 1. Set paths and class labels
# --------------------------------------------------

DATA_PATH = Path("data/processed/heartbeat_dataset.npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Keep a fixed order for the report and confusion matrix.
LABELS = ["N", "S", "V", "F", "Q"]


# --------------------------------------------------
# 2. Load the processed ECG dataset
# --------------------------------------------------

with np.load(DATA_PATH) as data:
    X = data["X"]
    y = data["y"]
    groups = data["groups"]

print("Dataset loaded successfully.")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique recordings:", len(np.unique(groups)))


# --------------------------------------------------
# 3. Split by recording to reduce data leakage
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

train_records = set(groups[train_idx])
test_records = set(groups[test_idx])

# Confirm that no recording occurs in both sets.
assert train_records.isdisjoint(test_records)

print("\nRecording-wise split check passed.")
print("Training recordings:", sorted(train_records))
print("Testing recordings:", sorted(test_records))
print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))

print("\nTest-set class counts:")
for label in LABELS:
    print(f"{label}: {np.sum(y_test == label)}")


# --------------------------------------------------
# 4. Train the Random Forest baseline
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=150,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Training complete.")


# --------------------------------------------------
# 5. Predict on held-out recordings
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# 6. Evaluate the model
# --------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

# Macro F1 over all five target classes.
macro_f1_all = f1_score(
    y_test,
    predictions,
    labels=LABELS,
    average="macro",
    zero_division=0,
)

# Macro F1 over classes that actually occur in the test set.
present_labels = [
    label for label in LABELS if np.any(y_test == label)
]

macro_f1_present = f1_score(
    y_test,
    predictions,
    labels=present_labels,
    average="macro",
    zero_division=0,
)

print("\n" + "=" * 50)
print("MODEL EVALUATION")
print("=" * 50)
print(f"Test accuracy: {accuracy:.4f}")
print(f"Macro F1 (all five classes): {macro_f1_all:.4f}")
print(f"Macro F1 (classes present in test set): {macro_f1_present:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=LABELS,
        zero_division=0,
    )
)

print("Confusion matrix")
print("Rows = actual classes; columns = predicted classes")
print("Class order:", LABELS)
print(confusion_matrix(y_test, predictions, labels=LABELS))


# --------------------------------------------------
# 7. Save the trained model
# --------------------------------------------------

model_path = MODEL_DIR / "random_forest.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")
print("\nBaseline training and evaluation finished.")
