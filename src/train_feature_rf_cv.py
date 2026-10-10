
from collections import Counter
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

from src.features import extract_features


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = Path("data/processed/heartbeat_dataset.npz")
OUTPUT_PATH = Path("data/processed/feature_rf_cv_predictions.npz")

LABELS = ["N", "S", "V", "F", "Q"]
N_SPLITS = 5


# --------------------------------------------------
# 2. Load dataset and extract features
# --------------------------------------------------

with np.load(DATA_PATH) as data:
    X_raw = data["X"]
    y = data["y"]
    groups = data["groups"]

print("Extracting statistical features...")
X = extract_features(X_raw)

print("Feature matrix shape:", X.shape)
print("Heartbeat labels:", len(y))
print("Unique recordings:", len(np.unique(groups)))


# --------------------------------------------------
# 3. Grouped cross-validation
# --------------------------------------------------

splitter = GroupKFold(n_splits=N_SPLITS)

oof_predictions = np.empty_like(y)

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups=groups), start=1
):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_records = set(groups[train_idx])
    test_records = set(groups[test_idx])

    assert train_records.isdisjoint(test_records)

    print(f"\n{'=' * 45}")
    print(f"Fold {fold}/{N_SPLITS}")
    print("Training records:", sorted(train_records, key=str))
    print("Testing records:", sorted(test_records, key=str))
    print("Training labels:", dict(Counter(y_train.tolist())))

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    oof_predictions[test_idx] = predictions

    print("Fold accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Fold macro F1:", round(
        f1_score(
            y_test,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        4,
    ))


# --------------------------------------------------
# 4. Aggregate held-out predictions
# --------------------------------------------------

print(f"\n{'=' * 45}")
print("AGGREGATED CROSS-VALIDATION RESULTS")
print("=" * 45)

print("Accuracy:", round(
    accuracy_score(y, oof_predictions), 4
))
print("Macro F1 (all five classes):", round(
    f1_score(
        y,
        oof_predictions,
        labels=LABELS,
        average="macro",
        zero_division=0,
    ),
    4,
))

print("\nClassification report:")
print(classification_report(
    y,
    oof_predictions,
    labels=LABELS,
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted)")
print("Class order:", LABELS)
print(confusion_matrix(y, oof_predictions, labels=LABELS))


# --------------------------------------------------
# 5. Save predictions for comparison
# --------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_PATH,
    y_true=y,
    y_pred=oof_predictions,
    groups=groups,
)

print("\nPredictions saved to:", OUTPUT_PATH)
