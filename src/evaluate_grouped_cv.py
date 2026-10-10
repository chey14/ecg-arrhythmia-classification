
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

DATA_PATH = Path("data/processed/heartbeat_dataset.npz")
LABELS = ["N", "S", "V", "F", "Q"]
N_SPLITS = 5

with np.load(DATA_PATH) as data:
    X = data["X"]
    y = data["y"]
    groups = data["groups"]

unique_groups = np.unique(groups)

if len(unique_groups) < N_SPLITS:
    raise ValueError("Not enough recordings for 5-fold GroupKFold.")

# Each recording is held out in exactly one fold.
splitter = GroupKFold(n_splits=N_SPLITS)

# Store one out-of-fold prediction for each heartbeat.
oof_predictions = np.empty(y.shape, dtype=y.dtype)
fold_number = np.zeros(len(y), dtype=int)

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups=groups), start=1
):
    train_records = set(groups[train_idx])
    test_records = set(groups[test_idx])

    assert train_records.isdisjoint(test_records)

    y_train = y[train_idx]
    y_test = y[test_idx]

    print(f"\n{'=' * 50}")
    print(f"FOLD {fold}/{N_SPLITS}")
    print("Training recordings:", sorted(train_records, key=str))
    print("Testing recordings:", sorted(test_records, key=str))
    print("Training class counts:", dict(Counter(y_train.tolist())))
    print("Testing class counts:", dict(Counter(y_test.tolist())))

    # A class absent from training cannot be learned in this fold.
    missing_classes = [
        label for label in LABELS if label not in set(y_train)
    ]
    if missing_classes:
        print("WARNING: classes absent from training:", missing_classes)

    model = RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X[train_idx], y_train)
    predictions = model.predict(X[test_idx])

    oof_predictions[test_idx] = predictions
    fold_number[test_idx] = fold

    print("Fold accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Fold macro F1 (all classes):", round(
        f1_score(
            y_test,
            predictions,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        4,
    ))

# Aggregate all held-out predictions.
print(f"\n{'=' * 50}")
print("AGGREGATED OUT-OF-FOLD EVALUATION")
print("=" * 50)

print("Overall accuracy:", round(
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

# Save predictions for later error analysis.
output_path = Path("data/processed/grouped_cv_predictions.npz")
np.savez_compressed(
    output_path,
    y_true=y,
    y_pred=oof_predictions,
    groups=groups,
    fold=fold_number,
)

print("\nOut-of-fold predictions saved to:", output_path)
