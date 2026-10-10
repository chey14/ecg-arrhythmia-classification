
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# All models must use the same five classes.
LABELS = ["N", "S", "V", "F", "Q"]

PREDICTION_FILES = {
    "Raw Random Forest": "data/processed/grouped_cv_predictions.npz",
    "Feature Random Forest": "data/processed/feature_rf_cv_predictions.npz",
    "CNN with focal loss": "data/processed/cnn_cv_predictions.npz",
}

results = []
reference_y = None
reference_groups = None

for model_name, file_path in PREDICTION_FILES.items():
    path = Path(file_path)

    if not path.exists():
        print(f"Skipping {model_name}: file not found at {path}")
        continue

    with np.load(path) as data:
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        groups = data["groups"]

    # Check that each file contains matching arrays.
    if not (len(y_true) == len(y_pred) == len(groups)):
        raise ValueError(f"Array lengths do not match for {model_name}")

    # Check that every model was evaluated on the same examples.
    if reference_y is None:
        reference_y = y_true.copy()
        reference_groups = groups.copy()
    else:
        if not np.array_equal(reference_y, y_true):
            raise ValueError(f"True labels differ for {model_name}")
        if not np.array_equal(reference_groups, groups):
            raise ValueError(f"Recording order differs for {model_name}")

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(
            y_true, y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "Macro Precision": precision_score(
            y_true, y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
        "Macro Recall": recall_score(
            y_true, y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0,
        ),
    })

    print(f"\n--- {model_name}: per-class report ---")
    print(classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
    ))

if not results:
    raise RuntimeError("No prediction files were found.")

comparison = pd.DataFrame(results).sort_values(
    "Macro F1", ascending=False
)

print("\n=== MODEL COMPARISON ===")
print(comparison.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.4f}".format,
        "Macro F1": "{:.4f}".format,
        "Macro Precision": "{:.4f}".format,
        "Macro Recall": "{:.4f}".format,
    },
))

output_path = Path("data/processed/model_comparison.csv")
comparison.to_csv(output_path, index=False)
print(f"\nComparison saved to: {output_path}")

