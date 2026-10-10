
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Load the dataset we just created.
dataset_path = Path("data/processed/heartbeat_dataset.npz")

with np.load(dataset_path) as data:
    X = data["X"]
    y = data["y"]
    groups = data["groups"]

# Split by recording, not by individual heartbeat.
# This prevents beats from the same recording appearing in
# both the training and test sets.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

train_records = sorted(set(groups[train_indices]))
test_records = sorted(set(groups[test_indices]))

print("Training records:", train_records)
print("Testing records:", test_records)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining class counts:")
print(dict(Counter(y_train.tolist())))

print("\nTesting class counts:")
print(dict(Counter(y_test.tolist())))

# Confirm no recording appears in both sets.
assert set(train_records).isdisjoint(set(test_records))
print("\nRecording-wise split check passed!")
