
from collections import Counter
from pathlib import Path
from src.focal_loss import FocalLoss
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_PATH = Path("data/processed/heartbeat_dataset.npz")
OUTPUT_PATH = Path("data/processed/cnn_cv_predictions.npz")

LABELS = ["N", "S", "V", "F", "Q"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

N_SPLITS = 5
EPOCHS = 8
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# Prefer Apple's Metal GPU when available.
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Training device:", DEVICE)


# --------------------------------------------------
# 2. Small 1D CNN
# --------------------------------------------------

class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

with np.load(DATA_PATH) as data:
    X = data["X"].astype(np.float32)
    y_text = data["y"]
    groups = data["groups"]

y = np.array([LABEL_TO_ID[label] for label in y_text], dtype=np.int64)

# Normalize each heartbeat independently.
# This helps numerical stability but removes absolute amplitude information.
means = X.mean(axis=1, keepdims=True)
stds = X.std(axis=1, keepdims=True)
X = (X - means) / np.maximum(stds, 1e-6)

# Conv1d expects (samples, channels, sequence length).
X = X[:, np.newaxis, :]

print("ECG tensor shape:", X.shape)
print("Total beats:", len(y))
print("Recordings:", len(np.unique(groups)))


# --------------------------------------------------
# 4. Grouped cross-validation
# --------------------------------------------------

splitter = GroupKFold(n_splits=N_SPLITS)
oof_predictions = np.full(len(y), -1, dtype=np.int64)

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups=groups), start=1
):
    print(f"\n{'=' * 50}")
    print(f"FOLD {fold}/{N_SPLITS}")

    train_records = set(groups[train_idx])
    test_records = set(groups[test_idx])
    assert train_records.isdisjoint(test_records)

    y_train = y[train_idx]
    y_test = y[test_idx]

    print("Training records:", sorted(train_records, key=str))
    print("Testing records:", sorted(test_records, key=str))
    print(
        "Training classes:",
        dict(Counter(y_text[train_idx].tolist())),
    )

    # Calculate class weights using training data only.
    counts = np.bincount(y_train, minlength=len(LABELS))
    weights = np.zeros(len(LABELS), dtype=np.float32)
    present = counts > 0

    # Inverse-square-root weighting is less extreme than inverse frequency.
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()

    class_weights = torch.tensor(weights, device=DEVICE)

    model = ECGCNN(num_classes=len(LABELS)).to(DEVICE)

    criterion = FocalLoss(
    class_weights=class_weights,
    gamma=2.0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    X_train_tensor = torch.from_numpy(X[train_idx])
    y_train_tensor = torch.from_numpy(y_train)

    loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    # Train only on the training recordings.
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total_samples += len(batch_y)

        average_loss = total_loss / total_samples
        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- loss: {average_loss:.4f}"
        )

    # Evaluate on held-out recordings.
    model.eval()
    test_tensor = torch.from_numpy(X[test_idx])
    batch_predictions = []

    with torch.no_grad():
        for start in range(0, len(test_tensor), BATCH_SIZE):
            batch = test_tensor[start:start + BATCH_SIZE].to(DEVICE)
            logits = model(batch)
            batch_predictions.extend(
                logits.argmax(dim=1).cpu().numpy().tolist()
            )

    predictions = np.asarray(batch_predictions, dtype=np.int64)
    oof_predictions[test_idx] = predictions

    print("Fold accuracy:", round(
        accuracy_score(y_test, predictions), 4
    ))
    print("Fold macro F1:", round(
        f1_score(
            y_test,
            predictions,
            labels=list(range(len(LABELS))),
            average="macro",
            zero_division=0,
        ),
        4,
    ))

    # Save the last fold's model for inspection only.
    # This model was not trained on the last fold's held-out recordings.
    if fold == N_SPLITS:
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), "models/ecg_cnn_last_fold.pth")


# --------------------------------------------------
# 5. Aggregate out-of-fold evaluation
# --------------------------------------------------

assert np.all(oof_predictions >= 0)

print(f"\n{'=' * 50}")
print("AGGREGATED CNN RESULTS")
print("=" * 50)

print("Accuracy:", round(
    accuracy_score(y, oof_predictions), 4
))
print("Macro F1:", round(
    f1_score(
        y,
        oof_predictions,
        labels=list(range(len(LABELS))),
        average="macro",
        zero_division=0,
    ),
    4,
))

print("\nClassification report:")
print(classification_report(
    y,
    oof_predictions,
    labels=list(range(len(LABELS))),
    target_names=LABELS,
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted)")
print("Class order:", LABELS)
print(confusion_matrix(
    y,
    oof_predictions,
    labels=list(range(len(LABELS))),
))

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUTPUT_PATH,
    y_true=y_text,
    y_pred=np.array([LABELS[i] for i in oof_predictions]),
    groups=groups,
)

print("\nPredictions saved to:", OUTPUT_PATH)
