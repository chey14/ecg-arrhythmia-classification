
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.focal_loss import FocalLoss


DATA_PATH = Path("data/processed/heartbeat_dataset.npz")
MODEL_PATH = Path("models/ecg_cnn_final.pth")

LABELS = ["N", "S", "V", "F", "Q"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

EPOCHS = 8
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Keep this architecture identical to the cross-validation model.
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
        return self.classifier(self.features(x))


def main():
    with np.load(DATA_PATH) as data:
        X = data["X"].astype(np.float32)
        y_text = data["y"]

    y = np.array(
        [LABEL_TO_ID[label] for label in y_text],
        dtype=np.int64,
    )

    # Apply the same per-heartbeat normalization used during evaluation.
    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)
    X = (X - means) / np.maximum(stds, 1e-6)
    X = X[:, np.newaxis, :]

    # Calculate weights from the full training dataset.
    counts = np.bincount(y, minlength=len(LABELS))
    weights = np.zeros(len(LABELS), dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()

    class_weights = torch.tensor(weights, device=DEVICE)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = ECGCNN(num_classes=len(LABELS)).to(DEVICE)
    criterion = FocalLoss(
        class_weights=class_weights,
        gamma=2.0,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("Training device:", DEVICE)
    print("Training on all available beats:", len(y))
    print("Class counts:", dict(zip(LABELS, counts.tolist())))

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

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- loss: {total_loss / total_samples:.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), MODEL_PATH)
    print("Final demo model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()

