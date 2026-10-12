
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from torch import nn


# Paths are relative to the project root.
DATA_PATH = Path("app/demo_heartbeats.npz")
MODEL_PATH = Path("models/ecg_cnn_final.pth")

LABELS = ["N", "S", "V", "F", "Q"]

CLASS_DESCRIPTIONS = {
    "N": "Normal and related beats",
    "S": "Supraventricular ectopic beats",
    "V": "Ventricular ectopic beats",
    "F": "Fusion beats",
    "Q": "Unknown or paced-related category",
}


# Keep this architecture identical to training.
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


@st.cache_data
def load_dataset():
    with np.load(DATA_PATH) as data:
        return (
            data["X"].astype(np.float32),
            data["y"].astype(str),
            data["groups"].astype(str),
        )


@st.cache_resource
def load_model():
    model = ECGCNN(num_classes=len(LABELS))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


st.set_page_config(
    page_title="ECG Beat Classification",
    page_icon="💓",
    layout="wide",
)

st.title("ECG Heartbeat Classification")




if not DATA_PATH.exists() or not MODEL_PATH.exists():
    st.error(
        "Required files are missing. Build the dataset and train the "
        "final CNN model before launching this app."
)
    st.stop()

X, y_true, groups = load_dataset()
model = load_model()

recordings = sorted(np.unique(groups).tolist())
selected_record = st.selectbox("Select recording", recordings)

indices = np.flatnonzero(groups == selected_record)
selected_index = st.selectbox(
    "Select heartbeat within this recording",
    indices.tolist(),
    format_func=lambda i: f"Heartbeat {i}",
)

signal = X[selected_index]

# Plot the original heartbeat segment.
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(np.arange(len(signal)) / 360.0, signal, linewidth=1)
ax.set_xlabel("Time within segment (seconds)")
ax.set_ylabel("ECG amplitude (stored units)")
ax.set_title(f"Recording {selected_record}, heartbeat index {selected_index}")
ax.grid(True, alpha=0.3)
st.pyplot(fig)
plt.close(fig)

if st.button("Classify heartbeat", type="primary"):
    # Match the per-heartbeat normalization used in training.
    mean = signal.mean()
    std = signal.std()
    normalized = (signal - mean) / max(float(std), 1e-6)

    tensor = torch.from_numpy(
        normalized.astype(np.float32)[None, None, :]
    )

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].numpy()

    predicted_index = int(np.argmax(probabilities))
    predicted_label = LABELS[predicted_index]

    st.subheader("Model output")
    st.metric("Predicted class", predicted_label)
    st.write(CLASS_DESCRIPTIONS[predicted_label])
    st.write(f"Reference annotation: **{y_true[selected_index]}**")

    

    st.subheader("Model scores")
    st.bar_chart(
        {
            "Score": {
                label: float(probabilities[i])
                for i, label in enumerate(LABELS)
            }
        }
    )

