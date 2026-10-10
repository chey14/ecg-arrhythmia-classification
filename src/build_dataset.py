
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb

from src.preprocessing import extract_heartbeat_segments

RECORDS = [
    "100", "101", "103", "105", "106",
    "108", "109", "111", "112", "113",
    "119", "200", "202", "208", "210",
]

DATA_DIR = Path("data/mit-bih")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

all_segments = []
all_labels = []
all_record_ids = []

for record_name in RECORDS:
    print(f"Processing record {record_name}...")

    record_path = DATA_DIR / record_name
    record = wfdb.rdrecord(str(record_path), physical=True)
    annotations = wfdb.rdann(str(record_path), extension="atr")

    segments, labels = extract_heartbeat_segments(
        record.p_signal,
        annotations,
        channel=0,
    )

    all_segments.append(segments.astype(np.float32))
    all_labels.append(labels)
    all_record_ids.extend([record_name] * len(labels))

# Combine the records after processing them individually.
X = np.concatenate(all_segments, axis=0)
y = np.concatenate(all_labels, axis=0)
groups = np.asarray(all_record_ids)

output_path = OUTPUT_DIR / "heartbeat_dataset.npz"

np.savez_compressed(
    output_path,
    X=X,
    y=y,
    groups=groups,
)

print("\nDataset saved successfully!")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Unique records:", len(np.unique(groups)))
print("Class counts:", dict(Counter(y.tolist())))
print("Saved to:", output_path)
