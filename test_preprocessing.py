
from collections import Counter

from src.data_loader import load_ecg_record
from src.preprocessing import extract_heartbeat_segments

# Load the ECG recording and reference annotations.
signal, sampling_frequency, annotations = load_ecg_record("100")

# Extract heartbeat windows from channel 0.
segments, labels = extract_heartbeat_segments(
    signal,
    annotations,
    channel=0,
)

print("Sampling frequency:", sampling_frequency)
print("Segments shape:", segments.shape)
print("Labels shape:", labels.shape)
print("Class counts:", Counter(labels))
print("First 10 labels:", labels[:10])

# Basic checks to catch common preprocessing mistakes.
assert segments.ndim == 2
assert segments.shape[1] == 252
assert len(segments) == len(labels)
assert len(labels) > 0

print("Preprocessing test passed!")
