
from src.data_loader import load_ecg_record


# Load record 100 from the MIT-BIH database.
signal, sampling_frequency, annotations = load_ecg_record("100")

# Inspect the basic properties of the recording.
print("Signal shape:", signal.shape)
print("Sampling frequency:", sampling_frequency)
print("Number of annotated beats:", len(annotations.sample))

# Look at the first few reference annotations.
print("First 10 beat positions:", annotations.sample[:10])
print("First 10 beat symbols:", annotations.symbol[:10])
