
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_ecg_record

# Load ECG record 100 and its reference annotations.
signal, sampling_frequency, annotations = load_ecg_record("100")

# Display the first 10 seconds of the first ECG channel.
duration_seconds = 10
number_of_samples = int(duration_seconds * sampling_frequency)

ecg = signal[:number_of_samples, 0]

# Convert sample positions into time in seconds.
time = np.arange(number_of_samples) / sampling_frequency

# Keep annotations that fall within these first 10 seconds.
mask = annotations.sample < number_of_samples
beat_positions = annotations.sample[mask]
beat_symbols = np.array(annotations.symbol)[mask]

# Plot the waveform.
plt.figure(figsize=(14, 5))
plt.plot(time, ecg, linewidth=0.8, label="ECG signal")

# Mark the reference annotations.
for position, symbol in zip(beat_positions, beat_symbols):
    plt.axvline(position / sampling_frequency, alpha=0.2, linewidth=0.7)

plt.title("MIT-BIH ECG Record 100 — First 10 Seconds")
plt.xlabel("Time (seconds)")
plt.ylabel("ECG amplitude (mV)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ecg_plot.png", dpi=150, bbox_inches="tight")
print("ECG plot saved to ecg_plot.png")
plt.show(block=True)

print("Annotations in the first 10 seconds:")
print(list(zip(beat_symbols, beat_positions)))
