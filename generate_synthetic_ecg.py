import numpy as np
import wfdb
from pathlib import Path

fs = 360
duration = 20
t = np.arange(fs * duration) / fs
rng = np.random.default_rng(42)

# Synthetic baseline signal — not real patient data.
signal = 0.015 * rng.normal(size=len(t))

# Generate simple heartbeat-like pulses.
beat_times = np.arange(0.8, duration - 0.5, 0.8)

for i, beat_time in enumerate(beat_times):
    center = int(beat_time * fs)

    # Vary pulse shape to create an illustrative waveform.
    width = 0.035 if i % 4 else 0.065
    amplitude = 0.8 if i % 4 else 1.1
    idx = np.arange(max(0, center - 30),
                    min(len(signal), center + 31))
    x = (idx - center) / fs
    signal[idx] += amplitude * np.exp(
        -(x ** 2) / (2 * width ** 2)
    )

out = Path("synthetic_ecg")
out.mkdir(exist_ok=True)

wfdb.wrsamp(
    record_name="synthetic_alert_demo",
    fs=fs,
    units=["mV"],
    sig_name=["ECG"],
    p_signal=signal.reshape(-1, 1),
    fmt=["16"],
    write_dir=str(out),
)

print("Created synthetic_ecg/synthetic_alert_demo.hea")
print("Created synthetic_ecg/synthetic_alert_demo.dat")
print("SYNTHETIC TEST SIGNAL — NOT REAL PATIENT DATA")
