
import numpy as np


def extract_features(X):
    """
    Extract simple statistical features from ECG heartbeat segments.

    Parameters
    ----------
    X : numpy.ndarray
        Shape: (number_of_beats, samples_per_beat)

    Returns
    -------
    numpy.ndarray
        One feature vector per heartbeat.
    """
    features = []

    for beat in X:
        beat_features = [
            np.mean(beat),
            np.std(beat),
            np.min(beat),
            np.max(beat),
            np.ptp(beat),
            np.median(beat),
            np.percentile(beat, 25),
            np.percentile(beat, 75),
            np.sqrt(np.mean(beat ** 2)),
        ]
        features.append(beat_features)

    return np.asarray(features, dtype=np.float32)
