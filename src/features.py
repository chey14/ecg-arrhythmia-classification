
import numpy as np


def extract_features(X):
    """
    Extract nine statistical features from each ECG heartbeat.

    Parameters
    ----------
    X : numpy.ndarray
        ECG segments with shape:
        (number_of_heartbeats, samples_per_heartbeat)

    Returns
    -------
    numpy.ndarray
        Feature matrix with nine columns per heartbeat.
    """
    # Calculate one statistical value per heartbeat.
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    minimum = np.min(X, axis=1)
    maximum = np.max(X, axis=1)
    peak_to_peak = np.ptp(X, axis=1)
    median = np.median(X, axis=1)
    percentile_25 = np.percentile(X, 25, axis=1)
    percentile_75 = np.percentile(X, 75, axis=1)
    rms = np.sqrt(np.mean(X ** 2, axis=1))

    # Combine the nine features into a single matrix.
    features = np.column_stack(
        [
            mean,
            std,
            minimum,
            maximum,
            peak_to_peak,
            median,
            percentile_25,
            percentile_75,
            rms,
        ]
    )

    return features.astype(np.float32)
