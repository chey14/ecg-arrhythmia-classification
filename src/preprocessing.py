
"""
Preprocess ECG recordings into individual heartbeat segments.
"""

import numpy as np


# Map original MIT-BIH annotation symbols to five target classes.
# Symbols not listed here will be ignored.
AAMI_CLASSES = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


def extract_heartbeat_segments(
    signal,
    annotations,
    channel=0,
    samples_before=90,
    samples_after=162,
):
    """
    Extract fixed-size ECG windows around valid annotated heartbeats.

    Returns
    -------
    segments : numpy.ndarray
        ECG windows with shape (number_of_beats, 252).
    labels : numpy.ndarray
        Target class for each extracted heartbeat.
    """
    segments = []
    labels = []

    # Visit every annotation and its symbol together.
    for sample, symbol in zip(annotations.sample, annotations.symbol):

        # Skip annotations that are not part of our five target classes.
        if symbol not in AAMI_CLASSES:
            continue

        # Define the beginning and end of this heartbeat window.
        start = sample - samples_before
        end = sample + samples_after

        # Skip windows that would extend beyond the recording.
        if start < 0 or end > signal.shape[0]:
            continue

        # Select the ECG window from the requested channel.
        segment = signal[start:end, channel]

        # Keep the segment only if it has the expected length.
        expected_length = samples_before + samples_after
        if len(segment) != expected_length:
            continue

        segments.append(segment)
        labels.append(AAMI_CLASSES[symbol])

    return np.asarray(segments), np.asarray(labels)
