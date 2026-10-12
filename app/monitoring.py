

from __future__ import annotations

import os
import tempfile
from math import gcd
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import butter, sosfiltfilt, find_peaks, resample_poly


WINDOW_LEFT = 90
WINDOW_RIGHT = 162
WINDOW_SIZE = WINDOW_LEFT + WINDOW_RIGHT  # 252 samples
TARGET_FS = 360
CLASSES = ["F", "N", "Q", "S", "V"]


# --------------------------------------------------
# Input readers
# --------------------------------------------------

def _read_csv(uploaded_file, signal_column: str) -> np.ndarray:
    df = pd.read_csv(uploaded_file)

    if signal_column not in df.columns:
        raise ValueError(f"Column {signal_column!r} was not found.")

    signal = pd.to_numeric(
        df[signal_column], errors="coerce"
    ).to_numpy(dtype=float)

    signal = signal[np.isfinite(signal)]

    if len(signal) < 3:
        raise ValueError(
            "The selected CSV column does not contain enough numeric samples."
        )

    return signal


def _read_wfdb(uploaded_files) -> tuple[np.ndarray, int, str]:
    """Read a WFDB record from an uploaded .hea + .dat pair."""
    try:
        import wfdb
    except ImportError as exc:
        raise ValueError(
            "WFDB support requires the 'wfdb' package. "
            "Install it with pip install wfdb."
        ) from exc

    file_map = {Path(f.name).name: f for f in uploaded_files}
    hea_names = [
        name for name in file_map if name.lower().endswith(".hea")
    ]

    if not hea_names:
        raise ValueError(
            "Upload the WFDB header (.hea) and matching signal file (.dat) together."
        )

    hea_name = hea_names[0]
    stem = Path(hea_name).stem

    if f"{stem}.dat" not in file_map:
        raise ValueError(
            f"Missing matching signal file: {stem}.dat"
        )

    with tempfile.TemporaryDirectory() as tmp:
        for name, uploaded_file in file_map.items():
            with open(os.path.join(tmp, name), "wb") as out:
                out.write(uploaded_file.getvalue())

        record = wfdb.rdrecord(os.path.join(tmp, stem))

        if record.p_signal is None:
            raise ValueError(
                "Could not read physical signal values from this WFDB record."
            )

        signal = np.asarray(record.p_signal, dtype=float)

        if signal.ndim == 2:
            signal = signal[:, 0]

        fs = int(round(float(record.fs)))
        lead = record.sig_name[0] if record.sig_name else "first lead"

        return signal, fs, lead


def _read_builtin_demo(
    demo_folder: str,
    record_name: str,
) -> tuple[np.ndarray, int, str]:
    """Read a bundled WFDB demo from the repository."""
    try:
        import wfdb
    except ImportError as exc:
        raise ValueError(
            "WFDB support requires the 'wfdb' package in requirements.txt."
        ) from exc

    # app/monitoring.py -> repository root
    root_dir = Path(__file__).resolve().parents[1]
    record_path = root_dir / demo_folder / record_name

    header_path = record_path.with_suffix(".hea")
    signal_path = record_path.with_suffix(".dat")

    if not header_path.is_file() or not signal_path.is_file():
        raise FileNotFoundError(
            "The built-in recording files are missing. Expected both "
            f"{header_path.relative_to(root_dir)} and "
            f"{signal_path.relative_to(root_dir)} in the repository."
        )

    record = wfdb.rdrecord(str(record_path))

    if record.p_signal is None:
        raise ValueError(
            f"Could not read signal data from {record_name}."
        )

    signal = np.asarray(record.p_signal, dtype=float)

    if signal.ndim == 2:
        signal = signal[:, 0]

    fs = int(round(float(record.fs)))
    lead = record.sig_name[0] if record.sig_name else "ECG"

    return signal, fs, lead


# --------------------------------------------------
# Candidate beat detection
# --------------------------------------------------

def _detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """Simple QRS candidate detector; not clinically validated."""
    x = np.asarray(signal, dtype=float)

    if fs < 50:
        raise ValueError(
            "Sampling rate is too low for this demo detector. Use at least 50 Hz."
        )

    if len(x) < int(fs * 2):
        raise ValueError(
            "Signal is too short; upload at least 2 seconds of ECG."
        )

    x = x - np.nanmedian(x)

    # QRS-focused bandpass for candidate detection only.
    low = 5.0
    high = min(18.0, fs * 0.45)

    if high <= low:
        raise ValueError(
            "Sampling rate is incompatible with the detector bandpass."
        )

    sos = butter(
        2,
        [low, high],
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    filtered = sosfiltfilt(sos, x)

    energy = np.abs(filtered)
    smooth_n = max(1, int(0.08 * fs))
    energy = np.convolve(
        energy,
        np.ones(smooth_n) / smooth_n,
        mode="same",
    )

    distance = max(1, int(0.25 * fs))
    prominence = max(float(np.std(energy) * 0.5), 1e-8)

    peaks, _ = find_peaks(
        energy,
        distance=distance,
        prominence=prominence,
    )

    # Align candidate peaks to nearby absolute filtered deflections.
    radius = max(1, int(0.08 * fs))
    aligned = []

    for peak in peaks:
        start = max(0, peak - radius)
        end = min(len(filtered), peak + radius + 1)

        if end > start:
            aligned.append(
                start + int(np.argmax(np.abs(filtered[start:end])))
            )

    if not aligned:
        return np.array([], dtype=int)

    aligned = np.array(sorted(set(aligned)), dtype=int)

    # De-duplicate detections closer than 200 ms.
    kept = []

    for peak in aligned:
        if not kept or peak - kept[-1] >= int(0.20 * fs):
            kept.append(int(peak))
        elif abs(filtered[peak]) > abs(filtered[kept[-1]]):
            kept[-1] = int(peak)

    return np.asarray(kept, dtype=int)


# --------------------------------------------------
# Beat classification
# --------------------------------------------------

def classify_uploaded_signal(
    signal: np.ndarray,
    fs: int,
    model,
    left: int = WINDOW_LEFT,
    right: int = WINDOW_RIGHT,
) -> pd.DataFrame:
    """Detect candidate beats and classify 252-sample windows at 360 Hz."""
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        raise ValueError("No finite signal samples found.")

    if fs != TARGET_FS:
        factor = gcd(int(fs), TARGET_FS)
        x = resample_poly(
            x,
            TARGET_FS // factor,
            int(fs) // factor,
        )
        fs = TARGET_FS

    peaks = _detect_r_peaks(x, fs)

    rows = []
    windows = []

    for peak in peaks:
        start = peak - left
        end = peak + right

        if start < 0 or end > len(x):
            continue

        segment = x[start:end]

        if len(segment) != WINDOW_SIZE:
            continue

        windows.append(segment.astype(np.float32))
        rows.append(
            {
                "sample_index": int(peak),
                "time_seconds": float(peak / fs),
            }
        )

    if not windows:
        raise ValueError(
            "No complete beat windows were detected. "
            "Try a longer or cleaner ECG signal."
        )

    X = np.stack(windows)
    predictions = model.predict(X)

    try:
        probabilities = model.predict_proba(X)
        classes = list(model.classes_)
    except Exception:
        probabilities = None
        classes = []

    results = pd.DataFrame(rows)
    results["prediction"] = predictions.astype(str)

    if probabilities is not None:
        results["confidence"] = [
            float(probabilities[i, classes.index(predictions[i])])
            for i in range(len(predictions))
        ]
    else:
        results["confidence"] = np.nan

    return results


# --------------------------------------------------
# Streamlit monitoring interface
# --------------------------------------------------

def render_ecg_monitoring(
    model_path: str = "models/random_forest.joblib",
) -> None:
    st.subheader("ECG Monitoring — Random Forest")

    

    

    root_dir = Path(__file__).resolve().parents[1]
    resolved_model_path = Path(model_path)

    if not resolved_model_path.is_absolute():
        resolved_model_path = root_dir / resolved_model_path

    if not resolved_model_path.is_file():
        st.error(
            f"Random Forest model not found at {resolved_model_path}. "
            "Ensure models/random_forest.joblib is committed to GitHub."
        )
        return

    try:
        model = joblib.load(resolved_model_path)
    except Exception as exc:
        st.error(f"Could not load the Random Forest model: {exc}")
        return

    st.markdown("### Choose an ECG recording")

    source = st.radio(
        "Recording source",
        [
            "Built-in demo: mixed ECG",
            "Built-in demo: repeated V beats",
            "Upload my own ECG",
        ],
        key="rf_ecg_source",
    )

    # Clear stale results when the user switches recording sources.
    previous_source = st.session_state.get("rf_ecg_previous_source")

    if previous_source != source:
        st.session_state.pop("rf_ecg_monitor_results", None)
        st.session_state.pop("rf_ecg_monitor_signal", None)
        st.session_state.pop("rf_ecg_monitor_fs", None)
        st.session_state.pop("rf_ecg_monitor_lead", None)
        st.session_state["rf_ecg_previous_source"] = source

    signal = None
    fs = TARGET_FS
    lead_name = "ECG"

    try:
        if source == "Built-in demo: mixed ECG":
            signal, fs, lead_name = _read_builtin_demo(
                "synthetic_mixed_demo",
                "synthetic_mixed_test",
            )
            st.info(
                "Loaded the bundled synthetic mixed recording. "
                "It is intended for demonstrating the upload-monitor workflow."
            )

        elif source == "Built-in demo: repeated V beats":
            signal, fs, lead_name = _read_builtin_demo(
                "synthetic_v_demo",
                "synthetic_v_test",
            )
            st.info(
                "Loaded the bundled synthetic V-beat recording. "
                "Predictions and alerts depend on the actual model output."
            )

        else:
            uploaded = st.file_uploader(
                "Upload an ECG CSV, or select both WFDB .hea and .dat files",
                type=["csv", "hea", "dat"],
                accept_multiple_files=True,
                key="rf_ecg_monitor_upload",
            )

            if not uploaded:
                st.info(
                    "Choose a built-in sample above, or upload your own ECG. "
                    "For WFDB, select the matching .hea and .dat files together."
                )
                return

            names = [f.name.lower() for f in uploaded]
            csv_files = [
                f for f in uploaded
                if f.name.lower().endswith(".csv")
            ]

            if csv_files:
                df = pd.read_csv(csv_files[0])

                numeric_columns = [
                    column
                    for column in df.columns
                    if pd.to_numeric(
                        df[column], errors="coerce"
                    ).notna().sum() > 0
                ]

                if not numeric_columns:
                    st.error("No numeric signal columns found in the CSV.")
                    return

                signal_column = st.selectbox(
                    "ECG signal column",
                    numeric_columns,
                )

                fs = int(
                    st.number_input(
                        "CSV sampling rate (Hz)",
                        min_value=50,
                        max_value=5000,
                        value=360,
                        step=1,
                    )
                )

                signal = pd.to_numeric(
                    df[signal_column],
                    errors="coerce",
                ).to_numpy(dtype=float)

                lead_name = signal_column

            elif any(name.endswith(".hea") for name in names):
                signal, fs, lead_name = _read_wfdb(uploaded)

                st.write(
                    f"Detected WFDB sampling rate: **{fs} Hz** · "
                    f"Lead: **{lead_name}**"
                )

            else:
                st.error(
                    "Unsupported upload. Provide a CSV or a matching "
                    "WFDB .hea + .dat pair."
                )
                return

        signal = np.asarray(signal, dtype=float)
        signal = signal[np.isfinite(signal)]

        if signal.size < 2:
            st.error("No usable numeric ECG samples found.")
            return

        duration = len(signal) / fs

        st.write(
            f"Signal duration: **{duration:.1f} seconds** · "
            f"Samples: **{len(signal):,}**"
        )

        if st.button(
            "Detect beats and classify",
            type="primary",
            key="rf_detect_classify_button",
        ):
            with st.spinner(
                "Detecting candidate beats and classifying segments..."
            ):
                # Convert once for both classification and plotting.
                if fs != TARGET_FS:
                    factor = gcd(int(fs), TARGET_FS)
                    signal_360 = resample_poly(
                        signal,
                        TARGET_FS // factor,
                        int(fs) // factor,
                    )
                else:
                    signal_360 = signal

                results = classify_uploaded_signal(
                    signal_360,
                    TARGET_FS,
                    model,
                )

            st.session_state["rf_ecg_monitor_results"] = results
            st.session_state["rf_ecg_monitor_signal"] = signal_360
            st.session_state["rf_ecg_monitor_fs"] = TARGET_FS
            st.session_state["rf_ecg_monitor_lead"] = lead_name

        results = st.session_state.get("rf_ecg_monitor_results")

        if results is None or results.empty:
            return

        signal_for_plot = st.session_state["rf_ecg_monitor_signal"]
        fs_plot = st.session_state["rf_ecg_monitor_fs"]
        plot_lead = st.session_state["rf_ecg_monitor_lead"]

        st.success(
            f"Classified {len(results)} candidate heartbeat segments."
        )

        # ECG waveform and detected candidate locations.
        fig, ax = plt.subplots(figsize=(12, 3.5))

        time_axis = np.arange(len(signal_for_plot)) / fs_plot

        ax.plot(
            time_axis,
            signal_for_plot,
            linewidth=0.7,
            label=plot_lead,
        )

        beat_indices = results["sample_index"].to_numpy(dtype=int)

        ax.scatter(
            results["time_seconds"],
            signal_for_plot[beat_indices],
            s=20,
            marker="x",
            label="Detected candidate beats",
        )

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Signal amplitude (input units)")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        # Prediction counts and per-beat results.
        st.markdown("### Predicted beat labels")

        counts = results["prediction"].value_counts()

        st.dataframe(
            counts.rename_axis("Class").reset_index(name="Count"),
            use_container_width=True,
            hide_index=True,
        )

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True,
        )

        # Repeated-V demonstration rule.
        st.markdown("### Repeated-V demo alert")

        col1, col2 = st.columns(2)

        with col1:
            window_n = int(
                st.number_input(
                    "Rolling window (beats)",
                    min_value=3,
                    max_value=50,
                    value=10,
                    key="rf_window_n",
                )
            )

        with col2:
            min_v = int(
                st.number_input(
                    "V predictions needed to show demo alert",
                    min_value=2,
                    max_value=50,
                    value=4,
                    key="rf_min_v",
                )
            )

        if min_v > window_n:
            st.error(
                "V threshold must be less than or equal to the rolling window."
            )
            return

        recent = results.tail(window_n)
        v_count = int((recent["prediction"] == "V").sum())

        st.write(
            f"Last {len(recent)} classified beats: "
            f"**{v_count} predicted V labels**"
        )

        if len(recent) >= window_n and v_count >= min_v:
            st.error(
                f"DEMO ALERT: {v_count} of the last {window_n} beats "
                "were classified as V."
            )
        else:
            st.info(
                "No demo alert under the selected rule."
            )

        st.download_button(
            "Download predictions as CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="ecg_rf_predictions_demo.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"Could not process this ECG: {exc}")
        st.caption(
            "Check the recording files, file format, sampling rate, units, "
            "and whether the signal is long and clean enough."
        )
