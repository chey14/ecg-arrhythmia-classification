from __future__ import annotations

import io

import os

import tempfile

from pathlib import Path

from typing import Optional

import numpy as np

import pandas as pd

import streamlit as st

import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks

import joblib

WINDOW_LEFT = 90

WINDOW_RIGHT = 162

WINDOW_SIZE = WINDOW_LEFT + WINDOW_RIGHT  # 252 samples

CLASSES = ["F", "N", "Q", "S", "V"]  # labels used by the project's saved RF

def _read_csv(uploaded_file, signal_column: str) -> np.ndarray:

    df = pd.read_csv(uploaded_file)

    if signal_column not in df.columns:

        raise ValueError(f"Column {signal_column!r} was not found.")

    signal = pd.to_numeric(df[signal_column], errors="coerce").to_numpy(dtype=float)

    signal = signal[np.isfinite(signal)]

    if len(signal) < 3:

        raise ValueError("The selected CSV column does not contain enough numeric samples.")

    return signal

def _read_wfdb(uploaded_files) -> tuple[np.ndarray, int, str]:

    """Read a WFDB record from a .hea + .dat pair uploaded together."""

    try:

        import wfdb

    except ImportError as exc:

        raise ValueError("WFDB support requires the 'wfdb' package. Install it with pip install wfdb.") from exc

    file_map = {Path(f.name).name: f for f in uploaded_files}

    hea_names = [n for n in file_map if n.lower().endswith(".hea")]

    if not hea_names:

        raise ValueError("Upload the WFDB header (.hea) and matching signal file (.dat) together.")

    hea_name = hea_names[0]

    stem = Path(hea_name).stem

    with tempfile.TemporaryDirectory() as tmp:

        for name, f in file_map.items():

            with open(os.path.join(tmp, name), "wb") as out:

                out.write(f.getvalue())

        record = wfdb.rdrecord(os.path.join(tmp, stem))

        if record.p_signal is None:

            raise ValueError("Could not read physical signal values from this WFDB record.")

        sig = np.asarray(record.p_signal, dtype=float)

        if sig.ndim == 2:

            sig = sig[:, 0]  # default to the first lead; user can use CSV for explicit lead selection

        fs = int(round(float(record.fs)))

        lead = record.sig_name[0] if record.sig_name else "first lead"

        return sig, fs, lead

def _detect_r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:

    """Simple QRS candidate detector; not a clinically validated beat detector."""

    x = np.asarray(signal, dtype=float)

    if fs < 50:

        raise ValueError("Sampling rate is too low for this demo detector (use at least 50 Hz).")

    if len(x) < int(fs * 2):

        raise ValueError("Signal is too short; upload at least 2 seconds of ECG.")

    x = x - np.nanmedian(x)

    # A QRS-focused bandpass used for detection only. Classification uses original samples.

    low = max(1.0, 5.0)

    high = min(18.0, fs * 0.45)

    if high <= low:

        raise ValueError("Sampling rate is incompatible with the detector bandpass.")

    sos = butter(2, [low, high], btype="bandpass", fs=fs, output="sos")

    filtered = sosfiltfilt(sos, x)

    # Absolute filtered energy makes the detector less sensitive to QRS polarity.

    energy = np.abs(filtered)

    smooth_n = max(1, int(0.08 * fs))

    energy = np.convolve(energy, np.ones(smooth_n) / smooth_n, mode="same")

    distance = max(1, int(0.25 * fs))

    prominence = max(float(np.std(energy) * 0.5), 1e-8)

    peaks, _ = find_peaks(energy, distance=distance, prominence=prominence)

    # Align each candidate to the strongest absolute filtered deflection nearby.

    radius = max(1, int(0.08 * fs))

    aligned = []

    for p in peaks:

        a, b = max(0, p-radius), min(len(filtered), p+radius+1)

        if b > a:

            aligned.append(a + int(np.argmax(np.abs(filtered[a:b]))))

    if not aligned:

        return np.array([], dtype=int)

    aligned = np.array(sorted(set(aligned)), dtype=int)

    # De-duplicate alignments closer than 200 ms.

    kept = []

    for p in aligned:

        if not kept or p - kept[-1] >= int(0.20 * fs):

            kept.append(int(p))

        elif abs(filtered[p]) > abs(filtered[kept[-1]]):

            kept[-1] = int(p)

    return np.asarray(kept, dtype=int)

def classify_uploaded_signal(

    signal: np.ndarray,

    fs: int,

    model,

    left: int = WINDOW_LEFT,

    right: int = WINDOW_RIGHT,

) -> pd.DataFrame:

    """Detect beats, extract 252-point windows at 360 Hz, and predict with the RF.

    The model was trained on 252 raw samples at 360 Hz. Resampling is therefore

    required when the uploaded signal is sampled at a different rate.

    """

    from scipy.signal import resample_poly

    x = np.asarray(signal, dtype=float)

    x = x[np.isfinite(x)]

    if x.size == 0:

        raise ValueError("No finite signal samples found.")

    if fs != 360:

        # Resample signal into training frequency so each window has compatible duration.

        from math import gcd

        g = gcd(int(fs), 360)

        x = resample_poly(x, 360 // g, int(fs) // g)

        fs = 360

    peaks = _detect_r_peaks(x, fs)

    rows, windows = [], []

    for p in peaks:

        start, end = p - left, p + right

        if start < 0 or end > len(x):

            continue

        segment = x[start:end]

        if len(segment) != WINDOW_SIZE:

            continue

        windows.append(segment.astype(np.float32))

        rows.append({"sample_index": int(p), "time_seconds": p / fs})

    if not windows:

        raise ValueError("No complete beat windows were detected. Try a longer/cleaner ECG signal.")

    X = np.stack(windows)

    preds = model.predict(X)

    try:

        probas = model.predict_proba(X)

        classes = list(model.classes_)

    except Exception:

        probas, classes = None, []

    out = pd.DataFrame(rows)

    out["prediction"] = preds.astype(str)

    if probas is not None:

        out["confidence"] = [float(probas[i, classes.index(preds[i])]) for i in range(len(preds))]

    else:

        out["confidence"] = np.nan

    return out

def render_ecg_monitoring(model_path: str = "models/random_forest.joblib") -> None:

    if not Path(model_path).exists():

        st.error(f"Random Forest model not found at {model_path}. Train/save it first.")

        return

    try:

        model = joblib.load(model_path)

    except Exception as exc:

        st.error(f"Could not load the Random Forest model: {exc}")

        return

    uploaded = st.file_uploader(

        "Upload ECG CSV, or select both WFDB .hea and .dat files",

        type=["csv", "hea", "dat"],

        accept_multiple_files=True,

        key="rf_ecg_monitor_upload",

    )

    if not uploaded:

        st.info("For WFDB, upload the matching .hea and .dat files together.")

        return

    names = [f.name.lower() for f in uploaded]

    fs = 360

    lead_name = "ECG"

    signal = None

    try:

        csvs = [f for f in uploaded if f.name.lower().endswith(".csv")]

        if csvs:

            csv_file = csvs[0]

            df = pd.read_csv(csv_file)

            numeric_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]

            if not numeric_cols:

                st.error("No numeric signal columns found in the CSV.")

                return

            signal_col = st.selectbox("ECG signal column", numeric_cols)

            fs = int(st.number_input("CSV sampling rate (Hz)", min_value=50, max_value=5000, value=360, step=1))

            signal = pd.to_numeric(df[signal_col], errors="coerce").to_numpy(dtype=float)

            lead_name = signal_col

        elif any(n.endswith(".hea") for n in names):

            signal, fs, lead_name = _read_wfdb(uploaded)

            st.write(f"Detected WFDB sampling rate: **{fs} Hz** · lead: **{lead_name}**")

        else:

            st.error("Unsupported upload. Provide a CSV or a WFDB .hea + .dat pair.")

            return

        signal = np.asarray(signal, dtype=float)

        finite = np.isfinite(signal)

        signal = signal[finite]

        if signal.size < 2:

            st.error("No usable numeric ECG samples found.")

            return

        duration = len(signal) / fs

        st.write(f"Signal duration: **{duration:.1f} seconds** · Samples: **{len(signal):,}**")

        if st.button("Detect beats and classify", type="primary"):

            with st.spinner("Detecting candidate beats and classifying segments..."):

                # Keep plotted samples aligned with returned beat indices.

                if fs != 360:

                    from math import gcd

                    from scipy.signal import resample_poly

                    g = gcd(int(fs), 360)

                    signal_360 = resample_poly(signal, 360 // g, int(fs) // g)

                else:

                    signal_360 = signal

                results = classify_uploaded_signal(signal_360, 360, model)

            st.session_state["rf_ecg_monitor_results"] = results

            st.session_state["rf_ecg_monitor_signal"] = signal_360

            st.session_state["rf_ecg_monitor_fs"] = 360

            st.session_state["rf_ecg_monitor_lead"] = lead_name

        results = st.session_state.get("rf_ecg_monitor_results")

        if results is None or results.empty:

            return

        signal_for_plot = st.session_state["rf_ecg_monitor_signal"]

        fs_plot = st.session_state["rf_ecg_monitor_fs"]

        st.success(f"Classified {len(results)} candidate heartbeat segments.")

        fig, ax = plt.subplots(figsize=(12, 3.5))

        t = np.arange(len(signal_for_plot)) / fs_plot

        ax.plot(t, signal_for_plot, linewidth=0.7, label=lead_name)

        ax.scatter(results["time_seconds"], signal_for_plot[results["sample_index"].to_numpy()],

                   s=20, marker="x", label="Detected candidate beats")

        ax.set_xlabel("Time (seconds)")

        ax.set_ylabel("Signal amplitude (input units)")

        ax.legend(loc="upper right")

        ax.grid(alpha=0.2)

        st.pyplot(fig)

        plt.close(fig)

        counts = results["prediction"].value_counts()

        st.write("Predicted beat-label counts")

        st.dataframe(counts.rename_axis("Class").reset_index(name="Count"), use_container_width=True)

        st.dataframe(results, use_container_width=True)

        st.markdown("#### Repeated-V demo alert")

        demo_mode = st.checkbox(
            "Enable synthetic V-beat demonstration",
            value=False,
            help="Adds artificial V labels for demonstrating the alert logic only.",
        )

        # Keep the model's actual predictions intact; use a separate copy for the demo.
        alert_results = results.copy()

        if demo_mode:
            if len(alert_results) < 5:
                st.warning("At least 5 detected beats are needed for this demo.")
            else:
                # Simulate three V labels among the last five beats.
                simulated_indices = alert_results.index[-5:-2]
                alert_results.loc[simulated_indices, "prediction"] = "V"
                st.warning(
                    "SIMULATION MODE: three V labels are artificially inserted "
                    "for a software demonstration. These are not model predictions "
                    "or clinical findings."
                )

        # Show V markers in red on the waveform when present.
        v_rows = alert_results[alert_results["prediction"] == "V"]
        if not v_rows.empty:
            fig_v, ax_v = plt.subplots(figsize=(12, 2.5))
            ax_v.plot(
                np.arange(len(signal_for_plot)) / fs_plot,
                signal_for_plot,
                linewidth=0.7,
                label="ECG waveform",
            )
            valid_v = v_rows[
                (v_rows["sample_index"] >= 0)
                & (v_rows["sample_index"] < len(signal_for_plot))
            ]
            if not valid_v.empty:
                ax_v.scatter(
                    valid_v["time_seconds"],
                    signal_for_plot[valid_v["sample_index"].to_numpy(dtype=int)],
                    color="red",
                    marker="x",
                    s=70,
                    label="V labels (simulated in demo mode)",
                )
            ax_v.set_xlabel("Time (seconds)")
            ax_v.set_ylabel("Signal amplitude")
            ax_v.legend()
            ax_v.grid(alpha=0.2)
            st.pyplot(fig_v)
            plt.close(fig_v)

        col1, col2 = st.columns(2)
        with col1:
            window_n = int(
                st.number_input(
                    "Rolling window (beats)",
                    min_value=3,
                    max_value=50,
                    value=10,
                )
            )
        with col2:
            min_v = int(
                st.number_input(
                    "V labels needed to show demo alert",
                    min_value=2,
                    max_value=50,
                    value=3,
                )
            )

        if min_v > window_n:
            st.error("V threshold must be less than or equal to the rolling window.")
        else:
            recent = alert_results.tail(window_n)
            v_count = int((recent["prediction"] == "V").sum())
            st.write(
                f"Last {len(recent)} classified beats: **{v_count} V labels**"
            )

            if len(recent) >= window_n and v_count >= min_v:
                st.error(
                    f"DEMO ALERT: {v_count} of the last {window_n} beats "
                    "are labeled V. This is a software demonstration, not a clinical alarm."
                )
            else:
                st.info(
                    "No alert under the selected rule. This does not establish "
                    "that the ECG is normal."
                )

        st.download_button(

            "Download predictions as CSV",

            data=results.to_csv(index=False).encode("utf-8"),

            file_name="ecg_rf_predictions_demo.csv",

            mime="text/csv",

        )

    except Exception as exc:

        st.error(f"Could not process this ECG: {exc}")

        st.caption("Check the file format, sampling rate, units, and whether the signal is long/clean enough.")
