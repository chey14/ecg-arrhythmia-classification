import wfdb

def load_ecg_record(record_name="100"):
    """Load a locally downloaded ECG record and its annotations."""

    record_path = f"data/mit-bih/{record_name}"

    record = wfdb.rdrecord(record_path, physical=True)
    annotations = wfdb.rdann(record_path, extension="atr")

    return record.p_signal, record.fs, annotations