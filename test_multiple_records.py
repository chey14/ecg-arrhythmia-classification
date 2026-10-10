
from collections import Counter
from pathlib import Path

import wfdb

from src.preprocessing import extract_heartbeat_segments

# These should match the records in src/download_data.py.
RECORDS = [
    "100", "101", "103", "105", "106",
    "108", "109", "111", "112", "113",
    "119", "200", "202", "208", "210",
]

DATA_DIR = Path("data/mit-bih")
total_counts = Counter()
successful_records = []

for record_name in RECORDS:
    record_path = DATA_DIR / record_name
    print(f"\nProcessing record {record_name}...")

    # Read one record at a time to conserve memory.
    record = wfdb.rdrecord(str(record_path), physical=True)
    annotations = wfdb.rdann(str(record_path), extension="atr")

    segments, labels = extract_heartbeat_segments(
        record.p_signal,
        annotations,
        channel=0,
    )

    counts = Counter(labels.tolist())
    total_counts.update(counts)
    successful_records.append(record_name)

    print(f"Heartbeat segments: {len(segments)}")
    print(f"Class counts: {dict(counts)}")

print("\n" + "=" * 45)
print("COMBINED CLASS COUNTS")
print("=" * 45)
print(dict(total_counts))
print("Total heartbeat segments:", sum(total_counts.values()))
print("Successfully processed records:", successful_records)
