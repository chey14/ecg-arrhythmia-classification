
"""
Download selected MIT-BIH records into data/mit-bih/.
"""

from pathlib import Path
import subprocess

BASE_URL = "https://physionet.org/files/mitdb/1.0.0/"
DATA_DIR = Path("data/mit-bih")

# Record 100 is already downloaded; add two more for now.
RECORDS = [
    "100", "101", "103", "105", "106",
    "108", "109", "111", "112", "113",
    "119", "200", "202", "208", "210",
]
DATA_DIR.mkdir(parents=True, exist_ok=True)

for record_name in RECORDS:
    for extension in ("hea", "dat", "atr"):
        output_path = DATA_DIR / f"{record_name}.{extension}"
        url = f"{BASE_URL}{record_name}.{extension}"

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Already exists: {output_path}")
            continue

        print(f"Downloading {url}")

        command = [
            "curl",
            "-fL",
            "-C", "-",
            "--connect-timeout", "15",
            "--max-time", "300",
            "--retry", "5",
            "--retry-delay", "2",
            "--progress-bar",
            url,
            "-o", str(output_path),
        ]

        result = subprocess.run(command)

        if result.returncode != 0:
            raise RuntimeError(
                f"Download failed for {url}. "
                "Run this script again to resume."
            )

        print(f"Downloaded: {output_path}")
