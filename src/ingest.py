"""Stage 1: fetch the raw dataset and assert its shape.

Run: python -m src.ingest
"""
import hashlib
import sys
import urllib.request

import pandas as pd

import src.config

EXPECTED_MIN_ROWS = 700
REQUIRED_COLS = {"dteday", "cnt", "casual", "registered", "temp", "hum", "windspeed"}


def download(force: bool = False) -> None:
    RAW = src.config.RAW
    if RAW.exists() and not force:
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(src.config.DATA_URL, RAW)


def load_raw() -> pd.DataFrame:
    """Load raw data and verify its integrity via SHA256 hash.

    Raises ValueError if the file's hash differs from the expected canonical value.
    This guard prevents model evaluation on tampered or corrupted data.

    Note: This function accesses src.config.RAW at call time, so monkeypatching
    src.config.RAW will affect this function at runtime (pytest uses this).
    """
    # Get RAW and RAW_SHA256 from config at call time for monkeypatchability
    RAW = src.config.RAW
    RAW_SHA256 = src.config.RAW_SHA256

    # Verify file hash before parsing
    with open(RAW, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    if actual_hash != RAW_SHA256:
        raise ValueError(
            f"Raw data integrity check failed.\n"
            f"Expected SHA256: {RAW_SHA256}\n"
            f"Actual SHA256:   {actual_hash}\n"
            "The raw dataset may have been modified or corrupted. Aborting."
        )

    df = pd.read_csv(RAW, parse_dates=["dteday"])
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"raw data is missing required columns: {sorted(missing)}")
    if len(df) < EXPECTED_MIN_ROWS:
        raise ValueError(f"raw data has only {len(df)} rows; expected >= {EXPECTED_MIN_ROWS}")
    return df


if __name__ == "__main__":
    download(force="--force" in sys.argv)
    df = load_raw()
    print(f"OK ingest: {len(df)} rows, {df.dteday.min().date()} -> {df.dteday.max().date()}")
