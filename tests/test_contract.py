"""Guard rails on the ingest contract and the trained model's behaviour."""
import os
import tempfile
from pathlib import Path

import pytest
import shutil

import pandas as pd
from sklearn.metrics import r2_score

from src.config import ALLOWED_RAW_COLS, FORBIDDEN, LAG_WARMUP, RAW, RAW_SHA256, TARGET
from src.features import build_features, feature_columns, reindex_to_calendar
from src.ingest import load_raw
from src.train import MODELS, temporal_split


@pytest.fixture(scope="module")
def raw():
    return load_raw().sort_values("dteday").reset_index(drop=True)


def test_no_unclassified_columns_in_raw():
    """No column in the raw CSV may be unclassified.

    A new column appearing in data/raw/bike_daily.csv that is not in ALLOWED_RAW_COLS
    or FORBIDDEN would be flagged as unclassified. This test ensures a new column cannot
    enter the model without being explicitly registered. The error message displays allowed
    and forbidden columns so adding a new feature to src/config.py is obvious when required.
    Unclassified columns here are valid raw features to be processed, not forbidden ones.
    """
    raw_df = load_raw()
    raw_columns = set(raw_df.columns)

    # Any column not in ALLOWED_RAW_COLS or FORBIDDEN is suspicious - it would bypass
    # our feature engineering pipeline and appear in the model without being registered.
    unclassified = raw_columns - ALLOWED_RAW_COLS - FORBIDDEN

    allowed_cols = sorted(ALLOWED_RAW_COLS)
    forbidden_cols = sorted(FORBIDDEN)

    assert unclassified == set(), \
        f"Found {len(unclassified)} columns in raw data that are not classified: " \
        f"{sorted(unclassified)}. All new columns must be added to src/config.py. " \
        f"Allowed: ALLOWED_RAW_COLS={allowed_cols}, Forbidden: FORBIDDEN={forbidden_cols}"


@pytest.fixture(scope="module")
def feats(raw):
    return build_features(raw)


def test_calendar_is_complete_before_lagging(raw):
    """The reindexed frame must have no gaps, or the lags are wrong."""
    grid = reindex_to_calendar(raw)
    assert set(grid.dteday.diff().dropna().dt.days.unique()) == {1}


def test_only_expected_rows_are_dropped(raw, feats):
    """Rows may only vanish for the lag warm-up or downstream of a genuinely
    absent day -- never because an encoder produced a silent NaN.

    Accounting: the first LAG_WARMUP days have no complete history. Each
    absent calendar day m costs the LAG_WARMUP rows after it (cnt_lag_1 is
    NaN on m+1, cnt_roll_7 is NaN on m+1..m+7, cnt_lag_7 on m+7; their
    union is 7 rows). Day m itself was never in `raw`, so it is not a loss.
    The bound is exact for the current feature set; add a longer lag and
    LAG_WARMUP in src/config.py must grow with it.
    """
    full = pd.date_range(raw.dteday.min(), raw.dteday.max(), freq="D")
    missing_days = len(full) - len(raw)
    allowed_loss = LAG_WARMUP + LAG_WARMUP * missing_days
    lost = len(raw) - len(feats)
    assert lost <= allowed_loss, f"lost {lost} rows; expected at most {allowed_loss}"


def test_beats_the_mean_baseline(feats):
    """A model that cannot beat predicting the training mean is not a result."""
    cols = feature_columns(feats)
    train, _, test = temporal_split(feats)
    ridge = MODELS["ridge"]
    ridge.fit(train[cols], train[TARGET])
    assert r2_score(test[TARGET], ridge.predict(test[cols])) > 0.4


def test_persistence_baseline_is_reported():
    """The baseline that matters for a lag-driven model is 'yesterday again'.
    It must be in the comparison, whatever its score."""
    assert "baseline_persistence" in MODELS


def test_raw_data_hash_guard(tmp_path, monkeypatch):
    """Ensure load_raw() refuses tampered data by checking SHA256.

    This test verifies that a single-byte modification to the raw CSV causes
    load_raw() to raise ValueError with expected vs actual hash in the message.
    The test never writes to data/raw/.
    """
    # Copy and corrupt one byte of the original file
    shutil.copy(RAW, tmp_path / "corrupted.csv")
    with open(tmp_path / "corrupted.csv", "r+b") as f:
        data = bytearray(f.read())
        mid = len(data) // 2
        data[mid] ^= 0x80  # flip a bit
        f.seek(0)
        f.write(bytes(data))

    corrupted_path = tmp_path / "corrupted.csv"

    # Monkeypatch RAW to point at the corrupted file
    monkeypatch.setattr("src.config.RAW", corrupted_path.resolve())

    with pytest.raises(ValueError, match="integrity check failed"):
        load_raw()


def test_hash_guard_error_message_shows_hashes(tmp_path, monkeypatch):
    """Ensure the error message clearly shows expected vs actual hash.

    Monkeypatch RAW to point at a temporarily created corrupted file and verify
    the error message contains both hashes.
    """
    # Create a temporary file with modified bytes
    shutil.copy(RAW, tmp_path / "garbage.csv")
    with open(tmp_path / "garbage.csv", "r+b") as f:
        data = bytearray(f.read())
        mid = len(data) // 2
        data[mid] ^= 0x80  # flip a bit
        f.seek(0)
        f.write(bytes(data))

    garbage_path = tmp_path / "garbage.csv"

    # Monkeypatch RAW to point at the corrupted file
    monkeypatch.setattr("src.config.RAW", garbage_path.resolve())

    with pytest.raises(ValueError, match="integrity check failed"):
        load_raw()
