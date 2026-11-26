"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
import pandas as pd
import pytest

from src.config import FORBIDDEN, TARGET
from src.features import build_features, feature_columns
from src.ingest import load_raw


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique


def test_cnt_roll_28_excludes_today_and_computes_correct_window_mean():
    """cnt_roll_28(d) must be the mean of cnt over calendar days [d-28, d-1] (28 days)."""
    from src.ingest import load_raw

    # Use a named date where we can compute by hand from strictly earlier dates.
    # 2012-03-15 is in the dataset and has 28 prior calendar days available.
    TARGET_DATE = "2012-03-15"

    raw = load_raw()
    feats = build_features(raw)
    by_date = feats.set_index("dteday")["cnt"]

    if TARGET_DATE not in by_date.index:
        pytest.skip(f"{TARGET_DATE} not in dataset; check data coverage")

    target_date = pd.Timestamp(TARGET_DATE)
    # Window: strictly prior 28 days, i.e., [d-28, d-1]
    window_dates = [target_date - pd.Timedelta(days=k) for k in range(1, 29)]

    # Verify all 28 prior dates exist (enforced by reindex_to_calendar)
    # Compare Timestamps directly, not .date() which produces plain date objects
    missing_ts = [dt for dt in window_dates if dt not in by_date.index]
    assert len(missing_ts) == 0, f"Missing {len(missing_ts)} of the 28 prior calendar days: {missing_ts}"

    # Check that the feature column exists before trying to read it
    if "cnt_roll_28" not in feats.columns:
        min_window_date = target_date - pd.Timedelta(days=27)
        max_window_date = target_date - pd.Timedelta(days=1)
        raise AssertionError(
            f"column 'cnt_roll_28' is missing from feature set. "
            f"It should be the mean of cnt over calendar days [{min_window_date.date()}, {max_window_date.date()}]."
        )

    cnts = [by_date[dt] for dt in window_dates]
    expected_mean = sum(cnts) / 28
    # feats has a RangeIndex after dropna(), so use .iloc with boolean mask
    # dteday contains Timestamps, so compare with string or Timestamp directly
    row_mask = feats.dteday == TARGET_DATE
    actual = feats.loc[row_mask, "cnt_roll_28"].iloc[0]

    assert actual == pytest.approx(expected_mean), \
        f"at {TARGET_DATE.date()}: got {actual}, expected {expected_mean:.4f}"


def test_cnt_lag_7_equals_target_from_exactly_7_days_prior():
    """cnt_lag_7(d) must equal cnt(d-7), not cnt(d-LAG_WARMUP)."""
    from src.ingest import load_raw

    TARGET_DATE = "2012-03-15"  # Has full history including d-7=2012-02-25

    raw = load_raw()
    feats = build_features(raw)
    by_date = feats.set_index("dteday")["cnt"]

    if TARGET_DATE not in by_date.index:
        pytest.skip(f"{TARGET_DATE} not in dataset; check data coverage")

    target_date = pd.Timestamp(TARGET_DATE)
    prior_date = target_date - pd.Timedelta(days=7)

    # Check that the prior date exists in our calendar
    if prior_date not in by_date.index:
        pytest.skip(f"prior date {prior_date.date()} missing from calendar")

    expected = by_date[prior_date]
    row_mask = feats.dteday == TARGET_DATE
    actual = feats.loc[row_mask, "cnt_lag_7"].iloc[0]

    assert actual == pytest.approx(expected), \
        f"at {target_date.date()}: cnt_lag_7={actual} != target from {prior_date.date()} ({expected})"


def test_encode_weather_raises_on_unknown_label():
    """encode_weather must raise ValueError on unseen labels, not silently map to NaN.

    An unseen weather label like 'SNOW' should trigger a loud failure listing
    the unmapped labels, so we don't drop interesting rows (e.g., cold winter days)
    with no explanation.
    """
    from src.features import encode_weather

    # Create a Series with known labels plus an unknown one
    series = pd.Series(["GOOD", "MISTY", "BAD", "SNOW", "GOOD"])

    # This should raise ValueError, not return NaN
    with pytest.raises(ValueError) as exc_info:
        encode_weather(series)

    # Verify the error message lists the unmapped label
    assert "SNOW" in str(exc_info.value), \
        f"Error message should list 'SNOW' as unmapped: {exc_info.value}"
    assert "unmapped" in str(exc_info.value).lower(), \
        f"Error message should mention 'unmapped': {exc_info.value}"
