"""Stage 3: temporal split, fit, evaluate, write metrics.

Two baselines and two models, on purpose.

  baseline_mean         predict the 2011 average for every day
  baseline_persistence  predict yesterday's count (cnt_lag_1)
  ridge                 standardised linear model on the audited features
  hgb                   gradient-boosted trees on the same features

The mean baseline shows what "the distribution moved" costs. Persistence
is the baseline that actually matters: ridge's features are dominated by
lagged demand, so if it cannot beat "yesterday again" it has learned
nothing worth reporting. The booster is there because it fails, and the
reason it fails is the headline result.

Run: python -m src.train [--target-transform NAME --out PATH]

Target transforms (for hgb only):
  --target-transform log      predict log(cnt + epsilon), output exp(pred)
  --target-transform diff     predict cnt - cnt_lag_1, output pred + cnt_lag_1
  --target-transform ratio    predict cnt / cnt_roll_28, output pred * cnt_roll_28

Every transform is invertible at prediction time using columns available in features.
Write results to PATH; never overwrite metrics.json.
"""
import argparse
import numpy as np
import json
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import METRICS, PROCESSED, TARGET, TRAIN_END, VALID_END
from src.features import feature_columns


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """Predict a single lagged column unchanged. Nothing is fit."""

    def __init__(self, column: str = "cnt_lag_1"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"persistence baseline needs column {self.column!r}")
        return self

    def predict(self, X):
        return X[self.column].to_numpy(dtype=float)


def target_transform_log(y: pd.Series, epsilon: float = 1e-6) -> pd.Series:
    """Log transform with epsilon to handle zeros.

    Inverse: exp(y).

    Note: We add epsilon because cnt=0 can occur (though rare in bikeshare).
    Using log(cnt + epsilon) keeps predictions bounded and handles edge cases.
    """
    return np.log(y + epsilon)


def target_transform_diff(X, y: pd.Series) -> pd.Series:
    """Difference transform: predict change from yesterday's count.

    Inverse: cnt_lag_1 (from X) + predicted change

    This requires cnt_lag_1 to be known at prediction time, which it is
    (one day of history).
    """
    return y - X["cnt_lag_1"]


def target_transform_ratio(X, y: pd.Series) -> pd.Series:
    """Ratio transform: normalize by 28-day rolling average.

    Inverse: cnt_roll_28 (from X) * predicted ratio

    This requires cnt_roll_28 to be known at prediction time, which it is.
    """
    return y / X["cnt_roll_28"]


TRANSFORMS = {
    "log": target_transform_log,
    "diff": target_transform_diff,
    "ratio": target_transform_ratio,
}


MODELS = {
    "baseline_mean": DummyRegressor(strategy="mean"),
    "baseline_persistence": PersistenceBaseline("cnt_lag_1"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "hgb": HistGradientBoostingRegressor(random_state=0, max_iter=300),
}


def inverse_transform_log(y_pred: pd.Series) -> pd.Series:
    """Inverse of log transform."""
    return np.exp(y_pred) - 1e-6


def inverse_transform_diff(y_pred: pd.Series, X) -> pd.Series:
    """Inverse of difference transform."""
    return y_pred + X["cnt_lag_1"]


def inverse_transform_ratio(y_pred: pd.Series, X) -> pd.Series:
    """Inverse of ratio transform."""
    return y_pred * X["cnt_roll_28"]


INVERSE_TRANSFORMS = {
    "log": inverse_transform_log,
    "diff": inverse_transform_diff,
    "ratio": inverse_transform_ratio,
}


def temporal_split(df: pd.DataFrame):
    """Split by date, never at random. A random split on a time series
    puts tomorrow in train and today in test."""
    train = df[df.dteday <= TRAIN_END]
    valid = df[(df.dteday > TRAIN_END) & (df.dteday <= VALID_END)]
    test = df[df.dteday > VALID_END]
    return train, valid, test


def fit_and_score(target_transform: str = None, out_path: str = None) -> dict:
    """Fit models and evaluate.

    Args:
        target_transform: One of "log", "diff", "ratio" (or None for no transform)
        out_path: Path to write results (if set, writes to separate file)

    Returns:
        Dictionary with metrics for all models on valid/test splits.
    """
    df = pd.read_parquet(PROCESSED)
    cols = feature_columns(df)
    train, valid, test = temporal_split(df)

    # Prepare target transform
    y_train_orig = train[TARGET].values
    y_valid_orig = valid[TARGET].values
    y_test_orig = test[TARGET].values

    X_train = train[cols]
    X_valid = valid[cols]
    X_test = test[cols]

    # Apply transform if specified (only for hgb)
    epsilon = 1e-6
    if target_transform == "log":
        y_train_hgb = np.log(y_train_orig + epsilon)
        y_valid_hgb = np.log(y_valid_orig + epsilon)
        y_test_hgb = np.log(y_test_orig + epsilon)
        transform_name = "log"
    elif target_transform == "diff":
        y_train_hgb = y_train_orig - X_train["cnt_lag_1"]
        y_valid_hgb = y_valid_orig - X_valid["cnt_lag_1"]
        y_test_hgb = y_test_orig - X_test["cnt_lag_1"]
        transform_name = "diff"
    elif target_transform == "ratio":
        # Use 27-day rolling mean at train time (excludes today)
        y_train_hgb = y_train_orig / X_train["cnt_roll_28"]
        y_valid_hgb = y_valid_orig / X_valid["cnt_roll_28"]
        y_test_hgb = y_test_orig / X_test["cnt_roll_28"]
        transform_name = "ratio"
    else:
        y_train_hgb = y_train_orig
        y_valid_hgb = y_valid_orig
        y_test_hgb = y_test_orig
        transform_name = None

    results = {
        "n_features": len(cols),
        "features": cols,
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "target_transform": target_transform,
    }

    # Compute metrics for all models on valid/test sets
    for name, model in MODELS.items():
        if name == "hgb" and target_transform:
            # Use transformed target for hgb with transform
            model.fit(X_train, y_train_hgb)
            y_pred_valid = model.predict(X_valid)
            y_pred_test = model.predict(X_test)

            # Inverse transform predictions back to original scale
            if transform_name == "log":
                y_pred_valid_inv = inverse_transform_log(y_pred_valid)
                y_pred_test_inv = inverse_transform_log(y_pred_test)
            elif transform_name == "diff":
                y_pred_valid_inv = inverse_transform_diff(y_pred_valid, X_valid)
                y_pred_test_inv = inverse_transform_diff(y_pred_test, X_test)
            elif transform_name == "ratio":
                y_pred_valid_inv = inverse_transform_ratio(y_pred_valid, X_valid)
                y_pred_test_inv = inverse_transform_ratio(y_pred_test, X_test)
            else:
                y_pred_valid_inv = y_pred_valid
                y_pred_test_inv = y_pred_test

            # Compute metrics on original scale
            results[name] = {
                split: {
                    "r2": round(float(r2_score(part[TARGET], y_pred_inv)), 4),
                    "mae": round(float(mean_absolute_error(part[TARGET], y_pred_inv)), 1),
                }
                for split, part, y_pred_inv in [
                    ("valid", valid, y_pred_valid_inv),
                    ("test", test, y_pred_test_inv),
                ]
            }
        else:
            # Standard models (no transform)
            model.fit(X_train, train[TARGET])
            results[name] = {
                split: {
                    "r2": round(float(r2_score(part[TARGET], model.predict(part[cols]))), 4),
                    "mae": round(float(mean_absolute_error(part[TARGET], model.predict(part[cols]))), 1),
                }
                for split, part in [("valid", valid), ("test", test)]
            }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on bike-share data")
    parser.add_argument(
        "--target-transform",
        choices=["log", "diff", "ratio", None],
        default=None,
        help="Target transformation for hgb model",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to write results (separate file, never overwrites metrics.json)",
    )

    args = parser.parse_args()
    res = fit_and_score(target_transform=args.target_transform, out_path=args.out)
    dest = Path(args.out) if args.out else METRICS
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2), newline="\n")
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, indent=2))
