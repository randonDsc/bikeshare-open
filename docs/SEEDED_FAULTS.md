# Seeded faults (Exercise 2)

Agent: Qwen 3.5 9B (`bikeshare-coder`, Ollama, 64k context) in headless Claude Code
sessions. Faults were introduced during the supervised seed-fault experiments. The first fault also triggered the hook during a live Qwen edit; other hook checks used the Windows Edit payload. Repairs were run in fresh Qwen sessions. "Turns" is the session's
`num_turns` from the Claude Code result message. Full transcripts are retained locally. Selected original outputs are included in `evidence/`.
Output below is pasted from those runs, not paraphrased.

## Exercise 1: failing test before the feature

The new feature test was committed before the implementation (commit `00103a4`).
The original failing output is in [evidence/E1_failing_test_red.txt](evidence/E1_failing_test_red.txt).
The assertion failed because `cnt_roll_28` was missing; 1 failed, 7 deselected.
The subsequent feature commit is `9252a75`. The additional lag regression evidence is in
[evidence/E1_lag7_bug_red.txt](evidence/E1_lag7_bug_red.txt).

## Fault 1 - a forbidden column added to `feature_columns`

**Change** (`src/features.py`):
```diff
@@ -83,7 +83,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
 def feature_columns(df: pd.DataFrame) -> list[str]:
     """Numeric columns that are safe to feed the model."""
     numeric = df.select_dtypes(include="number").columns
-    return [c for c in numeric if c not in FORBIDDEN]
+    return [c for c in numeric if c not in FORBIDDEN or c == "casual"]
 
 
 if __name__ == "__main__":
```

**Hook** (`.claude/hooks/check_leakage.sh`, fed the Windows Edit payload) - exit 2, stderr tail:
```
E       AssertionError: forbidden columns in feature set: {'casual'}
E       assert not ({'atemp', 'casual', 'cnt_lag_1', 'cnt_lag_7', 'cnt_roll_28', 'cnt_roll_7', ...} & {'casual', 'cnt', 'cnt_2d_bfr', 'days_since_2011', 'instant', 'registered', ...})

tests\test_leakage.py:31: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 8 passed in 0.88s
```

**`make test`** (last lines):
```
tests\test_leakage.py:31: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_leakage.py::test_no_forbidden_features - AssertionError: fo...
1 failed, 15 passed in 2.53s
make test exit=1
```

**Hook inside a live Qwen session.** When Qwen seeded this same fault itself via `/seed-fault`
(full transcript retained locally), Claude Code returned to the agent, immediately after its Edit:
`PostToolUse:Edit hook blocking error from command: "${CLAUDE_PROJECT_DIR}/.claude/hooks/check_leakage.sh": ... Leakage tests failed after editing C:/mine/ucsd/senior_fallq/dsc180_a/bikeshare-open/src/features.py: ... AssertionError: forbidden columns in feature set: {'casual'}`.
(That session also ignored "steps 1-3 only, do not restore" and undid its own fault, so its 1-turn "fix" is not counted.)

**Repair.** Fresh Qwen session, prompt: "make test is failing ... find the cause and fix it in src/".
It ran `make test`, read `src/features.py` and `src/config.py`, removed `or c == "casual"` in one Edit.
**9 turns, 36 s.** After: `16 passed in 1.71s`

## Fault 2 - `.shift(1)` deleted from `cnt_roll_7`

**Change** (`src/features.py`):
```diff
@@ -73,7 +73,7 @@ def build_features(df: pd.DataFrame) -> pd.DataFrame:
     out["cnt_lag_1"] = out[TARGET].shift(1)
     out["cnt_lag_7"] = out[TARGET].shift(7)  # fixed 7-day lag, independent of LAG_WARMUP
     # cnt_roll_7: mean of the 7 calendar days strictly before today
-    out["cnt_roll_7"] = out[TARGET].shift(1).rolling(7).mean()
+    out["cnt_roll_7"] = out[TARGET].rolling(7).mean()
     # cnt_roll_28: mean of the 28 calendar days strictly before today
     out["cnt_roll_28"] = out[TARGET].shift(1).rolling(28).mean()
 
```

**Hook** (`.claude/hooks/check_leakage.sh`, fed the Windows Edit payload) - hook exit=2, stderr tail:
```
E             Obtained: 4605.285714285715
E             Expected: 4724.428571428572 Â± 0.00472443

tests\test_leakage.py:57: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 8 passed in 0.92s
```

**`make test`** (last lines):
```
tests\test_leakage.py:57: AssertionError
=========================== short test summary info ===========================
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.fl...
1 failed, 15 passed in 2.60s
make test exit=1
```

Predicted catcher: `test_rolling_mean_excludes_today` (it checks one row, `iloc[200]`, and only if its 7-day window is present). Actual: the same.

**Repair.** Fresh Qwen session, same prompt. It ran `make test`, read `src/features.py`, and restored `.shift(1)` in one Edit (its explanation: `.rolling(7)` "includes today's value"). **7 turns, 41 s.** After: `16 passed in 2.13s`

## Fault 3 - the weather encoder's `raise` removed

**Change** (`src/features.py`):
```diff
@@ -26,8 +26,6 @@ def encode_weather(series: pd.Series) -> pd.Series:
     """
     known = series.dropna()
     unknown = set(known.unique()) - set(WEATHER_ORDER)
-    if unknown:
-        raise ValueError(f"unmapped weather labels: {sorted(unknown)}")
     return series.map(WEATHER_ORDER).astype(float)
 
 
```

**Hook** (`.claude/hooks/check_leakage.sh`, fed the Windows Edit payload) - hook exit=0, stderr tail:
```
(empty - the hook printed nothing)
```

**`make test`** (last lines):
```
................                                                         [100%]
16 passed in 2.14s
make test exit=0
```

Predicted: **nothing catches it** -- every label in the data is GOOD/MISTY/BAD, so without the guard the code behaves identically; it only matters for an unseen label. Actual: hook exit 0, 16 passed. **Uncaught**: the suite had no test of the guard itself (the row-loss bound only fires if a label is actually unmapped).

**Repair.** Fresh Qwen session told a reviewer reported the encoder no longer fails loudly. It added `test_encode_weather_raises_on_unknown_label` (`pd.Series(["GOOD","MISTY","BAD","SNOW","GOOD"])` must raise `ValueError` naming `SNOW`) and ran it against the faulty code first:
```text
E       Failed: DID NOT RAISE <class 'ValueError'>
tests\test_leakage.py:154: Failed
FAILED tests/test_leakage.py::test_encode_weather_raises_on_unknown_label - F...
```
Then restored the `raise`. **8 turns, 46 s.** After: `17 passed`. With this test, removing the `raise` now turns `make test` red.

## Fault 4 - overwriting `data/raw/` from Bash (the fault the PostToolUse hook cannot see)

Tampered copy: 2012-07-04 `temp` 29.090849 -> 5.000000, nothing else changed. Qwen 9B was told to try each route
(full transcripts retained locally):

| Route | Result |
|---|---|
| Edit tool on `data/raw/bike_daily.csv` | **Denied** by `Edit(./data/raw/**)`: `File is in a directory that is denied by your permission settings.` |
| (unprompted) `perl -i` / `sed -i` on the file | Qwen tried to work around the denial itself; refused only because they were not pre-approved in a headless session (interactive: an approval prompt) |
| `cp <tampered> data/raw/bike_daily.csv` (`Bash(cp:*)` pre-approved) | **Denied**: Claude Code 2.1.332 checks `cp`'s destination against the Edit deny rule, so the lesson's ""deny does not stop cp"" no longer holds verbatim here |
| `python -c "import shutil; shutil.copy('logs/evidence/bike_daily_tampered.csv', 'data/raw/bike_daily.csv')"` | **Ran.** `sha256sum` -> `97f504318d014c9cb96a4887ae3df6f478b65da05a2ad8dc1fe3b90e1fd35b95` (expected `a6bed157...`) |

With the raw file tampered: **no hook message** (the hook matches `Edit|Write`, not Bash), `make test` -> `17 passed`,
`make data` -> `OK ingest: 728 rows`. Nothing noticed. Restored with `git checkout -- data/raw/bike_daily.csv` (hash back to `a6bed157...`).

**What would catch it.** Permission rules bind tool calls, and an interpreter is a tool call whose writes they cannot see.
Options: (1) a hash check on the raw file inside the pipeline -- catches every route, including manual edits;
(2) a PreToolUse hook on Bash that rejects commands mentioning `data/raw` -- easy to evade with string building;
(3) a filesystem guard (read-only attribute / `chmod -w`) -- strongest, but outside the repo, so a grader's clone does not inherit it;
(4) a Stop hook running `make test` -- only useful once (1) exists.

**Implemented: (1).** Qwen 9B added `RAW_SHA256` and a check in `load_raw()` (fresh session; [result] subtype=success turns=41 duration=265s session=947a283b-3932-4906-b10a-60c5611821e4 is_error=False), plus a test that
corrupts a `tmp_path` copy and expects `ValueError`. Against the real tampered file:
```
ValueError: Raw data integrity check failed.
Expected SHA256: a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc
Actual SHA256:   97f504318d014c9cb96a4887ae3df6f478b65da05a2ad8dc1fe3b90e1fd35b95
The raw dataset may have been modified or corrupted. Aborting.
```
Diff:
```diff
diff --git a/src/config.py b/src/config.py
index c3ea819..15bfc4a 100644
--- a/src/config.py
+++ b/src/config.py
@@ -1,8 +1,13 @@
 """Single source of truth for paths, columns, and split dates."""
+import hashlib
 from pathlib import Path
 
 ROOT = Path(__file__).resolve().parents[1]
 RAW = ROOT / "data" / "raw" / "bike_daily.csv"
+
+# SHA256 hash of the canonical raw dataset (mirror copy, 2011-01-03 to 2012-12-31)
+# Expected value for https://github.com/christophM/interpretable-ml-book/raw/master/data/bike.csv
+RAW_SHA256 = "a6bed1577f6e4de56dcfae6d8315985073cc0ad3446c8c99ed60c86a0b5ad9dc"
 PROCESSED = ROOT / "data" / "processed" / "features.parquet"
 METRICS = ROOT / "reports" / "metrics.json"
 
diff --git a/src/ingest.py b/src/ingest.py
index 6654075..c16f68b 100644
--- a/src/ingest.py
+++ b/src/ingest.py
@@ -2,25 +2,51 @@
 
 Run: python -m src.ingest
 """
+import hashlib
 import sys
 import urllib.request
 
 import pandas as pd
 
-from src.config import DATA_URL, RAW
+import src.config
 
 EXPECTED_MIN_ROWS = 700
 REQUIRED_COLS = {"dteday", "cnt", "casual", "registered", "temp", "hum", "windspeed"}
 
 
 def download(force: bool = False) -> None:
+    RAW = src.config.RAW
     if RAW.exists() and not force:
         return
     RAW.parent.mkdir(parents=True, exist_ok=True)
-    urllib.request.urlretrieve(DATA_URL, RAW)
+    urllib.request.urlretrieve(src.config.DATA_URL, RAW)
 
 
 def load_raw() -> pd.DataFrame:
+    """Load raw data and verify its integrity via SHA256 hash.
+
+    Raises ValueError if the file's hash differs from the expected canonical value.
+    This guard prevents model evaluation on tampered or corrupted data.
+
+    Note: This function accesses src.config.RAW at call time, so monkeypatching
+    src.config.RAW will affect this function at runtime (pytest uses this).
+    """
+    # Get RAW and RAW_SHA256 from config at call time for monkeypatchability
+    RAW = src.config.RAW
+    RAW_SHA256 = src.config.RAW_SHA256
+
+    # Verify file hash before parsing
+    with open(RAW, "rb") as f:
+        actual_hash = hashlib.sha256(f.read()).hexdigest()
+
+    if actual_hash != RAW_SHA256:
+        raise ValueError(
+            f"Raw data integrity check failed.\n"
+            f"Expected SHA256: {RAW_SHA256}\n"
+            f"Actual SHA256:   {actual_hash}\n"
+            "The raw dataset may have been modified or corrupted. Aborting."
+        )
+
     df = pd.read_csv(RAW, parse_dates=["dteday"])
     missing = REQUIRED_COLS - set(df.columns)
     if missing:
```
