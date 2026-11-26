# Seeded faults (Exercise 2)

Agent: Qwen 3.5 9B (`bikeshare-coder`, Ollama, 64k context) in headless Claude Code
sessions. Each fault was seeded by Qwen through `/seed-fault` (so the edit went
through the Edit tool and the PostToolUse hook), then repaired by a *fresh*
Qwen session told only that `make test` was red. "Turns" is the session's
`num_turns` from the Claude Code result message. Transcripts: `logs/qwen/E2*`.
Output below is pasted from those runs, not paraphrased.

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
(`logs/qwen/E2_f1_seed.jsonl`), Claude Code returned to the agent, immediately after its Edit:
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
```nE       Failed: DID NOT RAISE <class 'ValueError'>
tests\test_leakage.py:154: Failed
FAILED tests/test_leakage.py::test_encode_weather_raises_on_unknown_label - F...
```n then restored the `raise`. **8 turns, 46 s.** After: `17 passed`. With this test, removing the `raise` now turns `make test` red.
