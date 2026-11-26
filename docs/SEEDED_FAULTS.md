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
