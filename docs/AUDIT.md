# Audit of a teammate's repository (Exercise 3)

**Repository audited:** https://github.com/Cheng-I-Lin/bikeshare-open, at commit `dab0405`.
**Diff under review:** `git diff d049708 HEAD` — everything after their unmodified starter
commit (14 files: `src/features.py`, `src/config.py`, `tests/`, `.claude/settings.json`,
`.gitignore`, `experiment_hgb_log.py`, docs, reports).
**Auditor:** the `leakage-auditor` subagent (`.claude/agents/leakage-auditor.md`), run from
this repo in a headless Claude Code session on Qwen 3.5 9B (`bikeshare-coder`), with read
access to the teammate's clone and Write/Edit denied. Full transcript retained locally (2 turns, 67 s; the subagent read files and ran their test suite). The original report is in `evidence/E3_auditor_report.txt`.

## Auditor output (verbatim)

```
All five checks pass:

1. **Target leakage**: Verified via tests (`test_no_forbidden_features`, `test_leakage_identity_holds`). `test_temp_hum_interaction` is derived from allowed columns (temp, hum), not the target.

2. **Temporal ordering**: `temporal_split()` uses date-based splitting (TRAIN_END=2011-12-31, VALID_END=2012-06-30). Models fit on train only; no validation/test leakage.

3. **Calendar gaps**: `reindex_to_calendar` is called before any lag feature, ensuring contiguous calendar index for `.shift()` semantics.

4. **Silent row loss**: `.dropna()` after lagging is bounded by LAG_WARMUP=7 in config.py; test bounds this explicitly.

5. **Unreported baseline**: Both `baseline_mean` and `baseline_persistence` are reported alongside every model in metrics.json (valid and test splits for all four models).

All 19 tests pass. The diff introduces one new feature (`test_temp_hum_interaction`) that is correctly handled per project rules: it's a derived feature from allowed raw columns, not forbidden, and its addition/deprecation is documented in docs/DECISIONS.md.
```

## Supplementary checks (run separately, not by the auditor)

- `make all` in the teammate's clone (same pinned versions) reproduces their committed
  `reports/metrics.json` exactly; `make test`: 19 passed.
- Their `docs/DECISIONS.md` records **DROP** for `test_temp_hum_interaction` (ridge validation
  R² 0.6475 → 0.6493, +0.0018, below the 0.01 threshold), but the feature is still built in
  `src/features.py`, so the committed metrics are the 15-feature pipeline (ridge test
  0.5835 / MAE 946.6 vs 0.5833 / 947.1 without it). Their REPORT discloses this.
- Committed raw CSV is unmodified (blob sha256 `a6bed157...`); with no `.gitattributes`, a
  Windows checkout with `core.autocrlf` converts it to CRLF and the working copy hashes to
  `13e33ae5...`. Portability only.
- `.claude/settings.json` replaces `Write(./data/raw/**)` with a duplicate `Edit(./data/raw/**)`.
  Does not change a number.

## My verdict

I agree that the new feature does not leak the target and that the data is split correctly.
The auditor missed one issue: my partner decided to drop the temperature-humidity feature, but
it is still in the final pipeline. Removing it changes ridge's test R² from 0.5835 to 0.5833.
The difference is small enough that my overall conclusion stays the same.
