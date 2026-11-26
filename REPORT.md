# Forecasting daily bikeshare demand — report

Daily Capital Bikeshare rides (`cnt`), Washington DC. Train = 2011, validation = H1 2012,
test = H2 2012, split by date. Reproduce everything below with `make all` (numbers in
`reports/metrics.json`); `make test` runs 20 tests.

## Results

Validation and test R² / MAE for both baselines and both models, before and after the
feature added in Exercise 1 (`cnt_roll_28`, the mean of `cnt` over the 28 days before *d*).
"Before" is the starter pipeline (commit `408cf60`, 14 features, 348 training days);
"after" is the committed `reports/metrics.json` (15 features, 306 training days — the longer
warm-up drops 42 early-2011 days).

| Model | Feature set | Valid R² | Valid MAE | Test R² | Test MAE |
|---|---|---|---|---|---|
| baseline_mean | before | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| baseline_persistence | before | 0.4507 | 900.6 | 0.5099 | 840.1 |
| ridge | before | 0.6475 | 832.9 | 0.5833 | 947.1 |
| hgb | before | 0.2185 | 1296.6 | -0.4975 | 1972.6 |
| baseline_mean | after | -0.6894 | 1877.3 | -1.7717 | 2659.6 |
| baseline_persistence | after | 0.4507 | 900.6 | 0.5099 | 840.1 |
| ridge | after | 0.6839 | 777.5 | 0.6239 | 879.3 |
| hgb | after | 0.2415 | 1296.6 | -0.4600 | 1952.4 |

The mean baseline's change is not the feature: it predicts the training mean, and the 42
dropped days are low-ridership January days. To isolate the feature, ridge and hgb were refit
on the same 306 training days with and without `cnt_roll_28` (validation only): ridge
0.6462 → 0.6839 (MAE 835.5 → 777.5), hgb 0.1929 → 0.2415. Dropping the rows alone moves
ridge 0.6475 → 0.6462. The test set was evaluated once, after all feature and model choices.

## Headline finding

The naive evaluation — every column, including `casual` and `registered` — gives R² = 1.000
and is invalid: `casual + registered == cnt` on every row (least-squares coefficients [1, 1],
max residual 7.3e-12), so the model learns addition. Under the temporal split the mean
baseline scores −1.77 on test, because ridership grew ~65% and the 2011 average is the wrong
level for late 2012. Ridge reaches test R² 0.624 against a persistence baseline of 0.510, but
loses to it on MAE (879 vs 840 rides). Gradient boosting fails (test R² −0.46, though it still
beats the mean baseline) because a tree cannot predict above the largest count it saw in
training, and 56% of test days exceed the 2011 maximum of 6,043. Adding `cnt_roll_28` raised
ridge's validation R² from 0.646 to 0.684 on identical training rows; I kept it.

## Exercise 4: target transformation

A tree fails here because the target drifts out of its training range, so the fix has to make
the target stationary. Three transforms were tried for hgb (code behind
`python -m src.train --target-transform {log,diff,ratio} --out PATH`; the committed metrics
are untransformed). Predictions are inverted before scoring, so R²/MAE are on rides.

| hgb target | Valid R² | Valid MAE |
|---|---|---|
| `cnt` (none) | 0.2415 | 1296.6 |
| `log(cnt)` | 0.1855 | 1324.2 |
| `cnt − cnt_lag_1` | 0.6138 | 829.9 |
| `cnt / cnt_roll_28` | **0.6754** | **732.4** |
| *ridge, untransformed* | *0.6839* | *777.5* |

`log` does not help: it compresses the scale but the drift is still out of range. The ratio
works for a measurable reason: 39% of validation days exceed the 2011 maximum count, but **0%**
exceed the 2011 maximum ratio (train 0.15–2.00, valid 0.17–1.85). Everything in the transform
is known at prediction time. Applied to ridge, the same ratio *hurts* (0.4039), because the
inverse multiplies a linear model's error by the level.

**Verdict: defensible, with limits.** The mechanism is sound — it removes the extrapolation
problem using a quantity known at prediction time, which is ordinary detrending, not leakage.
But the ratio was chosen as the best of three on the same 182 validation days and applied only
to the model that needed rescuing, so it does not support a fair "hgb ties ridge" claim.
Touched once at the end, hgb + ratio scored test R² 0.6392 / MAE 744.2 (ridge 0.6239 / 879.3);
a pre-registered choice and a fresh test period would be needed before believing that ordering.
Integrity note: during this exploration the agent printed test-set numbers despite explicit
instructions not to; no decision was taken on them.

## Exercise 5: ridge vs persistence

On test, ridge wins R² (0.624 vs 0.510) and loses MAE (879 vs 840). The two metrics disagree
because the errors are distributed differently (184 test days):

| | ridge | persistence |
|---|---|---|
| MAE / median abs. error | 879 / 807 | **840 / 507** |
| RMSE / 90th-pct error | **1068 / 1711** | 1219 / 1960 |
| 17 days with a >2,000-ride swing from yesterday (MAE) | **1823** | 3019 |
| the other 167 days (MAE) | 783 | **618** |
| mean bias (pred − actual) | −427 | +16 |

Persistence is closer on 57% of days; ridge is far better on the shocks. Ridge is not just
"yesterday again": besides `cnt_roll_7` (+403) and `cnt_lag_1` (+321), its largest standardised
coefficients are weather — `weather_severity` −290, `temp` +228, `windspeed` −160.

**I would deploy ridge.** The misses that cost money in a bikeshare system are the large ones —
a storm, a holiday, a sudden drop — because they decide where bikes and staff must be moved.
Ridge cuts those errors by about 40% (1823 vs 3019 on swing days; RMSE 1068 vs 1219) and uses
weather that persistence cannot. Its MAE loss is mostly a level bias: trained on 2011, it
under-predicts growing 2012 demand by 427 rides a day on average, which periodic retraining
corrects; persistence has no such lever.

## Attribution

**Model:** `qwen3.5:9b` via Ollama (built as `bikeshare-coder`, Q4_K_M, 64k context), driven
by Claude Code in headless sessions. I started on `qwen3.5:27b` and switched after Step 2: on
this laptop the 27B ran 56% on CPU (~1–2 min per action). The 27B's only run (Step 2) was
under a broken permission setup and is not compared. Claude Opus 5.5 orchestrated the Qwen
sessions, verified every result against the data, fixed infrastructure, and drafted the
write-ups for my review. Transcripts of every Qwen session are kept locally (`logs/qwen/`).

**Generated by the agent (Qwen 9B unless noted)**
- `src/features.py`: `cnt_roll_28`; `cnt_lag_7 = shift(7)` fix (after its first version silently made it a 28-day lag); weather `raise` restored.
- `tests/test_leakage.py`: `test_cnt_roll_28_…` (named date 2012-03-15), `test_cnt_lag_7_…`, `test_encode_weather_raises_on_unknown_label`.
- `tests/test_contract.py`: `test_no_unclassified_columns_in_raw`, the two raw-hash tests; `src/config.py`: `ALLOWED_RAW_COLS`, `RAW_SHA256`; `src/ingest.py`: hash check in `load_raw()`.
- `src/train.py`: `--target-transform` (log/diff/ratio); Opus fixed its ignored `--out`.
- Repairs of seeded faults 1–3 and the fault-4 guard (`docs/SEEDED_FAULTS.md` records turns).
- Claude Opus 5.5: `.claude/hooks/check_leakage.sh` Windows fix (after four failed Qwen attempts), `.gitattributes`, cleanup of an aborted Qwen edit, the `seed-fault` skill text, verification and ablation scripts, and drafts of `docs/*.md` and this report.

**Decisions that were mine** (each has a line in `docs/DECISIONS.md`)
- What to predict, what counts as leakage, how to split, which baselines: kept the starter's (`cnt`; `casual`/`registered` and time indexes excluded; 2011 / H1 2012 / H2 2012; mean and persistence), and kept `cnt_2d_bfr` out with new evidence.
- New raw columns must fail loudly (allowlist) rather than enter silently.
- Feature `cnt_roll_28`, and keeping it.
- Hook fix applied by Opus rather than by Qwen; model switch to 9B; graded-work-only scope.
- Exercise 4 verdict (defensible, with limits); Exercise 5 verdict (deploy ridge).
- Exercise 3 verdict on the teammate audit (`docs/AUDIT.md`), and on our own batch audit: `reports/audit.jsonl` flags nothing (leakage none, temporal and baselines OK), so there was nothing to fix or dispute.
- My skill: `seed-fault`.
- Whether the answer is believable: the ridge result yes, with the persistence and bias caveats above; the Exercise 4 hgb-vs-ridge ordering, not yet.
