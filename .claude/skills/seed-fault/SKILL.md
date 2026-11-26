---
name: seed-fault
description: Deliberately break the pipeline in one described way, prove a test goes red and the leakage hook fires, record the evidence in docs/SEEDED_FAULTS.md, then restore and prove green. Use when asked to seed, inject, or demonstrate a fault.
allowed-tools: Read, Edit, Bash(python -m pytest*), Bash(make*), Bash(git diff*), Bash(git status*)
---

# Current state (must be green before you break anything)

!`python -m pytest tests/ -q 2>&1 | tail -1`

# Seeding fault: $ARGUMENTS

If the line above is not "N passed", stop: you cannot attribute a red test to
your fault on a suite that is already red.

1. **Name the fault in one sentence** and predict which test should catch it.
   If you predict nothing will catch it, say so -- that is a finding.
2. **Make exactly one change** with the Edit tool, under `src/`, so the
   PostToolUse hook runs. Show the diff (`git diff src/`).
3. **Capture, verbatim**: the hook's stderr (it arrives as a system message
   after your edit), then `make test` output (last 15 lines).
   Do not summarise either. If the hook printed nothing, say so.
4. **Restore** with Edit (not `git checkout`), so the fix is also an edit the
   hook sees. Count your turns from the hook message to green.
5. **Verify green**: `make test` last line, and `git diff src/` is empty.
6. **Append to `docs/SEEDED_FAULTS.md`** one section: fault, exact diff,
   predicted vs actual catching test, hook stderr, `make test` output,
   turns to fix.

Never touch `data/raw/` or `tests/` to make a fault "pass". A fault that no
test catches is reported as uncaught, not hidden.
