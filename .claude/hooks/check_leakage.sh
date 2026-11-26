#!/usr/bin/env bash
# PostToolUse hook: re-run the leakage tests whenever src/ changes.
#
# This runs AFTER the edit has landed, so it cannot prevent the write. What
# exit 2 does is send stderr back to Claude as a system message, so the
# failure is in front of the agent immediately and it cannot carry on as if
# the edit were fine. To make "done" itself conditional on green tests, add
# a Stop hook that runs `make test` and exits 2 on failure (check
# stop_hook_active in the payload to avoid looping).
#
# Windows: under Git Bash `python3` is usually the Microsoft Store stub. The
# starter hook then parsed an empty path and exited 0 on every edit -- a
# guard that never fired. We prefer the project venv, normalise backslash
# paths, and fail loudly (exit 2) if the payload cannot be parsed.
set -uo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-.}"
PY=""
for c in "$ROOT/.venv/bin/python" "$ROOT/.venv/Scripts/python.exe" python3 python; do
  if "$c" -c 'import sys' >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "check_leakage hook: no working Python interpreter found; leakage tests NOT run." >&2
  exit 2
fi

payload=$(cat)
if ! path=$(printf '%s' "$payload" | "$PY" -c \
  'import json,sys; d=json.loads(sys.stdin.buffer.read().decode("utf-8-sig")); print(d.get("tool_input",{}).get("file_path",""))'); then
  echo "check_leakage hook: could not parse hook payload; leakage tests NOT run." >&2
  exit 2
fi
path=${path//\\//}   # C:\repo\src\x.py -> C:/repo/src/x.py

case "$path" in
  *src/*.py) ;;
  *) exit 0 ;;
esac

cd "$ROOT" || exit 0
if ! out=$("$PY" -m pytest tests/test_leakage.py -q 2>&1); then
  echo "Leakage tests failed after editing $path:" >&2
  echo "$out" | tail -20 >&2
  exit 2
fi
exit 0
