#!/usr/bin/env bash
# Append one row to bench/bench.jsonl. Never edits existing rows.
#
# bench.jsonl is the project's only admissible source for a performance claim,
# so rows are written by this script from real tool output — never by hand.
# Calibration rows carry the decode fields as null; B travels with every row so
# a row stays interpretable on a machine other than the one that produced it.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN=build/01_bandwidth
OUT=bench/bench.jsonl
GB="${1:-12}"
REPS="${2:-7}"
FILE="${3:-}"   # optional: measure the mmap paths over a real weight file

[ -x "$BIN" ] || { echo "build first: make" >&2; exit 1; }

if [ -n "$FILE" ]; then
  calib=$("$BIN" --gb "$GB" --reps "$REPS" --file "$FILE" --json)
else
  calib=$("$BIN" --gb "$GB" --reps "$REPS" --json)
fi

commit=$(git rev-parse --short HEAD 2>/dev/null || echo "uncommitted")
dirty=$(git status --porcelain 2>/dev/null | head -1)
[ -n "$dirty" ] && commit="${commit}-dirty"
ts=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
# grep -c prints 0 *and* exits 1 on no-match, so swallow the status only.
count=$(grep -c '"mode": "calib"' "$OUT" 2>/dev/null || true)
id=$(printf 'calib-%04d' "$(( ${count:-0} + 1 ))")

# Splice the calibration payload into the standard row schema.
python3 - "$id" "$ts" "$commit" "$calib" >> "$OUT" <<'PY'
import json, sys
id_, ts, commit, calib = sys.argv[1], sys.argv[2], sys.argv[3], json.loads(sys.argv[4])
B = calib["B_gbs"]
row = {
    "id": id_, "ts": ts, "commit": commit, "phase": "P0", "mode": "calib",
    "ctx": None, "slots": None, "quant": None, "spec_on": None,
    "raw_tok_s": None, "eff_tok_s": None, "accept_mean": None,
    "draft_gb_per_step": None,
    # For a calibration row bw_util is the loader path against the best path:
    # the P0 gate itself, expressed in the standard field.
    "bw_util_pct": round(100.0 * calib["loader_frac"], 2),
    "ttft_ms": None,
    "B_gbs": B,
    "paths_gbs": {k: v for k, v in calib.items()
                  if k not in ("kind", "gpu", "sm", "cuda_rt", "cuda_drv",
                               "buf_gib", "reps", "B_gbs", "loader_gbs",
                               "loader_frac")},
    "host": {"gpu": calib["gpu"], "sm": calib["sm"],
             "cuda_rt": calib["cuda_rt"], "cuda_drv": calib["cuda_drv"]},
    "buf_gib": calib["buf_gib"], "reps": calib["reps"],
    "notes": ("speed-of-light. bw_util_pct = registered-mmap / best path "
              "(P0 gate, needs >=90). B is the best path measured on THIS host."),
}
print(json.dumps(row))
PY

echo "appended $id to $OUT"
tail -1 "$OUT"
