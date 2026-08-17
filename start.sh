#!/usr/bin/env bash
# Bring qwengine up: fetch the model if needed, convert it once, then serve.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .env ] || cp .env.sample .env
set -a; . ./.env; set +a

MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
WEIGHTS="${WEIGHTS:-./out/qwengine.bin}"
PORT="${PORT:-8000}"; CTX="${CTX:-8192}"; DEPTH="${DEPTH:-4}"
NO_VISION="${NO_VISION:-0}"

command -v nvcc >/dev/null || { echo "nvcc not found -- install the CUDA toolkit"; exit 1; }

echo "==> building"
make -j"$(nproc)" >/dev/null

# 1. locate or fetch the checkpoint
if [ -d "$MODEL" ]; then
  SRC="$MODEL"
else
  echo "==> fetching $MODEL (first run only)"
  command -v hf >/dev/null || pip install -q --user huggingface_hub
  # Recent huggingface_hub prints "path=/the/snapshot/dir" on the last line;
  # older versions print the bare path. Accept either, or the whole thing is
  # handed to the converter as a directory name that does not exist.
  SRC="$(hf download "$MODEL" | tail -1)"
  SRC="${SRC#path=}"
fi
[ -d "$SRC" ] || { echo "checkpoint directory not found: $SRC"; exit 1; }
echo "    checkpoint: $SRC"

# 2. convert once
if [ ! -f "$WEIGHTS" ]; then
  echo "==> converting to $WEIGHTS (one time, a few minutes)"
  mkdir -p "$(dirname "$WEIGHTS")"
  VIS=""; [ "$NO_VISION" = "1" ] && VIS="--no-vision"
  ./build/convert "$SRC" "$WEIGHTS" $VIS
else
  echo "    weights already converted: $WEIGHTS"
fi

# 3. serve
echo "==> serving on http://0.0.0.0:$PORT"
nohup ./build/qwengine-serve "$WEIGHTS" --port "$PORT" --ctx "$CTX" --depth "$DEPTH" \
  > qwengine.log 2>&1 &
echo $! > qwengine.pid
for _ in $(seq 1 120); do
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "    ready:  curl http://127.0.0.1:$PORT/v1/models"
    exit 0
  fi
  sleep 2
done
echo "server did not come up; see qwengine.log"; exit 1
