#!/usr/bin/env python3
"""Run the reference image processor on a fixed image and dump what it feeds
the vision tower, so the C++ preprocessing can be checked against it."""
import sys, glob, struct, numpy as np
sys.path.insert(0, "/home/luc_gibson/infra/comfyui/.venv/lib/python3.12/site-packages")
from transformers import AutoProcessor
from PIL import Image

SNAP = glob.glob("/home/luc_gibson/infra/models/hub/models--Qwen--Qwen3.8-27B/snapshots/*/")[0]
proc = AutoProcessor.from_pretrained(SNAP)
ip = proc.image_processor
print("processor:", type(ip).__name__, file=sys.stderr)

rng = np.random.default_rng(7)
H, W = 96, 140
arr = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
img = Image.fromarray(arr, "RGB")
out = ip(images=[img], return_tensors="np")
pix = np.asarray(out["pixel_values"], dtype=np.float32)
grid = np.asarray(out["image_grid_thw"], dtype=np.int64)
print("pixel_values", pix.shape, "grid", grid.tolist(), file=sys.stderr)

with open("golden/image.bin", "wb") as f:
    f.write(struct.pack("<iiiii", H, W, int(grid[0][1]), int(grid[0][2]), pix.shape[0]))
    f.write(arr.tobytes())
    f.write(pix.reshape(-1).tobytes())
print("wrote golden/image.bin", file=sys.stderr)
