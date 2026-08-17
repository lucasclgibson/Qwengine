#!/usr/bin/env python3
"""Dump reference vision-tower inputs and outputs so the CUDA tower can be
checked against the real thing rather than against my reading of it."""
import sys, glob, struct, numpy as np
sys.path.insert(0, "/home/luc_gibson/infra/comfyui/.venv/lib/python3.12/site-packages")
import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SNAP = glob.glob("/home/luc_gibson/infra/models/hub/models--Qwen--Qwen3.8-27B/snapshots/*/")[0]
cfg = AutoConfig.from_pretrained(SNAP).vision_config
GH, GW = 4, 6                      # small grid -> 24 patches, 6 merged tokens
N = GH * GW

# Build the tower and load only the visual weights from the checkpoint.
model = Qwen3_5VisionModel._from_config(cfg).to(torch.float32).eval()
from safetensors import safe_open
import json
idx = json.load(open(SNAP + "model.safetensors.index.json"))["weight_map"]
want = {k: v for k, v in idx.items() if k.startswith("model.visual.")}
sd = {}
for shard in sorted(set(want.values())):
    with safe_open(SNAP + shard, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.visual."):
                sd[k[len("model.visual."):]] = f.get_tensor(k).to(torch.float32)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded visual weights: {len(sd)} tensors, missing={len(missing)}, unexpected={len(unexpected)}", file=sys.stderr)

pix = torch.randn(N, cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2,
                  dtype=torch.float32) * 0.5
grid = torch.tensor([[1, GH, GW]], dtype=torch.long)
with torch.no_grad():
    out = model(pix, grid)
merged = out.pooler_output.to(torch.float32).numpy()

# The tower's own helpers give us the position ids and bilinear weights, so the
# CUDA side is checked on the maths, not on my reconstruction of the indexing.
from transformers.vision_utils import (
    get_vision_position_ids, get_vision_bilinear_indices_and_weights)
pid = get_vision_position_ids(grid, cfg.spatial_merge_size).numpy().astype(np.int32)
side = int(cfg.num_position_embeddings ** 0.5)
bidx, bw = get_vision_bilinear_indices_and_weights(grid, side, cfg.spatial_merge_size)
bidx = bidx.numpy().astype(np.int32); bw = bw.numpy().astype(np.float32)

with open("golden/vision.bin", "wb") as f:
    f.write(struct.pack("<iii", N, GH, GW))
    f.write(pix.numpy().astype(np.float32).tobytes())
    f.write(pid.tobytes())
    f.write(bidx.tobytes())
    f.write(bw.tobytes())
    f.write(merged.astype(np.float32).tobytes())
print(f"wrote golden/vision.bin: N={N} merged={merged.shape}", file=sys.stderr)
