#!/usr/bin/env python3
"""Dev-only tokenizer bridge: text <-> token ids, using the checkpoint's own
tokenizer. Replaced by a native BPE later; for now it lets the engine be
driven with real text."""
import sys, os, glob
sys.path.insert(0, "/home/luc_gibson/infra/comfyui/.venv/lib/python3.12/site-packages")
from transformers import AutoTokenizer
SNAP = glob.glob("/home/luc_gibson/infra/models/hub/models--Qwen--Qwen3.8-27B/snapshots/*/")[0]
tk = AutoTokenizer.from_pretrained(SNAP)
mode = sys.argv[1]
if mode == "encode":
    print(",".join(str(i) for i in tk.encode(sys.argv[2])))
elif mode == "chat":
    msgs = [{"role": "user", "content": sys.argv[2]}]
    # tokenize=True returns a BatchEncoding (a dict), not a list — iterating it
    # yields the KEY NAMES. Render to text, then encode.
    txt = tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    print(",".join(str(i) for i in tk.encode(txt)))
elif mode == "decode":
    ids = [int(x) for x in sys.argv[2].split(",") if x.strip()]
    print(tk.decode(ids))
