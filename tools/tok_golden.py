#!/usr/bin/env python3
"""Encode a stress corpus with the reference tokenizer so the native one can be
checked against it: ASCII, CJK, emoji, code, whitespace pathologies, specials."""
import sys, glob, json
sys.path.insert(0, "/home/luc_gibson/infra/comfyui/.venv/lib/python3.12/site-packages")
from transformers import AutoTokenizer
SNAP = glob.glob("/home/luc_gibson/infra/models/hub/models--Qwen--Qwen3.8-27B/snapshots/*/")[0]
tk = AutoTokenizer.from_pretrained(SNAP)
cases = [
 "Hello, world!", "The capital of France is Paris.", "", " ", "  ", "\n", "\n\n",
 "  leading and trailing  ", "a\tb\tc", "don't can't won't I'll we've he'd it's",
 "DON'T CAN'T", "1234567890", "3.14159", "-42", "1,000,000",
 "def f(x):\n    return x*2\n", "for (int i=0;i<10;++i) { a[i] = b[i]; }",
 "SELECT * FROM t WHERE x > 1;", "https://example.com/a?b=c&d=e",
 "你好世界", "これはテストです", "안녕하세요", "Привет мир", "مرحبا بالعالم",
 "שלום עולם", "नमस्ते दुनिया", "Ελληνικά", "🚀🔥💡", "café naïve résumé",
 "é vs é", "a" * 200, "!@#$%^&*()", "<|im_start|>user\nhi<|im_end|>",
 "mixed 中文 and English 123 !!", "\r\n\r\n", "tab\there", "emoji 👨‍👩‍👧‍👦 family",
 "Ω≈ç√∫˜µ≤≥÷", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "０１２３", "ｆｕｌｌｗｉｄｔｈ",
]
cases += [f"number {i} test" for i in range(40)]
cases += [tk.apply_chat_template([{"role":"user","content":c}], tokenize=False,
          add_generation_prompt=True) for c in cases[:10]]
out = [{"text": c, "ids": tk.encode(c), "dec": tk.decode(tk.encode(c))} for c in cases]
json.dump(out, open("golden/tokenizer.json", "w"))
print(f"wrote {len(out)} cases", file=sys.stderr)
