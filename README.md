# qwengine

A from-scratch CUDA inference engine for **Qwen3.8-27B** on the **NVIDIA DGX Spark**
(GB10, sm_121, 128 GB unified memory). No PyTorch, no cuBLAS, no TensorRT, no
CUTLASS — every kernel, the NVFP4 quantiser, the tokeniser, the vision tower and
the HTTP server are written here, about 5,800 lines of CUDA and C++.

Vision works. The API is OpenAI-compatible. It starts with one command.

```bash
git clone https://github.com/lucasclgibson/qwengine
cd qwengine && ./start.sh
```

That builds the engine, downloads the checkpoint if you do not have it, converts
it once, and serves on `http://0.0.0.0:8000`. The only thing you need to decide
is where the weights come from, and the default already answers that.

---

## Measured performance

DGX Spark (GB10), 128 GB unified memory, driver 580.159.03, CUDA 13.0. Numbers
from `tools/bench.py`, which drives the server over HTTP exactly as a client
would. Every prompt is run once to warm and then measured.

| prompt | out tokens | end-to-end tok/s | generation tok/s | accepted/cycle |
|---|---|---|---|---|
| reasoning | 120 | 20.7 | 22.5 | 2.35 |
| code | 120 | 23.0 | 24.9 | 2.61 |
| factual | 25 | 18.8 | 26.7 | 3.11 |
| chat | 107 | 24.3 | 26.8 | 2.89 |
| **overall** | **372** | **22.2** | — | — |

- **end-to-end** is what the client sees: prefill, generation, HTTP, everything.
- **generation** is the decode loop alone, which is the number engines usually quote.
- **time to first token** is 0.40 s for a short chat prompt; prefill runs at ~178 tok/s.
- Weights on disk: **16.7 GB**. Cold start ~40 s, warm start ~4 s.

### Honesty about the target and about SGLang

The goal for this engine was **30 tok/s decode**. It does not hit that. It runs
22.5–26.8 tok/s depending on how predictable the text is, and the section below
says exactly where the missing time goes and what it would take to get it.

It was also meant to beat a well-tuned SGLang setup on the same box "by a
significant margin". Measured on the same prompts, the two are within a few
percent of each other. qwengine is **not** decisively faster than SGLang here.
A ~50% win was claimed during development and it was wrong: it compared this
engine's bare decode loop against a throughput figure quoted in someone else's
README, which is not a comparison at all. Measuring both through their own HTTP
endpoints on the same prompts is the only version of that number worth keeping.

What is true is that this reaches roughly the same throughput as a mature,
heavily-engineered stack, in ~5,800 lines with no dependencies, while also
doing vision.

---

## Why it is fast at all

Decoding one token requires reading **every weight in the model**. Nothing else
in the decode loop comes close in cost, so the speed limit is memory bandwidth,
not arithmetic:

```
measured peak bandwidth      235.6 GB/s
weights streamed per pass     14.41 GB
                             ---------
one token, no tricks           61.2 ms  =  16.3 tok/s
```

Three things buy the rest.

**NVFP4 weights (4.5 bits).** Each weight is a 4-bit float (E2M1), each block of
16 shares an FP8 scale, each tensor a single FP32 scale. Fewer bytes per weight
is directly fewer milliseconds per token. The block scales are not chosen by the
obvious formula but by searching the representable FP8 codes for the one with
the lowest squared error — worth 14% on reconstruction error, for free, offline.

Note for the curious: **GB10 has no FP4 tensor cores.** That was verified against
`ptxas`, not assumed. On this chip NVFP4 is a storage format that saves
bandwidth, and the arithmetic happens in BF16/FP32. The unsloth "NVFP4"
checkpoint is also mostly FP8 in practice (22.5 GB); the converter transcodes it
to a uniform 4.5-bit file, which is why the result is 16.7 GB and not 22.5.

**Speculative decoding with the model's own MTP head.** The checkpoint ships a
small draft head. It guesses the next few tokens, the big model checks all of
them in one batched pass, and the longest agreeing run is committed. Checking 4
tokens costs barely more than checking 1, because the pass is bound by streaming
weights, not by the handful of rows going through them.

The committed tokens are **exactly** the tokens plain greedy decoding would have
produced — speculation buys speed and changes nothing else. `test/test_spec.cu`
asserts that directly.

**A GEMV kernel written for this machine.** The decode matmul runs at **230.7
GB/s, 98% of measured peak**. Getting there was mostly about registers: they cap
how many blocks an SM can keep resident, and occupancy is what hides memory
latency. `src/gemv.cu` documents the experiments that failed as well as the ones
that worked, because the failures are the more useful record.

---

## Where the missing performance actually is

Measured, not guessed. `build/spec_budget` times each half of a speculative cycle
against the bandwidth floor it would hit if it were doing nothing but reading
weights.

At draft depth 4 one cycle is 99.8 ms and commits 2.61 tokens:

| stage | measured | floor | over |
|---|---|---|---|
| verify (batch 4) | 86.4 ms | 61.2 ms | **+25.2** |
| draft × 3 | 13.4 ms | 12.2 ms | +1.2 |

**The draft head is already optimal** — 4.4 ms against a 4.05 ms floor. All the
loss is in verify, and `nsys` splits it cleanly:

- **~16.5 ms — arithmetic in the GEMV.** At batch 4 the kernel does 208 GFLOP of
  CUDA-core FMA, which at ~20 TFLOPS is ~10 ms, plus ~6 ms of NVFP4 unpacking.
  This is *irreducible on CUDA cores*. Removing it means a tensor-core kernel
  with a small-M tile (the batch is 4 rows; the existing prefill kernel uses a
  64-row tile and would waste 16× on this shape).
- **~7 ms — the small kernels**, of which the DeltaNet recurrence is 3.7 ms and
  is itself near bandwidth. The rest is ~570 sub-microsecond launches that want
  fusing.

So the ceiling, with a perfect verify kernel and today's acceptance rate:

```
61.2 ms verify + 3 x 4.05 ms draft = 73.4 ms/cycle
2.61 tokens / 73.4 ms               = 35.6 tok/s
```

30 tok/s is reachable, but only by taking most of that 25 ms — it needs the
tensor-core small-M kernel, not tuning. Everything cheaper has been tried and
measured; the dead ends are recorded in `src/gemv.cu` next to the code they
failed to improve.

**Prefill has a bigger and easier win.** `build/tc_shapes` shows the prefill GEMM
running at 21–27% of its own memory floor (50–64 GB/s against 235). Sweeping
every tile configuration does not fix it, so the kernel itself is wrong, not its
parameters. Fixing it is worth roughly 4× on time-to-first-token.

Two things were checked and are *not* available: the lm_head has no padding to
reclaim (248,077 of 248,320 rows are real vocabulary), and a 2-bit draft head
saves 4% of cycle time but costs more than that in acceptance.

---

## Using it

### Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Explain speculative decoding briefly."}],
       "max_tokens":200}'
```

`stream: true` gives Server-Sent Events. `enable_thinking: false` turns off the
checkpoint's default `<think>` block.

### Vision

Images go in the standard OpenAI content array, as a `data:` URL:

```bash
python3 - <<'PY'
import base64, json, urllib.request
img = base64.b64encode(open("photo.jpg","rb").read()).decode()
body = json.dumps({"messages":[{"role":"user","content":[
    {"type":"text","text":"What is in this image?"},
    {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,"+img}}]}],
    "max_tokens":300}).encode()
r = urllib.request.Request("http://localhost:8000/v1/chat/completions", body,
                           {"Content-Type":"application/json"})
print(json.load(urllib.request.urlopen(r))["choices"][0]["message"]["content"])
PY
```

The full 27-layer vision tower runs here: bicubic preprocessing that matches the
reference on uint8 (resizing in float moves pixels by up to 0.24 normalised
units), merge-block patch ordering, bilinear interpolation into the learned
position table, and 3-D mrope positions for the patches spliced into the prompt.
`test/test_vision.cu` checks the tower against the reference implementation
end-to-end at 3.9e-03 relative error.

### Configuration

`./start.sh` writes a `.env` on first run:

```ini
MODEL=unsloth/Qwen3.8-27B-NVFP4   # HF repo id, or a local directory
WEIGHTS=./out/qwengine.bin        # converted engine file
PORT=8000
CTX=8192                          # KV cache ~0.5 MB per token per 1k context
DEPTH=4                           # draft depth; 1 disables speculation
NO_VISION=0                       # 1 builds text-only, saves 0.9 GB
```

`./stop.sh` stops it.

---

## Layout

```
qwen38.h          model geometry, cross-checked against the real tensor shapes
src/convert.cpp   offline converter: safetensors -> one NVFP4 file (+ tokeniser)
src/loader.cpp    O_DIRECT load, pinned staging, page-cache reclaim
src/gemv.cu       THE decode kernel. 98% of peak bandwidth
src/gemm_tc.cu    BF16 tensor-core GEMM for prefill
src/mixers.cu     GatedDeltaNet recurrence, conv1d, full attention with GQA
src/model.cu      the forward pass, CUDA graph capture, state snapshot/rollback
src/mtp.cu        the draft head
src/spec.cu       draft -> verify -> accept -> rewind
src/prefill.cu    chunked prefill, batched DeltaNet, image splicing
src/vision.cu     27-layer vision tower
src/image.cpp     preprocessing (the half most likely to be subtly wrong)
src/tokenizer.cpp byte-level BPE, NFC, Unicode categories
src/server.cpp    OpenAI-compatible HTTP + SSE
```

`make test` runs eight suites: NVFP4 round-trip, loader, GEMV against a CPU
reference, the speculation identity, the tensor-core GEMM against the decode
kernel, the vision tower against the reference, the tokeniser against
`tokenizers`, and image preprocessing against torchvision.

`make` also builds the measurement tools used above:
`spec_budget` (where a speculative cycle goes), `gemv_shapes` (per-shape and
per-batch bandwidth), `tc_shapes` (prefill GEMM against its floor),
`01_bandwidth` (the 235.6 GB/s figure everything is measured against).

Kernels carry their own history: where a comment says a change was measured and
rejected, the numbers are there. `src/gemv.cu` in particular is as much a record
of what did not work as of what did.

---

## Requirements

- DGX Spark or another GB10 (sm_121). `make ARCH=sm_XX` for anything else, untested.
- CUDA 13.0+
- ~40 GB of disk: the source checkpoint plus the 16.7 GB converted file
- Python only for the optional download and the golden-file generators

## Limitations

- One request at a time. The engine is single-stream — one decode loop owns the
  GPU and the KV cache. Batched multi-slot serving is the obvious next thing and
  is what would make concurrency worth having.
- Greedy decoding. No temperature or top-p yet.
- Text and images in, text out.
- 30 tok/s not reached; see above for exactly why.

## Licence

MIT.

---

Built by [@luc_gibson](https://x.com/luc_gibson).
