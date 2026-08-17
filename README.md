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
| reasoning | 120 | 20.5 | 22.5 | 2.35 |
| code | 120 | 23.2 | 24.9 | 2.61 |
| factual | 25 | 19.9 | 26.6 | 3.11 |
| chat | 107 | 25.4 | 27.4 | 2.97 |
| **overall** | **372** | **22.5** | — | — |

- **end-to-end** is what the client sees: prefill, generation, HTTP, everything.
- **generation** is the decode loop alone, which is the number engines usually quote.
- **time to first token** is 0.31 s for a short chat prompt. Prefill runs at
  ~250 tok/s on a 900-token prompt, ~215 tok/s on a 2800-token one.
- Weights on disk: **16.7 GB**. Cold start ~40 s, warm start ~4 s.

### Where this actually stands against SGLang

Both engines run on this machine and are driven by the same script over their
own HTTP endpoints. Two methodology notes, because the naive versions of these
measurements are all wrong in SGLang's favour or ours:

- **Decode is measured by differencing.** Each prompt is run to 24 tokens and to
  184, and the rate is the difference over the difference, which cancels prefill
  exactly. Dividing total tokens by total wall time (the obvious version)
  charges prefill to decode and flatters whichever engine caches prompts.
- **Prefill uses a fresh random document every time**, so neither engine's
  prefix cache can serve it. Re-using prompts across runs let SGLang's radix
  cache answer a 2700-token prefill in 0.13 s, which is not prefill at all.

| | qwengine | SGLang | |
|---|---|---|---|
| decode, reasoning | **23.9** | 19.2 | tok/s |
| decode, code | 33.5 | **36.2** | tok/s |
| decode, chat | **28.0** | 27.4 | tok/s |
| decode, essay | **22.2** | 21.2 | tok/s |
| **decode overall** | **26.2** | 26.0 | tok/s |
| prefill, 2267 tokens | 551 | **2157** | tok/s |
| prefill, 6775 tokens | 346 | **1937** | tok/s |

**Decode is at parity**, winning three prompts of four. **Prefill is still 5-6x
behind**, and that is the honest headline: this engine is competitive at
generating tokens and not yet competitive at reading them.

In a chat client the prefix cache hides most of that, because only the first
message of a conversation pays full prefill (see below) -- but "hidden" is not
"fixed", and a long paste still costs real seconds.

What it would take to close the rest is known and measured, not mysterious.
Prefill at a 2048-token chunk now divides as GEMM 54%, attention 18%,
DeltaNet 10%, and each has a specific answer:

**The GEMM is bound by SHARED-MEMORY WRITES.** Ablating just the stores takes
the whole model pass from 282 ms to 144, and 47 TFLOP/s to 93 -- against a 250
TFLOP/s machine peak. Everything else has been measured away: not the global
loads (ablating them changes nothing now the k-loop is pipelined), not bank
conflicts (pads of 8/16/24/32 all within 3%), not occupancy (4 blocks/SM is
*worse*), not the MMA-to-load ratio. Two cheap escapes were tried and both lose:
halving the store width doubles the store count (423 ms), and loading activation
fragments straight from global is uncoalesced because a fragment's 16 rows are
10 KB apart (420 ms). The real fix is to stop staging weights in shared at all
-- dequantise NVFP4 directly into `mma.sync` fragment registers, whose layout is
documented where `wmma`'s is opaque. That is a different kernel, not a tuning of
this one.

**Attention needs tensor cores**, not a bigger tile. It is instruction-bound on
the warp reduction behind every (query, key) dot. Widening the query tile makes
it worse, not better (tile 8 = 3407 ms, tile 32 = 4025), and so does splitting
the warp into lane-groups per query (4187 ms), because that trades shuffles for
uncoalesced key reads. Flash attention removes the reductions entirely by making
QK-transpose a matmul.

**DeltaNet is the one already taken.** See below.

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

### Time to first token, and the prefix cache

In a chat client this was the worst thing about the engine, and it was not a
kernel problem. Open WebUI re-sends the whole conversation on every message, and
qwengine re-read all of it every time. Time to first token grew without limit:
about 10 seconds on a 2200-token history, 30 seconds on 5400.

The engine now remembers the exact token sequence its state reflects — the last
prompt plus every token the model consumed answering it — and when the next
request begins with that sequence, it reads only the new tail. Turn 2 onwards
costs the same as turn 1 of an empty chat:

| turn | before | after | SGLang |
|---|---|---|---|
| 1 (cold, 2200 tokens) | 9.8 s | 9.8 s | 1.0 s |
| 2 | 9.9 s | **0.40 s** | 0.14 s |
| 5 | 10.3 s | **0.40 s** | 0.14 s |

Only a whole-cache match counts. The KV cache could be truncated anywhere, but
the DeltaNet recurrence cannot: its state is a running summary with nothing to
roll back to, so the only position it can resume from is the one it is already
at. An edited message or a regeneration falls back to a full read.

Two things had to be true for this to work at all, and the second was a bug:

- Every token in the generated run really has been fed to the model — `spec_step`
  advances the position by exactly the number of tokens it commits — so prompt
  plus output is precisely what the state has seen.
- **The chat template has to replay history exactly as it was generated.** The
  assistant turn is opened with a `<think>` block, and that block was being
  dropped when the same turn was later re-rendered as history. Consecutive
  prompts therefore stopped agreeing *before the first reply*, and no cache
  could ever hit. Rendering history with the opener it was generated with is
  both more faithful and what makes the cache work.

Cold prefill is still about 9× slower than SGLang and is the largest remaining
gap in the engine. The section below says where that time goes.

### Prefill throughput

Prefill was ~178 tok/s and is now ~250. Two things were wrong, and both are
worth describing because neither showed up in a tile-size sweep.

**Every shared-memory row started on bank 0.** The tiles are stored row-major at
their natural stride of `TC_BK` bf16 = 128 bytes, and shared memory is exactly
32 four-byte banks = 128 bytes wide. So row *r* began at bank `(r*32) % 32 = 0`,
and each `wmma::load_matrix_sync` — which reads 16 rows at once — serialised 16
ways. This is invisible to a sweep because *every* power-of-two tile width has
the property, so every configuration measured equally badly. Padding the stride
by 8 elements (the smallest pad wmma allows for bf16) makes rows land on banks
`4r % 32`, turning a 16-way conflict into a 2-way one. With the conflict gone
the tile optimum moved too, from a 64-token tile to a 128-token one.

**A 6240-wide projection was falling off the tensor-core path entirely.** The
launcher required `N % 128 == 0` and sent anything else to the *decode* kernel,
eight prompt tokens at a time. The comment claimed this only caught two tiny
gate projections. It did not: the fused `in_zab` projection is [6240, 5120], and
6240 is not a multiple of 128, so an 18 MB matrix was re-read once per group of
8 tokens — 64 times for a 512-token chunk, across 48 layers, about **55 GB of
pure waste per chunk**. It was a quarter of all prefill time. Every N in this
model is a multiple of 32, so the tile width is now chosen per shape (128, 96,
64 or 32, widest that divides) and the ragged case costs 131 ms per chunk
instead of 555.

What is left, per 512-token chunk:

| | ms | |
|---|---|---|
| `gemm_tc` | 999 | 56% — at 23 TFLOP/s; each warp does one 32×32 tile, so fragment loads and MMAs are 1:1. A wider per-warp tile would amortise the loads. |
| `k_delta_step` | 432 | 24% — the DeltaNet recurrence, one step per token per layer, re-reading 3.1 MB of state each time. Needs a chunked formulation to batch it. |
| `gemm_tc<96>` | 131 | 7% — the ragged projection above |
| `k_attn_decode` | 122 | 7% |

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
