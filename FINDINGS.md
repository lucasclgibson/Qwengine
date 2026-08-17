# qwengine — engineering findings

Everything below was measured on this machine: a DGX Spark (GB10, sm_121, 48 SMs,
128 GB unified LPDDR5X, 65536 registers/SM, 99 KB opt-in shared/SM), driver
580.159.03, CUDA 13.0, running Qwen3.8-27B at NVFP4.

The point of this document is that **most of it is negative results**. The
positive changes are in the git history with their numbers; what is hard to
recover later is the list of things that look obviously right and are not. Where
a number appears, a benchmark in `bench/` produced it.

---

## 1. Machine ceilings

Every ratio in this engine is quoted against these two numbers, so both were
measured rather than taken from a spec sheet.

### Memory: 236.8 GB/s for streaming reads (`bench/02_readmax.cu`)

Fifteen variants, best of seven runs each:

| variant | GB/s |
|---|---|
| `uint4` grid-stride, 256 threads | **236.8** |
| `uint4`, unrolled 2 / 4 / 8 / 16 | 235.3 / 235.4 / 234.6 / 232.8 |
| 32-byte loads (`ld.global.v4.b64`) | 234.1–236.2 |
| 32-byte loads + `L2::evict_first` | 234.4–235.5 |
| 128 / 512 threads, 8–32 blocks/SM | 235.7 / 236.5 |

**Nothing beats a plain `uint4` loop.** 87% of the 273 GB/s spec figure is
simply what this LPDDR5X part delivers to a reader. Two consequences:

- The decode floor is **60.8 ms/token** (14.41 GB of weights per pass). Decode
  cannot be made faster by reading *better*, only by reading *less*, accepting
  more tokens per verify, or removing arithmetic that isn't overlapped.
- The decode GEMV at 230.7 GB/s is at **97% of achievable**, not 98% of a
  theoretical number. It is finished.

Discovered incidentally: sm_121 accepts 256-bit vector loads
(`ld.global.v4.b64`), and the `L2::evict_first` hint is *only* legal at that
width. Neither helps here.

### Compute: ~250 TFLOP/s BF16 tensor cores (`bench/03_mmamax.cu`)

| variant | TFLOP/s |
|---|---|
| back-to-back `mma`, operands in registers | 236.8–241.6 |
| **operands reloaded from shared each step** | **249.7–250.8** |

The second row is the important one: **fragment loads from shared are free** —
the reload variant measured *higher* than registers-only. So a GEMM that is slow
is not slow because of shared-memory *reads*.

---

## 2. Where it stands

Measured with `tools/bench.py` and `bench/prof_prefill.cu`; the SGLang column is
`lmsysorg/sglang:qwen38-27b` on the same box.

| | qwengine | SGLang |
|---|---|---|
| decode | **25.5–26.2 tok/s** | 24.9–26.0 |
| prefill, ~2270 tokens | 553 tok/s | **2168** |
| prefill, ~6750 tokens | 348 tok/s | **2022** |
| chat TTFT, turn 1 (2200 tok) | 1.42 s | 1.01 s |
| chat TTFT, turns 2+ | **0.29 s** | 0.14 s |

Decode is at parity. **Prefill is ~4x behind and that is the outstanding
problem.**

Prefill at the 2048-token chunk the server actually uses divides as:

| kernel | ms | share |
|---|---|---|
| `gemm_tc` | 1995 | 54% |
| `k_attn_prefill` | 654 | 18% |
| `k_delta_chunk` | 371 | 10% |
| `gemm_tc<96>` (ragged N) | 119 | 3% |
| `k_swiglu_fused` | 132 | 4% |

---

## 3. Measurement traps

Five separate times a measurement was wrong in a way that pointed the work in the
wrong direction. These cost more time than any kernel.

**A stale server answered for an hour.** `start.sh` logged `Address already in
use` and exited into a file nobody read, leaving an older binary serving. Every
end-to-end number over that period was flat while the kernels underneath got 26%
faster. `start.sh` now refuses to start when the port is taken.

**A hand-built profiler binary went stale.** `prof_prefill` was compiled by hand
rather than through the Makefile, so it reported the *old* behaviour of a change
already made. Same failure class as the stale object files that `-MMD -MP` was
added to prevent, one level up. It is in the Makefile now.

**Counting SSE chunks is not counting tokens.** SGLang flushes several tokens per
chunk, so a per-chunk count put it at 9 tok/s instead of 25.

**Abandoning a stream leaves the server generating.** A TTFT probe that read the
first token and disconnected left the server producing into a broken pipe; the
next request queued behind it and the "decode" measurement was garbage.

**Dividing tokens by wall time charges prefill to decode**, which flatters
whichever engine cached the prompt. Decode is now measured by differencing a
24-token run against a 184-token one, which cancels prefill exactly. Prefill uses
a fresh random document every time — reusing prompts let SGLang's radix cache
answer a 2700-token prefill in 0.13 s, which is not prefill.

**An ablation can delete more than you think.** `TC_ABLATE=3` removes the
`tc_store` call, which lets the compiler dead-code the weight *fetch* as well. It
appeared to show "shared stores cost half the kernel"; it actually showed a
kernel with no weights. The fragment experiment (§5) is what exposed this.

---

## 4. What worked

| change | effect |
|---|---|
| Prefix caching (KV + DeltaNet state reuse) | chat TTFT 9.8 s → **0.29 s** |
| Never commit past the stop token | made the cache work at all (see below) |
| Shared-stride padding in `gemm_tc` | 864 → 696 ms |
| Vectorised staging (2-byte → 16-byte copies) | 604 → 434 ms |
| Software-pipelined k-loop | 376 → 279 ms |
| Warp-per-timestep attention scores | chunk 5329 → 4164 ms @ T=2048 |
| Chunked DeltaNet (state resident in shared) | 665 → 371 ms |
| Ragged-N tile ladder (128/96/64/32) | that shape 555 → 131 ms |
| Draft depth 3 → 4 | 23.7 → 24.9 tok/s |
| `dk`-split in the per-token delta rule | 6% → 48% GPU utilisation |

Four of these deserve their reasons recorded:

**Every shared row started on bank 0.** Tiles were stored at their natural stride
of `TC_BK` bf16 = 128 bytes, and shared memory is exactly 32 four-byte banks =
128 bytes. So row *r* began at bank `(r*32) % 32 = 0` and each
`load_matrix_sync` — which reads 16 rows — serialised 16 ways. This is invisible
to a tile sweep because *every* power-of-two width has the property, so every
configuration measured equally badly.

**The GEMM was copying two bytes at a time.** Both staging loops moved one bf16
per thread, turning a 128×64 tile into 8192 separate 2-byte loads and 8192 2-byte
shared stores per k-tile per block. It is not the bytes — the tile is
L2-resident — it is the instruction count.

**A 6240-wide projection was falling off the tensor-core path.** The launcher
required `N % 128 == 0` and sent anything else to the *decode* kernel, eight
tokens at a time. A comment claimed this only caught "two tiny gate
projections"; the fused `in_zab` projection is [6240, 5120], and 6240 is not a
multiple of 128, so an 18 MB matrix was re-read once per group of 8 tokens — 64
times per chunk, 48 layers deep, **~55 GB of pure waste per chunk**.

**Speculation committed tokens past the stop token.** A cycle commits 1+k tokens
at once, so replies ran past `<|im_end|>` into `<|endoftext|>`, and those extra
tokens were fed to the model. The engine's state then matched a sequence no
client would ever send back, so the prefix cache missed on every naturally-ended
reply. This was invisible because every benchmark capped `max_tokens` at 64,
truncating replies before they ended — the cache *looked* like it worked while in
real use it would have missed every turn.

**The tile optimum moved three times**, which is the most transferable lesson
here. A sweep is only valid for the kernel it was run against:

| | warp 2×2 | warp 2×4 |
|---|---|---|
| original (2-byte staging) | 736 ms | 603 ms |
| vectorised staging | 376 | 442 |
| + software-pipelined k-loop | 295 | **279** |

---

## 5. What failed

All of these are implemented, measured, and kept in the tree with their numbers.

### Fragment-order weights (`src/gemm_frag.cu`) — 339 ms vs 261

Store NVFP4 in exactly the order a `wmma` fragment wants, so a lane reads its
eight values with one 4-byte load and dequantises straight into `.x[]` — no
shared staging of weights at all. The mapping was probed from hardware, not
assumed: for a 16×16 `matrix_a` row-major fragment, lane L holds rows
`{L/4, L/4+8}` and columns `{2(L%4), +1, +8, +9}`. Density is identical (144
bytes per 16×16 tile = 4.5 bits). Output is **bit-identical** (rel-err 0.000e+00).

It is still slower. **Shared staging is not overhead — it is a latency-hiding
buffer shared by every warp in the block.** Removing it puts a *global* load in
the innermost loop where `gemm_tc` has a shared one. Pipelining those global
loads to hide the latency is worse again (509 ms): the prefetch registers spill.

Tile ordering does matter and is kept: k-major beats n-major (339 vs 364),
because a block's n-tiles for one k step must be adjacent — ordered n-major they
sit `ktiles*144` bytes apart, 45 KB on a K=5120 matrix.

### Tensor cores for the decode verify pass — 275 ms vs 75

A `BT=16` instantiation wastes only 4× on rows against a ~10× arithmetic rate,
which sounds like a win. It leaves 2 warps per block and is latency-starved. At
small M the dequant and staging costs are per-weight and do not amortise.

### Attention, three attempts

| attempt | result |
|---|---|
| query tile 8 → 16 → 32 → 64 (queries in shared) | 3407 → 3513 → 4025 → 4751 ms |
| 8 lanes per query instead of 32 (fewer shuffles) | 4187 ms |
| baseline (tile 8, queries in registers) | **3407 ms** |

Attention is *not* memory-bound on K/V re-reads, as I assumed — it is
instruction-bound on the warp reduction behind every (query, key) dot. Widening
the tile adds shared traffic and shuffles faster than it saves key reads;
lane-groups trade shuffles for uncoalesced key reads. Only tensor cores remove
the reduction, by making QKᵀ a matmul.

### GEMM, everything else

| attempt | result |
|---|---|
| `cp.async` for the activation tile | 282 ms (neutral) |
| activation fragments straight from global | 420 ms — uncoalesced, a fragment's 16 rows are 10 KB apart |
| 8-element (16-byte) shared stores | 423 ms — halves store width, doubles store count |
| deeper K tile (BK=128) | 316 ms |
| wider output tile (BN=256) | 300 ms |
| wider token tile (BT=256) | 403 ms |
| more blocks/SM via BK=32 (4 blocks) | 336 ms |
| higher MMA:load ratio (warp 4×4) | 320 ms |
| shared pad 8 / 16 / 24 / 32 | 290 / 282 / 282 / 283 — **bank conflicts are not the cap**, so XOR swizzling would buy nothing |

### Prefill chunk size

A single-chunk measurement argues for a smaller chunk (486 tok/s at 256 tokens
vs 340 at 2048, because attention within a chunk is quadratic). End-to-end
refutes it (344 tok/s at 2048 vs 322 at 256): shortening chunks does not remove
the quadratic work, it only moves it — every later chunk still attends over all
the history before it.

### Other closed avenues

- **lm_head has no padding to reclaim** — 248,077 of 248,320 rows are real
  vocabulary (`bench/vocab_probe.cu`).
- **A 2-bit draft head** saves 4% of cycle time and loses more than that in
  acceptance (2.67 → 2.56 accepted).
- **Draft depth beyond 5** buys nothing: depth 5 and 6 both accept 2.86.
- **The draft head is already optimal** — 4.4 ms against a 4.05 ms bandwidth
  floor (`bench/spec_budget.cu`). All the loss in a speculative cycle is in
  verify.

---

## 6. What is left, and what it needs

At depth 4 a cycle is 96 ms and commits 2.61 tokens. Verify is 84.9 ms against a
61.2 ms floor, so **if verify were bandwidth-bound, depth 5 would give 2.86
tokens per 83.9 ms = 34 tok/s** — and a candidate tree, whose extra rows would
then be nearly free, considerably more. The whole decode question is whether the
batched GEMV can be made to cost what its bytes cost.

It currently cannot, and the ablations bound the prize: at B=4, dequant costs
4.1 ms and the batch FMAs 5.5 ms, so removing both lands at ~66 ms against the
61.9 floor. That route caps decode near 29 tok/s. Beating that needs the verify
cost to stop scaling with batch at all — which is tensor cores at small M, which
loses today for the reasons in §5.

For prefill, the remaining work is genuinely three specialist kernels:

1. **A flash-attention kernel** with online softmax and tensor-core QKᵀ (18% of
   prefill, and the reduction it removes is the measured bottleneck).
2. **A multi-stage `cp.async` GEMM** driven by `mbarrier`, with warp
   specialisation — keeping the shared buffer (which §5 shows is load-bearing)
   and adding asynchrony to the chain, rather than trying to remove the buffer.
3. **A chunked WY DeltaNet**, turning the recurrence into GEMMs.

### The thing that would most change the odds

`ncu` is blocked (`ERR_NVGPUCTRPERM`), so there are no hardware counters. Every
remaining hypothesis has been tested by ablation, and ablation has become
ambiguous — §3 records one that deleted more than intended and sent the work
down a blind alley. The GEMM moves 81 GB per pass at 288 GB/s combined traffic,
which is above the DRAM ceiling and therefore partly L2-served, and *which pipe
is saturated cannot be determined from timings alone*.

Enabling counters:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf
sudo update-initramfs -u && sudo reboot
```

With stall-reason and pipe-utilisation counters, the next step stops being a
guess.

---

## 7. Instruments

All of these ship with the engine and produced the numbers above.

| tool | question it answers |
|---|---|
| `bench/01_bandwidth.cu` | what is streaming bandwidth on this box |
| `bench/02_readmax.cu` | is 235.6 GB/s really the read ceiling (yes) |
| `bench/03_mmamax.cu` | what is the BF16 tensor-core peak (~250 TFLOP/s) |
| `bench/spec_budget.cu` | where does a speculative cycle go, against its floor |
| `bench/gemv_shapes.cu` | per-shape and per-batch decode bandwidth |
| `bench/tc_shapes.cu` | the prefill GEMM against its memory floor |
| `bench/verify_kernel.cu` | CUDA cores vs tensor cores for the verify pass |
| `bench/frag_bench.cu` | fragment-order weights vs shared staging |
| `bench/prof_prefill.cu` | one prefill chunk, for `nsys` |
| `bench/vocab_probe.cu` | is any of lm_head padding (no) |
| `tools/bench.py` | end-to-end tok/s over HTTP, as a client sees it |

`make test` runs nine suites against reference implementations: NVFP4
round-trip, loader, GEMV against a CPU reference, the speculation identity, the
tensor-core GEMM against the decode kernel, the delta rule against its
per-token form, the vision tower, the tokeniser against `tokenizers`, and image
preprocessing against torchvision.
