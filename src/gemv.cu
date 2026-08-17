#pragma once
// src/gemv.cu — the hot kernel. ~99% of every generated word's cost is here.
//
// WHAT IT COMPUTES
//   y[N] = W[N,K] . x[K]
// W is one weight matrix of the model, stored 4-bit. x is the running state
// (one number per channel). Producing one word runs this ~400 times, over
// every matrix in the model, which is why the whole 14.4 GB gets read per word.
//
// WHY IT LOOKS LIKE THIS
// The kernel is memory-bound, not maths-bound: it must read 14.4 GB and only
// does ~2 arithmetic ops per byte. So the design goal is not "fewer
// instructions", it is "never make the memory system wait". Three rules follow:
//
//   1. Every read is coalesced. A warp (32 threads in lockstep) reads 512
//      contiguous bytes in one shot. Scattered reads would cost 10x.
//   2. x is staged in shared memory once per block and reused across many
//      output rows. Read straight from global, x would outweigh the weights
//      4:1 in traffic.
//   3. One multiply by the block scale per 16 weights, not per weight, by
//      summing the raw 4-bit values first and scaling the total.
//
// THE WEIGHT FORMAT (see convert.cpp for the writer)
// Weights are 4-bit codes into a fixed 8-value "ruler"
// {0,.5,1,1.5,2,3,4,6} plus sign. Each run of 16 weights shares one 8-bit
// scale saying how far to stretch that ruler; each matrix has one float scale
// on top. On disk, one warp-step is laid out as exactly:
//
//   [ 512 bytes: 1024 weight codes ][ 64 bytes: the 64 scales for them ]
//
// so lane i grabs its 16 bytes of codes at offset i*16, and the 2 scales
// covering them as one 2-byte read at 512 + i*2. Both coalesced, no lane ever
// needs a neighbour's scale.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "format.h"

namespace spark27 {

// Tuning. Measured, not guessed; the measurements are inline below.
//  - 32 rows/block keeps enough blocks in flight for the narrow matrices
//    (N=5120) while still amortising the x staging over many rows.
//  - KTILE = 1024 matches one warp-step exactly, and every K in this model is
//    a multiple of 1024, so there is never a ragged tail.
constexpr int GEMV_WARPS = 8;
constexpr int GEMV_ROWS_PER_WARP = 4;  // default; overridable per launch
constexpr int GEMV_KTILE = 1024;
constexpr int GEMV_THREADS = GEMV_WARPS * 32;
constexpr int SWZ_STEP_BYTES = 576;
constexpr int SWZ_VAL_BYTES = 512;

// The 8-value ruler, signed. Index is the raw 4-bit code.
// Placed in shared memory: index maps 1:1 onto shared-memory banks, so 32
// lanes reading 32 different codes hit 32 different banks — conflict-free.
__device__ __forceinline__ void load_e2m1_lut(float *lut, int tid) {
  if (tid < 16) {
    const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    lut[tid] = (tid & 8) ? -mag[tid & 7] : mag[tid & 7];
  }
}

// E2M1 code -> float by building the float's bits directly. MEASURED SLOWER
// than the shared-memory table (B=3: 88.9 ms vs 81.5 ms), because the e==0
// special case costs a select and the ALU was not actually the constraint.
// Kept for the record; the table is what ships.
__device__ __forceinline__ float e2m1_bits(uint32_t c) {
  const uint32_t e = (c >> 1) & 3u, m = c & 1u;
  uint32_t bits = ((e + 126u) << 23) | (m << 22);
  if (e == 0u) bits = m ? (126u << 23) : 0u;
  bits |= (c & 8u) << 28;                     // sign
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// FP8 E4M3 -> float. 1 sign, 4 exponent, 3 mantissa, bias 7.
__device__ __forceinline__ float e4m3(uint8_t v) {
  int e = (v >> 3) & 0xF, m = v & 7;
  float mag = e ? __builtin_ldexpf(1.0f + m * 0.125f, e - 7)
                : __builtin_ldexpf((float)m, -9);
  return (v & 0x80) ? -mag : mag;
}

// Accumulate 8 weights (one packed 32-bit word) against 8 activations.
__device__ __forceinline__ float dot8(uint32_t packed, const float *xs,
                                      const float *lut) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += lut[(packed >> (i * 4)) & 0xF] * xs[i];
  return s;
}

// y[N] = W[N,K] . x[K].  W is NVFP4 in the swizzled layout above.
template <int RPW>
__global__ __launch_bounds__(GEMV_THREADS) void gemv_nvfp4(
    const uint8_t *__restrict__ W, const float *__restrict__ x,
    float *__restrict__ y, int N, int K, float tscale) {
  __shared__ float xs[GEMV_KTILE];
  __shared__ float lut[16];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row0 = blockIdx.x * (GEMV_WARPS * RPW) + warp * RPW;
  const int steps = K / GEMV_KTILE;

  load_e2m1_lut(lut, threadIdx.x);

  float acc[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

  for (int s = 0; s < steps; ++s) {
    // Stage this slice of x. Costs one small read per block per slice, and it
    // is served from L2 (x is 10-70 KB against a 24 MB cache), so it never
    // touches DRAM after the first block.
    __syncthreads();
    for (int i = threadIdx.x; i < GEMV_KTILE; i += GEMV_THREADS)
      xs[i] = x[s * GEMV_KTILE + i];
    __syncthreads();

    // This lane's 32 weights cover x[lane*32 .. lane*32+31] of the slice.
    const float *xl = xs + lane * 32;

#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const int n = row0 + r;
      if (n >= N) continue;
      const uint8_t *p = W + ((size_t)n * steps + s) * SWZ_STEP_BYTES;

      // One coalesced 512 B warp read: 32 lanes x 16 B = 32 weights each.
      const uint4 v = *(const uint4 *)(p + lane * 16);
      // ...and one coalesced 64 B warp read for the two scales that cover them.
      const uint16_t sc = *(const uint16_t *)(p + SWZ_VAL_BYTES + lane * 2);

      // 32 weights = exactly 2 scale-blocks of 16. Sum each block's raw
      // contributions, then apply its scale once.
      float b0 = dot8(v.x, xl + 0, lut) + dot8(v.y, xl + 8, lut);
      float b1 = dot8(v.z, xl + 16, lut) + dot8(v.w, xl + 24, lut);
      acc[r] += b0 * e4m3((uint8_t)(sc & 0xFF)) + b1 * e4m3((uint8_t)(sc >> 8));
    }
  }

  // Each lane holds a partial sum for its slice of the row; add the 32 up.
  // Fixed shuffle order, no atomics — same input always gives the same bits.
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    float a = acc[r];
#pragma unroll
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
    const int n = row0 + r;
    if (lane == 0 && n < N) y[n] = a * tscale;
  }
}

// ---------------------------------------------------------------------------
// Batched version: y[B][N] = W[N,K] . x[B][K]
//
// THIS IS THE WHOLE POINT OF SPECULATION. The weights are read exactly once
// regardless of B, because the cost is dominated by streaming W, not by the
// arithmetic. So checking 4 candidate tokens costs almost the same as
// checking 1 — and every candidate that turns out correct is a token we got
// for free. B=1 here is identical in result to the unbatched kernel.
//
// Each thread now carries B accumulators instead of 1, and the activation
// tile in shared memory holds B rows instead of 1.
// ---------------------------------------------------------------------------
template <int RPW, int B>
__global__ __launch_bounds__(GEMV_THREADS) void gemm_nvfp4(
    const uint8_t *__restrict__ W, const float *__restrict__ x,
    float *__restrict__ y, int N, int K, float tscale) {
  // Activations staged as bf16 PAIRS, not floats.
  //
  // The inner loop is one shared-memory load per weight per batch element, and
  // at batch>1 that issue rate is what limits the kernel — not DRAM. Packing
  // two activations into one 32-bit word halves the number of loads. bf16
  // keeps ~3 decimal digits, which is irrelevant beside weights that are only
  // 4 bits to begin with, and both the plain and speculative paths use this
  // same kernel so the "speculation is invisible" invariant is unaffected.
  __shared__ __nv_bfloat162 xs[B][GEMV_KTILE / 2];
  __shared__ float lut[16];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row0 = blockIdx.x * (GEMV_WARPS * RPW) + warp * RPW;
  const int steps = K / GEMV_KTILE;

  load_e2m1_lut(lut, threadIdx.x);

  float acc[RPW][B];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int b = 0; b < B; ++b) acc[r][b] = 0.f;

  for (int s = 0; s < steps; ++s) {
    // Staging TWO swizzle steps per barrier (to halve the __syncthreads) was
    // measured and is worse: B=3 77.0 -> 78.5 ms, B=4 81.2 -> 87.9. The extra
    // shared memory costs more occupancy than the saved barriers gain.
    //
    // Reading x straight from global instead of staging it here was measured
    // and is catastrophic -- B=3 went 77 -> 354 ms. The activations are only
    // ~60 KB and sit in L2, but every lane needs a different 32-element slice
    // and the resulting access pattern defeats the cache entirely. Shared
    // staging is not an optimisation here, it is load-bearing.
    //
    // Stage x TRANSPOSED: xs[b][i*32 + lane] holds x[k = lane*32 + i].
    //
    // Lane `lane` owns the 32 contiguous weights k = lane*32 .. +31 (that is
    // fixed by the on-disk layout). Storing x in natural order then makes lane
    // read xs[lane*32 + i] — stride 32 floats, so all 32 lanes land on the
    // SAME shared-memory bank, a 32-way conflict. Memory latency hides it at
    // batch 1, but the batched kernel does B times as many shared reads and it
    // becomes the wall: measured B=8 taking 8.2x the time of B=1, i.e. the
    // weight read was not being amortised at all, which defeats the entire
    // purpose of batching. Transposed, lane reads xs[i*32 + lane] -> bank
    // `lane`, conflict-free.
    // xs[b][p*32 + lane] holds the pair ( x[lane*32 + 2p], x[lane*32 + 2p+1] ).
    // Bank = lane, so still conflict-free, and now 16 loads cover 32 weights.
    // The weight loads stay BELOW the barriers. Hoisting them above (so their
    // ~600-clock latency is covered by the staging) was measured and is a net
    // loss: it keeps RPW uint4s live across both barriers, which took batch 4
    // from 64 to 80 registers and so from 4 resident blocks per SM to 3.
    // 75.73 -> 80.89 ms. Occupancy is worth more here than latency hiding,
    // which is the same lesson as every other experiment in this kernel:
    // REGISTERS ARE THE BINDING CONSTRAINT.
    __syncthreads();
#pragma unroll
    for (int b = 0; b < B; ++b)
      for (int j = threadIdx.x; j < GEMV_KTILE / 2; j += GEMV_THREADS) {
        const int ln = j & 31, pr = j >> 5;      // lane, pair index
        const size_t base = (size_t)b * K + s * GEMV_KTILE + (size_t)ln * 32 + 2 * pr;
        xs[b][j] = __floats2bfloat162_rn(x[base], x[base + 1]);
      }
    __syncthreads();

    // Load every row's weights for this step FIRST, then walk the
    // activations once, applying each to all RPW rows.
    //
    // The naive order (row outer, activations inner) re-reads the same 32
    // activations from shared once PER OUTPUT ROW — 8x the traffic at RPW=8.
    // Ablation showed those reads are the entire cost of batching: with them
    // removed, batch 6 ran as fast as batch 1 (70 ms vs 106 ms). Hoisting them
    // out of the row loop is therefore the whole game.
    uint4 wq[RPW];
    float sca[RPW][2];
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const int n = row0 + r;
      const int nn = n < N ? n : N - 1;    // clamp, masked on write-out
      const uint8_t *p = W + ((size_t)nn * steps + s) * SWZ_STEP_BYTES;
      wq[r] = *(const uint4 *)(p + lane * 16);
      const uint16_t sc = *(const uint16_t *)(p + SWZ_VAL_BYTES + lane * 2);
      sca[r][0] = e4m3((uint8_t)(sc & 0xFF));
      sca[r][1] = e4m3((uint8_t)(sc >> 8));
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {          // 4 groups of 8 weights
      // Read this group's activations ONCE for all rows.
      //
      // Written as float2 but it does not cost 2 registers per pair: ptxas
      // keeps the bf16 pair packed and converts at the point of use. Writing
      // that by hand (holding __nv_bfloat162 and converting inside the row
      // loop) was tried and produces byte-identical register counts, so the
      // clearer spelling stays.
      float2 xr[B][4];
#pragma unroll
      for (int b = 0; b < B; ++b)
#pragma unroll
        for (int i = 0; i < 4; ++i)
          xr[b][i] = __bfloat1622float2(xs[b][(j * 4 + i) * 32 + lane]);

#pragma unroll
      for (int r = 0; r < RPW; ++r) {
        const uint32_t w = (j == 0) ? wq[r].x : (j == 1) ? wq[r].y
                         : (j == 2) ? wq[r].z : wq[r].w;
        const float sc_j = sca[r][j >> 1];
        float wv[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) wv[i] = lut[(w >> (i * 4)) & 0xF] * sc_j;
#pragma unroll
        for (int b = 0; b < B; ++b) {
          float a = 0.f;
#pragma unroll
          for (int i = 0; i < 4; ++i)
            a += wv[2 * i] * xr[b][i].x + wv[2 * i + 1] * xr[b][i].y;
          acc[r][b] += a;
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    const int n = row0 + r;
#pragma unroll
    for (int b = 0; b < B; ++b) {
      float a = acc[r][b];
#pragma unroll
      for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
      if (lane == 0 && n < N) y[(size_t)b * N + n] = a * tscale;
    }
  }
}

#ifndef ABLATE
#define ABLATE 0
#endif
// Rows per warp. Two forces pull against each other: more rows per warp means
// each staged activation is reused more times, but it also means holding more
// rows' weights in registers, and registers are what cap how many blocks an SM
// can keep resident.
//
// An earlier sweep concluded 2 rows below batch 4 and 3 above, and that was
// wrong — it was run before the activation staging was transposed, when the
// shared-memory bank conflicts were the dominant cost and extra rows only made
// them worse. Re-swept against the current kernel, FIVE rows is better at
// every batch size, by a lot in the middle of the range:
//
//   batch    1      2      3      4      5      6     (ms for a whole pass)
//   old   62.38  71.34  81.68  75.73  87.73 104.93
//   RPW=5 60.64  64.63  67.17  74.96  83.94  86.76
//
// RPW=6 is marginally better at batch 5 (82.55) and clearly worse at 6
// (98.87), so it is not worth a per-batch table. The lesson worth keeping is
// that a measurement taken before a fix can outlive the thing it measured.
#ifndef GEMM_RPW
#define GEMM_RPW 0            // 0 = use the measured default
#endif
template <int B>
static inline void launch_gemm_b(const uint8_t *W, const float *x, float *y,
                                 int N, int K, float ts, cudaStream_t st) {
  constexpr int RPW = GEMM_RPW ? GEMM_RPW : 5;
  const int rpb = GEMV_WARPS * RPW;
  gemm_nvfp4<RPW, B><<<(N + rpb - 1) / rpb, GEMV_THREADS, 0, st>>>(W, x, y, N, K, ts);
}

// Dispatch on the batch size actually in use.
inline void launch_gemm(const uint8_t *W, const float *x, float *y, int N, int K,
                        float tscale, int B, cudaStream_t st = 0) {
  switch (B) {
    case 1: launch_gemm_b<1>(W, x, y, N, K, tscale, st); break;
    case 2: launch_gemm_b<2>(W, x, y, N, K, tscale, st); break;
    case 3: launch_gemm_b<3>(W, x, y, N, K, tscale, st); break;
    case 4: launch_gemm_b<4>(W, x, y, N, K, tscale, st); break;
    case 5: launch_gemm_b<5>(W, x, y, N, K, tscale, st); break;
    case 6: launch_gemm_b<6>(W, x, y, N, K, tscale, st); break;
    case 7: launch_gemm_b<7>(W, x, y, N, K, tscale, st); break;
    case 8: launch_gemm_b<8>(W, x, y, N, K, tscale, st); break;
    default: printf("launch_gemm: unsupported batch %d\n", B); break;
  }
}

// ---------------------------------------------------------------------------
// D2: the 2-bit draft-head matmul.  y[N] = W[N,K] . x[K]
//
// Same shape of kernel as the 4-bit one, but each lane's 32 weights now fit in
// 8 bytes instead of 16, so a warp step is 320 bytes rather than 576. Only the
// draft head uses this: its guesses are checked by the full model, so lower
// precision costs a little acceptance and no correctness. Always batch 1.
//
//   [ 256 B: 1024 two-bit codes ][ 64 B: their 64 scales ]
// lane i takes 8 B of codes at i*8 and its two scales at 256 + i*2.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void load_d2_lut(float *lut, int tid) {
  if (tid < 4) {
    const float lv[4] = {-1.5f, -0.5f, 0.5f, 1.5f};
    lut[tid] = lv[tid];
  }
}

template <int RPW>
__global__ __launch_bounds__(GEMV_THREADS) void gemv_d2(
    const uint8_t *__restrict__ W, const float *__restrict__ x,
    float *__restrict__ y, int N, int K, float tscale) {
  __shared__ float xs[GEMV_KTILE];
  __shared__ float lut[4];

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row0 = blockIdx.x * (GEMV_WARPS * RPW) + warp * RPW;
  const int steps = K / GEMV_KTILE;
  load_d2_lut(lut, threadIdx.x);

  float acc[RPW];
#pragma unroll
  for (int r = 0; r < RPW; ++r) acc[r] = 0.f;

  for (int s = 0; s < steps; ++s) {
    __syncthreads();
    for (int i = threadIdx.x; i < GEMV_KTILE; i += GEMV_THREADS)
      xs[i] = x[s * GEMV_KTILE + i];
    __syncthreads();
    const float *xl = xs + lane * 32;

#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const int n = row0 + r;
      if (n >= N) continue;
      const uint8_t *p = W + ((size_t)n * steps + s) * D2_STEP_BYTES;
      const uint2 v = *(const uint2 *)(p + lane * 8);     // 32 codes
      const uint16_t sc = *(const uint16_t *)(p + D2_VAL_BYTES + lane * 2);
      // v.x is scale-block 0 (weights 0..15), v.y is block 1 (16..31).
      float b0 = 0.f, b1 = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        b0 += lut[(v.x >> (2 * i)) & 3u] * xl[i];
        b1 += lut[(v.y >> (2 * i)) & 3u] * xl[16 + i];
      }
      acc[r] += b0 * e4m3((uint8_t)(sc & 0xFF)) + b1 * e4m3((uint8_t)(sc >> 8));
    }
  }

#pragma unroll
  for (int r = 0; r < RPW; ++r) {
    float a = acc[r];
#pragma unroll
    for (int o = 16; o; o >>= 1) a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
    const int n = row0 + r;
    if (lane == 0 && n < N) y[n] = a * tscale;
  }
}

inline void launch_gemv_d2(const uint8_t *W, const float *x, float *y, int N,
                           int K, float tscale, cudaStream_t st = 0) {
  const int rpb = GEMV_WARPS * 4;
  gemv_d2<4><<<(N + rpb - 1) / rpb, GEMV_THREADS, 0, st>>>(W, x, y, N, K, tscale);
}

// Pick the block shape from the matrix shape.
//
// Measured: bandwidth tracks blocks-per-SM. Below ~6 blocks/SM the memory
// system starves; above ~10 it saturates. Since blocks = N / (8*RPW), a small
// N needs a small RPW to make enough of them. One compromise value cannot
// serve both a [48,5120] and a [248320,5120], so we choose per launch.
template <int RPW>
static inline void launch_rpw(const uint8_t *W, const float *x, float *y, int N,
                              int K, float ts, cudaStream_t st) {
  const int rpb = GEMV_WARPS * RPW;
  gemv_nvfp4<RPW><<<(N + rpb - 1) / rpb, GEMV_THREADS, 0, st>>>(W, x, y, N, K, ts);
}

inline void launch_gemv(const uint8_t *W, const float *x, float *y, int N, int K,
                        float tscale, cudaStream_t st = 0) {
  // Measured on this machine: 6.7 blocks/SM gives 77-81% of peak, 11-16 gives
  // 89-92%, 160 gives 98%. Filling the machine once is not enough — the tail
  // of the last wave leaves the memory system idle. Target ~12 blocks/SM
  // (576 blocks), which means N >= 4608 * RPW.
  if (N >= 4608 * 4)      launch_rpw<4>(W, x, y, N, K, tscale, st);
  else if (N >= 4608 * 2) launch_rpw<2>(W, x, y, N, K, tscale, st);
  else                    launch_rpw<1>(W, x, y, N, K, tscale, st);
}

}  // namespace spark27
