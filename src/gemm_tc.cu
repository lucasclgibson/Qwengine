#pragma once
// src/gemm_tc.cu — tensor-core GEMM for prefill.
//
// WHY PREFILL NEEDS A DIFFERENT KERNEL
// Generating a token reads all 14.65 GB of weights to do ~29 billion
// multiply-adds: about 3.6 operations per byte. That is hopelessly
// memory-bound, which is why the decode kernel is built entirely around
// streaming bytes and never touches a tensor core.
//
// Reading a PROMPT inverts that. The same 14.65 GB now serves every token in
// the prompt at once — 2000 tokens means 2000x the arithmetic for the same
// bytes. Prefill is compute-bound, and on CUDA cores alone (~18 TFLOP/s) a
// 2000-token prompt is ~6 s of pure maths. Tensor cores are the answer.
//
// WHY BF16 AND NOT FP4
// GB10 (sm_121) has NO FP4 tensor core instructions — verified against ptxas:
//   "Instruction 'mma with block scale' not supported on .target 'sm_121'"
//   "Feature '.kind::mxf4' not supported on .target 'sm_121'"
// Native FP4 matrix maths is datacenter Blackwell only. On this chip NVFP4 can
// only ever be a storage format, so we unpack it to BF16 in shared memory and
// feed the BF16 tensor cores, which sm_121 does have. Accumulation is fp32.
//
// C[T,N] = A[T,K] . W[N,K]^T   with W in the swizzled NVFP4 layout.
// W is [N,K] row-major, which is exactly a column-major B operand, so no
// transpose is needed anywhere.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include "format.h"
#include "gemv.cu"   // shares e4m3() and the E2M1 table layout

namespace spark27 {
namespace wmma = nvcuda::wmma;

// 64x64 output tile per block, K consumed 32 at a time.
// 4 warps in a 2x2 grid; each warp owns a 32x32 quadrant = 2x2 wmma fragments.
#ifndef TC_ABLATE
#define TC_ABLATE 0
#endif
#ifndef TC_BK_CFG
#define TC_BK_CFG 64
#endif
#ifndef TC_WARPS_CFG
#define TC_WARPS_CFG 16
#endif
// Output tile per block, and how much K is consumed per barrier. A wider K
// step means fewer __syncthreads and more tensor-core work per load.
//
// TC_BT is the number of PROMPT TOKENS a block handles, and it is the one that
// matters most: a block streams the whole K dimension of its weight slice, so
// a prompt of T tokens reads every weight ceil(T / TC_BT) times. Doubling
// TC_BT halves the weight traffic for any prompt longer than one tile.
//
// Re-swept after the shared-memory stride was padded (see TC_LD below), on a
// 256-token chunk over every weight shape in the model:
//
//   BT   BN   BK  warps      ms    tok/s   TFLOP/s
//   32  128   64      4  1037.9    246.7      12.1
//   64  128   32      8   875.0    292.6      14.3
//   64  128   64      8   691.1    370.4      18.2
//  128  128   32     16   678.3    377.4      18.5
//  128  128   64     16   541.6    472.7      23.2   <- default
//  256   64   64     16   737.3    347.2      17.0
//
// TC_BT=256 with TC_BN=128 would halve the traffic again but needs 32 warps
// and more shared memory than an SM has; ptxas rejects it.
#ifndef TC_BN_CFG
#define TC_BN_CFG 128
#endif
#ifndef TC_BT_CFG
#define TC_BT_CFG 128
#endif
constexpr int TC_BN = TC_BN_CFG, TC_BT = TC_BT_CFG, TC_BK = TC_BK_CFG;

// SHARED-MEMORY ROW STRIDE — PADDED, AND THAT PADDING IS THE WHOLE KERNEL.
//
// Both tiles are stored row-major with one row per matrix row. Stored at their
// natural stride of TC_BK bf16 = 128 bytes, every row begins at byte offset
// r*128, and shared memory has exactly 32 four-byte banks = 128 bytes. So
// row r starts at bank (r*32) % 32 = 0 -- EVERY ROW STARTS ON BANK 0.
// wmma::load_matrix_sync reads 16 rows of a fragment at once, so all 16 land on
// the same bank and serialise 16 ways.
//
// This is invisible to a tile-size sweep: every power-of-two TC_BK has the
// property, so every configuration was equally bad (21-27% of the memory floor,
// whatever the tile).
//
// Padding by 8 elements makes the stride 72 bf16 = 144 bytes = 36 banks, so
// row r starts at bank (r*36) % 32 = 4r % 32: rows 0..7 hit eight different
// banks and only rows 8 apart collide. 16-way becomes 2-way. 8 is the smallest
// pad that works because wmma requires ldm to be a multiple of 8 for bf16.
// Row-stride pad. 8 was chosen to break the 16-way bank conflict that came
// from a 128-byte (exactly 32-bank) stride, and that fix was worth 864 -> 696
// ms. Beyond breaking it, the exact pad does not matter -- swept 8/16/24/32,
// all within 290/282/282/283 ms -- so residual bank conflicts are NOT what
// caps this kernel and XOR swizzling would buy nothing. 16 by a nose.
#ifndef TC_PAD
#define TC_PAD 16
#endif
constexpr int TC_LD = TC_BK + TC_PAD;
// How many 16x16 fragments each warp owns, in tokens and in outputs. A warp
// loads WFT + WFN fragments per K-step and issues WFT*WFN MMAs against them, so
// this ratio is how much tensor-core work each shared-memory read pays for.
// At 2x2 it was 4 loads for 4 MMAs -- 1:1, and the kernel ran at 23 TFLOP/s.
// Swept THREE times, and the answer moved twice -- which is the lesson worth
// keeping. A tile sweep is only valid for the kernel it was run against.
//
//                              warp 2x2   warp 2x4
//   original (2-byte staging)     736 ms     603 ms
//   vectorised staging            376        442
//   + software-pipelined k-loop   295        279     <- default
//
// Fat warps won at first because staging was so expensive that minimising the
// threads doing it beat parallelism; thin warps won once staging was cheap; fat
// warps win again now that the pipeline wants registers more than it wants
// threads. Re-sweep after every structural change.
//
// Still unhelpful, all three times: a wider token tile (BT=256), a deeper K
// tile (BK=128 = 316 ms), a wider output tile (BN=256 = 300 ms). Each buys less
// re-reading at the cost of occupancy, and occupancy wins here.
//
// Beware 1024-thread configs: they exceed the 65536-register block budget and
// the launch fails SILENTLY, which the benchmark reports as an absurd
// 593 TFLOP/s. Always sanity-check a result against the 250 TFLOP/s machine
// peak measured by bench/03_mmamax.
#ifndef TC_WFT_CFG
#define TC_WFT_CFG 2
#endif
#ifndef TC_WFN_CFG
#define TC_WFN_CFG 4
#endif
constexpr int TC_WFT = TC_WFT_CFG, TC_WFN = TC_WFN_CFG;

// Unpack this block's slice of W into shared BF16.
//
// For a 32-wide K-tile aligned to 32, all 32 weights of a row live in 16
// contiguous bytes and share exactly two block scales, so the whole tile is a
// dense little read rather than a gather.
// The weights and activations for ONE k-tile, held in registers between the
// global read and the shared-memory store.
//
// Splitting the tile load into fetch-then-store is what lets the k-loop
// software-pipeline: the global reads for tile k+1 are issued BEFORE the MMAs
// of tile k, so ~600 clocks of DRAM latency are paid while the tensor cores are
// busy instead of while the whole block sits at a barrier. It costs about a
// dozen registers per thread and NO extra shared memory, so the block count per
// SM is unchanged -- which matters, because double-buffering the shared tiles
// instead would have halved it.
template <int WITER, int AITER>
struct TileRegs {
  uint2 wv[WITER];        // 16 packed 4-bit weights per group
  float sc[WITER];        // that group's block scale
  bool live[WITER];       // false for rows past N
  uint4 av[AITER];        // 8 activations each
};

// Phase 1: global -> registers. Touches no shared memory, so it can be issued
// while other warps are still reading the previous tile out of shared.
template <int BN, int BK, int BT, int THREADS, int WITER, int AITER>
__device__ __forceinline__ void tc_fetch(TileRegs<WITER, AITER> &r,
                                         const uint8_t *__restrict__ W,
                                         const __nv_bfloat16 *__restrict__ A,
                                         int n0, int t0, int N, int T, int K,
                                         int k0, int tid) {
  const int steps = K / 1024;
  const int s = k0 >> 10, j0 = k0 & 1023;
  constexpr int GROUPS = BK / 16;
#pragma unroll
  for (int it = 0; it < WITER; ++it) {
    const int i = tid + it * THREADS;
    r.live[it] = false;
    if (i >= BN * GROUPS) continue;
    const int row = i / GROUPS, g = i % GROUPS;
    const int n = n0 + row;
    if (n >= N) continue;
    const uint8_t *p = W + ((size_t)n * steps + s) * SWZ_STEP_BYTES;
    r.wv[it] = *(const uint2 *)(p + ((j0 + g * 16) >> 1));
    r.sc[it] = e4m3(p[SWZ_VAL_BYTES + ((j0 + g * 16) >> 4)]);
    r.live[it] = true;
  }
  constexpr int VEC = 8;
#pragma unroll
  for (int it = 0; it < AITER; ++it) {
    const int i = (tid + it * THREADS) * VEC;
    r.av[it] = make_uint4(0, 0, 0, 0);
    if (i >= BT * BK) continue;
    const int t = t0 + i / BK;
    if (t < T) r.av[it] = *(const uint4 *)(A + (size_t)t * K + k0 + (i % BK));
  }
}

// Phase 2: registers -> shared, unpacking NVFP4 to BF16 on the way.
//
// The per-tensor scale is deliberately NOT applied here. An E2M1 level has 1
// mantissa bit and an E4M3 block scale has 3, so their product needs at most 6
// -- and BF16 has 8. Dequantised weights are therefore EXACT in BF16, and this
// GEMM agrees with the fp32 decode kernel to ~5e-7. Multiplying by the
// arbitrary per-tensor float here destroys that exactness and costs three
// orders of magnitude of accuracy (measured: 5e-7 -> 1.5e-3). It goes on the
// fp32 accumulator instead.
template <int BN, int BK, int BT, int LD, int THREADS, int WITER, int AITER>
__device__ __forceinline__ void tc_store(const TileRegs<WITER, AITER> &r,
                                         __nv_bfloat16 *ws, __nv_bfloat16 *as,
                                         const float *lut, int tid) {
  constexpr int GROUPS = BK / 16;
#pragma unroll
  for (int it = 0; it < WITER; ++it) {
    const int i = tid + it * THREADS;
    if (i >= BN * GROUPS) continue;
    __nv_bfloat16 *dst = ws + (i / GROUPS) * LD + (i % GROUPS) * 16;
    if (!r.live[it]) {
      *(uint4 *)(dst + 0) = make_uint4(0, 0, 0, 0);
      *(uint4 *)(dst + 8) = make_uint4(0, 0, 0, 0);
      continue;
    }
    const uint32_t w0 = r.wv[it].x, w1 = r.wv[it].y;
    const float sc = r.sc[it];
    __nv_bfloat16 tmp[16];
#pragma unroll
    for (int c = 0; c < 8; ++c) tmp[c] = __float2bfloat16(lut[(w0 >> (c * 4)) & 0xF] * sc);
#pragma unroll
    for (int c = 0; c < 8; ++c) tmp[8 + c] = __float2bfloat16(lut[(w1 >> (c * 4)) & 0xF] * sc);
    *(uint4 *)(dst + 0) = *(const uint4 *)(tmp + 0);
    *(uint4 *)(dst + 8) = *(const uint4 *)(tmp + 8);
  }
  constexpr int VEC = 8;
#pragma unroll
  for (int it = 0; it < AITER; ++it) {
    const int i = (tid + it * THREADS) * VEC;
    if (i >= BT * BK) continue;
    *(uint4 *)(as + (i / BK) * LD + (i % BK)) = r.av[it];
  }
}

// BT tokens x BN outputs per block, K consumed BK at a time.
//
// SHARED MEMORY IS DYNAMIC, and that is not a stylistic choice. A static
// __shared__ array is capped at 48 KB per block; the opt-in limit on this chip
// is 99 KB. That cap was silently deciding the tile size: BT=256 needs ~55 KB
// and ptxas simply refused it ("uses too much shared data"), which read like a
// hardware limit and is not one. BT is exactly the knob that decides how often
// the weights get re-read -- a T-token prompt streams all 13.8 GB ceil(T/BT)
// times -- so the 48 KB cap was costing a factor of two on every long prompt.
template <int BT, int BN, int BK, int WFT, int WFN>
__global__ void gemm_tc(const uint8_t *__restrict__ W,
                        const __nv_bfloat16 *__restrict__ A,
                        float *__restrict__ C, int T, int N, int K,
                        float tscale) {
  constexpr int LD = BK + 8;              // padded stride, see TC_LD
  constexpr int WTILE_T = WFT * 16, WTILE_N = WFN * 16;
  constexpr int WARPS = (BT / WTILE_T) * (BN / WTILE_N);
  constexpr int THREADS = WARPS * 32;
  constexpr int WPR = BN / WTILE_N;       // warps across the output dimension

  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16 *ws = smem;
  __nv_bfloat16 *as = smem + (size_t)BN * LD;
  __shared__ float lut[16];

  const int tid = threadIdx.x;
  if (tid < 16) {
    const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    lut[tid] = (tid & 8) ? -mag[tid & 7] : mag[tid & 7];
  }

  const int n0 = blockIdx.x * BN;
  const int t0 = blockIdx.y * BT;
  const int warp = tid >> 5;
  const int wt = (warp / WPR) * WTILE_T, wn = (warp % WPR) * WTILE_N;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WFT][WFN];
#pragma unroll
  for (int i = 0; i < WFT; ++i)
#pragma unroll
    for (int j = 0; j < WFN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  // SOFTWARE-PIPELINED K-LOOP.
  //
  // The loop used to be [barrier][read global + dequant into shared][barrier]
  // [MMA], which serialises DRAM latency against tensor-core work: the whole
  // block arrives at the barrier together, issues its loads together, and
  // stalls together. Ablation put the cost of that staging at HALF the kernel.
  //
  // Now the global reads for tile k+1 are issued between the two barriers of
  // tile k, so they are in flight across both the barrier and the MMAs that
  // follow. Nothing is double-buffered in shared -- only in registers -- so
  // blocks per SM are unchanged.
  constexpr int GROUPS = BK / 16;
  constexpr int WITER = (BN * GROUPS + THREADS - 1) / THREADS;
  constexpr int AITER = (BT * BK / 8 + THREADS - 1) / THREADS;
  TileRegs<WITER, AITER> reg;

  tc_fetch<BN, BK, BT, THREADS, WITER, AITER>(reg, W, A, n0, t0, N, T, K, 0, tid);
  for (int k0 = 0; k0 < K; k0 += BK) {
    __syncthreads();                       // previous tile is finished being read
    tc_store<BN, BK, BT, LD, THREADS, WITER, AITER>(reg, ws, as, lut, tid);
    // Issue the NEXT tile's global reads now. They have no dependency on shared
    // memory, so they overlap the barrier below and the MMAs after it.
    if (k0 + BK < K)
      tc_fetch<BN, BK, BT, THREADS, WITER, AITER>(reg, W, A, n0, t0, N, T, K,
                                                  k0 + BK, tid);
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa[WFT];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb[WFN];
#pragma unroll
      for (int i = 0; i < WFT; ++i)
        wmma::load_matrix_sync(fa[i], as + (wt + i * 16) * LD + kk, LD);
#pragma unroll
      for (int j = 0; j < WFN; ++j)
        wmma::load_matrix_sync(fb[j], ws + (wn + j * 16) * LD + kk, LD);
#pragma unroll
      for (int i = 0; i < WFT; ++i)
#pragma unroll
        for (int j = 0; j < WFN; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
  }

  // Apply the per-tensor scale to the fp32 accumulator, then store straight to
  // global. C is padded to whole tiles, so no bounds check and no staging.
#pragma unroll
  for (int i = 0; i < WFT; ++i)
#pragma unroll
    for (int j = 0; j < WFN; ++j) {
#pragma unroll
      for (int e = 0; e < acc[i][j].num_elements; ++e) acc[i][j].x[e] *= tscale;
      wmma::store_matrix_sync(C + (size_t)(t0 + wt + i * 16) * N + n0 + wn + j * 16,
                              acc[i][j], N, wmma::mem_row_major);
    }
}

// Rows of A and C must be padded up to TC_BT. N needs no padding: the tile
// WIDTH adapts to it.
//
// This used to demand N % 128 == 0 and hand anything else to the decode kernel,
// eight prompt tokens at a time. That looked harmless -- the comment here said
// it only caught "the two [48,5120] DeltaNet gate projections, which are tiny".
// It was wrong, and expensively so. The fused in_zab projection is
// [6240, 5120], and 6240 is not a multiple of 128. So an 18 MB matrix was
// re-read once per group of 8 tokens: 64 times for a 512-token chunk, 48 layers
// deep, about 55 GB of pure waste per chunk. It was 25% of all prefill time.
//
// Every N in this model is a multiple of 32, so a 32-wide tile always fits.
// Take the widest that divides N -- 128 where it does, 32 where it does not --
// and the fallback reads the matrix ceil(T/128) times like everything else
// instead of ceil(T/8).
inline bool gemm_tc_ok(int N) { return N % 32 == 0; }

// Launch one instantiation, opting in to the >48 KB shared budget first.
// cudaFuncSetAttribute has to be called once per kernel before it can use more
// than 48 KB; the static flag keeps it off the per-launch path.
template <int BN, int WFN>
static inline void launch_tc(const uint8_t *W, const __nv_bfloat16 *A, float *C,
                             int T, int N, int K, float ts, int ty,
                             cudaStream_t st) {
  constexpr int LD = TC_BK + 8;
  constexpr int SMEM = (BN + TC_BT) * LD * (int)sizeof(__nv_bfloat16);
  constexpr int WARPS = (TC_BT / (TC_WFT * 16)) * (BN / (WFN * 16));
  auto k = gemm_tc<TC_BT, BN, TC_BK, TC_WFT, WFN>;
  static bool optin = false;
  if (!optin) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    optin = true;
  }
  k<<<dim3(N / BN, ty), WARPS * 32, SMEM, st>>>(W, A, C, T, N, K, ts);
}

inline void launch_gemm_tc(const uint8_t *W, const __nv_bfloat16 *A, float *C,
                           int T, int N, int K, float tscale,
                           cudaStream_t st = 0) {
  const int ty = (T + TC_BT - 1) / TC_BT;
  // The narrow fallbacks shrink the per-warp output tile to match, so a 32-wide
  // N never asks for a 32-wide warp tile it cannot fill.
  if (N % TC_BN == 0)      launch_tc<TC_BN, TC_WFN>(W, A, C, T, N, K, tscale, ty, st);
  else if (N % 96 == 0)    launch_tc<96, 2>(W, A, C, T, N, K, tscale, ty, st);
  else if (N % 64 == 0)    launch_tc<64, 2>(W, A, C, T, N, K, tscale, ty, st);
  else                     launch_tc<32, 2>(W, A, C, T, N, K, tscale, ty, st);
}

// ---------------------------------------------------------------------------
// The DECODE verify pass on tensor cores.
//
// A speculative verify is a GEMM with M = draft depth = 4..8 token rows. It has
// always run on the CUDA-core GEMV in gemv.cu, which at batch 4 costs 77.65 ms
// against a 62.2 ms bandwidth floor -- the 15 ms gap being 208 GFLOP of FP32
// FMA at a ~20.9 TFLOPS CUDA-core peak.
//
// This was dismissed once on the grounds that the prefill tile is 128 tokens
// wide and M=4 would waste 32x. That reasoning was wrong: the tile is a
// template parameter, and one WMMA fragment is 16 rows, so a BT=16
// instantiation wastes at most 4x at M=4 and 2x at M=8 -- against a tensor-core
// rate roughly an order of magnitude above the CUDA cores. The arithmetic that
// matters is not the FLOP ratio but whether the arithmetic disappears behind
// the weight stream, because the pass is bandwidth bound either way.
//
// Activations must be BF16 here, where the decode kernel takes fp32; the caller
// converts once per pass, which is B*K elements against 14.41 GB of weights.
inline void launch_gemm_tc_decode(const uint8_t *W, const __nv_bfloat16 *A,
                                  float *C, int N, int K, float ts,
                                  cudaStream_t st = 0) {
  constexpr int BT = 16, BK = TC_BK, LD = BK + 8;
  if (N % 128 == 0) {
    constexpr int SMEM = (128 + BT) * LD * (int)sizeof(__nv_bfloat16);
    auto k = gemm_tc<BT, 128, BK, 1, 4>;
    static bool o = false;
    if (!o) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); o = true; }
    k<<<dim3(N / 128, 1), (128 / 64) * 32, SMEM, st>>>(W, A, C, BT, N, K, ts);
  } else {
    constexpr int SMEM = (32 + BT) * LD * (int)sizeof(__nv_bfloat16);
    auto k = gemm_tc<BT, 32, BK, 1, 2>;
    static bool o = false;
    if (!o) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM); o = true; }
    k<<<dim3(N / 32, 1), (32 / 32) * 32, SMEM, st>>>(W, A, C, BT, N, K, ts);
  }
}

}  // namespace spark27
