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
constexpr int TC_LD = TC_BK + 8;
constexpr int TC_WARPS = TC_WARPS_CFG;
constexpr int TC_THREADS = TC_WARPS * 32;
// Warps tile the output as (TC_BT/32) x (TC_BN/32) quadrants of 32x32.
constexpr int TC_WT = TC_BT / 32, TC_WN = TC_BN / 32;
static_assert(TC_WT * TC_WN == TC_WARPS, "warp grid must cover the tile");

// Unpack this block's slice of W into shared BF16.
//
// For a 32-wide K-tile aligned to 32, all 32 weights of a row live in 16
// contiguous bytes and share exactly two block scales, so the whole tile is a
// dense little read rather than a gather.
template <int BN, int THREADS>
__device__ __forceinline__ void tc_load_w(__nv_bfloat16 *ws,
                                          const uint8_t *__restrict__ W,
                                          int n0, int N, int K, int k0,
                                          const float *lut, int tid) {
  // NOTE: the per-tensor scale is deliberately NOT applied here.
  //
  // An E2M1 level has 1 mantissa bit and an E4M3 block scale has 3, so their
  // product needs at most 6 — and BF16 has 8. Dequantised weights are
  // therefore EXACT in BF16, and this GEMM agrees with the fp32 decode kernel
  // to ~5e-7. Multiplying by the arbitrary per-tensor float here destroys that
  // exactness and costs three orders of magnitude of accuracy (measured:
  // 5e-7 -> 1.5e-3). It is applied to the fp32 accumulator instead.
  const int steps = K / 1024;
  const int s = k0 >> 10, j0 = k0 & 1023;
  // Each unit of work is one row's 16-weight group (one block scale).
  constexpr int GROUPS = TC_BK / 16;
  for (int i = tid; i < BN * GROUPS; i += THREADS) {
    const int r = i / GROUPS, g = i % GROUPS;
    const int n = n0 + r;
    __nv_bfloat16 *dst = ws + r * TC_LD + g * 16;
    if (n >= N) {
#pragma unroll
      for (int c = 0; c < 16; ++c) dst[c] = __float2bfloat16(0.f);
      continue;
    }
    const uint8_t *p = W + ((size_t)n * steps + s) * SWZ_STEP_BYTES;
    const uint8_t *v = p + ((j0 + g * 16) >> 1);             // 8 bytes = 16 weights
    const float sc = e4m3(p[SWZ_VAL_BYTES + ((j0 + g * 16) >> 4)]);
#if TC_ABLATE == 1
#pragma unroll
    for (int c = 0; c < 16; ++c) dst[c] = __float2bfloat16(sc);
#else
    // One 8-byte read for all 16 weights rather than 16 byte loads. Unpacking
    // is ~35% of this kernel's time (measured by ablation: 15.4 -> 23.7
    // TFLOP/s with it removed), so the loads are worth doing properly.
    const uint32_t w0 = ((const uint32_t *)v)[0], w1 = ((const uint32_t *)v)[1];
#pragma unroll
    for (int c = 0; c < 8; ++c)
      dst[c] = __float2bfloat16(lut[(w0 >> (c * 4)) & 0xF] * sc);
#pragma unroll
    for (int c = 0; c < 8; ++c)
      dst[8 + c] = __float2bfloat16(lut[(w1 >> (c * 4)) & 0xF] * sc);
#endif
  }
}

template <int BN>
__global__ __launch_bounds__((TC_BT / 32) * (BN / 32) * 32) void gemm_tc(
    const uint8_t *__restrict__ W, const __nv_bfloat16 *__restrict__ A,
    float *__restrict__ C, int T, int N, int K, float tscale) {
  constexpr int WN = BN / 32, WT = TC_BT / 32;
  constexpr int WARPS = WT * WN, THREADS = WARPS * 32;
  __shared__ __nv_bfloat16 ws[BN * TC_LD];
  __shared__ __nv_bfloat16 as[TC_BT * TC_LD];
  __shared__ float lut[16];

  const int tid = threadIdx.x;
  if (tid < 16) {
    const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    lut[tid] = (tid & 8) ? -mag[tid & 7] : mag[tid & 7];
  }

  const int n0 = blockIdx.x * BN;
  const int t0 = blockIdx.y * TC_BT;
  const int warp = tid >> 5;
  const int wt = (warp / WN) * 32, wn = (warp % WN) * 32;  // this warp's quadrant

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  for (int k0 = 0; k0 < K; k0 += TC_BK) {
    __syncthreads();
    tc_load_w<BN, THREADS>(ws, W, n0, N, K, k0, lut, tid);
    for (int i = tid; i < TC_BT * TC_BK; i += THREADS) {
      const int r = i / TC_BK, c = i % TC_BK;
      const int t = t0 + r;
      as[r * TC_LD + c] = (t < T) ? A[(size_t)t * K + k0 + c] : __float2bfloat16(0.f);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TC_BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(fa[i], as + (wt + i * 16) * TC_LD + kk, TC_LD);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(fb[j], ws + (wn + j * 16) * TC_LD + kk, TC_LD);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
  }

  // Apply the per-tensor scale to the fp32 accumulator, then store straight to
  // global. C is padded to whole tiles (see launch_gemm_tc), so there is no
  // bounds check and no shared staging — staging 64x128 floats cost 32 KB and
  // blew the shared budget by itself.
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
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

inline void launch_gemm_tc(const uint8_t *W, const __nv_bfloat16 *A, float *C,
                           int T, int N, int K, float tscale,
                           cudaStream_t st = 0) {
  const int ty = (T + TC_BT - 1) / TC_BT;
  if (N % TC_BN == 0) {
    constexpr int THREADS = (TC_BT / 32) * (TC_BN / 32) * 32;
    gemm_tc<TC_BN><<<dim3(N / TC_BN, ty), THREADS, 0, st>>>(W, A, C, T, N, K, tscale);
  } else if (N % 96 == 0) {   // 6240 = 96 * 65, the in_zab shape
    constexpr int THREADS = (TC_BT / 32) * 3 * 32;
    gemm_tc<96><<<dim3(N / 96, ty), THREADS, 0, st>>>(W, A, C, T, N, K, tscale);
  } else if (N % 64 == 0) {
    constexpr int THREADS = (TC_BT / 32) * 2 * 32;
    gemm_tc<64><<<dim3(N / 64, ty), THREADS, 0, st>>>(W, A, C, T, N, K, tscale);
  } else {
    constexpr int THREADS = (TC_BT / 32) * 32;
    gemm_tc<32><<<dim3(N / 32, ty), THREADS, 0, st>>>(W, A, C, T, N, K, tscale);
  }
}

}  // namespace spark27
