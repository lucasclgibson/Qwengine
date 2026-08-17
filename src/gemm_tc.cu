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
#define TC_WARPS_CFG 8
#endif
// Output tile per block, and how much K is consumed per barrier. A wider K
// step means fewer __syncthreads and more tensor-core work per load.
#ifndef TC_BN_CFG
#define TC_BN_CFG 128
#endif
#ifndef TC_BT_CFG
#define TC_BT_CFG 64
#endif
constexpr int TC_BN = TC_BN_CFG, TC_BT = TC_BT_CFG, TC_BK = TC_BK_CFG;
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
  for (int i = tid; i < TC_BN * GROUPS; i += TC_THREADS) {
    const int r = i / GROUPS, g = i % GROUPS;
    const int n = n0 + r;
    __nv_bfloat16 *dst = ws + r * TC_BK + g * 16;
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

__global__ __launch_bounds__(TC_THREADS) void gemm_tc(
    const uint8_t *__restrict__ W, const __nv_bfloat16 *__restrict__ A,
    float *__restrict__ C, int T, int N, int K, float tscale) {
  __shared__ __nv_bfloat16 ws[TC_BN * TC_BK];
  __shared__ __nv_bfloat16 as[TC_BT * TC_BK];
  __shared__ float lut[16];

  const int tid = threadIdx.x;
  if (tid < 16) {
    const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    lut[tid] = (tid & 8) ? -mag[tid & 7] : mag[tid & 7];
  }

  const int n0 = blockIdx.x * TC_BN;
  const int t0 = blockIdx.y * TC_BT;
  const int warp = tid >> 5;
  const int wt = (warp / TC_WN) * 32, wn = (warp % TC_WN) * 32;  // this warp's quadrant

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  for (int k0 = 0; k0 < K; k0 += TC_BK) {
    __syncthreads();
    tc_load_w(ws, W, n0, N, K, k0, lut, tid);
    for (int i = tid; i < TC_BT * TC_BK; i += TC_THREADS) {
      const int r = i / TC_BK, c = i % TC_BK;
      const int t = t0 + r;
      as[i] = (t < T) ? A[(size_t)t * K + k0 + c] : __float2bfloat16(0.f);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TC_BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma::load_matrix_sync(fa[i], as + (wt + i * 16) * TC_BK + kk, TC_BK);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma::load_matrix_sync(fb[j], ws + (wn + j * 16) * TC_BK + kk, TC_BK);
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

// Rows of A and C must be padded up to TC_BT, and N must be a multiple of
// TC_BN. Every weight matrix in this model satisfies the N condition except
// the two [48,5120] DeltaNet gate projections, which are tiny and stay on the
// decode kernel.
inline bool gemm_tc_ok(int N) { return N % TC_BN == 0; }

inline void launch_gemm_tc(const uint8_t *W, const __nv_bfloat16 *A, float *C,
                           int T, int N, int K, float tscale,
                           cudaStream_t st = 0) {
  dim3 grid(N / TC_BN, (T + TC_BT - 1) / TC_BT);
  gemm_tc<<<grid, TC_THREADS, 0, st>>>(W, A, C, T, N, K, tscale);
}

}  // namespace spark27
