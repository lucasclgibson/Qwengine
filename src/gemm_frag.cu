#pragma once
// src/gemm_frag.cu — a GEMM that never stages weights in shared memory.
//
// WHY THIS EXISTS
// The tensor-core GEMM in gemm_tc.cu tops out at 47 TFLOP/s against a 250
// TFLOP/s machine, and ablation says exactly why: deleting the shared-memory
// STORES takes it to 93. Not the global loads, not bank conflicts, not
// occupancy, not the MMA-to-load ratio -- all four were measured away. It is
// the volume of shared writes, 32 KB per block per k-tile, half of it weights.
//
// The weights are only staged in shared because NVFP4 has to be unpacked to
// BF16 somewhere and `wmma::load_matrix_sync` reads from a plain strided array.
// But a fragment is just registers, and `wmma::fragment` exposes them as `.x[]`.
// If the weights are STORED in the order the fragment wants, each lane can read
// its eight values with one 4-byte load, dequantise them into `.x[]`, and go
// straight to `mma_sync` -- no shared memory involved at all.
//
// THE LAYOUT
// Probed from the hardware (not assumed): for a 16x16 `matrix_a` row-major
// fragment, lane L holds rows {L/4, L/4+8} and columns {(L%4)*2, +1, +8, +9},
// in the element order
//     e0,e1 -> (r=L/4,     c=2(L%4), +1)      e4,e5 -> same rows, c+8
//     e2,e3 -> (r=L/4+8,   c=2(L%4), +1)      e6,e7 -> same rows, c+8
// so a 16(n) x 16(k) tile is written as 32 lanes x 8 nibbles = 128 bytes, plus
// one FP8 scale per row = 16 bytes. 144 bytes for 256 weights is 4.5 bits,
// exactly the density of the swizzled layout it replaces.
//
// A warp then reads a whole tile as 32 consecutive 4-byte words: one 128-byte
// transaction, perfectly coalesced, and nothing is written to shared.
//
// THE RESULT: IT LOSES. 339 ms against gemm_tc's 261 on the same shapes, 37
// TFLOP/s against 48, with output that is BIT-IDENTICAL (rel-err 0.000e+00), so
// the layout and the fragment mapping are both right -- it is simply slower.
//
// Why, and why the ablation that motivated it was misleading: deleting the
// shared stores from gemm_tc does give 93 TFLOP/s, but it also deletes the
// dependency chain global-load -> dequant -> store -> fragment-load. What that
// ablation measured was not "stores are expensive" but "this whole chain is on
// the critical path". Moving the chain out of shared and into registers does
// not shorten it: it puts a GLOBAL load in the innermost loop, where gemm_tc
// has a shared one, and global latency is an order of magnitude worse. Staging
// in shared is not overhead, it is a latency-hiding buffer that every warp in
// the block shares.
//
// Software-pipelining the global loads to hide that latency was tried and is
// worse again (509 ms): the prefetch registers spill. Tile ordering does
// matter and is kept -- k-major beats n-major, 339 against 364 -- because a
// block's n-tiles for one k step must be adjacent.
//
// Kept, unused, because the negative result is worth more than the code: it
// says the next attempt at this kernel should attack the LATENCY of the chain
// (multi-stage cp.async into shared, which keeps the sharing and adds the
// asynchrony) rather than trying to remove the shared buffer.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include "format.h"
#include "gemv.cu"      // e4m3(), and the E2M1 value table

namespace spark27 {
namespace wmma = nvcuda::wmma;

// Bytes per 16x16 tile: 128 of packed values, then 16 scales.
constexpr int FRAG_TILE_BYTES = 128 + 16;

// Where lane L's element e sits inside a 16x16 tile, as (row, col). Used by the
// repacker on the host/device side and by the test; the kernel itself never
// needs it, because the data is already in this order.
__host__ __device__ __forceinline__ void frag_rc(int lane, int e, int &r, int &c) {
  r = (lane >> 2) + ((e & 2) ? 8 : 0);
  c = (lane & 3) * 2 + (e & 1) + ((e & 4) ? 8 : 0);
}

// ---------------------------------------------------------------------------
// Repack a swizzled NVFP4 matrix into fragment order.
//
// One block per 16x16 tile. Reading the source is a gather -- the swizzled
// layout puts a row's 1024 weights in one 576-byte step -- but this runs once
// at load time, not per token.
// ---------------------------------------------------------------------------
__global__ void k_repack_frag(const uint8_t *__restrict__ src,
                              uint8_t *__restrict__ dst, int N, int K) {
  // Tiles are ordered K-MAJOR: index = kt * ntiles + nt. A block consumes many
  // n-tiles at one k step, so those must be adjacent in memory; ordering them
  // n-major instead put them ktiles*144 bytes apart -- 45 KB on a K=5120
  // matrix -- and the kernel touched 128 bytes out of every 45 KB.
  const int ntiles = N / 16;
  const int tile = blockIdx.x;                 // one tile per block
  const int kt = tile / ntiles, nt = tile % ntiles;
  const int n0 = nt * 16, k0 = kt * 16;
  const int steps = K / 1024;
  const int lane = threadIdx.x;                // 32 threads

  // Values: this lane's 8 nibbles, in fragment element order.
  uint32_t packed = 0;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int r, c;
    frag_rc(lane, e, r, c);
    const int n = n0 + r, k = k0 + c;
    const uint8_t *p = src + ((size_t)n * steps + (k >> 10)) * SWZ_STEP_BYTES;
    const int j = k & 1023;
    const uint8_t byte = p[j >> 1];
    const uint32_t nib = (j & 1) ? (byte >> 4) : (byte & 0xF);
    packed |= nib << (e * 4);
  }
  *(uint32_t *)(dst + (size_t)tile * FRAG_TILE_BYTES + lane * 4) = packed;

  // Scales: one per row of the tile. Every column of the tile lies in the same
  // 16-wide NVFP4 block, so a row has exactly one scale here.
  if (lane < 16) {
    const int n = n0 + lane;
    const uint8_t *p = src + ((size_t)n * steps + (k0 >> 10)) * SWZ_STEP_BYTES;
    dst[(size_t)tile * FRAG_TILE_BYTES + 128 + lane] =
        p[SWZ_VAL_BYTES + ((k0 & 1023) >> 4)];
  }
}

inline size_t frag_bytes(int N, int K) {
  return (size_t)(N / 16) * (K / 16) * FRAG_TILE_BYTES;
}

inline void repack_frag(const uint8_t *src, uint8_t *dst, int N, int K,
                        cudaStream_t st = 0) {
  const int tiles = (N / 16) * (K / 16);
  k_repack_frag<<<tiles, 32, 0, st>>>(src, dst, N, K);
}

// ---------------------------------------------------------------------------
// C[T,N] = A[T,K] . W[N,K]^T, with W in fragment order.
//
// Only the ACTIVATIONS are staged in shared. The weights go global -> registers
// -> mma, which is the whole point.
// ---------------------------------------------------------------------------
#ifndef FG_BT_CFG
#define FG_BT_CFG 64
#endif
#ifndef FG_BN_CFG
#define FG_BN_CFG 128
#endif
#ifndef FG_BK_CFG
#define FG_BK_CFG 32
#endif
constexpr int FG_BT = FG_BT_CFG, FG_BN = FG_BN_CFG, FG_BK = FG_BK_CFG;
constexpr int FG_LD = FG_BK + 8;

template <int BT, int BN, int BK, int WFT, int WFN>
__global__ void gemm_frag(const uint8_t *__restrict__ W,
                          const __nv_bfloat16 *__restrict__ A,
                          float *__restrict__ C, int T, int N, int K,
                          float tscale) {
  constexpr int LD = BK + 8;
  constexpr int WARPS = (BT / (WFT * 16)) * (BN / (WFN * 16));
  constexpr int THREADS = WARPS * 32;
  constexpr int WPR = BN / (WFN * 16);

  extern __shared__ __nv_bfloat16 as[];        // [BT][LD] activations only
  __shared__ float lut[16];
  const int tid = threadIdx.x;
  if (tid < 16) {
    const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    lut[tid] = (tid & 8) ? -mag[tid & 7] : mag[tid & 7];
  }

  const int n0 = blockIdx.x * BN, t0 = blockIdx.y * BT;
  const int warp = tid >> 5, lane = tid & 31;
  const int wt = (warp / WPR) * WFT * 16, wn = (warp % WPR) * WFN * 16;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WFT][WFN];
#pragma unroll
  for (int i = 0; i < WFT; ++i)
#pragma unroll
    for (int j = 0; j < WFN; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

  // Which scales this lane needs, and which of its 8 elements use each.
  const int srow0 = lane >> 2, srow1 = srow0 + 8;

  for (int k0 = 0; k0 < K; k0 += BK) {
    __syncthreads();
    for (int i = tid * 8; i < BT * BK; i += THREADS * 8) {
      const int r = i / BK, c = i % BK;
      const int t = t0 + r;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (t < T) v = *(const uint4 *)(A + (size_t)t * K + k0 + c);
      *(uint4 *)(as + r * LD + c) = v;
    }
    __syncthreads();

    // SOFTWARE-PIPELINED over the k steps.
    //
    // Unlike gemm_tc, whose fragments come from shared memory, this kernel's
    // weight fragments come from GLOBAL. Loading them in the innermost loop put
    // full DRAM latency between consecutive MMAs and cost more than the shared
    // stores it was meant to avoid (339 ms against gemm_tc's 261). The packed
    // word and its two scales for step kk+1 are therefore fetched before the
    // MMAs of step kk, which costs WFN registers and hides the latency behind
    // tensor-core work.
    uint32_t pw[WFN];
    uint8_t ps0[WFN], ps1[WFN];
    auto fetch = [&](int kk, uint32_t *w, uint8_t *s0, uint8_t *s1) {
      const int ktl = (k0 + kk) / 16;
#pragma unroll
      for (int j = 0; j < WFN; ++j) {
        const size_t tile = (size_t)ktl * (N / 16) + ((n0 + wn) / 16 + j);
        const uint8_t *p = W + tile * FRAG_TILE_BYTES;
        w[j] = *(const uint32_t *)(p + lane * 4);
        s0[j] = p[128 + srow0];
        s1[j] = p[128 + srow1];
      }
    };
    fetch(0, pw, ps0, ps1);
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t nw[WFN]; uint8_t ns0[WFN], ns1[WFN];
      if (kk + 16 < BK) fetch(kk + 16, nw, ns0, ns1);

      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fw[WFN];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fa[WFT];
#pragma unroll
      for (int j = 0; j < WFN; ++j) {
        const float sa = e4m3(ps0[j]), sb = e4m3(ps1[j]);
#pragma unroll
        for (int e = 0; e < 8; ++e)
          fw[j].x[e] = __float2bfloat16(lut[(pw[j] >> (e * 4)) & 0xF] * ((e & 2) ? sb : sa));
      }
#pragma unroll
      for (int i = 0; i < WFT; ++i)
        wmma::load_matrix_sync(fa[i], as + (wt + i * 16) * LD + kk, LD);
#pragma unroll
      for (int i = 0; i < WFT; ++i)
#pragma unroll
        for (int j = 0; j < WFN; ++j)
          wmma::mma_sync(acc[i][j], fw[j], fa[i], acc[i][j]);
#pragma unroll
      for (int j = 0; j < WFN; ++j) { pw[j] = nw[j]; ps0[j] = ns0[j]; ps1[j] = ns1[j]; }
    }
  }

#pragma unroll
  for (int i = 0; i < WFT; ++i)
#pragma unroll
    for (int j = 0; j < WFN; ++j) {
#pragma unroll
      for (int e = 0; e < acc[i][j].num_elements; ++e) acc[i][j].x[e] *= tscale;
      wmma::store_matrix_sync(C + (size_t)(t0 + wt + i * 16) * N + n0 + wn + j * 16,
                              acc[i][j], N, wmma::mem_col_major);
    }
}

#ifndef FG_WFT_CFG
#define FG_WFT_CFG 2
#endif
#ifndef FG_WFN_CFG
#define FG_WFN_CFG 2
#endif

inline void launch_gemm_frag(const uint8_t *W, const __nv_bfloat16 *A, float *C,
                             int T, int N, int K, float ts, cudaStream_t st = 0) {
  constexpr int WFT = FG_WFT_CFG, WFN = FG_WFN_CFG;
  constexpr int WARPS = (FG_BT / (WFT * 16)) * (FG_BN / (WFN * 16));
  constexpr int SMEM = FG_BT * FG_LD * (int)sizeof(__nv_bfloat16);
  auto k = gemm_frag<FG_BT, FG_BN, FG_BK, WFT, WFN>;
  static bool o = false;
  if (!o) {
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    o = true;
  }
  k<<<dim3(N / FG_BN, (T + FG_BT - 1) / FG_BT), WARPS * 32, SMEM, st>>>(
      W, A, C, T, N, K, ts);
}

}  // namespace spark27
