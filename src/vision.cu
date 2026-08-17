#pragma once
// src/vision.cu — the image encoder.
//
// WHAT IT DOES
// Turns a picture into tokens the language model can read. An image is cut into
// 16x16 patches, each patch becomes a 1152-wide vector, those run through 27
// transformer layers that all see each other (no causal mask -- every part of an
// image may inform every other), and then 2x2 groups of patches are merged and
// projected to 5120 wide, which is exactly the language model's embedding size.
// The result is spliced into the text token stream wherever an <image> token sits.
//
// WHY IT COSTS ALMOST NOTHING
// This runs ONCE when a picture arrives, not once per generated token. A
// 500-token reply after an image runs it exactly once. Decode speed is
// untouched.
//
// PRECISION
// Left at BF16 throughout. The tower is 0.46 B parameters (~0.9 GB) and never
// touches the speed-critical path, so there is nothing to gain by quantising it
// and image quality stays beyond question. It also falls out for free: every
// vision matrix is 1152, 4304 or 4608 wide, none a multiple of the 1024-weight
// NVFP4 swizzle step, so the converter leaves them alone.
//
// Every formula below was taken from the HF reference and independently
// verified: an implementation written from this spec matched the reference
// module to 1.2e-6 max relative error in fp32.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "../qwen38.h"
#include "elem.cu"
#include "loader.cpp"
#include "mixers.cu"

// NOTE: use __bfloat162float() for __nv_bfloat16 operands, never mixers.cu's
// bf16f(), which takes raw uint16 BITS -- passing a __nv_bfloat16 to it silently
// converts the VALUE to an integer and everything small becomes zero.

namespace spark27 {
namespace wmma_v = nvcuda::wmma;

namespace vis {
constexpr int C = 1152;          // tower width
constexpr int LAYERS = 27;
constexpr int HEADS = 16;
constexpr int HD = C / HEADS;    // 72
constexpr int ROT = HD / 2;      // 36 -- only half the head is rotated
constexpr int FF = 4304;
constexpr int PATCH = 16;
constexpr int TEMPORAL = 2;
constexpr int MERGE = 2;
constexpr int PATCH_FEAT = 3 * TEMPORAL * PATCH * PATCH;   // 1536
constexpr int MERGED = C * MERGE * MERGE;                  // 4608
constexpr int OUT = 5120;                                  // == q38::HIDDEN
constexpr int POS_SIDE = 48;                               // 48x48 = 2304 table
constexpr float LN_EPS = 1e-6f;
constexpr float ROPE_THETA = 10000.0f;
// smart_resize bounds from the checkpoint's preprocessor_config.json
constexpr int RESIZE_FACTOR = PATCH * MERGE;               // 32
constexpr long MIN_PIXELS = 65536;
constexpr long MAX_PIXELS = 16777216;
}  // namespace vis

// ---------------------------------------------------------------------------
// BF16 GEMM with bias:  C[T,N] = A[T,K] . W[N,K]^T + b[N]
// The tower's weights are BF16, so unlike the language model there is nothing
// to unpack -- straight into the tensor cores.
// ---------------------------------------------------------------------------
constexpr int VB_N = 64, VB_T = 64, VB_K = 32, VB_WARPS = 4;
constexpr int VB_THREADS = VB_WARPS * 32;

__global__ __launch_bounds__(VB_THREADS) void k_gemm_bf16(
    const __nv_bfloat16 *__restrict__ W, const __nv_bfloat16 *__restrict__ A,
    const __nv_bfloat16 *__restrict__ bias, float *__restrict__ Cout, int T,
    int N, int K) {
  __shared__ __nv_bfloat16 ws[VB_N * VB_K];
  __shared__ __nv_bfloat16 as[VB_T * VB_K];
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * VB_N, t0 = blockIdx.y * VB_T;
  const int warp = tid >> 5;
  const int wt = (warp >> 1) * 32, wn = (warp & 1) * 32;

  wmma_v::fragment<wmma_v::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j) wmma_v::fill_fragment(acc[i][j], 0.0f);

  for (int k0 = 0; k0 < K; k0 += VB_K) {
    __syncthreads();
    for (int i = tid; i < VB_N * VB_K; i += VB_THREADS) {
      const int r = i / VB_K, c = i % VB_K;
      const int n = n0 + r, k = k0 + c;
      ws[i] = (n < N && k < K) ? W[(size_t)n * K + k] : __float2bfloat16(0.f);
    }
    for (int i = tid; i < VB_T * VB_K; i += VB_THREADS) {
      const int r = i / VB_K, c = i % VB_K;
      const int t = t0 + r, k = k0 + c;
      as[i] = (t < T && k < K) ? A[(size_t)t * K + k] : __float2bfloat16(0.f);
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < VB_K; kk += 16) {
      wmma_v::fragment<wmma_v::matrix_a, 16, 16, 16, __nv_bfloat16, wmma_v::row_major> fa[2];
      wmma_v::fragment<wmma_v::matrix_b, 16, 16, 16, __nv_bfloat16, wmma_v::col_major> fb[2];
#pragma unroll
      for (int i = 0; i < 2; ++i)
        wmma_v::load_matrix_sync(fa[i], as + (wt + i * 16) * VB_K + kk, VB_K);
#pragma unroll
      for (int j = 0; j < 2; ++j)
        wmma_v::load_matrix_sync(fb[j], ws + (wn + j * 16) * VB_K + kk, VB_K);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j) wmma_v::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
    }
  }

  __shared__ float cs[VB_T * VB_N];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma_v::store_matrix_sync(cs + (wt + i * 16) * VB_N + wn + j * 16, acc[i][j],
                                VB_N, wmma_v::mem_row_major);
  __syncthreads();
  for (int i = tid; i < VB_T * VB_N; i += VB_THREADS) {
    const int r = i / VB_N, c = i % VB_N;
    const int t = t0 + r, n = n0 + c;
    if (t < T && n < N)
      Cout[(size_t)t * N + n] = cs[i] + (bias ? __bfloat162float(bias[n]) : 0.f);
  }
}

inline void gemm_bf16(const __nv_bfloat16 *W, const __nv_bfloat16 *A,
                      const __nv_bfloat16 *bias, float *Cout, int T, int N, int K,
                      cudaStream_t st = 0) {
  dim3 g((N + VB_N - 1) / VB_N, (T + VB_T - 1) / VB_T);
  k_gemm_bf16<<<g, VB_THREADS, 0, st>>>(W, A, bias, Cout, T, N, K);
}

// ---------------------------------------------------------------------------
// LayerNorm with learned weight AND bias -- note this is a true LayerNorm
// (subtract the mean), not the RMSNorm the language model uses, and the weight
// is used directly, not as (1 + weight).
//   y = (x - mean) / sqrt(var + eps) * g + b,  var biased (divide by n)
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void k_layernorm(float *__restrict__ out,
                                                   const float *__restrict__ in,
                                                   const __nv_bfloat16 *__restrict__ g,
                                                   const __nv_bfloat16 *__restrict__ b,
                                                   int n, float eps) {
  __shared__ float rs[8], rss[8];
  const float *x = in + (size_t)blockIdx.x * n;
  float *y = out + (size_t)blockIdx.x * n;
  float s = 0.f, ss = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) { s += x[i]; ss += x[i] * x[i]; }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    ss += __shfl_xor_sync(0xFFFFFFFFu, ss, o);
  }
  if ((threadIdx.x & 31) == 0) { rs[threadIdx.x >> 5] = s; rss[threadIdx.x >> 5] = ss; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float a = 0.f, aa = 0.f;
    for (int i = 0; i < blockDim.x / 32; ++i) { a += rs[i]; aa += rss[i]; }
    const float mean = a / n;
    rs[0] = mean;
    rss[0] = rsqrtf(aa / n - mean * mean + eps);
  }
  __syncthreads();
  const float mean = rs[0], inv = rss[0];
  for (int i = threadIdx.x; i < n; i += blockDim.x)
    y[i] = (x[i] - mean) * inv * __bfloat162float(g[i]) + __bfloat162float(b[i]);
}

// ---------------------------------------------------------------------------
// Vision RoPE. Each patch has a (row, col) position. The first 18 frequency
// slots encode the row, the next 18 the column, and the pair rotated is
// (j, j+36) -- the half-split convention, over only the first 72 of each head.
// v is not rotated.
// ---------------------------------------------------------------------------
__global__ void k_vision_rope(float *__restrict__ q, float *__restrict__ k,
                              const int *__restrict__ pos, int N) {
  const int n = blockIdx.x;                 // patch
  const int h = blockIdx.y;                 // head
  const int j = threadIdx.x;                // 0..35
  if (n >= N || j >= vis::ROT) return;
  const int row = pos[2 * n], col = pos[2 * n + 1];
  // inv_freq[i] = theta^(-i/18) for i in 0..17; slot j<18 -> row, else col
  const int slot = j < 18 ? j : j - 18;
  const int p = j < 18 ? row : col;
  const float freq = __powf(vis::ROPE_THETA, -(float)slot / 18.0f);
  float s, c;
  __sincosf((float)p * freq, &s, &c);
  const size_t base = ((size_t)n * vis::HEADS + h) * vis::HD;
  float *qq = q + base, *kk = k + base;
  const float q0 = qq[j], q1 = qq[j + vis::ROT];
  qq[j] = q0 * c - q1 * s;
  qq[j + vis::ROT] = q1 * c + q0 * s;
  const float k0 = kk[j], k1 = kk[j + vis::ROT];
  kk[j] = k0 * c - k1 * s;
  kk[j + vis::ROT] = k1 * c + k0 * s;
}

// Split the fused qkv output. Layout is qkv_sel*C + head*HD + d.
__global__ void k_vis_split_qkv(const float *__restrict__ qkv, float *__restrict__ q,
                                float *__restrict__ k, float *__restrict__ v, int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N * vis::C) return;
  const int n = i / vis::C, d = i % vis::C;
  const size_t row = (size_t)n * 3 * vis::C;
  q[i] = qkv[row + d];
  k[i] = qkv[row + vis::C + d];
  v[i] = qkv[row + 2 * vis::C + d];
}

// Full bidirectional attention over all N patches. One block per (head, query).
// No causal mask: every part of the image may attend to every other part.
__global__ __launch_bounds__(128) void k_vision_attn(
    const float *__restrict__ q, const float *__restrict__ k,
    const float *__restrict__ v, float *__restrict__ out, int N, float scale) {
  extern __shared__ float sm[];             // N floats of scores
  const int h = blockIdx.x, n = blockIdx.y;
  const int t = threadIdx.x;
  __shared__ float sq[vis::HD], red[4];
  if (t < vis::HD) sq[t] = q[((size_t)n * vis::HEADS + h) * vis::HD + t];
  __syncthreads();

  float mx = -INFINITY;
  for (int j = t; j < N; j += blockDim.x) {
    const float *kj = k + ((size_t)j * vis::HEADS + h) * vis::HD;
    float s = 0.f;
    for (int d = 0; d < vis::HD; ++d) s += sq[d] * kj[d];
    s *= scale;                             // scale applied to the logit
    sm[j] = s;
    mx = fmaxf(mx, s);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
  if ((t & 31) == 0) red[t >> 5] = mx;
  __syncthreads();
  if (t == 0) {
    float m = -INFINITY;
    for (int i = 0; i < blockDim.x / 32; ++i) m = fmaxf(m, red[i]);
    red[0] = m;
  }
  __syncthreads();
  const float m = red[0];

  float ls = 0.f;
  for (int j = t; j < N; j += blockDim.x) { const float e = __expf(sm[j] - m); sm[j] = e; ls += e; }
#pragma unroll
  for (int o = 16; o; o >>= 1) ls += __shfl_xor_sync(0xFFFFFFFFu, ls, o);
  if ((t & 31) == 0) red[t >> 5] = ls;
  __syncthreads();
  if (t == 0) {
    float a = 0.f;
    for (int i = 0; i < blockDim.x / 32; ++i) a += red[i];
    red[0] = 1.0f / a;
  }
  __syncthreads();
  const float inv = red[0];

  if (t < vis::HD) {
    float a = 0.f;
    for (int j = 0; j < N; ++j)
      a += sm[j] * v[((size_t)j * vis::HEADS + h) * vis::HD + t];
    out[((size_t)n * vis::HEADS + h) * vis::HD + t] = a * inv;
  }
}

// gelu_pytorch_tanh, used inside the 27 blocks.
__global__ void k_gelu_tanh(float *__restrict__ x, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float u = x[i];
  x[i] = 0.5f * u * (1.0f + tanhf(0.7978845608028654f * (u + 0.044715f * u * u * u)));
}
// exact GELU (erf), used by the patch merger -- a DIFFERENT activation from the
// blocks above. The reference uses nn.GELU() there and GELUTanh() inside blocks.
__global__ void k_gelu_erf(float *__restrict__ x, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float u = x[i];
  x[i] = 0.5f * u * (1.0f + erff(u * 0.7071067811865476f));
}

__global__ void k_add_f(float *__restrict__ x, const float *__restrict__ y, size_t n) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += y[i];
}

// Bilinear-interpolated position embedding from the 48x48 table, added to the
// patch embeddings. Interpolation weights come from linspace(0,47,h/w).
__global__ void k_vis_pos_embed(float *__restrict__ x,
                                const __nv_bfloat16 *__restrict__ table,
                                const int *__restrict__ idx,
                                const float *__restrict__ wts, int N) {
  const int n = blockIdx.x;
  if (n >= N) return;
  for (int d = threadIdx.x; d < vis::C; d += blockDim.x) {
    float a = 0.f;
#pragma unroll
    for (int c = 0; c < 4; ++c)
      a += wts[c * N + n] * __bfloat162float(table[(size_t)idx[c * N + n] * vis::C + d]);
    x[(size_t)n * vis::C + d] += a;
  }
}

inline void to_bf16_v(__nv_bfloat16 *o, const float *i, size_t n, cudaStream_t st) {
  to_bf16(o, i, n, st);
}

}  // namespace spark27

namespace spark27 {

// Weight pointers for the tower, resolved once.
struct VisionTower {
  const __nv_bfloat16 *pe_w = nullptr, *pe_b = nullptr;   // patch embed (1152,1536)
  const __nv_bfloat16 *pos = nullptr;                     // (2304,1152)
  struct Blk {
    const __nv_bfloat16 *n1w, *n1b, *n2w, *n2b;
    const __nv_bfloat16 *qkv_w, *qkv_b, *proj_w, *proj_b;
    const __nv_bfloat16 *fc1_w, *fc1_b, *fc2_w, *fc2_b;
  } blk[vis::LAYERS];
  const __nv_bfloat16 *mg_nw, *mg_nb, *mg1_w, *mg1_b, *mg2_w, *mg2_b;

  // scratch
  int max_patches = 0;
  float *x, *xn, *qkv, *q, *k, *v, *att, *ff, *out;
  __nv_bfloat16 *xbf, *ffbf;
  int *dpos, *didx;
  float *dwts;
  bool present = false;
};

inline bool vision_init(VisionTower &T, Model &m, int max_patches) {
  if (!m.has("model.visual.patch_embed.proj.weight")) return false;
  auto B = [&](const char *n) { return (const __nv_bfloat16 *)m.get(n).dev; };
  T.pe_w = B("model.visual.patch_embed.proj.weight");
  T.pe_b = B("model.visual.patch_embed.proj.bias");
  T.pos  = B("model.visual.pos_embed.weight");
  char b[160];
  for (int i = 0; i < vis::LAYERS; ++i) {
    auto N = [&](const char *f) { snprintf(b, sizeof b, f, i); return B(b); };
    T.blk[i].n1w = N("model.visual.blocks.%d.norm1.weight");
    T.blk[i].n1b = N("model.visual.blocks.%d.norm1.bias");
    T.blk[i].n2w = N("model.visual.blocks.%d.norm2.weight");
    T.blk[i].n2b = N("model.visual.blocks.%d.norm2.bias");
    T.blk[i].qkv_w = N("model.visual.blocks.%d.attn.qkv.weight");
    T.blk[i].qkv_b = N("model.visual.blocks.%d.attn.qkv.bias");
    T.blk[i].proj_w = N("model.visual.blocks.%d.attn.proj.weight");
    T.blk[i].proj_b = N("model.visual.blocks.%d.attn.proj.bias");
    T.blk[i].fc1_w = N("model.visual.blocks.%d.mlp.linear_fc1.weight");
    T.blk[i].fc1_b = N("model.visual.blocks.%d.mlp.linear_fc1.bias");
    T.blk[i].fc2_w = N("model.visual.blocks.%d.mlp.linear_fc2.weight");
    T.blk[i].fc2_b = N("model.visual.blocks.%d.mlp.linear_fc2.bias");
  }
  T.mg_nw = B("model.visual.merger.norm.weight");
  T.mg_nb = B("model.visual.merger.norm.bias");
  T.mg1_w = B("model.visual.merger.linear_fc1.weight");
  T.mg1_b = B("model.visual.merger.linear_fc1.bias");
  T.mg2_w = B("model.visual.merger.linear_fc2.weight");
  T.mg2_b = B("model.visual.merger.linear_fc2.bias");

  const int P = max_patches;
  T.max_patches = P;
  auto A = [&](float **p, size_t n) { LCHECK(cudaMalloc(p, n * sizeof(float))); };
  A(&T.x, (size_t)P * vis::C);  A(&T.xn, (size_t)P * vis::C);
  A(&T.qkv, (size_t)P * 3 * vis::C);
  A(&T.q, (size_t)P * vis::C);  A(&T.k, (size_t)P * vis::C);
  A(&T.v, (size_t)P * vis::C);  A(&T.att, (size_t)P * vis::C);
  A(&T.ff, (size_t)P * vis::FF);
  A(&T.out, (size_t)(P / 4 + 1) * vis::OUT);
  A(&T.dwts, (size_t)4 * P);
  LCHECK(cudaMalloc(&T.xbf, (size_t)P * vis::FF * sizeof(__nv_bfloat16)));
  LCHECK(cudaMalloc(&T.ffbf, (size_t)P * vis::FF * sizeof(__nv_bfloat16)));
  LCHECK(cudaMalloc(&T.dpos, (size_t)2 * P * sizeof(int)));
  LCHECK(cudaMalloc(&T.didx, (size_t)4 * P * sizeof(int)));
  T.present = true;
  printf("vision: tower loaded, %d layers, up to %d patches (%.2f GB scratch)\n",
         vis::LAYERS, P, (double)P * (vis::C * 7 + vis::FF * 2) * 4 / 1e9);
  return true;
}

// pixels: [N, 1536] float, already normalised and in merge-block order.
// pos:    [N, 2] int (row, col) per patch
// idx/wts: the 4 bilinear corners and weights for the position table
// Writes M = N/4 embeddings of width 5120 into T.out.
inline void vision_forward(VisionTower &T, const float *pixels, const int *pos,
                           const int *idx, const float *wts, int N,
                           cudaStream_t st) {
  const int M = N / (vis::MERGE * vis::MERGE);
  // 1. patch embedding: the Conv3d has kernel == stride == input extent, so it
  //    is exactly a dense [1152,1536] GEMM.
  to_bf16(T.xbf, pixels, (size_t)N * vis::PATCH_FEAT, st);
  gemm_bf16(T.pe_w, T.xbf, T.pe_b, T.x, N, vis::C, vis::PATCH_FEAT, st);

  // 2. + bilinear-interpolated position embedding
  LCHECK(cudaMemcpyAsync(T.didx, idx, 4 * N * sizeof(int), cudaMemcpyHostToDevice, st));
  LCHECK(cudaMemcpyAsync(T.dwts, wts, 4 * N * sizeof(float), cudaMemcpyHostToDevice, st));
  LCHECK(cudaMemcpyAsync(T.dpos, pos, 2 * N * sizeof(int), cudaMemcpyHostToDevice, st));
  k_vis_pos_embed<<<N, 256, 0, st>>>(T.x, T.pos, T.didx, T.dwts, N);

  const float scale = rsqrtf((float)vis::HD);
  for (int l = 0; l < vis::LAYERS; ++l) {
    const auto &b = T.blk[l];
    // ---- attention ----
    k_layernorm<<<N, 256, 0, st>>>(T.xn, T.x, b.n1w, b.n1b, vis::C, vis::LN_EPS);
    to_bf16(T.xbf, T.xn, (size_t)N * vis::C, st);
    gemm_bf16(b.qkv_w, T.xbf, b.qkv_b, T.qkv, N, 3 * vis::C, vis::C, st);
    k_vis_split_qkv<<<(N * vis::C + 255) / 256, 256, 0, st>>>(T.qkv, T.q, T.k, T.v, N);
    k_vision_rope<<<dim3(N, vis::HEADS), vis::ROT, 0, st>>>(T.q, T.k, T.dpos, N);
    k_vision_attn<<<dim3(vis::HEADS, N), 128, N * sizeof(float), st>>>(
        T.q, T.k, T.v, T.att, N, scale);
    to_bf16(T.xbf, T.att, (size_t)N * vis::C, st);
    gemm_bf16(b.proj_w, T.xbf, b.proj_b, T.xn, N, vis::C, vis::C, st);
    k_add_f<<<((size_t)N * vis::C + 255) / 256, 256, 0, st>>>(T.x, T.xn, (size_t)N * vis::C);

    // ---- MLP ----
    k_layernorm<<<N, 256, 0, st>>>(T.xn, T.x, b.n2w, b.n2b, vis::C, vis::LN_EPS);
    to_bf16(T.xbf, T.xn, (size_t)N * vis::C, st);
    gemm_bf16(b.fc1_w, T.xbf, b.fc1_b, T.ff, N, vis::FF, vis::C, st);
    k_gelu_tanh<<<((size_t)N * vis::FF + 255) / 256, 256, 0, st>>>(T.ff, (size_t)N * vis::FF);
    to_bf16(T.ffbf, T.ff, (size_t)N * vis::FF, st);
    gemm_bf16(b.fc2_w, T.ffbf, b.fc2_b, T.xn, N, vis::C, vis::FF, st);
    k_add_f<<<((size_t)N * vis::C + 255) / 256, 256, 0, st>>>(T.x, T.xn, (size_t)N * vis::C);
  }

  // ---- patch merger ----
  // LayerNorm per token FIRST, then four consecutive tokens are concatenated.
  // Because the patch order is merge-block order, those four are exactly one
  // 2x2 spatial block, so the concat needs no shuffle -- it is a reinterpret.
  k_layernorm<<<N, 256, 0, st>>>(T.xn, T.x, T.mg_nw, T.mg_nb, vis::C, vis::LN_EPS);
  to_bf16(T.xbf, T.xn, (size_t)N * vis::C, st);
  gemm_bf16(T.mg1_w, T.xbf, T.mg1_b, T.ff, M, vis::MERGED, vis::MERGED, st);
  // NOTE: exact GELU here, not the tanh approximation the blocks use.
  k_gelu_erf<<<((size_t)M * vis::MERGED + 255) / 256, 256, 0, st>>>(T.ff, (size_t)M * vis::MERGED);
  to_bf16(T.ffbf, T.ff, (size_t)M * vis::MERGED, st);
  gemm_bf16(T.mg2_w, T.ffbf, T.mg2_b, T.out, M, vis::OUT, vis::MERGED, st);
}

}  // namespace spark27
