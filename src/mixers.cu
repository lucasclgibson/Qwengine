#pragma once
// src/mixers.cu — the two kinds of "look at earlier tokens" layer.
//
// This model alternates: three GatedDeltaNet layers, then one full-attention
// layer, sixteen times over. They solve the same problem in opposite ways.
//
//   FULL ATTENTION keeps every previous token's key and value in a cache, and
//   compares the new token against all of them. Exact, but the cost and the
//   memory grow with the conversation length.
//
//   GATED DELTANET keeps a fixed-size summary instead — a 128x128 matrix per
//   head that is updated as each token arrives, like a running memory that
//   decays and gets written over. Cost per token is constant no matter how
//   long the conversation. Cheaper, but lossy.
//
// Mixing them is why this model can claim a 262k context without the memory
// bill that 64 full-attention layers would bring: only 16 layers keep a cache.
//
// Every formula below was read from the reference implementation
// (transformers/models/qwen3_5/modeling_qwen3_5.py); line references given.

#include <cuda_runtime.h>
#include <stdint.h>

#include "../qwen38.h"

namespace spark27 {

__device__ __forceinline__ float bf16f(uint16_t h) {
  const uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

// ===========================================================================
// GATED DELTANET
// ===========================================================================

// Causal depthwise conv over the fused QKV stream, kernel width 4, then SiLU.
// Each channel has its own 4-tap filter and sees only the current token and
// the three before it — hence "causal". The three previous values live in
// conv_state, which we roll forward here.
// Ref: torch_causal_conv1d_update (modeling_qwen3_5.py:222).
__global__ void k_conv1d_step(float *__restrict__ x,        // [C] in-place
                              float *__restrict__ state,    // [C][3]
                              const uint16_t *__restrict__ w,  // [C][4] BF16
                              int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float *st = state + (size_t)c * 3;
  const float h0 = st[0], h1 = st[1], h2 = st[2], h3 = x[c];
  const uint16_t *wc = w + (size_t)c * 4;
  const float acc = h0 * bf16f(wc[0]) + h1 * bf16f(wc[1]) + h2 * bf16f(wc[2]) +
                    h3 * bf16f(wc[3]);
  st[0] = h1; st[1] = h2; st[2] = h3;
  x[c] = silu(acc);
}

// Chunked causal conv: the same 4-tap filter, but computed for every token of
// a prefill chunk at once.
//
// The decode version is inherently one-at-a-time because it only ever has one
// new token. During prefill we hold the whole chunk, and a 4-tap filter is
// just a window over it — so this is fully parallel over (token, channel).
// The three tokens before the chunk come from the carried conv state.
// Writes to a separate buffer: each output reads three inputs behind it, so
// updating in place would clobber its neighbours' inputs.
__global__ void k_conv1d_chunk(const float *__restrict__ x,
                               float *__restrict__ out,
                               float *__restrict__ state,
                               const uint16_t *__restrict__ w, int C, int T) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (c >= C || t >= T) return;
  const uint16_t *wc = w + (size_t)c * 4;
  const float *st = state + (size_t)c * 3;
  float acc = 0.f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int u = t - 3 + j;                       // absolute token index
    const float v = (u < 0) ? st[3 + u] : x[(size_t)u * C + c];
    acc += v * bf16f(wc[j]);
  }
  out[(size_t)t * C + c] = silu(acc);
}

// Roll the conv state forward past a whole chunk: keep the last three RAW
// (pre-conv) inputs, taking them from the carried state if the chunk is short.
__global__ void k_conv1d_carry(const float *__restrict__ x,
                               float *__restrict__ state, int C, int T) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float *st = state + (size_t)c * 3;
  const float prev[3] = {st[0], st[1], st[2]};
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int u = T - 3 + j;
    st[j] = (u < 0) ? prev[3 + u] : x[(size_t)u * C + c];
  }
}

// beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)
// Ref: modeling_qwen3_5.py:516-518. The .float() there matters — in half
// precision exp(A_log) can overflow to inf — so we do it all in float.
__global__ void k_delta_gates(const float *__restrict__ a,
                              const float *__restrict__ b,
                              const uint16_t *__restrict__ A_log,
                              const uint16_t *__restrict__ dt_bias,
                              float *__restrict__ g, float *__restrict__ beta,
                              int H) {
  // blockIdx.y selects the token, so a whole prefill chunk's gates are one
  // launch instead of one per token.
  const int tk = blockIdx.y;
  a += (size_t)tk * H; b += (size_t)tk * H;
  g += (size_t)tk * H; beta += (size_t)tk * H;
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H) return;
  beta[h] = 1.0f / (1.0f + __expf(-b[h]));
  const float t = a[h] + bf16f(dt_bias[h]);
  const float sp = t > 20.0f ? t : log1pf(__expf(t));  // softplus, stable
  g[h] = -__expf(bf16f(A_log[h])) * sp;
}

// L2-normalise each head's vector, then scale queries by 1/sqrt(head_dim).
// Ref: l2norm (modeling_qwen3_5.py:240) — note it is sum, not mean, + eps.
// blockIdx.y is the token and `row` is its stride, because Q and K live inside
// a wider fused row (10240) -- indexing heads contiguously across tokens would
// silently normalise the wrong memory for every token after the first.
__global__ void k_l2norm_heads(float *__restrict__ x, int dim, float extra,
                               int row = 0) {
  __shared__ float red[32];
  const int h = blockIdx.x, d = threadIdx.x;
  float *p = x + (size_t)blockIdx.y * row + (size_t)h * dim;
  float ss = 0.f;
  for (int i = d; i < dim; i += blockDim.x) ss += p[i] * p[i];
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, o);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  if (d == 0) {
    float t = 0.f;
    for (int i = 0; i < blockDim.x / 32; ++i) t += red[i];
    red[0] = rsqrtf(t + 1e-6f) * extra;
  }
  __syncthreads();
  const float s = red[0];
  for (int i = d; i < dim; i += blockDim.x) p[i] *= s;
}

// One step of the delta rule. One block per value head, one thread per output
// channel; thread dv owns column dv of that head's state matrix S.
//
// The maths, from torch_recurrent_gated_delta_rule (modeling_qwen3_5.py:327):
//   S      <- S * exp(g)                       decay the memory
//   kv_mem  = S^T k                            what the memory already knows about k
//   delta   = (v - kv_mem) * beta              the surprise, scaled by a learned rate
//   S      <- S + k (x) delta                  write the surprise in
//   out     = S^T q                            read the memory back with q
//
// Written naively that reads S twice. Substituting the update into the read:
//   out = exp(g) * (S^T q) + delta * (k . q)
// so one pass over S yields both S^T k and S^T q, and the second pass only
// writes. S is 3 MB per layer and L2 is 24 MB, so the write pass hits cache.
// DKSPLIT threads cooperate on each output channel instead of one.
//
// This kernel is launched once per token per layer -- 24576 times for a
// 512-token prefill chunk -- and at one block per value head that is 48 blocks
// of 128 threads: 6144 threads on a part that holds 98304, or 6% of the
// machine. The dk loop was 128 serial iterations inside each thread while the
// GPU sat idle. Splitting dk DKSPLIT ways multiplies the threads by DKSPLIT and
// shortens the serial loop by the same factor; the partial sums reduce through
// shared memory, which costs one extra barrier per step.
//
// The state is [DK][DV] with dv contiguous, so every thread still reads a
// coalesced run and the access pattern is unchanged -- only how many threads
// share the work.
#ifndef DELTA_DKSPLIT
#define DELTA_DKSPLIT 8
#endif
__global__ __launch_bounds__(128 * DELTA_DKSPLIT) void k_delta_step(
    float *__restrict__ S,          // [HV][DK][DV]
    const float *__restrict__ q,    // [HK][DK], already l2-normed and scaled
    const float *__restrict__ k,    // [HK][DK], already l2-normed
    const float *__restrict__ v,    // [HV][DV]
    const float *__restrict__ g,    // [HV]
    const float *__restrict__ beta, // [HV]
    float *__restrict__ out,        // [HV][DV]
    int DK, int DV, int group) {
  constexpr int SP = DELTA_DKSPLIT;
  const int h = blockIdx.x;
  const int dv = threadIdx.x % 128;          // output channel
  const int part = threadIdx.x / 128;        // which slice of dk this thread owns
  float *Sh = S + (size_t)h * DK * DV;
  const float *qh = q + (size_t)(h / group) * DK;   // 48 v-heads share 16 k-heads
  const float *kh = k + (size_t)(h / group) * DK;

  __shared__ float sq[128], sk[128], red[32];
  __shared__ float pk[SP][128], pq[SP][128];
  if (part == 0) { sq[dv] = qh[dv]; sk[dv] = kh[dv]; }
  __syncthreads();

  const float gh = __expf(g[h]);

  // Single pass over this thread's slice of S: both S^T k and S^T q at once.
  const int dk0 = part * (DK / SP), dk1 = dk0 + (DK / SP);
  float ak = 0.f, aq = 0.f;
  for (int dk = dk0; dk < dk1; ++dk) {
    const float s = Sh[(size_t)dk * DV + dv];  // coalesced: dv is the fast axis
    ak += s * sk[dk];
    aq += s * sq[dk];
  }
  pk[part][dv] = ak; pq[part][dv] = aq;

  // k . q, reduced across the block. Only the first slice's threads take part,
  // so the reduction is over exactly 128 lanes however wide the block is.
  float p = (part == 0) ? sk[dv] * sq[dv] : 0.f;
#pragma unroll
  for (int o = 16; o; o >>= 1) p += __shfl_xor_sync(0xFFFFFFFFu, p, o);
  if (part == 0 && (dv & 31) == 0) red[dv >> 5] = p;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < 4; ++i) t += red[i];
    red[0] = t;
  }
  __syncthreads();
  const float kq = red[0];

  // Recombine the dk slices.
  float akt = 0.f, aqt = 0.f;
#pragma unroll
  for (int i = 0; i < SP; ++i) { akt += pk[i][dv]; aqt += pq[i][dv]; }

  const float kv_mem = gh * akt;
  const float delta = (v[(size_t)h * DV + dv] - kv_mem) * beta[h];
  if (part == 0) out[(size_t)h * DV + dv] = gh * aqt + delta * kq;

  for (int dk = dk0; dk < dk1; ++dk)
    Sh[(size_t)dk * DV + dv] = Sh[(size_t)dk * DV + dv] * gh + sk[dk] * delta;
}

// Gated RMSNorm, per head.  out = norm(x) * weight * silu(z)
// Ref: Qwen3_5RMSNormGated (modeling_qwen3_5.py:187). Note this one uses
// `weight *`, NOT the `(1 + weight)` the layer norms use.
__global__ __launch_bounds__(128) void k_rmsnorm_gated(
    float *__restrict__ x, const float *__restrict__ z,
    const uint16_t *__restrict__ w, int dim, float eps) {
  __shared__ float red[32];
  const int h = blockIdx.x, d = threadIdx.x;
  float *p = x + (size_t)h * dim;
  const float *zp = z + (size_t)h * dim;
  float ss = 0.f;
  for (int i = d; i < dim; i += blockDim.x) ss += p[i] * p[i];
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, o);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  if (d == 0) {
    float t = 0.f;
    for (int i = 0; i < blockDim.x / 32; ++i) t += red[i];
    red[0] = rsqrtf(t / dim + eps);
  }
  __syncthreads();
  const float s = red[0];
  for (int i = d; i < dim; i += blockDim.x)
    p[i] = p[i] * s * bf16f(w[i]) * silu(zp[i]);
}

// ===========================================================================
// FULL ATTENTION
// ===========================================================================

// q_proj emits 24 blocks of 512, each [query(256) | gate(256)] — the split is
// per head, not one big Q followed by one big gate.
// Ref: modeling_qwen3_5.py:678 view(..., -1, head_dim*2) then chunk(2, dim=-1).
__global__ void k_split_qgate(const float *__restrict__ qproj,
                              float *__restrict__ q, float *__restrict__ gate,
                              int heads, int dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= heads * dim) return;
  const int h = i / dim, d = i % dim;
  q[i] = qproj[(size_t)h * 2 * dim + d];
  gate[i] = qproj[(size_t)h * 2 * dim + dim + d];
}

// Per-head RMSNorm with the (1 + weight) convention, used by q_norm/k_norm.
__global__ __launch_bounds__(256) void k_rmsnorm_head(float *__restrict__ x,
                                                      const uint16_t *__restrict__ w,
                                                      int dim, float eps) {
  __shared__ float red[32];
  const int h = blockIdx.x, d = threadIdx.x;
  float *p = x + (size_t)h * dim;
  float ss = 0.f;
  for (int i = d; i < dim; i += blockDim.x) ss += p[i] * p[i];
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, o);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  if (d == 0) {
    float t = 0.f;
    for (int i = 0; i < blockDim.x / 32; ++i) t += red[i];
    red[0] = rsqrtf(t / dim + eps);
  }
  __syncthreads();
  const float s = red[0];
  for (int i = d; i < dim; i += blockDim.x) p[i] = p[i] * s * (1.0f + bf16f(w[i]));
}

// Write this token's key and value into the cache at `pos`.
__global__ void k_kv_append(const float *__restrict__ k, const float *__restrict__ v,
                            float *__restrict__ kc, float *__restrict__ vc,
                            const int *__restrict__ dpos, int kvdim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kvdim) return;
  const int b = blockIdx.y;                 // token b of this batch
  const int pos = *dpos + b;
  kc[(size_t)pos * kvdim + i] = k[(size_t)b * kvdim + i];
  vc[(size_t)pos * kvdim + i] = v[(size_t)b * kvdim + i];
}

// Decode attention. One block per query head.
//
//   score[t] = (q . k[t]) / sqrt(head_dim)     how well token t matches
//   p        = softmax(score)                  turn scores into weights
//   out      = sum_t p[t] * v[t]               weighted blend of past values
//
// 24 query heads share 4 key/value heads (GQA): head h reads kv head h/6.
// Softmax subtracts the max first, which changes nothing mathematically but
// stops exp() overflowing.
__global__ __launch_bounds__(256) void k_attn_decode(
    const float *__restrict__ q, const float *__restrict__ kc,
    const float *__restrict__ vc, float *__restrict__ out,
    const int *__restrict__ dpos, int dim, int kv_heads, int q_heads,
    float scale, float *__restrict__ scratch, int ctx) {
  extern __shared__ float sh[];
  // Token b of the batch sits at position pos+b and may attend to everything
  // up to and including itself — that is the causal mask, and it is why a
  // batch of speculative tokens can be verified in one pass without any of
  // them seeing a token that comes after it.
  const int b = blockIdx.y;
  const int T = *dpos + b + 1;
  const int h = blockIdx.x, d = threadIdx.x;
  const int kvh = h / (q_heads / kv_heads);
  const int kvdim = kv_heads * dim;
  const float *qh = q + (size_t)b * q_heads * dim + (size_t)h * dim;
  float *scores = scratch + ((size_t)b * q_heads + h) * ctx;

  __shared__ __align__(16) float sq[256];
  __shared__ float red[32];
  if (d < dim) sq[d] = qh[d];
  __syncthreads();

  // Scores: one WARP per timestep, not one thread.
  //
  // Giving each thread its own timestep looks tidy and is a disaster for
  // memory. Consecutive lanes then read keys 'kvdim' floats apart -- 4 KB on
  // this model -- so all 32 lanes of a warp land in 32 different cache lines
  // and a single warp load costs 32 memory wavefronts instead of one. The
  // compiler cannot rescue it either: the dot loop emits 256 scalar loads per
  // lane, none of them vectorised.
  //
  // Inverted, a warp owns a timestep and its 32 lanes split the 256 channels
  // eight apiece, so each lane issues two float4 loads from ONE contiguous
  // 1 KB key vector: one wavefront per load, and the dot reduces over the warp.
  const int lane = d & 31, warp = d >> 5;
  const int nwarp = blockDim.x >> 5;
  const float4 qa = *(const float4 *)(sq + lane * 4);
  const float4 qb = *(const float4 *)(sq + 128 + lane * 4);
  float mx = -INFINITY;
  for (int t = warp; t < T; t += nwarp) {
    const float *kt = kc + (size_t)t * kvdim + (size_t)kvh * dim;
    const float4 ka = *(const float4 *)(kt + lane * 4);
    const float4 kb = *(const float4 *)(kt + 128 + lane * 4);
    float s = qa.x * ka.x + qa.y * ka.y + qa.z * ka.z + qa.w * ka.w
            + qb.x * kb.x + qb.y * kb.y + qb.z * kb.z + qb.w * kb.w;
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xFFFFFFFFu, s, o);
    s *= scale;
    if (lane == 0) scores[t] = s;
    mx = fmaxf(mx, s);            // s is warp-uniform after the reduction
  }
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  if (d == 0) {
    float m = -INFINITY;
    for (int i = 0; i < nwarp; ++i) m = fmaxf(m, red[i]);
    red[0] = m;
  }
  __syncthreads();
  const float m = red[0];

  float ls = 0.f;
  for (int t = d; t < T; t += blockDim.x) {
    const float e = __expf(scores[t] - m);
    scores[t] = e;
    ls += e;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) ls += __shfl_xor_sync(0xFFFFFFFFu, ls, o);
  if ((d & 31) == 0) red[d >> 5] = ls;
  __syncthreads();
  if (d == 0) {
    float t = 0.f;
    for (int i = 0; i < blockDim.x / 32; ++i) t += red[i];
    red[0] = 1.0f / t;
  }
  __syncthreads();
  const float inv = red[0];

  // Blend values. Thread d owns output channel d; reads across t are
  // coalesced because d is the fast axis of the cache.
  if (d < dim) {
    float acc = 0.f;
    for (int t = 0; t < T; ++t)
      acc += scores[t] * vc[(size_t)t * kvdim + (size_t)kvh * dim + d];
    out[(size_t)b * q_heads * dim + (size_t)h * dim + d] = acc * inv;
  }
  (void)sh;
}

// out *= sigmoid(gate)   — the attention output gate, applied before o_proj.
// Ref: modeling_qwen3_5.py:714  attn_output = attn_output * torch.sigmoid(gate)
__global__ void k_gate_sigmoid(float *__restrict__ x, const float *__restrict__ gate,
                               int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= 1.0f / (1.0f + __expf(-gate[i]));
}

}  // namespace spark27
