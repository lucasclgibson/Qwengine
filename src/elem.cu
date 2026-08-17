#pragma once
// src/elem.cu — the small kernels between the big matrix multiplies.
//
// These touch kilobytes where the GEMV touches gigabytes, so they are not
// where the time goes. They only have to be correct and cheap to launch.
// (Later they get folded into the GEMV's prologue/epilogue and captured in a
// CUDA graph, which removes the launch cost entirely.)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace spark27 {

// ---------------------------------------------------------------------------
// RMSNorm.  y = x / sqrt(mean(x^2) + eps) * (1 + weight)
//
// NOTE THE (1 + weight). This model stores the norm weight as a *delta from
// one* and initialises it to zeros, so multiplying by `weight` directly would
// scale every activation to nothing. modeling_qwen3_5.py Qwen3_5RMSNorm:
//     output = output * (1.0 + self.weight.float())
// DeltaNet's internal Qwen3_5RMSNormGated does NOT do this -- it uses plain
// `weight *`. Two different norms in one model; see k_rmsnorm_gated below.
//
// "Normalise": rescale a vector so its typical magnitude is 1, then let a
// learned per-channel weight decide how loud each channel should actually be.
// Every layer does this to its input to stop values exploding across 64 layers.
//
// One block, whole vector. Sum of squares accumulates in float with a fixed
// tree order, so the answer is bit-identical every run.
// ---------------------------------------------------------------------------
template <int THREADS>
__global__ __launch_bounds__(THREADS) void k_rmsnorm(float *__restrict__ out_,
                                                     const float *__restrict__ in_,
                                                     const uint16_t *__restrict__ w,
                                                     int n, float eps) {
  // blockIdx.x selects the batch row; each row normalises independently.
  const float *in = in_ + (size_t)blockIdx.x * n;
  float *out = out_ + (size_t)blockIdx.x * n;
  __shared__ float red[THREADS / 32];
  float ss = 0.f;
  for (int i = threadIdx.x; i < n; i += THREADS) {
    const float v = in[i];
    ss += v * v;
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.f;
    for (int i = 0; i < THREADS / 32; ++i) t += red[i];
    red[0] = rsqrtf(t / n + eps);
  }
  __syncthreads();
  const float scale = red[0];
  // Weights are stored BF16: the top 16 bits of the float.
  for (int i = threadIdx.x; i < n; i += THREADS) {
    const uint32_t bits = (uint32_t)w[i] << 16;
    float wf;
    memcpy(&wf, &bits, 4);
    out[i] = in[i] * scale * (1.0f + wf);
  }
}

// ---------------------------------------------------------------------------
// SwiGLU.  out = silu(gate) * up,  silu(x) = x * sigmoid(x)
//
// The model's "thinking" step. Two projections of the same input: one decides
// how much of each channel to let through (the gate), the other is the content.
// ---------------------------------------------------------------------------
__global__ void k_swiglu(float *__restrict__ out, const float *__restrict__ gate,
                         const float *__restrict__ up, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float g = gate[i];
  out[i] = (g / (1.0f + __expf(-g))) * up[i];
}

// x += y
__global__ void k_add(float *__restrict__ x, const float *__restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] += y[i];
}

__global__ void k_copy(float *__restrict__ d, const float *__restrict__ s, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = s[i];
}

// ---------------------------------------------------------------------------
// Rotary position embedding (RoPE).
//
// How the model knows word order. Each pair of channels is treated as a point
// on a circle and rotated by an angle proportional to the token's position;
// pairs further along the vector rotate more slowly. Attention then compares
// rotated vectors, so the comparison naturally depends on the *distance*
// between two tokens rather than their absolute positions.
//
// This model rotates only the first ROT of each head's DIM channels
// (partial_rotary_factor 0.25 -> 64 of 256); the rest pass through untouched.
// Pairs are (i, i+ROT/2) — the half-split convention, not adjacent pairs.
// ---------------------------------------------------------------------------
// mrope: each token has THREE positions — time, height, width — and each
// frequency slot uses one of them, interleaved T,H,W,T,H,W,... The section
// sizes [11,11,10] sum to the 32 rotary pairs.
//
// For text every token has t == h == w, so this reduces exactly to ordinary
// RoPE and the text path is unaffected. It stops being equivalent the moment
// an image is in the sequence, because image patches get distinct height and
// width positions — which is the whole point of the scheme.
__device__ __forceinline__ int mrope_component(int slot) {
  const int m = slot % 3;
  if (m == 1 && slot <= 31) return 1;   // H: slots 1,4,...,31   (11 of them)
  if (m == 2 && slot <= 29) return 2;   // W: slots 2,5,...,29   (10 of them)
  return 0;                             // T: everything else    (11 of them)
}

__global__ void k_rope(float *__restrict__ v, int heads, int dim, int rot,
                       const int *__restrict__ dpos, float theta,
                       const int *__restrict__ mpos) {
  const int h = blockIdx.x;
  const int i = threadIdx.x;         // 0 .. rot/2-1
  const int b = blockIdx.y;          // batch row: token b is at position pos+b
  if (h >= heads || i >= rot / 2) return;
  // mpos, when present, gives (t,h,w) per batch row and overrides the scalar
  // position. Text-only callers pass nullptr.
  const int pos = mpos ? mpos[b * 3 + mrope_component(i)] : (*dpos + b);
  float *p = v + (size_t)b * heads * dim + (size_t)h * dim;
  const float freq = __powf(theta, -2.0f * i / (float)rot);
  const float a = pos * freq;
  float s, c;
  __sincosf(a, &s, &c);
  const float x = p[i], y = p[i + rot / 2];
  p[i] = x * c - y * s;
  p[i + rot / 2] = x * s + y * c;
}

// ---------------------------------------------------------------------------
// Greedy sampling: pick the highest-scoring token.
//
// The model outputs one score per vocabulary entry (248320 of them). Greedy
// decoding just takes the largest. Ties break to the lowest index so the
// result is reproducible.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(1024) void k_argmax(const float *__restrict__ logits_,
                                                 int n, int *__restrict__ out) {
  const int b = blockIdx.x;
  const float *logits = logits_ + (size_t)b * n;
  __shared__ float bv[32];
  __shared__ int bi[32];
  float best = -INFINITY;
  int besti = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float v = logits[i];
    if (v > best || (v == best && i < besti)) { best = v; besti = i; }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, o);
    const int oi = __shfl_xor_sync(0xFFFFFFFFu, besti, o);
    if (ov > best || (ov == best && oi < besti)) { best = ov; besti = oi; }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) { bv[warp] = best; bi[warp] = besti; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float bb = bv[0];
    int bidx = bi[0];
    for (int i = 1; i < blockDim.x / 32; ++i)
      if (bv[i] > bb || (bv[i] == bb && bi[i] < bidx)) { bb = bv[i]; bidx = bi[i]; }
    out[b] = bidx;
  }
}

// Look up one token's embedding row and widen it to float.
// The embedding table is NVFP4 like everything else, so this unpacks one
// swizzled row rather than doing a plain memcpy.
__global__ void k_embed(float *__restrict__ out_, const uint8_t *__restrict__ tab,
                        const int *__restrict__ dtoken, int K, float tscale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= K) return;
  const int b = blockIdx.y;
  const int token = dtoken[b];
  float *out = out_ + (size_t)b * K;
  const int steps = K / 1024;
  const int s = i >> 10, j = i & 1023;
  const uint8_t *p = tab + ((size_t)token * steps + s) * 576;
  const uint8_t byte = p[j >> 1];
  const uint8_t code = (j & 1) ? (byte >> 4) : (byte & 0xF);
  const float mag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float w = (code & 8) ? -mag[code & 7] : mag[code & 7];
  const uint8_t sb = p[512 + (j >> 4)];
  const int e = (sb >> 3) & 0xF, mm = sb & 7;
  float sc = e ? ldexpf(1.0f + mm * 0.125f, e - 7) : ldexpf((float)mm, -9);
  if (sb & 0x80) sc = -sc;
  out[i] = w * sc * tscale;
}

// Move the sequence position on by n, on the device, so a captured graph
// carries its own state forward.
__global__ void k_advance_by(int *pos, int n) { *pos += n; }

// fp32 -> bf16. The tensor-core GEMM takes bf16 operands, so activations are
// converted on the way in. Weights are NOT converted here: unpacked NVFP4 is
// already exact in bf16 (see gemm_tc.cu).
__global__ void k_to_bf16(__nv_bfloat16 *__restrict__ o,
                          const float *__restrict__ i, size_t n) {
  const size_t x = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (x < n) o[x] = __float2bfloat16(i[x]);
}

// SwiGLU straight off a fused gate|up projection: row b of `f` is
// [gate(n) | up(n)], so no unpacking copy is needed.
__global__ void k_swiglu_fused(float *__restrict__ out,
                               const float *__restrict__ f, int n, int B) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * B) return;
  const int b = i / n, j = i % n;
  const float g = f[(size_t)b * 2 * n + j];
  out[i] = (g / (1.0f + __expf(-g))) * f[(size_t)b * 2 * n + n + j];
}

// Scatter a fused projection's rows into up to three separate buffers.
// The copy is tiny (tens of KB) next to the GB the fused matmul just read.
__global__ void k_unpack3(const float *__restrict__ f, float *__restrict__ a,
                          float *__restrict__ b, float *__restrict__ c,
                          int na, int nb, int nc, int B) {
  const int tot = na + nb + nc;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= tot * B) return;
  const int r = i / tot, j = i % tot;
  const float v = f[(size_t)r * tot + j];
  if (j < na)            a[(size_t)r * na + j] = v;
  else if (j < na + nb)  b[(size_t)r * nb + (j - na)] = v;
  else                   c[(size_t)r * nc + (j - na - nb)] = v;
}

// ---- launchers -------------------------------------------------------------
inline void rmsnorm(float *out, const float *in, const uint16_t *w, int n,
                    float eps, cudaStream_t st = 0, int B = 1) {
  k_rmsnorm<1024><<<B, 1024, 0, st>>>(out, in, w, n, eps);
}
inline void swiglu(float *out, const float *gate, const float *up, int n,
                   cudaStream_t st = 0, int B = 1) {
  k_swiglu<<<(n * B + 255) / 256, 256, 0, st>>>(out, gate, up, n * B);
}
inline void add_into(float *x, const float *y, int n, cudaStream_t st = 0,
                     int B = 1) {
  k_add<<<(n * B + 255) / 256, 256, 0, st>>>(x, y, n * B);
}
inline void copy_vec(float *d, const float *s, int n, cudaStream_t st = 0) {
  k_copy<<<(n + 255) / 256, 256, 0, st>>>(d, s, n);
}
inline void rope(float *v, int heads, int dim, int rot, const int *dpos,
                 float theta, cudaStream_t st = 0, int B = 1,
                 const int *mpos = nullptr) {
  k_rope<<<dim3(heads, B), rot / 2, 0, st>>>(v, heads, dim, rot, dpos, theta, mpos);
}
inline void argmax(const float *logits, int n, int *out, cudaStream_t st = 0,
                   int B = 1) {
  k_argmax<<<B, 1024, 0, st>>>(logits, n, out);
}
inline void swiglu_fused(float *out, const float *f, int n, int B,
                         cudaStream_t st = 0) {
  k_swiglu_fused<<<(n * B + 255) / 256, 256, 0, st>>>(out, f, n, B);
}
inline void unpack3(const float *f, float *a, float *b, float *c, int na, int nb,
                    int nc, int B, cudaStream_t st = 0) {
  const int tot = na + nb + nc;
  k_unpack3<<<(tot * B + 255) / 256, 256, 0, st>>>(f, a, b, c, na, nb, nc, B);
}

inline void to_bf16(__nv_bfloat16 *o, const float *i, size_t n,
                    cudaStream_t st = 0) {
  k_to_bf16<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(o, i, n);
}

inline void embed(float *out, const uint8_t *tab, const int *dtoken, int K,
                  float ts, cudaStream_t st = 0, int B = 1) {
  k_embed<<<dim3((K + 255) / 256, B), 256, 0, st>>>(out, tab, dtoken, K, ts);
}

}  // namespace spark27
