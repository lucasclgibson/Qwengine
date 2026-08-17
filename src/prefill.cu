#pragma once
// src/prefill.cu — read the prompt in chunks instead of one token at a time.
//
// WHAT WAS WRONG
// The engine fed prompts through the decode path, one token per full pass of
// the model. That is ~14.65 GB of weight traffic PER PROMPT TOKEN, and it ran
// at ~14 tok/s — a 2000-token prompt took over two minutes before the first
// word came out.
//
// WHAT THIS DOES
// Push a whole chunk of the prompt through together. The weights are then read
// once for the entire chunk rather than once per token, which flips the work
// from memory-bound to compute-bound — and compute is what tensor cores are
// for (see gemm_tc.cu).
//
// Only the final token's logits are wanted, so lm_head runs on one row.
//
// The DeltaNet recurrence still has to walk tokens in order — token n+1's
// state depends on token n's — but its state is 3 MB per layer against a
// 24 MB L2, and we process a whole chunk within one layer before moving on,
// so those sequential steps stay in cache instead of going to DRAM.

#include "gemm_tc.cu"
#include "model.cu"
#include "vision.cu"

namespace spark27 {

// Scratch sized for one chunk. Allocated once, reused for every chunk.
struct PrefillBuf {
  int chunk = 0;
  float *h, *hn, *mix;
  __nv_bfloat16 *hbf;              // bf16 copy of the activations for the GEMM
  float *qkv, *zb, *ab, *bb, *dnout;
  float *qproj, *qh, *gateh, *kh, *vh, *attn_out;
  float *mlp_g, *mlp_a, *fused;
  __nv_bfloat16 *mlp_abf;
  float *scores;
  float *convout;      // conv writes here, not in place
  float *gb, *betab;   // per-token gates for the whole chunk
  int *dtok;
};

// Row count rounded up to a whole tensor-core tile; the GEMM writes whole
// tiles and skips bounds checks, so every buffer it touches is padded.
static inline int pad_t(int t) { return (t + TC_BT - 1) / TC_BT * TC_BT; }

inline void prefill_init(PrefillBuf &p, int chunk, int ctx) {
  p.chunk = chunk;
  const int P = pad_t(chunk);
  auto A = [&](float **q, size_t n) { LCHECK(cudaMalloc(q, n * sizeof(float))); };
  auto Ab = [&](__nv_bfloat16 **q, size_t n) {
    LCHECK(cudaMalloc(q, n * sizeof(__nv_bfloat16)));
    LCHECK(cudaMemset(*q, 0, n * sizeof(__nv_bfloat16)));
  };
  A(&p.h, (size_t)P * HIDDEN);  A(&p.hn, (size_t)P * HIDDEN);
  A(&p.mix, (size_t)P * HIDDEN);
  // hbf is reused as the bf16 staging buffer for several projections, so it
  // must hold the WIDEST of them, not just HIDDEN: o_proj consumes Q_DIM and
  // out_proj consumes LIN_V_DIM, both 6144.
  constexpr int HBF_W = Q_DIM > HIDDEN ? Q_DIM : HIDDEN;
  static_assert(HBF_W >= LIN_V_DIM, "hbf must cover every projection input");
  Ab(&p.hbf, (size_t)P * HBF_W);
  A(&p.qkv, (size_t)P * LIN_QKV_DIM); A(&p.zb, (size_t)P * LIN_Z_DIM);
  A(&p.ab, (size_t)P * LIN_V_HEADS);  A(&p.bb, (size_t)P * LIN_V_HEADS);
  A(&p.dnout, (size_t)P * LIN_V_DIM);
  A(&p.qproj, (size_t)P * QGATE_DIM); A(&p.qh, (size_t)P * Q_DIM);
  A(&p.gateh, (size_t)P * Q_DIM);
  A(&p.kh, (size_t)P * KV_DIM); A(&p.vh, (size_t)P * KV_DIM);
  A(&p.attn_out, (size_t)P * Q_DIM);
  A(&p.mlp_g, (size_t)P * 2 * INTERMEDIATE);   // gate|up fused
  A(&p.mlp_a, (size_t)P * INTERMEDIATE);
  A(&p.fused, (size_t)P * (QGATE_DIM + 2 * KV_DIM));
  Ab(&p.mlp_abf, (size_t)P * INTERMEDIATE);
  A(&p.scores, (size_t)P * N_Q_HEADS * ctx);
  A(&p.convout, (size_t)P * LIN_QKV_DIM);
  A(&p.gb, (size_t)P * LIN_V_HEADS);
  A(&p.betab, (size_t)P * LIN_V_HEADS);
  LCHECK(cudaMalloc(&p.dtok, (size_t)P * sizeof(int)));
}

// One projection. Uses the tensor-core path when the output width tiles
// cleanly, and falls back to the decode kernel for the two tiny [48,5120]
// DeltaNet gate projections.
static inline void proj(const uint8_t *W, const __nv_bfloat16 *abf,
                        const float *afp, float *out, int T, int N, int K,
                        float sc, cudaStream_t st) {
  if (gemm_tc_ok(N)) { launch_gemm_tc(W, abf, out, T, N, K, sc, st); return; }
  // Fallback for widths the tensor-core tile does not divide — only the two
  // [48,5120] DeltaNet gate projections. The decode kernel tops out at batch
  // 8, so walk the chunk in groups of 8.
  for (int b = 0; b < T; b += 8) {
    const int n = (T - b) < 8 ? (T - b) : 8;
    launch_gemm(W, afp + (size_t)b * K, out + (size_t)b * N, N, K, sc, n, st);
  }
}

// Overwrite the embeddings of image-placeholder tokens with what the vision
// tower produced. The language model then reads the picture exactly where the
// <|image_pad|> tokens sat in the prompt.
__global__ void k_splice_image(float *__restrict__ h, const float *__restrict__ img,
                               const int *__restrict__ slot, int n, int dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * dim) return;
  const int k = i / dim, d = i % dim;
  h[(size_t)slot[k] * dim + d] = img[(size_t)k * dim + d];
}

// Push `T` tokens (already in p.dtok) through the model, starting at the
// current position. Leaves the last token's logits in rt.logits.
//
// `img` / `islot` / `nimg` optionally splice vision embeddings in; `mpos` gives
// per-token (t,h,w) positions for mrope. Both are null for pure text.
inline void prefill_chunk(Runtime &rt, PrefillBuf &p, int T,
                          const float *img = nullptr, const int *islot = nullptr,
                          int nimg = 0, const int *mpos = nullptr) {
  cudaStream_t st = rt.stream;
  const int P = pad_t(T);

  // p.dtok, not rt.dtok_in: the runtime's token buffer is sized for the
  // speculative batch (8), while a prefill chunk is hundreds of tokens.
  embed(p.h, rt.embed, p.dtok, HIDDEN, rt.s_embed, st, T);
  if (img && nimg > 0)
    k_splice_image<<<(nimg * HIDDEN + 255) / 256, 256, 0, st>>>(p.h, img, islot,
                                                                nimg, HIDDEN);

  for (int l = 0; l < N_LAYERS; ++l) {
    const LayerPtrs &L = rt.L[l];

    rmsnorm(p.hn, p.h, L.in_norm, HIDDEN, RMS_EPS, st, T);
    to_bf16(p.hbf, p.hn, (size_t)T * HIDDEN, st);

    if (L.full) {
      proj(L.qkv_proj, p.hbf, p.hn, p.fused, T, QGATE_DIM + 2 * KV_DIM, HIDDEN,
           L.sqkv, st);
      unpack3(p.fused, p.qproj, p.kh, p.vh, QGATE_DIM, KV_DIM, KV_DIM, T, st);

      k_split_qgate<<<(T * Q_DIM + 255) / 256, 256, 0, st>>>(
          p.qproj, p.qh, p.gateh, T * N_Q_HEADS, HEAD_DIM);
      k_rmsnorm_head<<<T * N_Q_HEADS, 256, 0, st>>>(p.qh, L.q_norm, HEAD_DIM, RMS_EPS);
      k_rmsnorm_head<<<T * N_KV_HEADS, 256, 0, st>>>(p.kh, L.k_norm, HEAD_DIM, RMS_EPS);
      rope(p.qh, N_Q_HEADS, HEAD_DIM, ROTARY_DIM, rt.dpos, ROPE_THETA, st, T, mpos);
      rope(p.kh, N_KV_HEADS, HEAD_DIM, ROTARY_DIM, rt.dpos, ROPE_THETA, st, T, mpos);

      const int fi = full_index(l);
      float *kc = rt.kcache + (size_t)fi * rt.ctx * KV_DIM;
      float *vc = rt.vcache + (size_t)fi * rt.ctx * KV_DIM;
      k_kv_append<<<dim3((KV_DIM + 255) / 256, T), 256, 0, st>>>(p.kh, p.vh, kc,
                                                                 vc, rt.dpos, KV_DIM);
      // Every query in the chunk is served by the same kernel decode uses;
      // query t sees positions 0..pos+t, which is the causal mask.
      k_attn_decode<<<dim3(N_Q_HEADS, T), 256, 0, st>>>(
          p.qh, kc, vc, p.attn_out, rt.dpos, HEAD_DIM, N_KV_HEADS, N_Q_HEADS,
          1.0f / sqrtf((float)HEAD_DIM), p.scores, rt.ctx);
      k_gate_sigmoid<<<(T * Q_DIM + 255) / 256, 256, 0, st>>>(p.attn_out, p.gateh,
                                                              T * Q_DIM);
      to_bf16(p.hbf, p.attn_out, (size_t)T * Q_DIM, st);
      proj(L.o_proj, p.hbf, p.attn_out, p.mix, T, HIDDEN, Q_DIM, L.so, st);
    } else {
      proj(L.in_qkv, p.hbf, p.hn, p.qkv, T, LIN_QKV_DIM, HIDDEN, L.s_qkv, st);
      proj(L.in_zab, p.hbf, p.hn, p.fused, T, LIN_Z_DIM + 2 * LIN_V_HEADS, HIDDEN,
           L.s_zab, st);
      unpack3(p.fused, p.zb, p.ab, p.bb, LIN_Z_DIM, LIN_V_HEADS, LIN_V_HEADS, T, st);

      const int li = linear_index(l);
      float *conv = rt.dn_conv + (size_t)li * CONV_CHANNELS * 3;
      float *S = rt.dn_state + (size_t)li * LIN_V_HEADS * LIN_HEAD_DIM * LIN_HEAD_DIM;

      // Only the state update is genuinely sequential. The conv, the gates and
      // both norms are independent per token, so they run once for the whole
      // chunk instead of once per token -- that alone turns ~6*T launches per
      // layer into 4 + T.
      k_conv1d_chunk<<<dim3((CONV_CHANNELS + 255) / 256, T), 256, 0, st>>>(
          p.qkv, p.convout, conv, L.conv_w, CONV_CHANNELS, T);
      k_conv1d_carry<<<(CONV_CHANNELS + 255) / 256, 256, 0, st>>>(
          p.qkv, conv, CONV_CHANNELS, T);
      k_delta_gates<<<dim3(1, T), 64, 0, st>>>(p.ab, p.bb, L.A_log, L.dt_bias,
                                               p.gb, p.betab, LIN_V_HEADS);
      k_l2norm_heads<<<dim3(LIN_K_HEADS, T), 128, 0, st>>>(
          p.convout, LIN_HEAD_DIM, 1.0f / sqrtf((float)LIN_HEAD_DIM), LIN_QKV_DIM);
      k_l2norm_heads<<<dim3(LIN_K_HEADS, T), 128, 0, st>>>(
          p.convout + LIN_Q_DIM, LIN_HEAD_DIM, 1.0f, LIN_QKV_DIM);

      for (int b = 0; b < T; ++b) {
        const float *qkv_b = p.convout + (size_t)b * LIN_QKV_DIM;
        k_delta_step<<<LIN_V_HEADS, 128, 0, st>>>(
            S, qkv_b, qkv_b + LIN_Q_DIM, qkv_b + LIN_Q_DIM + LIN_K_DIM,
            p.gb + (size_t)b * LIN_V_HEADS, p.betab + (size_t)b * LIN_V_HEADS,
            p.dnout + (size_t)b * LIN_V_DIM, LIN_HEAD_DIM, LIN_HEAD_DIM,
            LIN_V_HEADS / LIN_K_HEADS);
      }
      k_rmsnorm_gated<<<LIN_V_HEADS * T, 128, 0, st>>>(p.dnout, p.zb, L.dn_norm,
                                                        LIN_HEAD_DIM, RMS_EPS);
      to_bf16(p.hbf, p.dnout, (size_t)T * LIN_V_DIM, st);
      proj(L.out_proj, p.hbf, p.dnout, p.mix, T, HIDDEN, LIN_V_DIM, L.s_out, st);
    }
    add_into(p.h, p.mix, HIDDEN, st, T);

    rmsnorm(p.hn, p.h, L.post_norm, HIDDEN, RMS_EPS, st, T);
    to_bf16(p.hbf, p.hn, (size_t)T * HIDDEN, st);
    proj(L.gate_up, p.hbf, p.hn, p.mlp_g, T, 2 * INTERMEDIATE, HIDDEN, L.sgu, st);
    swiglu_fused(p.mlp_a, p.mlp_g, INTERMEDIATE, T, st);
    to_bf16(p.mlp_abf, p.mlp_a, (size_t)T * INTERMEDIATE, st);
    proj(L.down_proj, p.mlp_abf, p.mlp_a, p.mix, T, HIDDEN, INTERMEDIATE, L.sd, st);
    add_into(p.h, p.mix, HIDDEN, st, T);
  }

  // Only the last token's prediction is wanted, so the 715 MB lm_head runs on
  // exactly one row.
  copy_vec(rt.mix, p.h + (size_t)(T - 1) * HIDDEN, HIDDEN, st);
  rmsnorm(rt.hn, rt.mix, rt.final_norm, HIDDEN, RMS_EPS, st, 1);
  copy_vec(rt.hlast, rt.hn, HIDDEN, st);   // post-norm, as the draft head wants
  launch_gemm(rt.lm_head, rt.hn, rt.logits, VOCAB, HIDDEN, rt.s_lm, 1, st);
  argmax(rt.logits, VOCAB, rt.dtok_out, st, 1);
  k_advance_by<<<1, 1, 0, st>>>(rt.dpos, T);
  rt.pos += T;
  (void)P;
}

// Feed a whole prompt. Returns the token the model predicts next.
inline int prefill(Runtime &rt, PrefillBuf &p, const std::vector<int> &toks) {
  int next = 0;
  for (size_t i = 0; i < toks.size(); i += (size_t)p.chunk) {
    const int T = (int)((toks.size() - i) < (size_t)p.chunk ? toks.size() - i
                                                            : (size_t)p.chunk);
    LCHECK(cudaMemcpyAsync(p.dtok, toks.data() + i, T * sizeof(int),
                           cudaMemcpyHostToDevice, rt.stream));
    prefill_chunk(rt, p, T);
    LCHECK(cudaMemcpyAsync(&next, rt.dtok_out, sizeof(int), cudaMemcpyDeviceToHost,
                           rt.stream));
    LCHECK(cudaStreamSynchronize(rt.stream));
  }
  return next;
}

}  // namespace spark27
