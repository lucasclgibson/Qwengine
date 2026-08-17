#pragma once
// src/mtp.cu — the draft head, and the reason this engine can beat its own
// bandwidth ceiling.
//
// THE IDEA
// Producing one token costs one full read of the model: 14.65 GB, ~75 ms.
// That is a hard floor. But *checking* several candidate tokens costs almost
// the same as checking one, because the cost is streaming the weights, not the
// arithmetic (see the batched kernel in gemv.cu).
//
// So: a small, cheap head guesses the next few tokens. Then one pass of the
// big model checks them all at once. Every guess the big model agrees with is
// a token we got for free. Crucially the big model has the final say, so the
// output is exactly what greedy decoding would have produced — speculation
// buys speed, never a different answer.
//
// THE HEAD
// Qwen3.8 ships its own draft head in the checkpoint (15 tensors, `mtp.*`):
// one transformer block that takes the main model's last hidden state plus the
// embedding of the token just produced, and predicts the token after it.
//
//   x = fc( concat[ norm_e(embed(token)) , norm_h(hidden) ] )   10240 -> 5120
//   x = one transformer block (own KV cache)
//   draft = argmax( lm_head( norm(x) ) )
//
// Chaining it (feed x and the drafted token back in) drafts more than one
// token ahead.

#ifndef MTP_CHAIN_PRENORM
#define MTP_CHAIN_PRENORM 0
#endif
#include <cuda_runtime.h>

#include "../qwen38.h"
#include "elem.cu"
#include "gemv.cu"
#include "loader.cpp"
#include "mixers.cu"

namespace spark27 {

using namespace q38;

struct MtpHead {
  const uint16_t *pre_fc_norm_h = nullptr, *pre_fc_norm_e = nullptr;
  const uint8_t *fc = nullptr;
  float s_fc = 1;
  const uint16_t *in_norm = nullptr, *post_norm = nullptr, *out_norm = nullptr;
  const uint8_t *q_proj = nullptr, *k_proj = nullptr, *v_proj = nullptr, *o_proj = nullptr;
  float sq = 1, sk = 1, sv = 1, so = 1;
  const uint16_t *q_norm = nullptr, *k_norm = nullptr;
  const uint8_t *gate_proj = nullptr, *up_proj = nullptr, *down_proj = nullptr;
  float sg = 1, su = 1, sd = 1;

  // scratch
  float *e = nullptr, *cat = nullptr, *x = nullptr, *xn = nullptr, *mix = nullptr;
  float *qproj = nullptr, *qh = nullptr, *gateh = nullptr, *kh = nullptr, *vh = nullptr;
  float *attn = nullptr, *mg = nullptr, *mu = nullptr, *ma = nullptr;
  float *logits = nullptr, *scores = nullptr;
  float *kc = nullptr, *vc = nullptr;   // its own KV cache, separate from the trunk
  int *dtok = nullptr, *dpos = nullptr;
  int *dtok_in = nullptr;     // the committed token the chain starts from
  int *dchain = nullptr;      // the drafted tokens, one per depth
  int ctx = 0;
  // The draft's own copy of lm_head at 2.5 bits instead of 4.5 (397 MB vs
  // 715 MB). Read once per draft step, so it is most of the draft's cost.
  const uint8_t *lm_draft = nullptr;
  float s_lm_draft = 1.0f;
  bool have_draft_head = false;
  // The whole draft chain, recorded once. Without this the chain is ~25 kernel
  // launches per step and the launch overhead is comparable to the weight
  // traffic it is trying to hide.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  int depth = 0;
};

// Concatenate two 5120-vectors. Which half is which matters: mtp.fc's right
// half is close to a scaled identity, so the hidden state — the signal that
// should pass through nearly unchanged — belongs in the SECOND half.
// Order is therefore [embedding, hidden]. Flipping it is a one-line change and
// the draft accuracy immediately tells you if it is wrong.
__global__ void k_concat2(float *__restrict__ out, const float *__restrict__ a,
                          const float *__restrict__ b, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * n) return;
  out[i] = (i < n) ? a[i] : b[i - n];
}

inline void mtp_init(MtpHead &M, Model &m, int ctx) {
  M.ctx = ctx;
  auto T = [&](const char *n) { return m.get(n); };
  M.pre_fc_norm_h = (const uint16_t *)T("mtp.pre_fc_norm_hidden.weight").dev;
  M.pre_fc_norm_e = (const uint16_t *)T("mtp.pre_fc_norm_embedding.weight").dev;
  Tensor t = T("mtp.fc.weight");                 M.fc = t.dev;        M.s_fc = t.scale;
  M.out_norm = (const uint16_t *)T("mtp.norm.weight").dev;
  M.in_norm  = (const uint16_t *)T("mtp.layers.0.input_layernorm.weight").dev;
  M.post_norm= (const uint16_t *)T("mtp.layers.0.post_attention_layernorm.weight").dev;
  t = T("mtp.layers.0.self_attn.q_proj.weight"); M.q_proj = t.dev;    M.sq = t.scale;
  t = T("mtp.layers.0.self_attn.k_proj.weight"); M.k_proj = t.dev;    M.sk = t.scale;
  t = T("mtp.layers.0.self_attn.v_proj.weight"); M.v_proj = t.dev;    M.sv = t.scale;
  t = T("mtp.layers.0.self_attn.o_proj.weight"); M.o_proj = t.dev;    M.so = t.scale;
  M.q_norm = (const uint16_t *)T("mtp.layers.0.self_attn.q_norm.weight").dev;
  M.k_norm = (const uint16_t *)T("mtp.layers.0.self_attn.k_norm.weight").dev;
  t = T("mtp.layers.0.mlp.gate_proj.weight");    M.gate_proj = t.dev; M.sg = t.scale;
  t = T("mtp.layers.0.mlp.up_proj.weight");      M.up_proj = t.dev;   M.su = t.scale;
  t = T("mtp.layers.0.mlp.down_proj.weight");    M.down_proj = t.dev; M.sd = t.scale;

  auto A = [&](float **p, size_t n) { LCHECK(cudaMalloc(p, n * sizeof(float))); };
  A(&M.e, HIDDEN); A(&M.cat, 2 * HIDDEN); A(&M.x, HIDDEN); A(&M.xn, HIDDEN);
  A(&M.mix, HIDDEN);
  A(&M.qproj, QGATE_DIM); A(&M.qh, Q_DIM); A(&M.gateh, Q_DIM);
  A(&M.kh, KV_DIM); A(&M.vh, KV_DIM); A(&M.attn, Q_DIM);
  A(&M.mg, INTERMEDIATE); A(&M.mu, INTERMEDIATE); A(&M.ma, INTERMEDIATE);
  A(&M.logits, VOCAB); A(&M.scores, (size_t)N_Q_HEADS * ctx);
  A(&M.kc, (size_t)ctx * KV_DIM); A(&M.vc, (size_t)ctx * KV_DIM);
  LCHECK(cudaMalloc(&M.dtok, sizeof(int)));
  LCHECK(cudaMalloc(&M.dpos, sizeof(int)));
  LCHECK(cudaMalloc(&M.dtok_in, sizeof(int)));
  LCHECK(cudaMalloc(&M.dchain, 16 * sizeof(int)));
  LCHECK(cudaMemset(M.dpos, 0, sizeof(int)));

  // MEASURED NET LOSS — off by default.
  //
  // The 2-bit draft head does save bytes (0.397 GB vs 0.715), but the draft is
  // only ~11% of a speculative cycle, so that is ~4% of the time, while the
  // accuracy it costs is worse: mean accepted tokens fell 2.67 -> 2.56 and
  // 2.20 -> 2.00 on the two test prompts, i.e. 28.43/21.05 tok/s against
  // 29.79/23.16 with the 4-bit head. Drafting quality is worth more than draft
  // bandwidth at this depth.
  //
  // The D2 format, packer and kernel are kept: they work, and the useful thing
  // learned is that 2 bits costs ~7% accuracy, which is the number that
  // matters when considering low-bit weights for the TRUNK.
#ifdef USE_D2_DRAFT_HEAD
  if (m.has("lm_head.weight.draft")) {
    Tensor d = m.get("lm_head.weight.draft");
    M.lm_draft = d.dev;
    M.s_lm_draft = d.scale;
    M.have_draft_head = true;
    printf("mtp: using 2-bit draft lm_head (%.3f GB)\n", d.bytes / 1e9);
  }
#endif
}

// One draft step.
//   hidden : the trunk's last hidden state (pre-final-norm) for the token we
//            just committed to
//   dtoken : that token's id, on the device
// Leaves the drafted token in M.dtok and the block output in M.x, so chaining
// is just calling this again with (M.x, M.dtok).
inline void mtp_draft(MtpHead &M, const uint8_t *embed_tab, float embed_scale,
                      const uint8_t *lm_head, float lm_scale,
                      const float *hidden, const int *dtoken, int *out_tok,
                      cudaStream_t st) {
  embed(M.e, embed_tab, dtoken, HIDDEN, embed_scale, st, 1);

  // norm each half separately, then join: [embedding, hidden]
  rmsnorm(M.cat, M.e, M.pre_fc_norm_e, HIDDEN, RMS_EPS, st, 1);
  rmsnorm(M.cat + HIDDEN, hidden, M.pre_fc_norm_h, HIDDEN, RMS_EPS, st, 1);
  launch_gemm(M.fc, M.cat, M.x, HIDDEN, MTP_FC_IN, M.s_fc, 1, st);

  // ---- one transformer block, same geometry as a trunk attention layer ----
  rmsnorm(M.xn, M.x, M.in_norm, HIDDEN, RMS_EPS, st, 1);
  launch_gemm(M.q_proj, M.xn, M.qproj, QGATE_DIM, HIDDEN, M.sq, 1, st);
  launch_gemm(M.k_proj, M.xn, M.kh, KV_DIM, HIDDEN, M.sk, 1, st);
  launch_gemm(M.v_proj, M.xn, M.vh, KV_DIM, HIDDEN, M.sv, 1, st);
  k_split_qgate<<<(Q_DIM + 255) / 256, 256, 0, st>>>(M.qproj, M.qh, M.gateh,
                                                     N_Q_HEADS, HEAD_DIM);
  k_rmsnorm_head<<<N_Q_HEADS, 256, 0, st>>>(M.qh, M.q_norm, HEAD_DIM, RMS_EPS);
  k_rmsnorm_head<<<N_KV_HEADS, 256, 0, st>>>(M.kh, M.k_norm, HEAD_DIM, RMS_EPS);
  rope(M.qh, N_Q_HEADS, HEAD_DIM, ROTARY_DIM, M.dpos, ROPE_THETA, st, 1);
  rope(M.kh, N_KV_HEADS, HEAD_DIM, ROTARY_DIM, M.dpos, ROPE_THETA, st, 1);
  k_kv_append<<<dim3((KV_DIM + 255) / 256, 1), 256, 0, st>>>(M.kh, M.vh, M.kc,
                                                             M.vc, M.dpos, KV_DIM);
  k_attn_decode<<<dim3(N_Q_HEADS, 1), 256, 0, st>>>(
      M.qh, M.kc, M.vc, M.attn, M.dpos, HEAD_DIM, N_KV_HEADS, N_Q_HEADS,
      1.0f / sqrtf((float)HEAD_DIM), M.scores, M.ctx);
  k_gate_sigmoid<<<(Q_DIM + 255) / 256, 256, 0, st>>>(M.attn, M.gateh, Q_DIM);
  launch_gemm(M.o_proj, M.attn, M.mix, HIDDEN, Q_DIM, M.so, 1, st);
  add_into(M.x, M.mix, HIDDEN, st, 1);

  rmsnorm(M.xn, M.x, M.post_norm, HIDDEN, RMS_EPS, st, 1);
  launch_gemm(M.gate_proj, M.xn, M.mg, INTERMEDIATE, HIDDEN, M.sg, 1, st);
  launch_gemm(M.up_proj, M.xn, M.mu, INTERMEDIATE, HIDDEN, M.su, 1, st);
  swiglu(M.ma, M.mg, M.mu, INTERMEDIATE, st, 1);
  launch_gemm(M.down_proj, M.ma, M.mix, HIDDEN, INTERMEDIATE, M.sd, 1, st);
  add_into(M.x, M.mix, HIDDEN, st, 1);

  // read out through the head's own norm and the shared lm_head
  rmsnorm(M.xn, M.x, M.out_norm, HIDDEN, RMS_EPS, st, 1);
  if (M.have_draft_head)
    launch_gemv_d2(M.lm_draft, M.xn, M.logits, VOCAB, HIDDEN, M.s_lm_draft, st);
  else
    launch_gemm(lm_head, M.xn, M.logits, VOCAB, HIDDEN, lm_scale, 1, st);
  argmax(M.logits, VOCAB, out_tok, st, 1);
  k_advance_by<<<1, 1, 0, st>>>(M.dpos, 1);
}

// Record the whole chain: depth guesses, each feeding the next. Fixed pointers
// throughout, so it captures cleanly and replays as one submission.
inline void mtp_capture(MtpHead &M, const uint8_t *embed_tab, float embed_scale,
                        const uint8_t *lm_head, float lm_scale,
                        const float *trunk_hlast, int depth, cudaStream_t st) {
  M.depth = depth;
  LCHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
  const float *h = trunk_hlast;
  const int *tok = M.dtok_in;
  for (int d = 0; d < depth; ++d) {
    mtp_draft(M, embed_tab, embed_scale, lm_head, lm_scale, h, tok, M.dchain + d, st);
    // Chain on the head's own normed output, matching the fact that step 1
    // consumes the trunk's POST-final-norm hidden state. The two are the same
    // choice and must be made together; measured, post-norm raises first-draft
    // accuracy from 87.2% to 95.7% on easy text.
#if MTP_CHAIN_PRENORM
    h = M.x;
#else
    h = M.xn;
#endif
    tok = M.dchain + d;   // and on the token just drafted
  }
  LCHECK(cudaStreamEndCapture(st, &M.graph));
  LCHECK(cudaGraphInstantiate(&M.graph_exec, M.graph, nullptr, nullptr, 0));
  size_t nodes = 0;
  LCHECK(cudaGraphGetNodes(M.graph, nullptr, &nodes));
  printf("cuda graph: %zu nodes for a depth-%d draft chain\n", nodes, depth);
}

}  // namespace spark27
