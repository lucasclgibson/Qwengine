#pragma once
// src/model.cu — one token in, one token out.
//
// THE SHAPE OF A FORWARD PASS
// A token id comes in. We look up its 5120-number "meaning" vector, then push
// that vector through 64 layers. Each layer does two things, and each adds its
// result back onto the vector rather than replacing it (a "residual" — it lets
// 64 layers refine a signal instead of 64 chances to destroy it):
//
//   1. a MIXER, which looks at earlier tokens (DeltaNet or attention)
//   2. an MLP, which thinks about this token on its own
//
// At the end, multiply by lm_head to score all 248320 possible next tokens,
// and take the highest.
//
// Every layer is preceded by a norm, and the mixer alternates 3 DeltaNet
// layers to 1 attention layer. See mixers.cu for what those do.

#include <cuda_runtime.h>
#include <stdio.h>

#include "../qwen38.h"
#include "elem.cu"
#include "gemv.cu"
#include "loader.cpp"
#include "mixers.cu"

namespace spark27 {

using namespace q38;

// Pointers resolved once at startup. Looking tensors up by name inside the
// decode loop would mean 400 string hashes per token.
struct LayerPtrs {
  bool full;
  const uint16_t *in_norm, *post_norm;
  // full attention
  const uint8_t *qkv_proj, *o_proj;      // q|k|v fused into one matrix
  float sqkv, so;
  const uint16_t *q_norm, *k_norm;
  // deltanet
  const uint8_t *in_qkv, *in_zab, *out_proj;   // z|a|b fused
  float s_qkv, s_zab, s_out;
  const uint16_t *conv_w, *A_log, *dt_bias, *dn_norm;
  // mlp
  const uint8_t *gate_up, *down_proj;          // gate|up fused
  float sgu, sd;
};

struct Runtime {
  Model *m = nullptr;
  int ctx = 0;
  LayerPtrs L[N_LAYERS];
  const uint8_t *embed = nullptr, *lm_head = nullptr;
  float s_embed = 1, s_lm = 1;
  const uint16_t *final_norm = nullptr;

  // activations
  float *h, *hn, *mix;
  float *qkv, *zb, *ab, *bb, *gb, *betab, *dnout;
  float *qproj, *qh, *gateh, *kh, *vh, *attn_out;
  float *mlp_g, *mlp_a, *fused;
  float *logits, *scores;
  float *hlast;   // last layer's hidden state, before the final norm — this is
                  // what the MTP draft head consumes
  int *dtok_in;   // token being processed  (device-resident so a graph can read it)
  int *dtok_out;  // token argmax chose
  int *dpos;      // position, incremented on device at the end of each step

  // persistent state
  float *dn_state;  // [48 linear layers][48 heads][128][128]
  float *dn_conv;   // [48 linear layers][10240][3]
  float *kcache, *vcache;  // [16 full layers][ctx][KV_DIM]

  // Speculation needs to be able to UNDO a step.
  //
  // The KV cache heals itself: entries are written by position, so a rejected
  // token's slot is simply overwritten next cycle. The DeltaNet recurrence
  // does not — its state has already absorbed the rejected token, and the
  // update is not safely invertible (undoing means dividing by the decay,
  // which amplifies error and would break exactness).
  //
  // So: snapshot the state before a speculative pass, and record what each
  // layer fed into its recurrence. On a rejection, restore the snapshot and
  // replay only the accepted tokens. 151 MB + 6 MB of snapshot, 16 MB of
  // recorded inputs — against 14.65 GB of weight traffic per pass, it is noise.
  float *dn_backup = nullptr, *conv_backup = nullptr;
  float *dn_save = nullptr;   // [linear layer][b][ raw qkv 10240 | a 48 | b 48 ]
  int pos = 0;

  // One token's work, recorded once and replayed. A token is ~1465 kernel
  // launches; at a few microseconds each that is milliseconds of pure driver
  // overhead per token. A CUDA graph submits the whole lot as a single unit.
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  bool captured = false;
  // A second graph for the fixed-size speculative verify batch.
  cudaGraph_t vgraph = nullptr;
  cudaGraphExec_t vgraph_exec = nullptr;
  int vbatch = 0;
};


static inline int linear_index(int layer) { return layer - layer / 4; }
static inline int full_index(int layer) { return layer / 4; }

// Widest speculative batch we will capture a graph for.
constexpr int MAX_B = 8;

#define DALLOC(p, n) LCHECK(cudaMalloc(&(p), (size_t)(n) * sizeof(float)))

inline void rt_capture(Runtime &rt);

inline void rt_init(Runtime &rt, Model &m, int ctx) {
  rt.m = &m;
  rt.ctx = ctx;

  auto W = [&](const char *fmt, int l) {
    char b[128];
    snprintf(b, sizeof b, fmt, l);
    return m.get(b);
  };

  for (int l = 0; l < N_LAYERS; ++l) {
    LayerPtrs &p = rt.L[l];
    p.full = is_full_attn(l);
    p.in_norm = (const uint16_t *)W("model.language_model.layers.%d.input_layernorm.weight", l).dev;
    p.post_norm = (const uint16_t *)W("model.language_model.layers.%d.post_attention_layernorm.weight", l).dev;
    Tensor t;
    if (p.full) {
      t = W("model.language_model.layers.%d.self_attn.qkv_proj.weight", l); p.qkv_proj = t.dev; p.sqkv = t.scale;
      t = W("model.language_model.layers.%d.self_attn.o_proj.weight", l); p.o_proj = t.dev; p.so = t.scale;
      p.q_norm = (const uint16_t *)W("model.language_model.layers.%d.self_attn.q_norm.weight", l).dev;
      p.k_norm = (const uint16_t *)W("model.language_model.layers.%d.self_attn.k_norm.weight", l).dev;
    } else {
      t = W("model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", l); p.in_qkv = t.dev; p.s_qkv = t.scale;
      t = W("model.language_model.layers.%d.linear_attn.in_proj_zab.weight", l); p.in_zab = t.dev; p.s_zab = t.scale;
      t = W("model.language_model.layers.%d.linear_attn.out_proj.weight", l);    p.out_proj = t.dev; p.s_out = t.scale;
      p.conv_w  = (const uint16_t *)W("model.language_model.layers.%d.linear_attn.conv1d.weight", l).dev;
      p.A_log   = (const uint16_t *)W("model.language_model.layers.%d.linear_attn.A_log", l).dev;
      p.dt_bias = (const uint16_t *)W("model.language_model.layers.%d.linear_attn.dt_bias", l).dev;
      p.dn_norm = (const uint16_t *)W("model.language_model.layers.%d.linear_attn.norm.weight", l).dev;
    }
    t = W("model.language_model.layers.%d.mlp.gate_up_proj.weight", l); p.gate_up = t.dev; p.sgu = t.scale;
    t = W("model.language_model.layers.%d.mlp.down_proj.weight", l); p.down_proj = t.dev; p.sd = t.scale;
  }
  Tensor e = m.get("model.language_model.embed_tokens.weight");
  rt.embed = e.dev; rt.s_embed = e.scale;
  Tensor lh = m.get("lm_head.weight");
  rt.lm_head = lh.dev; rt.s_lm = lh.scale;
  rt.final_norm = (const uint16_t *)m.get("model.language_model.norm.weight").dev;

  DALLOC(rt.h, MAX_B * HIDDEN);       DALLOC(rt.hn, MAX_B * HIDDEN);
  DALLOC(rt.mix, MAX_B * HIDDEN);
  DALLOC(rt.qkv, MAX_B * LIN_QKV_DIM); DALLOC(rt.zb, MAX_B * LIN_Z_DIM);
  DALLOC(rt.ab, MAX_B * LIN_V_HEADS);  DALLOC(rt.bb, MAX_B * LIN_V_HEADS);
  DALLOC(rt.gb, MAX_B * LIN_V_HEADS);  DALLOC(rt.betab, MAX_B * LIN_V_HEADS);
  DALLOC(rt.dnout, MAX_B * LIN_V_DIM);
  DALLOC(rt.qproj, MAX_B * QGATE_DIM); DALLOC(rt.qh, MAX_B * Q_DIM);
  DALLOC(rt.gateh, MAX_B * Q_DIM);
  DALLOC(rt.kh, MAX_B * KV_DIM);       DALLOC(rt.vh, MAX_B * KV_DIM);
  DALLOC(rt.attn_out, MAX_B * Q_DIM);
  DALLOC(rt.mlp_g, MAX_B * 2 * INTERMEDIATE);   // holds gate|up fused
  DALLOC(rt.mlp_a, MAX_B * INTERMEDIATE);
  DALLOC(rt.fused, MAX_B * (QGATE_DIM + 2 * KV_DIM));
  DALLOC(rt.logits, (size_t)MAX_B * VOCAB);
  DALLOC(rt.scores, (size_t)MAX_B * N_Q_HEADS * ctx);
  DALLOC(rt.hlast, MAX_B * HIDDEN);
  LCHECK(cudaMalloc(&rt.dtok_in, MAX_B * sizeof(int)));
  LCHECK(cudaMalloc(&rt.dtok_out, MAX_B * sizeof(int)));
  LCHECK(cudaMalloc(&rt.dpos, sizeof(int)));
  LCHECK(cudaMemset(rt.dpos, 0, sizeof(int)));
  LCHECK(cudaStreamCreate(&rt.stream));

  const size_t st = (size_t)N_LINEAR_LAYERS * LIN_V_HEADS * LIN_HEAD_DIM * LIN_HEAD_DIM;
  DALLOC(rt.dn_state, st);
  LCHECK(cudaMemset(rt.dn_state, 0, st * sizeof(float)));
  const size_t cs = (size_t)N_LINEAR_LAYERS * CONV_CHANNELS * 3;
  DALLOC(rt.dn_conv, cs);
  LCHECK(cudaMemset(rt.dn_conv, 0, cs * sizeof(float)));
  const size_t kv = (size_t)N_FULL_LAYERS * ctx * KV_DIM;
  DALLOC(rt.kcache, kv);
  DALLOC(rt.vcache, kv);
  DALLOC(rt.dn_backup, st);
  DALLOC(rt.conv_backup, cs);
  const size_t sv = (size_t)N_LINEAR_LAYERS * MAX_B * (LIN_QKV_DIM + 2 * LIN_V_HEADS);
  DALLOC(rt.dn_save, sv);

  printf("runtime: ctx %d, deltanet state %.1f MB, kv cache %.1f MB\n", ctx,
         st * 4 / 1e6, kv * 2 * 4 / 1e6);
  rt_capture(rt);
}

inline void rt_reset(Runtime &rt) {
  rt.pos = 0;
  LCHECK(cudaMemset(rt.dpos, 0, sizeof(int)));
  const size_t st = (size_t)N_LINEAR_LAYERS * LIN_V_HEADS * LIN_HEAD_DIM * LIN_HEAD_DIM;
  LCHECK(cudaMemset(rt.dn_state, 0, st * sizeof(float)));
  const size_t cs = (size_t)N_LINEAR_LAYERS * CONV_CHANNELS * 3;
  LCHECK(cudaMemset(rt.dn_conv, 0, cs * sizeof(float)));
}

// Push one token through the whole model. Returns nothing; the chosen next
// token lands in rt.dtok on the device.
inline void record_step(Runtime &rt, int B, bool save = false) {
  cudaStream_t st = rt.stream;

  embed(rt.h, rt.embed, rt.dtok_in, HIDDEN, rt.s_embed, st, B);

  for (int l = 0; l < N_LAYERS; ++l) {
    const LayerPtrs &p = rt.L[l];

    // ---- mixer -------------------------------------------------------------
    rmsnorm(rt.hn, rt.h, p.in_norm, HIDDEN, RMS_EPS, st, B);

    if (p.full) {
      // One matmul for q, k and v: same input, so they are one taller matrix.
      launch_gemm(p.qkv_proj, rt.hn, rt.fused, QGATE_DIM + 2 * KV_DIM, HIDDEN,
                  p.sqkv, B, st);
      unpack3(rt.fused, rt.qproj, rt.kh, rt.vh, QGATE_DIM, KV_DIM, KV_DIM, B, st);

      k_split_qgate<<<(B * Q_DIM + 255) / 256, 256, 0, st>>>(
          rt.qproj, rt.qh, rt.gateh, B * N_Q_HEADS, HEAD_DIM);
      k_rmsnorm_head<<<B * N_Q_HEADS, 256, 0, st>>>(rt.qh, p.q_norm, HEAD_DIM, RMS_EPS);
      k_rmsnorm_head<<<B * N_KV_HEADS, 256, 0, st>>>(rt.kh, p.k_norm, HEAD_DIM, RMS_EPS);

      rope(rt.qh, N_Q_HEADS, HEAD_DIM, ROTARY_DIM, rt.dpos, ROPE_THETA, st, B);
      rope(rt.kh, N_KV_HEADS, HEAD_DIM, ROTARY_DIM, rt.dpos, ROPE_THETA, st, B);

      const int fi = full_index(l);
      float *kc = rt.kcache + (size_t)fi * rt.ctx * KV_DIM;
      float *vc = rt.vcache + (size_t)fi * rt.ctx * KV_DIM;
      k_kv_append<<<dim3((KV_DIM + 255) / 256, B), 256, 0, st>>>(rt.kh, rt.vh, kc,
                                                                 vc, rt.dpos, KV_DIM);
      k_attn_decode<<<dim3(N_Q_HEADS, B), 256, 0, st>>>(
          rt.qh, kc, vc, rt.attn_out, rt.dpos, HEAD_DIM, N_KV_HEADS, N_Q_HEADS,
          1.0f / sqrtf((float)HEAD_DIM), rt.scores, rt.ctx);
      k_gate_sigmoid<<<(B * Q_DIM + 255) / 256, 256, 0, st>>>(rt.attn_out, rt.gateh,
                                                              B * Q_DIM);
      launch_gemm(p.o_proj, rt.attn_out, rt.mix, HIDDEN, Q_DIM, p.so, B, st);
    } else {
      launch_gemm(p.in_qkv, rt.hn, rt.qkv, LIN_QKV_DIM, HIDDEN, p.s_qkv, B, st);
      // z, a and b likewise. a and b are [48,5120] on their own, which makes
      // 6 blocks for 48 SMs and runs at 6% of bandwidth; folded in, they cost
      // nothing.
      launch_gemm(p.in_zab, rt.hn, rt.fused, LIN_Z_DIM + 2 * LIN_V_HEADS, HIDDEN,
                  p.s_zab, B, st);
      unpack3(rt.fused, rt.zb, rt.ab, rt.bb, LIN_Z_DIM, LIN_V_HEADS, LIN_V_HEADS,
              B, st);

      const int li = linear_index(l);
      float *conv = rt.dn_conv + (size_t)li * CONV_CHANNELS * 3;
      float *S = rt.dn_state + (size_t)li * LIN_V_HEADS * LIN_HEAD_DIM * LIN_HEAD_DIM;

      // The projections batch, but the recurrence cannot: token b+1's state
      // depends on token b's. So we step the state B times. It is cheap next
      // to the weight streaming — the state is 3 MB against 14.65 GB of
      // weights — so the sequential part costs us very little.
      const size_t SAVE_STRIDE = LIN_QKV_DIM + 2 * LIN_V_HEADS;
      for (int b = 0; b < B; ++b) {
        float *qkv_b = rt.qkv + (size_t)b * LIN_QKV_DIM;
        if (save) {
          // Record the RAW projection outputs, before the conv consumes them,
          // so a replay can reproduce this step exactly.
          float *sv = rt.dn_save + ((size_t)li * MAX_B + b) * SAVE_STRIDE;
          copy_vec(sv, qkv_b, LIN_QKV_DIM, st);
          copy_vec(sv + LIN_QKV_DIM, rt.ab + (size_t)b * LIN_V_HEADS, LIN_V_HEADS, st);
          copy_vec(sv + LIN_QKV_DIM + LIN_V_HEADS, rt.bb + (size_t)b * LIN_V_HEADS,
                   LIN_V_HEADS, st);
        }
        k_conv1d_step<<<(CONV_CHANNELS + 255) / 256, 256, 0, st>>>(qkv_b, conv,
                                                                    p.conv_w,
                                                                    CONV_CHANNELS);
        k_delta_gates<<<1, 64, 0, st>>>(rt.ab + (size_t)b * LIN_V_HEADS,
                                        rt.bb + (size_t)b * LIN_V_HEADS, p.A_log,
                                        p.dt_bias, rt.gb + (size_t)b * LIN_V_HEADS,
                                        rt.betab + (size_t)b * LIN_V_HEADS,
                                        LIN_V_HEADS);
        k_l2norm_heads<<<LIN_K_HEADS, 128, 0, st>>>(qkv_b, LIN_HEAD_DIM,
                                                    1.0f / sqrtf((float)LIN_HEAD_DIM));
        k_l2norm_heads<<<LIN_K_HEADS, 128, 0, st>>>(qkv_b + LIN_Q_DIM, LIN_HEAD_DIM, 1.0f);
        k_delta_step<<<LIN_V_HEADS, 128, 0, st>>>(
            S, qkv_b, qkv_b + LIN_Q_DIM, qkv_b + LIN_Q_DIM + LIN_K_DIM,
            rt.gb + (size_t)b * LIN_V_HEADS, rt.betab + (size_t)b * LIN_V_HEADS,
            rt.dnout + (size_t)b * LIN_V_DIM, LIN_HEAD_DIM, LIN_HEAD_DIM,
            LIN_V_HEADS / LIN_K_HEADS);
        k_rmsnorm_gated<<<LIN_V_HEADS, 128, 0, st>>>(
            rt.dnout + (size_t)b * LIN_V_DIM, rt.zb + (size_t)b * LIN_Z_DIM,
            p.dn_norm, LIN_HEAD_DIM, RMS_EPS);
      }
      launch_gemm(p.out_proj, rt.dnout, rt.mix, HIDDEN, LIN_V_DIM, p.s_out, B, st);
    }
    add_into(rt.h, rt.mix, HIDDEN, st, B);  // residual

    // ---- MLP ---------------------------------------------------------------
    rmsnorm(rt.hn, rt.h, p.post_norm, HIDDEN, RMS_EPS, st, B);
    launch_gemm(p.gate_up, rt.hn, rt.mlp_g, 2 * INTERMEDIATE, HIDDEN, p.sgu, B, st);
    swiglu_fused(rt.mlp_a, rt.mlp_g, INTERMEDIATE, B, st);
    launch_gemm(p.down_proj, rt.mlp_a, rt.mix, HIDDEN, INTERMEDIATE, p.sd, B, st);
    add_into(rt.h, rt.mix, HIDDEN, st, B);
  }

  // The draft head reads the hidden state AFTER the trunk's final norm.
  // The checkpoint does not state which; mtp.norm existing as the head's own
  // read-out norm argues for pre-norm, but that was inference. Measured:
  // post-norm lifts first-draft accuracy 87.2% -> 95.7% (easy) and
  // 78.7% -> 80.9% (prose), so post-norm it is.
  rmsnorm(rt.hn, rt.h, rt.final_norm, HIDDEN, RMS_EPS, st, B);
  copy_vec(rt.hlast, rt.hn, B * HIDDEN, st);
  launch_gemm(rt.lm_head, rt.hn, rt.logits, VOCAB, HIDDEN, rt.s_lm, B, st);
  argmax(rt.logits, VOCAB, rt.dtok_out, st, B);
  k_advance_by<<<1, 1, 0, st>>>(rt.dpos, B);
}

// Record the step once. Capture does not execute anything, so it leaves the
// caches and the position untouched; it just writes down the work.
inline void rt_capture(Runtime &rt) {
  LCHECK(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed));
  record_step(rt, 1);
  LCHECK(cudaStreamEndCapture(rt.stream, &rt.graph));
  LCHECK(cudaGraphInstantiate(&rt.graph_exec, rt.graph, nullptr, nullptr, 0));
  rt.captured = true;
  size_t nodes = 0;
  LCHECK(cudaGraphGetNodes(rt.graph, nullptr, &nodes));
  printf("cuda graph: %zu nodes captured for one token\n", nodes);
}

// Snapshot the recurrent + conv state before a speculative pass.
inline void dn_snapshot(Runtime &rt) {
  const size_t stn = (size_t)N_LINEAR_LAYERS * LIN_V_HEADS * LIN_HEAD_DIM * LIN_HEAD_DIM;
  const size_t csn = (size_t)N_LINEAR_LAYERS * CONV_CHANNELS * 3;
  LCHECK(cudaMemcpyAsync(rt.dn_backup, rt.dn_state, stn * 4, cudaMemcpyDeviceToDevice, rt.stream));
  LCHECK(cudaMemcpyAsync(rt.conv_backup, rt.dn_conv, csn * 4, cudaMemcpyDeviceToDevice, rt.stream));
}

// Rewind the recurrence to the snapshot, then re-apply exactly `n` tokens
// using the inputs recorded during the speculative pass. Only the state
// matters here, so the output norm and out_proj are skipped.
inline void dn_replay(Runtime &rt, int n) {
  cudaStream_t st = rt.stream;
  const size_t stn = (size_t)N_LINEAR_LAYERS * LIN_V_HEADS * LIN_HEAD_DIM * LIN_HEAD_DIM;
  const size_t csn = (size_t)N_LINEAR_LAYERS * CONV_CHANNELS * 3;
  LCHECK(cudaMemcpyAsync(rt.dn_state, rt.dn_backup, stn * 4, cudaMemcpyDeviceToDevice, st));
  LCHECK(cudaMemcpyAsync(rt.dn_conv, rt.conv_backup, csn * 4, cudaMemcpyDeviceToDevice, st));
  if (n <= 0) return;
  const size_t SAVE_STRIDE = LIN_QKV_DIM + 2 * LIN_V_HEADS;
  for (int l = 0; l < N_LAYERS; ++l) {
    if (is_full_attn(l)) continue;
    const LayerPtrs &p = rt.L[l];
    const int li = linear_index(l);
    float *conv = rt.dn_conv + (size_t)li * CONV_CHANNELS * 3;
    float *S = rt.dn_state + (size_t)li * LIN_V_HEADS * LIN_HEAD_DIM * LIN_HEAD_DIM;
    for (int b = 0; b < n; ++b) {
      const float *sv = rt.dn_save + ((size_t)li * MAX_B + b) * SAVE_STRIDE;
      // conv and l2norm work in place, so replay through scratch
      copy_vec(rt.qkv, sv, LIN_QKV_DIM, st);
      k_conv1d_step<<<(CONV_CHANNELS + 255) / 256, 256, 0, st>>>(rt.qkv, conv,
                                                                  p.conv_w,
                                                                  CONV_CHANNELS);
      k_delta_gates<<<1, 64, 0, st>>>(sv + LIN_QKV_DIM,
                                      sv + LIN_QKV_DIM + LIN_V_HEADS, p.A_log,
                                      p.dt_bias, rt.gb, rt.betab, LIN_V_HEADS);
      k_l2norm_heads<<<LIN_K_HEADS, 128, 0, st>>>(rt.qkv, LIN_HEAD_DIM,
                                                  1.0f / sqrtf((float)LIN_HEAD_DIM));
      k_l2norm_heads<<<LIN_K_HEADS, 128, 0, st>>>(rt.qkv + LIN_Q_DIM, LIN_HEAD_DIM, 1.0f);
      k_delta_step<<<LIN_V_HEADS, 128, 0, st>>>(
          S, rt.qkv, rt.qkv + LIN_Q_DIM, rt.qkv + LIN_Q_DIM + LIN_K_DIM, rt.gb,
          rt.betab, rt.dnout, LIN_HEAD_DIM, LIN_HEAD_DIM,
          LIN_V_HEADS / LIN_K_HEADS);
    }
  }
}

__global__ void k_set_pos(int *pos, int v) { *pos = v; }

// Capture the verify pass for a fixed batch size.
inline void rt_capture_verify(Runtime &rt, int B) {
  rt.vbatch = B;
  LCHECK(cudaStreamBeginCapture(rt.stream, cudaStreamCaptureModeRelaxed));
  record_step(rt, B, /*save=*/true);
  LCHECK(cudaStreamEndCapture(rt.stream, &rt.vgraph));
  LCHECK(cudaGraphInstantiate(&rt.vgraph_exec, rt.vgraph, nullptr, nullptr, 0));
  size_t nodes = 0;
  LCHECK(cudaGraphGetNodes(rt.vgraph, nullptr, &nodes));
  printf("cuda graph: %zu nodes for a %d-token verify pass\n", nodes, B);
}

// One token in, one token out.
inline int step(Runtime &rt, int token) {
  LCHECK(cudaMemcpyAsync(rt.dtok_in, &token, sizeof(int), cudaMemcpyHostToDevice,
                         rt.stream));
  LCHECK(cudaGraphLaunch(rt.graph_exec, rt.stream));
  int out = 0;
  LCHECK(cudaMemcpyAsync(&out, rt.dtok_out, sizeof(int), cudaMemcpyDeviceToHost,
                         rt.stream));
  LCHECK(cudaStreamSynchronize(rt.stream));
  rt.pos++;
  return out;
}

}  // namespace spark27
