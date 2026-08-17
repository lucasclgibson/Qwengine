// qwen38.h — all Qwen3.8-27B geometry, constexpr. Text path only.
//
// Every value here was read from the BF16 checkpoint's config.json and then
// cross-checked against the actual tensor shapes in the safetensors headers,
// because config.json does not state all of them. convert.cpp re-validates
// config.json against this header
// and refuses to run on mismatch — that is the guard against a wrong or
// updated checkpoint silently producing garbage.
//
// Nothing machine-specific belongs in this file. Bandwidth B is calibrated at
// runtime (bench/01_bandwidth.cu), not baked in, so the engine runs on any
// DGX Spark.
#pragma once
#include <stdint.h>

namespace q38 {

// ---- core geometry --------------------------------------------------------
constexpr int HIDDEN       = 5120;
constexpr int N_LAYERS     = 64;
constexpr int INTERMEDIATE = 17408;   // SwiGLU inner dim
constexpr int VOCAB        = 248320;
constexpr int MAX_POS      = 262144;  // native
constexpr int MAX_CTX      = 32768;   // engine v1 cap
constexpr float RMS_EPS    = 1e-6f;

// ---- layer pattern --------------------------------------------------------
// config: full_attention_interval = 4, layer_types = 16x(lin,lin,lin,full).
// Layer i is full attention iff i % 4 == 3.
constexpr int FULL_ATTN_INTERVAL = 4;
constexpr bool is_full_attn(int layer) {
  return layer % FULL_ATTN_INTERVAL == FULL_ATTN_INTERVAL - 1;
}
constexpr int N_FULL_LAYERS   = N_LAYERS / FULL_ATTN_INTERVAL;        // 16
constexpr int N_LINEAR_LAYERS = N_LAYERS - N_FULL_LAYERS;             // 48

// ---- full attention (gated GQA) -------------------------------------------
constexpr int N_Q_HEADS  = 24;
constexpr int N_KV_HEADS = 4;
constexpr int HEAD_DIM   = 256;
constexpr int Q_DIM      = N_Q_HEADS * HEAD_DIM;   // 6144
constexpr int KV_DIM     = N_KV_HEADS * HEAD_DIM;  // 1024
// attn_output_gate=true: q_proj emits Q and the output gate concatenated, so
// its out_features is 2*Q_DIM. Checkpoint confirms q_proj = [12288, 5120].
constexpr int QGATE_DIM  = 2 * Q_DIM;              // 12288
// q_norm/k_norm are per-head RMSNorm over HEAD_DIM (QK-norm), weight [256].

// ---- RoPE -----------------------------------------------------------------
// partial_rotary_factor 0.25 => rotation covers only the first 64 of 256 dims;
// the remaining 192 pass through unrotated. rope_theta 1e7 (not the usual 1e4).
// mrope_section [11,11,10] sums to 32 = ROTARY_DIM/2 pairs; with vision
// dropped, all three sections carry the same text position, so mrope reduces
// to standard RoPE. Derived from the checkpoint, not assumed.
constexpr float PARTIAL_ROTARY = 0.25f;
constexpr int   ROTARY_DIM     = (int)(HEAD_DIM * PARTIAL_ROTARY);  // 64
constexpr float ROPE_THETA     = 1e7f;

// ---- gated DeltaNet (linear attention) ------------------------------------
// Q and K share LIN_K_HEADS; V has 3x as many heads. All head dims 128.
constexpr int LIN_K_HEADS   = 16;
constexpr int LIN_V_HEADS   = 48;
constexpr int LIN_HEAD_DIM  = 128;   // linear_{key,value}_head_dim both 128
constexpr int LIN_Q_DIM     = LIN_K_HEADS * LIN_HEAD_DIM;  // 2048
constexpr int LIN_K_DIM     = LIN_K_HEADS * LIN_HEAD_DIM;  // 2048
constexpr int LIN_V_DIM     = LIN_V_HEADS * LIN_HEAD_DIM;  // 6144
constexpr int LIN_QKV_DIM   = LIN_Q_DIM + LIN_K_DIM + LIN_V_DIM;  // 10240
constexpr int LIN_Z_DIM     = LIN_V_DIM;                   // 6144, output gate
// Causal depthwise conv over the fused QKV stream, one filter per channel.
constexpr int CONV_KERNEL   = 4;
constexpr int CONV_CHANNELS = LIN_QKV_DIM;                 // 10240
// Per-V-head decay params: A_log[48], dt_bias[48]; recurrent state in FP32.
constexpr int LIN_STATE_ELEMS = LIN_V_HEADS * LIN_HEAD_DIM * LIN_HEAD_DIM;

// ---- MTP head -------------------------------------------------------------
// mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=false (shares
// embed_tokens and lm_head). Wiring, from the checkpoint's tensor set:
//   fc( concat[ pre_fc_norm_hidden(h), pre_fc_norm_embedding(e) ] )  10240->5120
//   -> one full-attention block (identical geometry to a main full layer)
//   -> mtp.norm -> shared lm_head
constexpr int MTP_LAYERS  = 1;
constexpr int MTP_FC_IN   = 2 * HIDDEN;  // 10240

// ---- NVFP4 format ---------------------------------------------------------
// E2M1 values (4 bit), one FP8 E4M3 scale per 16-element block, one FP32
// per-tensor scale. 4 + 8/16 = 4.5 bits/weight.
constexpr int NVFP4_BLOCK    = 16;
constexpr int NVFP4_VAL_BITS = 4;

// ---- on-disk swizzle, co-designed with gemv_nvfp4.cu ----------------------
// One warp step reads SWZ_WEIGHTS contiguous weights of one output row:
//   [ 512 B packed values | 64 B block scales ]
// 32 lanes x uint4 = 512 B fully coalesced for the values; each lane then
// takes the 2 scales covering its own 32 weights as one ushort, so the scale
// read is a single coalesced 64 B transaction. Blocks of 16 never straddle a
// lane boundary (32 weights/lane = exactly 2 blocks).
// Every K in this model is a multiple of 1024 (5120/6144/10240/17408), so no
// tensor needs padding — verified over all 506 matrices.
constexpr int SWZ_WEIGHTS     = 1024;
constexpr int SWZ_VAL_BYTES   = SWZ_WEIGHTS * NVFP4_VAL_BITS / 8;      // 512
constexpr int SWZ_SCALE_BYTES = SWZ_WEIGHTS / NVFP4_BLOCK;             // 64
constexpr int SWZ_STEP_BYTES  = SWZ_VAL_BYTES + SWZ_SCALE_BYTES;       // 576

// Packed bytes for a [N,K] matrix under this layout.
constexpr int64_t nvfp4_bytes(int64_t N, int64_t K) {
  return N * (K / SWZ_WEIGHTS) * SWZ_STEP_BYTES;
}

// ---- consistency asserts --------------------------------------------------
static_assert(N_FULL_LAYERS == 16 && N_LINEAR_LAYERS == 48, "layer split");
static_assert(Q_DIM == 6144 && KV_DIM == 1024 && QGATE_DIM == 12288, "attn dims");
static_assert(ROTARY_DIM == 64, "partial rotary");
static_assert(LIN_QKV_DIM == 10240 && LIN_Z_DIM == 6144, "deltanet dims");
static_assert(MTP_FC_IN == 10240, "mtp fc in");
static_assert(SWZ_VAL_BYTES == 512 && SWZ_SCALE_BYTES == 64, "swizzle step");
static_assert(SWZ_WEIGHTS % NVFP4_BLOCK == 0, "blocks tile the warp step");
static_assert(SWZ_VAL_BYTES % 32 == 0, "warp-wide uint4 coalescing");
static_assert(HIDDEN % SWZ_WEIGHTS == 0 && INTERMEDIATE % SWZ_WEIGHTS == 0 &&
                  LIN_V_DIM % SWZ_WEIGHTS == 0 && LIN_QKV_DIM % SWZ_WEIGHTS == 0,
              "every K tiles without padding");

}  // namespace q38
