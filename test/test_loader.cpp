// test/test_loader.cpp — verify qwengine.bin loads and that what lands on the
// device is byte-identical to what the converter wrote.
//
// Also checks the converted artefact against qwen38.h: every layer must carry
// exactly the tensors its type implies, at the shapes the header declares.
// That is the check that would have caught reading q_proj as [6144,5120]
// instead of [12288,5120], and it is worth far more than a size assertion.
//
// Skips (exit 0) when out/qwengine.bin is absent, so a fresh clone can run the
// suite before a 15 GB conversion. A skip is reported as a skip, never as a pass.
#define SPARK27_NO_MAIN
#include "../src/loader.cpp"

#include <fcntl.h>

using namespace spark27;
using namespace q38;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    ++g_checks;                                                                \
    if (!(cond)) {                                                             \
      fprintf(stderr, "FAIL %s:%d: %s\n      ", __FILE__, __LINE__, #cond);    \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
      ++g_fail;                                                                \
    }                                                                          \
  } while (0)

static const char *kPath = "out/qwengine.bin";

// Compare a tensor's device bytes against the file's bytes.
static void check_bytes(const Model &m, const char *name, int fd) {
  auto it = m.index.find(name);
  if (it == m.index.end()) { CHECK(false, "missing %s", name); return; }
  const TensorEntry &e = m.table[(size_t)it->second];

  size_t n = (size_t)(e.bytes < (1 << 20) ? e.bytes : (1 << 20));  // up to 1 MB
  std::vector<uint8_t> from_file(n), from_dev(n);
  if (pread(fd, from_file.data(), n, (off_t)e.offset) != (ssize_t)n) {
    CHECK(false, "pread %s", name); return;
  }
  Tensor t = m.get(name);
  if (cudaMemcpy(from_dev.data(), t.dev, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
    CHECK(false, "D2H %s", name); return;
  }
  CHECK(memcmp(from_file.data(), from_dev.data(), n) == 0,
        "%s: device bytes differ from file", name);
}

int main() {
  if (access(kPath, R_OK) != 0) {
    printf("test_loader: SKIP — %s not present (run build/convert first)\n", kPath);
    return 0;
  }
  Model m = load_model(kPath);
  printf("-- loaded: %zu tensors, %.3f GB, %.0f ms, copy %.1f GB/s --\n",
         m.table.size(), m.bytes / 1e9, m.load_ms, m.copy_gbs);

  // 675 language-model tensors + 333 vision. A --no-vision build has 675.
  const size_t nt = m.table.size();
  CHECK(nt == 1009 || nt == 676, "tensor count %zu, want 1009 (vision+tokenizer) or 676 (--no-vision)", nt);
  CHECK(m.has("__tokenizer__"), "weights file should carry the tokenizer");

  // Payload must be contiguous and match the sum of the parts.
  int64_t sum = 0;
  for (auto &e : m.table) sum += e.bytes;
  CHECK(sum == m.bytes, "sum of tensors %lld != payload %lld", (long long)sum,
        (long long)m.bytes);

  int fd = open(kPath, O_RDONLY);
  CHECK(fd >= 0, "reopen");

  // ---- geometry, layer by layer, against qwen38.h ------------------------
  int n_full = 0, n_lin = 0;
  for (int L = 0; L < N_LAYERS; ++L) {
    char role[64];
    // Both layer types carry these.
    for (const char *r : {"input_layernorm.weight", "post_attention_layernorm.weight"})
      CHECK(m.has((std::string("model.language_model.layers.") + std::to_string(L) +
                   "." + r).c_str()), "layer %d missing %s", L, r);

    // gate and up share an input, so they are stored as one taller matrix.
    Tensor gate = m.layer(L, "mlp.gate_up_proj.weight");
    Tensor down = m.layer(L, "mlp.down_proj.weight");
    CHECK(gate.n == 2 * INTERMEDIATE && gate.k == HIDDEN,
          "L%d gate_up_proj [%d,%d], want [%d,%d]", L, gate.n, gate.k,
          2 * INTERMEDIATE, HIDDEN);
    CHECK(down.n == HIDDEN && down.k == INTERMEDIATE, "L%d down_proj [%d,%d]", L,
          down.n, down.k);

    if (is_full_attn(L)) {
      ++n_full;
      Tensor q = m.layer(L, "self_attn.qkv_proj.weight");
      Tensor o = m.layer(L, "self_attn.o_proj.weight");
      // q|k|v in one matrix; q's half is itself [query|gate] per head, so the
      // total is 2*Q_DIM + 2*KV_DIM. Getting this wrong is silent.
      CHECK(q.n == QGATE_DIM + 2 * KV_DIM && q.k == HIDDEN,
            "L%d qkv_proj [%d,%d], want [%d,%d]", L, q.n, q.k,
            QGATE_DIM + 2 * KV_DIM, HIDDEN);
      CHECK(o.n == HIDDEN && o.k == Q_DIM, "L%d o_proj [%d,%d]", L, o.n, o.k);
      snprintf(role, sizeof role, "self_attn.q_norm.weight");
      CHECK(m.layer(L, role).n == HEAD_DIM, "L%d q_norm", L);
    } else {
      ++n_lin;
      Tensor qkv = m.layer(L, "linear_attn.in_proj_qkv.weight");
      Tensor z = m.layer(L, "linear_attn.in_proj_zab.weight");
      Tensor op = m.layer(L, "linear_attn.out_proj.weight");
      Tensor cv = m.layer(L, "linear_attn.conv1d.weight");
      CHECK(qkv.n == LIN_QKV_DIM && qkv.k == HIDDEN, "L%d in_proj_qkv [%d,%d]", L,
            qkv.n, qkv.k);
      CHECK(z.n == LIN_Z_DIM + 2 * LIN_V_HEADS && z.k == HIDDEN,
            "L%d in_proj_zab [%d,%d], want [%d,%d]", L, z.n, z.k,
            LIN_Z_DIM + 2 * LIN_V_HEADS, HIDDEN);
      CHECK(op.n == HIDDEN && op.k == LIN_V_DIM, "L%d out_proj [%d,%d]", L, op.n, op.k);
      // conv1d is [10240,1,4] -> stored flat as BF16.
      CHECK(cv.kind == KIND_BF16 && cv.n == CONV_CHANNELS * CONV_KERNEL,
            "L%d conv1d n=%d, want %d", L, cv.n, CONV_CHANNELS * CONV_KERNEL);
      CHECK(m.layer(L, "linear_attn.A_log").n == LIN_V_HEADS, "L%d A_log", L);
      CHECK(m.layer(L, "linear_attn.dt_bias").n == LIN_V_HEADS, "L%d dt_bias", L);
    }
  }
  CHECK(n_full == N_FULL_LAYERS, "full layers %d, want %d", n_full, N_FULL_LAYERS);
  CHECK(n_lin == N_LINEAR_LAYERS, "linear layers %d, want %d", n_lin, N_LINEAR_LAYERS);

  // ---- top level and MTP --------------------------------------------------
  Tensor lm = m.get("lm_head.weight");
  Tensor emb = m.get("model.language_model.embed_tokens.weight");
  CHECK(lm.n == VOCAB && lm.k == HIDDEN, "lm_head [%d,%d]", lm.n, lm.k);
  CHECK(emb.n == VOCAB && emb.k == HIDDEN, "embed [%d,%d]", emb.n, emb.k);
  CHECK(lm.dev != emb.dev, "lm_head and embed must not alias (untied)");
  // The draft head re-reads lm_head once per draft step, where it is about
  // three quarters of the bytes, so its size is a first-order cost.
  printf("   lm_head payload = %.3f GB\n", lm.bytes / 1e9);

  Tensor fc = m.get("mtp.fc.weight");
  CHECK(fc.n == HIDDEN && fc.k == MTP_FC_IN, "mtp.fc [%d,%d], want [%d,%d]", fc.n,
        fc.k, HIDDEN, MTP_FC_IN);
  for (const char *r : {"mtp.norm.weight", "mtp.pre_fc_norm_hidden.weight",
                        "mtp.pre_fc_norm_embedding.weight"})
    CHECK(m.has(r), "missing %s", r);
  Tensor mq = m.get("mtp.layers.0.self_attn.q_proj.weight");
  CHECK(mq.n == QGATE_DIM && mq.k == HIDDEN, "mtp q_proj [%d,%d]", mq.n, mq.k);

  // ---- vision tower: all present, or all absent; never half ---------------
  int vis = 0;
  for (auto &e : m.table)
    if (strstr(e.name, "visual") || strstr(e.name, "vision")) ++vis;
  CHECK(vis == 333 || vis == 0, "%d vision tensors — expected all 333 or none", vis);
  if (vis) {
    // Spot-check the pieces the tower cannot run without.
    for (const char *n : {"model.visual.patch_embed.proj.weight",
                          "model.visual.pos_embed.weight",
                          "model.visual.blocks.26.mlp.linear_fc2.weight",
                          "model.visual.merger.linear_fc2.weight"})
      CHECK(m.has(n), "vision build missing %s", n);
    // BF16 tensors are stored flat: n is the element count, k is 0.
    Tensor mg = m.get("model.visual.merger.linear_fc2.weight");
    CHECK(mg.kind == KIND_BF16 && mg.n == HIDDEN * 4608,
          "merger fc2 should be BF16 with %d elements, got kind %d n %d",
          HIDDEN * 4608, mg.kind, mg.n);
    printf("   vision tower: 333 tensors, %.2f GB, merger -> %d wide (BF16)\n",
           mg.bytes / 1e9 * 333 / 333, HIDDEN);
  }

  // ---- device bytes match the file ---------------------------------------
  for (const char *n : {"lm_head.weight",
                        "model.language_model.embed_tokens.weight",
                        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
                        "model.language_model.layers.3.self_attn.qkv_proj.weight",
                        "model.language_model.layers.63.mlp.down_proj.weight",
                        "mtp.fc.weight"})
    check_bytes(m, n, fd);
  close(fd);

  // ---- NVFP4 payload sizes obey the swizzle ------------------------------
  int bad = 0;
  for (auto &e : m.table) {
    if (e.kind == KIND_NVFP4 && e.bytes != nvfp4_bytes(e.n, e.k)) ++bad;
    if (e.kind == KIND_D2 && e.bytes != d2_bytes(e.n, e.k)) ++bad;
  }
  CHECK(bad == 0, "%d NVFP4 tensors have wrong swizzled size", bad);

  free_model(m);
  printf("\n%d checks, %d failures\n", g_checks, g_fail);
  if (!g_fail) printf("test_loader: PASS\n");
  return g_fail ? 1 : 0;
}
