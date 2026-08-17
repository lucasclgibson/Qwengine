// test/test_gemv.cu — is the hot kernel correct, and does it run at memory speed?
//
// Two questions, in order of importance:
//   1. Does it compute the right answer? (checked against a plain CPU version
//      that reads the same bytes the slow, obvious way)
//   2. Does it hit the machine's memory bandwidth? Anything below ~90% of B
//      means the rest of the engine is pointless until it is fixed.
//
// The headline number is "passes/sec": how many times per second we can push
// the whole model through this kernel. That is the physical ceiling on words
// per second before any speculation trick is applied.
#define SPARK27_NO_MAIN
#include "../src/loader.cpp"
#include "../src/gemv.cu"

#include <math.h>
#include <vector>

using namespace spark27;

static int g_fail = 0, g_checks = 0;
#define CHECK(c, ...)                                                          \
  do {                                                                         \
    ++g_checks;                                                                \
    if (!(c)) {                                                                \
      fprintf(stderr, "FAIL %s:%d: %s\n      ", __FILE__, __LINE__, #c);       \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
      ++g_fail;                                                                \
    }                                                                          \
  } while (0)

static const float kMag[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static float e4m3_host(uint8_t v) {
  int e = (v >> 3) & 0xF, m = v & 7;
  float mag = e ? ldexpf(1.0f + m * 0.125f, e - 7) : ldexpf((float)m, -9);
  return (v & 0x80) ? -mag : mag;
}

// The obvious, slow version. Deliberately written to mirror the on-disk format
// literally rather than to be fast — its whole job is to disagree with the
// kernel if the kernel is wrong.
static void gemv_cpu(const uint8_t *W, const float *x, float *y, int N, int K,
                     float tscale) {
  const int steps = K / 1024;
  for (int n = 0; n < N; ++n) {
    double acc = 0.0;
    for (int s = 0; s < steps; ++s) {
      const uint8_t *p = W + ((size_t)n * steps + s) * 576;
      for (int b = 0; b < 64; ++b) {              // 64 scale-blocks per step
        const float sc = e4m3_host(p[512 + b]);
        double blk = 0.0;
        for (int i = 0; i < 16; ++i) {            // 16 weights per block
          const int idx = b * 16 + i;
          const uint8_t byte = p[idx >> 1];
          const uint8_t code = (idx & 1) ? (byte >> 4) : (byte & 0xF);
          const float w = (code & 8) ? -kMag[code & 7] : kMag[code & 7];
          blk += (double)w * x[s * 1024 + idx];
        }
        acc += blk * sc;
      }
    }
    y[n] = (float)(acc * tscale);
  }
}

int main() {
  const char *path = "out/qwengine.bin";
  if (access(path, R_OK) != 0) {
    printf("test_gemv: SKIP — %s not present\n", path);
    return 0;
  }
  Model m = load_model(path);
  printf("model: %zu tensors, %.3f GB\n\n", m.table.size(), m.bytes / 1e9);

  int maxK = 0;
  for (auto &e : m.table)
    if (e.kind == KIND_NVFP4 && e.k > maxK) maxK = e.k;

  float *dx = nullptr, *dy = nullptr;
  LCHECK(cudaMalloc(&dx, (size_t)maxK * 4));
  LCHECK(cudaMalloc(&dy, (size_t)q38::VOCAB * 4));  // widest output is lm_head
  std::vector<float> hx((size_t)maxK);
  // Deterministic, non-trivial activations. A constant vector would hide sign
  // and index bugs entirely.
  for (int i = 0; i < maxK; ++i) hx[i] = sinf(i * 0.017f) * 0.9f;
  LCHECK(cudaMemcpy(dx, hx.data(), (size_t)maxK * 4, cudaMemcpyHostToDevice));

  // ---- 1. correctness ------------------------------------------------------
  printf("-- correctness vs CPU reference --\n");
  const char *probes[] = {
      "model.language_model.layers.0.linear_attn.in_proj_zab.weight",   // K=5120
      "model.language_model.layers.0.mlp.down_proj.weight",           // K=17408
      "model.language_model.layers.3.self_attn.qkv_proj.weight",        // N=12288
      "model.language_model.layers.7.self_attn.qkv_proj.weight",        // N=1024
  };
  for (const char *name : probes) {
    Tensor t = m.get(name);
    const int N = t.n, K = t.k;
    // Only the first rows, or the CPU reference takes minutes.
    const int Nc = N < 512 ? N : 512;

    launch_gemv(t.dev, dx, dy, N, K, t.scale);
    LCHECK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)Nc);
    LCHECK(cudaMemcpy(got.data(), dy, (size_t)Nc * 4, cudaMemcpyDeviceToHost));

    std::vector<uint8_t> hw((size_t)Nc * (K / 1024) * 576);
    LCHECK(cudaMemcpy(hw.data(), t.dev, hw.size(), cudaMemcpyDeviceToHost));
    std::vector<float> want((size_t)Nc);
    gemv_cpu(hw.data(), hx.data(), want.data(), Nc, K, t.scale);

    double worst = 0.0, denom = 0.0;
    for (int i = 0; i < Nc; ++i) {
      worst = fmax(worst, fabs((double)got[i] - want[i]));
      denom = fmax(denom, fabs((double)want[i]));
    }
    const double rel = denom > 0 ? worst / denom : 0.0;
    printf("   %-58s [%6d,%6d] max rel diff %.2e\n",
           strrchr(name, 'y') ? name + strlen(name) - 34 : name, N, K, rel);
    // Kernel sums in float, reference in double: small drift is expected,
    // a real indexing bug is not subtle.
    CHECK(rel < 2e-3, "%s: rel diff %.3e — kernel disagrees with reference",
          name, rel);
  }

  // ---- 2. determinism ------------------------------------------------------
  printf("\n-- determinism --\n");
  {
    Tensor t = m.get("model.language_model.layers.0.mlp.gate_up_proj.weight");
    std::vector<float> a((size_t)t.n), b((size_t)t.n);
    launch_gemv(t.dev, dx, dy, t.n, t.k, t.scale);
    LCHECK(cudaDeviceSynchronize());
    LCHECK(cudaMemcpy(a.data(), dy, (size_t)t.n * 4, cudaMemcpyDeviceToHost));
    LCHECK(cudaMemset(dy, 0, (size_t)t.n * 4));
    launch_gemv(t.dev, dx, dy, t.n, t.k, t.scale);
    LCHECK(cudaDeviceSynchronize());
    LCHECK(cudaMemcpy(b.data(), dy, (size_t)t.n * 4, cudaMemcpyDeviceToHost));
    CHECK(memcmp(a.data(), b.data(), (size_t)t.n * 4) == 0,
          "same input gave different bits across runs");
    printf("   identical bits across runs: yes\n");
  }

  // ---- 3. the number that matters -----------------------------------------
  // One "pass" = every matrix the model multiplies by, once. embed_tokens is
  // excluded: generating a word looks up a single row of it, it is not a
  // matmul, so counting its 0.7 GB would understate our real speed.
  printf("\n-- full-model pass --\n");
  std::vector<const TensorEntry *> mats;
  int64_t pass_bytes = 0;
  for (auto &e : m.table) {
    if (e.kind != KIND_NVFP4) continue;
    if (strstr(e.name, "embed_tokens")) continue;
    mats.push_back(&e);
    pass_bytes += e.bytes;
  }
  printf("   %zu matrices, %.3f GB per pass\n", mats.size(), pass_bytes / 1e9);

  auto run_pass = [&] {
    for (const TensorEntry *e : mats) {
      Tensor t = m.get(e->name);
      launch_gemv(t.dev, dx, dy, t.n, t.k, t.scale);
    }
  };
  run_pass();  // warm up
  LCHECK(cudaDeviceSynchronize());

  cudaEvent_t a, b;
  LCHECK(cudaEventCreate(&a));
  LCHECK(cudaEventCreate(&b));
  double best_ms = 1e30;
  for (int i = 0; i < 5; ++i) {
    LCHECK(cudaEventRecord(a));
    run_pass();
    LCHECK(cudaEventRecord(b));
    LCHECK(cudaEventSynchronize(b));
    float ms = 0;
    LCHECK(cudaEventElapsedTime(&ms, a, b));
    if (ms < best_ms) best_ms = ms;
  }
  const double gbs = pass_bytes / (best_ms * 1e-3) / 1e9;
  const double B = 235.0;  // measured on this host, bench calib-0001..3
  printf("   time per pass : %.2f ms\n", best_ms);
  printf("   bandwidth     : %.1f GB/s  (%.0f%% of B=%.0f)\n", gbs,
         100.0 * gbs / B, B);
  printf("   passes/sec    : %.2f   <-- ceiling on words/sec before speculation\n",
         1000.0 / best_ms);
  CHECK(gbs > 0.75 * B, "kernel at %.0f%% of memory bandwidth — too slow to build on",
        100.0 * gbs / B);

  LCHECK(cudaFree(dx));
  LCHECK(cudaFree(dy));
  free_model(m);
  printf("\n%d checks, %d failures\n", g_checks, g_fail);
  if (!g_fail) printf("test_gemv: PASS\n");
  return g_fail ? 1 : 0;
}
