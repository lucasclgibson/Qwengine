// bench/frag_bench.cu — does dequantising straight into mma fragments beat
// staging the weights in shared memory?
//
// gemm_tc.cu unpacks NVFP4 into a shared BF16 tile and lets wmma read it back.
// Ablation says those shared WRITES are what caps it at 47 TFLOP/s: delete them
// and it runs at 93. gemm_frag.cu stores the weights in fragment order instead,
// so a lane reads its eight values with one 4-byte load and writes them into
// the fragment registers directly, and nothing goes through shared.
//
// This checks both halves of the claim on the model's real weights: that the
// two kernels agree numerically, and that the second is faster.
#define SPARK27_NO_MAIN
#include "../src/loader.cpp"
#include "../src/gemm_tc.cu"
#include "../src/gemm_frag.cu"

#include <algorithm>
#include <map>
#include <vector>

using namespace spark27;

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "out/qwengine.bin";
  const int T = argc > 2 ? atoi(argv[2]) : 256;
  Model m = load_model(path);

  struct Info { int count; int64_t bytes; };
  std::map<std::pair<int, int>, Info> shapes;
  std::map<std::pair<int, int>, const TensorEntry *> sample;
  double total = 0;
  for (auto &e : m.table) {
    if (e.kind != KIND_NVFP4 || strstr(e.name, "embed_tokens")) continue;
    if (e.n % FG_BN || e.k % FG_BK) continue;      // shapes the tile divides
    auto key = std::make_pair(e.n, e.k);
    shapes[key].count++;
    shapes[key].bytes += e.bytes;
    total += e.bytes;
    if (!sample.count(key)) sample[key] = &e;
  }

  int maxK = 0, maxN = 0;
  for (auto &kv : shapes) { maxK = std::max(maxK, kv.first.second); maxN = std::max(maxN, kv.first.first); }
  const int TP = (T + 127) / 128 * 128;
  __nv_bfloat16 *dA; float *dC1, *dC2; uint8_t *dF;
  LCHECK(cudaMalloc(&dA, (size_t)maxK * TP * 2));
  LCHECK(cudaMalloc(&dC1, (size_t)maxN * TP * 4));
  LCHECK(cudaMalloc(&dC2, (size_t)maxN * TP * 4));
  LCHECK(cudaMalloc(&dF, frag_bytes(maxN, maxK)));
  // Deterministic pseudo-random activations, small so products stay in range.
  {
    std::vector<__nv_bfloat16> h((size_t)maxK * TP);
    unsigned r = 7;
    for (auto &v : h) { r = r * 1664525u + 1013904223u; v = __float2bfloat16(((float)((r >> 9) & 1023) / 512.f - 1.f) * 0.5f); }
    LCHECK(cudaMemcpy(dA, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
  }

  printf("pass = %.3f GB over %zu shapes, T = %d\n", total / 1e9, shapes.size(), T);
  printf("tile: BT=%d BN=%d BK=%d warp=%dx%d\n\n", FG_BT, FG_BN, FG_BK,
         FG_WFT_CFG, FG_WFN_CFG);

  auto time_it = [&](bool frag) {
    double ms_total = 0;
    for (auto &kv : shapes) {
      const int N = kv.first.first, K = kv.first.second;
      Tensor t = m.get(sample[kv.first]->name);
      if (frag) { repack_frag(t.dev, dF, N, K); LCHECK(cudaDeviceSynchronize()); }
      auto go = [&]() {
        if (frag) launch_gemm_frag(dF, dA, dC2, T, N, K, t.scale);
        else launch_gemm_tc(t.dev, dA, dC1, T, N, K, t.scale);
      };
      for (int i = 0; i < 3; ++i) go();
      LCHECK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      LCHECK(cudaEventCreate(&a)); LCHECK(cudaEventCreate(&b));
      double best = 1e30;
      for (int i = 0; i < 3; ++i) {
        LCHECK(cudaEventRecord(a));
        for (int r = 0; r < 6; ++r) go();
        LCHECK(cudaEventRecord(b));
        LCHECK(cudaEventSynchronize(b));
        float ms = 0; LCHECK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, (double)ms / 6);
      }
      LCHECK(cudaEventDestroy(a)); LCHECK(cudaEventDestroy(b));
      ms_total += best * (kv.second.bytes / (double)t.bytes);
    }
    return ms_total;
  };

  // Correctness on one real matrix before any timing is believed.
  {
    auto kv = shapes.begin();
    const int N = kv->first.first, K = kv->first.second;
    Tensor t = m.get(sample[kv->first]->name);
    repack_frag(t.dev, dF, N, K);
    LCHECK(cudaDeviceSynchronize());
    LCHECK(cudaMemset(dC1, 0, (size_t)N * TP * 4));
    LCHECK(cudaMemset(dC2, 0, (size_t)N * TP * 4));
    launch_gemm_tc(t.dev, dA, dC1, T, N, K, t.scale);
    launch_gemm_frag(dF, dA, dC2, T, N, K, t.scale);
    LCHECK(cudaDeviceSynchronize());
    LCHECK(cudaGetLastError());
    std::vector<float> c1((size_t)T * N), c2((size_t)T * N);
    LCHECK(cudaMemcpy(c1.data(), dC1, c1.size() * 4, cudaMemcpyDeviceToHost));
    LCHECK(cudaMemcpy(c2.data(), dC2, c2.size() * 4, cudaMemcpyDeviceToHost));
    double se = 0, sr = 0, worst = 0;
    for (size_t i = 0; i < c1.size(); ++i) {
      const double d = (double)c2[i] - c1[i];
      se += d * d; sr += (double)c1[i] * c1[i];
      worst = fmax(worst, fabs(d));
    }
    const double rel = sqrt(se / (sr > 0 ? sr : 1));
    printf("correctness on [%d,%d]: rel-err %.3e, worst abs %.3e  %s\n\n", N, K,
           rel, worst, rel < 1e-5 ? "OK" : "*** MISMATCH ***");
    if (!(rel < 1e-5)) { free_model(m); return 1; }
  }

  const double t_tc = time_it(false);
  const double t_fg = time_it(true);
  const double fl = 2.0 * (total / 0.5625) * T;
  printf("%-22s %10s %12s\n", "kernel", "ms/pass", "TFLOP/s");
  printf("%-22s %10.1f %12.1f\n", "gemm_tc (shared)", t_tc, fl / (t_tc * 1e-3) / 1e12);
  printf("%-22s %10.1f %12.1f\n", "gemm_frag (registers)", t_fg, fl / (t_fg * 1e-3) / 1e12);
  printf("\n%.2fx\n", t_tc / t_fg);

  cudaFree(dA); cudaFree(dC1); cudaFree(dC2); cudaFree(dF);
  free_model(m);
  return 0;
}
