// bench/tc_shapes.cu — how close is the prefill GEMM to its own floor?
//
// The decode kernel is bandwidth-bound and measurably near peak. The prefill
// kernel is a different animal: it processes TC_BT tokens per pass over the
// weights, so it has TC_BT times the arithmetic and the same memory traffic.
// The floor is therefore whichever is larger,
//
//   memory : bytes(W) * ceil(T / TC_BT) / bandwidth      (each token tile
//                                                         re-reads the weights)
//   compute: 2 * params * T / tensor-core-rate
//
// and for the tile sizes here the memory term wins, which means prefill should
// be running at roughly the same GB/s as decode. This measures whether it is.
#define SPARK27_NO_MAIN
#include "../src/loader.cpp"
#include "../src/gemm_tc.cu"

#include <algorithm>
#include <map>
#include <vector>

using namespace spark27;

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "out/qwengine.bin";
  Model m = load_model(path);
  const double BW = 235.6;

  struct Info { int count; int64_t bytes; };
  std::map<std::pair<int, int>, Info> shapes;
  std::map<std::pair<int, int>, const TensorEntry *> sample;
  int64_t total = 0;
  for (auto &e : m.table) {
    if (e.kind != KIND_NVFP4 || strstr(e.name, "embed_tokens")) continue;
    if (!gemm_tc_ok(e.n)) continue;
    auto key = std::make_pair(e.n, e.k);
    shapes[key].count++;
    shapes[key].bytes += e.bytes;
    total += e.bytes;
    if (!sample.count(key)) sample[key] = &e;
  }
  printf("tile: BT=%d BN=%d BK=%d warps=%d   pass = %.3f GB\n\n",
         TC_BT, TC_BN, TC_BK, TC_WARPS, total / 1e9);

  int maxK = 0, maxN = 0;
  for (auto &kv : shapes) { maxK = std::max(maxK, kv.first.second); maxN = std::max(maxN, kv.first.first); }

  const int TMAX = argc > 2 ? atoi(argv[2]) : 256;
  __nv_bfloat16 *dA; float *dC;
  LCHECK(cudaMalloc(&dA, (size_t)maxK * TMAX * 2));
  LCHECK(cudaMalloc(&dC, (size_t)maxN * TMAX * 4));
  LCHECK(cudaMemset(dA, 0x3C, (size_t)maxK * TMAX * 2));

  printf("%6s %10s %10s %10s %10s %10s\n", "T", "ms/pass", "tok/s", "GB/s",
         "TFLOP/s", "% of floor");
  for (int T = 16; T <= TMAX; T *= 2) {
    double ms_total = 0;
    for (auto &kv : shapes) {
      const int N = kv.first.first, K = kv.first.second;
      Tensor t = m.get(sample[kv.first]->name);
      for (int i = 0; i < 2; ++i) launch_gemm_tc(t.dev, dA, dC, T, N, K, t.scale);
      LCHECK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      LCHECK(cudaEventCreate(&a)); LCHECK(cudaEventCreate(&b));
      double best = 1e30;
      const int reps = 5;
      for (int i = 0; i < 2; ++i) {
        LCHECK(cudaEventRecord(a));
        for (int r = 0; r < reps; ++r) launch_gemm_tc(t.dev, dA, dC, T, N, K, t.scale);
        LCHECK(cudaEventRecord(b));
        LCHECK(cudaEventSynchronize(b));
        float ms = 0; LCHECK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, (double)ms / reps);
      }
      LCHECK(cudaEventDestroy(a)); LCHECK(cudaEventDestroy(b));
      ms_total += best * (kv.second.bytes / (double)t.bytes);
    }
    // Each of the ceil(T/TC_BT) token tiles streams the weights once.
    const int tiles = (T + TC_BT - 1) / TC_BT;
    const double floor_ms = total * tiles / (BW * 1e9) * 1e3;
    const double gbs = total * tiles / (ms_total * 1e-3) / 1e9;
    const double tflops = 2.0 * (total / 0.5625) * T / (ms_total * 1e-3) / 1e12;
    printf("%6d %10.2f %10.1f %10.1f %10.1f %9.0f%%\n", T, ms_total,
           T * 1000.0 / ms_total, gbs, tflops, 100.0 * floor_ms / ms_total);
  }
  printf("\n'%% of floor' is how much of the achievable time is real work:\n"
         "100%% means the kernel is streaming weights as fast as the memory\n"
         "system allows and there is nothing left to win.\n");
  LCHECK(cudaFree(dA)); LCHECK(cudaFree(dC));
  free_model(m);
  return 0;
}
