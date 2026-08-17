// bench/verify_kernel.cu — CUDA cores or tensor cores for the verify pass?
//
// A speculative verify is a GEMM with M = draft depth (4..8) token rows against
// every weight matrix in the model. Today it runs on the CUDA-core GEMV, which
// at batch 4 takes ~77 ms against a 62.2 ms bandwidth floor; the gap is FP32
// FMA throughput. A tensor-core kernel pads M up to a 16-row WMMA fragment --
// wasting up to 4x on rows -- but runs the arithmetic an order of magnitude
// faster. Whether that trade wins is an empirical question about whether the
// arithmetic hides behind the weight stream, so this measures both over the
// real shapes and weights.
#define SPARK27_NO_MAIN
#include "../src/loader.cpp"
#include "../src/gemv.cu"
#include "../src/gemm_tc.cu"

#include <algorithm>
#include <functional>
#include <map>
#include <vector>

using namespace spark27;

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "out/qwengine.bin";
  const int BMIN = argc > 2 ? atoi(argv[2]) : 3;
  const int BMAX = argc > 3 ? atoi(argv[3]) : 6;
  const double BW = 236.8;                      // measured read ceiling
  Model m = load_model(path);

  struct Info { int count; int64_t bytes; };
  std::map<std::pair<int, int>, Info> shapes;
  std::map<std::pair<int, int>, const TensorEntry *> sample;
  double total = 0;
  for (auto &e : m.table) {
    if (e.kind != KIND_NVFP4 || strstr(e.name, "embed_tokens")) continue;
    auto key = std::make_pair(e.n, e.k);
    shapes[key].count++;
    shapes[key].bytes += e.bytes;
    total += e.bytes;
    if (!sample.count(key)) sample[key] = &e;
  }
  int maxK = 0, maxN = 0;
  for (auto &kv : shapes) { maxK = std::max(maxK, kv.first.second); maxN = std::max(maxN, kv.first.first); }

  float *dx, *dy;
  __nv_bfloat16 *dxb;
  LCHECK(cudaMalloc(&dx, (size_t)maxK * 16 * 4));
  LCHECK(cudaMalloc(&dxb, (size_t)maxK * 16 * 2));
  LCHECK(cudaMalloc(&dy, (size_t)maxN * 16 * 4));
  LCHECK(cudaMemset(dx, 0x3C, (size_t)maxK * 16 * 4));
  LCHECK(cudaMemset(dxb, 0x3C, (size_t)maxK * 16 * 2));

  printf("pass = %.3f GB, bandwidth floor = %.2f ms at %.1f GB/s\n\n",
         total / 1e9, total / (BW * 1e9) * 1e3, BW);

  auto time_it = [&](const std::function<void(const Tensor &, int, int)> &fn) {
    double ms_total = 0;
    for (auto &kv : shapes) {
      const int N = kv.first.first, K = kv.first.second;
      Tensor t = m.get(sample[kv.first]->name);
      for (int i = 0; i < 3; ++i) fn(t, N, K);
      LCHECK(cudaDeviceSynchronize());
      cudaEvent_t a, b;
      LCHECK(cudaEventCreate(&a)); LCHECK(cudaEventCreate(&b));
      double best = 1e30;
      for (int i = 0; i < 3; ++i) {
        LCHECK(cudaEventRecord(a));
        for (int r = 0; r < 8; ++r) fn(t, N, K);
        LCHECK(cudaEventRecord(b));
        LCHECK(cudaEventSynchronize(b));
        float ms = 0; LCHECK(cudaEventElapsedTime(&ms, a, b));
        best = std::min(best, (double)ms / 8);
      }
      LCHECK(cudaEventDestroy(a)); LCHECK(cudaEventDestroy(b));
      ms_total += best * (kv.second.bytes / (double)t.bytes);
    }
    return ms_total;
  };

  // Tensor-core path pads M to one 16-row fragment, so its cost is the same for
  // every batch from 1 to 16 -- which is exactly why it might win at 4.
  const double tc = time_it([&](const Tensor &t, int N, int K) {
    launch_gemm_tc_decode(t.dev, dxb, dy, N, K, t.scale);
  });

  printf("%6s %14s %14s %10s\n", "batch", "CUDA cores", "tensor cores", "winner");
  for (int B = BMIN; B <= BMAX && B <= 8; ++B) {
    const double cc = time_it([&](const Tensor &t, int N, int K) {
      launch_gemm(t.dev, dx, dy, N, K, t.scale, B);
    });
    printf("%6d %11.2f ms %11.2f ms %10s  (%+.0f%%)\n", B, cc, tc,
           tc < cc ? "tensor" : "cuda", 100.0 * (cc - tc) / cc);
  }
  printf("\ntensor-core cost is flat in batch: one 16-row fragment covers any\n"
         "draft depth up to 16, so deeper speculation is free on that path.\n");

  cudaFree(dx); cudaFree(dxb); cudaFree(dy);
  free_model(m);
  return 0;
}
