// bench/gemv_shapes.cu — where does the GEMV kernel actually lose bandwidth?
//
// The full-model pass runs at 77% of B. That average hides the answer. This
// breaks it down per matrix shape, so the fix is aimed at a measured problem
// rather than a guessed one.
#define SPARK27_NO_MAIN
#include "../src/loader.cpp"
#include "../src/gemv.cu"

#include <algorithm>
#include <map>
#include <vector>

using namespace spark27;

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "out/qwengine.bin";
  // Batch sweep: a verify pass in speculative decoding runs at batch = draft
  // depth, so what matters is not the batch-1 pass but how the cost GROWS with
  // batch. If an extra row were free the depth could be raised for free too.
  const int BMAX = argc > 2 ? atoi(argv[2]) : 1;
  const int BMIN = argc > 3 ? atoi(argv[3]) : 1;   // narrow the sweep while iterating
  const bool SHAPES = argc <= 4;                   // 5th arg: skip the per-shape table
  Model m = load_model(path);
  const double B = 235.0;

  // Group the model's matrices by shape; count how many of each there are and
  // how many bytes per pass they account for.
  struct Info { int count; int64_t bytes; };
  std::map<std::pair<int, int>, Info> shapes;
  std::map<std::pair<int, int>, const TensorEntry *> sample;
  for (auto &e : m.table) {
    if (e.kind != KIND_NVFP4 || strstr(e.name, "embed_tokens")) continue;
    auto key = std::make_pair(e.n, e.k);
    shapes[key].count++;
    shapes[key].bytes += e.bytes;
    if (!sample.count(key)) sample[key] = &e;
  }

  int maxK = 0, maxN = 0;
  for (auto &kv : shapes) { maxK = std::max(maxK, kv.first.second); maxN = std::max(maxN, kv.first.first); }
  // Batched: x is [B,K] and y is [B,N], so both buffers scale with the batch.
  const int BA = BMAX < 1 ? 1 : BMAX > 8 ? 8 : BMAX;
  float *dx, *dy;
  LCHECK(cudaMalloc(&dx, (size_t)maxK * BA * 4));
  LCHECK(cudaMalloc(&dy, (size_t)maxN * BA * 4));
  LCHECK(cudaMemset(dx, 0x3C, (size_t)maxK * BA * 4));

  cudaDeviceProp prop;
  LCHECK(cudaGetDeviceProperties(&prop, 0));
  printf("SMs = %d\n\n", prop.multiProcessorCount);
  printf("%8s %8s %6s %10s %9s %8s %7s  %s\n", "N", "K", "count", "GB/pass",
         "blocks", "blk/SM", "GB/s", "% of B");

  double total_bytes = 0, total_ms = 0;
  std::vector<std::pair<double, std::string>> losses;

  for (auto &kv : shapes) {
    const int N = kv.first.first, K = kv.first.second;
    Tensor t = m.get(sample[kv.first]->name);
    if (!SHAPES) { total_bytes += kv.second.bytes; continue; }
    const int rpb = GEMV_WARPS * (N >= 4608*4 ? 4 : N >= 4608*2 ? 2 : 1); const int blocks = (N + rpb - 1) / rpb;

    for (int i = 0; i < 3; ++i) launch_gemv(t.dev, dx, dy, N, K, t.scale);
    LCHECK(cudaDeviceSynchronize());

    cudaEvent_t a, b;
    LCHECK(cudaEventCreate(&a)); LCHECK(cudaEventCreate(&b));
    double best = 1e30;
    const int reps = 20;
    for (int i = 0; i < 5; ++i) {
      LCHECK(cudaEventRecord(a));
      for (int r = 0; r < reps; ++r) launch_gemv(t.dev, dx, dy, N, K, t.scale);
      LCHECK(cudaEventRecord(b));
      LCHECK(cudaEventSynchronize(b));
      float ms = 0; LCHECK(cudaEventElapsedTime(&ms, a, b));
      best = std::min(best, (double)ms / reps);
    }
    const double gbs = t.bytes / (best * 1e-3) / 1e9;
    printf("%8d %8d %6d %10.3f %9d %8.1f %7.1f  %5.0f%%%s\n", N, K, kv.second.count,
           kv.second.bytes / 1e9, blocks, (double)blocks / prop.multiProcessorCount,
           gbs, 100.0 * gbs / B, gbs < 0.8 * B ? "  <-- slow" : "");

    // Time this shape contributes to a full pass, and the time it would take
    // at full bandwidth. The gap is what is recoverable.
    const double ms_actual = kv.second.bytes / (gbs * 1e9) * 1e3;
    const double ms_ideal = kv.second.bytes / (B * 1e9) * 1e3;
    total_bytes += kv.second.bytes;
    total_ms += ms_actual;
    char buf[128];
    snprintf(buf, sizeof buf, "[%d,%d] x%d", N, K, kv.second.count);
    losses.push_back({ms_actual - ms_ideal, buf});
  }

  if (SHAPES)
  printf("\nper-pass total: %.3f GB in %.2f ms = %.1f GB/s (%.0f%% of B), %.2f passes/s\n",
         total_bytes / 1e9, total_ms, total_bytes / (total_ms * 1e-3) / 1e9,
         100.0 * total_bytes / (total_ms * 1e-3) / 1e9 / B, 1000.0 / total_ms);

  if (SHAPES) {
  std::sort(losses.begin(), losses.end(), [](auto &a, auto &b) { return a.first > b.first; });
  printf("\nwhere the time is lost (ms per pass above the bandwidth floor):\n");
  double tot_loss = 0; for (auto &l : losses) tot_loss += l.first;
  for (size_t i = 0; i < losses.size() && i < 8; ++i)
    printf("   %-22s %6.2f ms  (%.0f%% of all loss)\n", losses[i].second.c_str(),
           losses[i].first, 100.0 * losses[i].first / tot_loss);
  printf("   total recoverable: %.2f ms of %.2f ms\n", tot_loss, total_ms);
  }

  // ---- batch sweep ---------------------------------------------------------
  // Same shapes, same weights, batch 1..BMAX. The number to watch is the
  // marginal cost of row b: in a bandwidth-bound kernel it should be near
  // zero, because the weights are read once no matter how many rows use them.
  if (BMAX > 1) {
    printf("\nbatch sweep (whole-pass time, and the cost of the row that was added):\n");
    printf("%5s %10s %10s %9s %10s\n", "batch", "ms/pass", "tok/s", "GB/s", "marginal");
    double prev = 0;
    for (int b = BMIN; b <= BMAX && b <= 8; ++b) {
      double ms_total = 0;
      for (auto &kv : shapes) {
        const int N = kv.first.first, K = kv.first.second;
        Tensor t = m.get(sample[kv.first]->name);
        for (int i = 0; i < 3; ++i) launch_gemm(t.dev, dx, dy, N, K, t.scale, b);
        LCHECK(cudaDeviceSynchronize());
        cudaEvent_t a2, b2;
        LCHECK(cudaEventCreate(&a2)); LCHECK(cudaEventCreate(&b2));
        double best = 1e30;
        const int reps = 8;
        for (int i = 0; i < 2; ++i) {
          LCHECK(cudaEventRecord(a2));
          for (int r = 0; r < reps; ++r) launch_gemm(t.dev, dx, dy, N, K, t.scale, b);
          LCHECK(cudaEventRecord(b2));
          LCHECK(cudaEventSynchronize(b2));
          float ms = 0; LCHECK(cudaEventElapsedTime(&ms, a2, b2));
          best = std::min(best, (double)ms / reps);
        }
        LCHECK(cudaEventDestroy(a2)); LCHECK(cudaEventDestroy(b2));
        ms_total += best * (kv.second.bytes / (double)t.bytes);
      }
      char marg[32];
      if (b == BMIN) snprintf(marg, sizeof marg, "%10s", "-");
      else snprintf(marg, sizeof marg, "%+7.2f ms", ms_total - prev);
      printf("%5d %10.2f %10.2f %9.1f %s\n", b, ms_total, b * 1000.0 / ms_total,
             b * total_bytes / (ms_total * 1e-3) / 1e9, marg);
      prev = ms_total;
    }
  }

  LCHECK(cudaFree(dx)); LCHECK(cudaFree(dy));
  free_model(m);
  return 0;
}
