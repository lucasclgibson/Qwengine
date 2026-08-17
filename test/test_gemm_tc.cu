// test/test_gemm_tc.cu — does the tensor-core prefill GEMM agree with the
// decode kernel, and how fast is it?
//
// The decode kernel is the reference: it is the one validated by producing
// correct text. This checks the prefill path against it on real weights.
// Exact agreement is not expected — prefill multiplies in BF16 with fp32
// accumulation where decode uses fp32 throughout — so the bar is "close enough
// that the model cannot tell", not bit-equality.
#define SPARK27_NO_MAIN
#include "../src/loader.cpp"
#include "../src/gemm_tc.cu"

#include <math.h>
#include <vector>

using namespace spark27;

static int g_fail = 0;

int main() {
  if (access("out/qwengine.bin", R_OK) != 0) {
    printf("test_gemm_tc: SKIP — no weights\n");
    return 0;
  }
  Model m = load_model("out/qwengine.bin");

  const char *names[] = {
      "model.language_model.layers.0.mlp.gate_up_proj.weight",    // [17408,5120]
      "model.language_model.layers.0.mlp.down_proj.weight",    // [5120,17408]
      "model.language_model.layers.3.self_attn.qkv_proj.weight", // [12288,5120]
  };

  for (const char *nm : names) {
    Tensor t = m.get(nm);
    const int N = t.n, K = t.k, T = 64;

    std::vector<float> hx((size_t)T * K);
    for (size_t i = 0; i < hx.size(); ++i) hx[i] = sinf(i * 0.013f) * 0.7f;
    std::vector<__nv_bfloat16> hxb(hx.size());
    for (size_t i = 0; i < hx.size(); ++i) hxb[i] = __float2bfloat16(hx[i]);

    float *dx = nullptr, *dc = nullptr, *dref = nullptr;
    __nv_bfloat16 *dxb = nullptr;
    LCHECK(cudaMalloc(&dx, hx.size() * 4));
    LCHECK(cudaMalloc(&dxb, (size_t)((T + 63) / 64 * 64) * K * 2));
    LCHECK(cudaMemset(dxb, 0, (size_t)((T + 63) / 64 * 64) * K * 2));
    LCHECK(cudaMalloc(&dc, (size_t)((T + 63) / 64 * 64) * N * 4));
    LCHECK(cudaMalloc(&dref, (size_t)N * 4));
    LCHECK(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice));
    LCHECK(cudaMemcpy(dxb, hxb.data(), hx.size() * 2, cudaMemcpyHostToDevice));

    launch_gemm_tc(t.dev, dxb, dc, T, N, K, t.scale);
    launch_gemm(t.dev, dx, dref, N, K, t.scale, 1);   // decode kernel, row 0
    LCHECK(cudaDeviceSynchronize());

    std::vector<float> got((size_t)N), ref((size_t)N);
    LCHECK(cudaMemcpy(got.data(), dc, (size_t)N * 4, cudaMemcpyDeviceToHost));
    LCHECK(cudaMemcpy(ref.data(), dref, (size_t)N * 4, cudaMemcpyDeviceToHost));

    double se = 0, sr = 0;
    for (int i = 0; i < N; ++i) {
      const double d = (double)got[i] - ref[i];
      se += d * d;
      sr += (double)ref[i] * ref[i];
    }
    const double rel = sqrt(se / sr);
    printf("  %-54s [%6d,%6d] rel-err vs decode %.2e\n", nm, N, K, rel);
    if (!(rel < 5e-3)) {
      printf("      FAIL: prefill disagrees with decode\n");
      ++g_fail;
    }

    // ---- throughput at a realistic prompt width ----------------------------
    const int TT = 512;
    float *dc2 = nullptr;
    __nv_bfloat16 *dxb2 = nullptr;
    LCHECK(cudaMalloc(&dc2, (size_t)((TT + 63) / 64 * 64) * N * 4));
    LCHECK(cudaMalloc(&dxb2, (size_t)TT * K * 2));
    LCHECK(cudaMemset(dxb2, 0x3C, (size_t)TT * K * 2));
    for (int i = 0; i < 3; ++i) launch_gemm_tc(t.dev, dxb2, dc2, TT, N, K, t.scale);
    LCHECK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    LCHECK(cudaEventCreate(&a));
    LCHECK(cudaEventCreate(&b));
    double best = 1e30;
    for (int r = 0; r < 5; ++r) {
      LCHECK(cudaEventRecord(a));
      launch_gemm_tc(t.dev, dxb2, dc2, TT, N, K, t.scale);
      LCHECK(cudaEventRecord(b));
      LCHECK(cudaEventSynchronize(b));
      float ms = 0;
      LCHECK(cudaEventElapsedTime(&ms, a, b));
      best = fmin(best, (double)ms);
    }
    printf("      T=%d: %6.2f ms = %5.1f TFLOP/s\n", TT, best,
           2.0 * TT * N * K / (best * 1e-3) / 1e12);

    cudaFree(dx); cudaFree(dxb); cudaFree(dc); cudaFree(dref);
    cudaFree(dc2); cudaFree(dxb2);
  }

  free_model(m);
  printf("test_gemm_tc: %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
