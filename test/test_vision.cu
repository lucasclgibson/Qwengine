// test/test_vision.cu — does the CUDA vision tower match the reference?
//
// tools/vision_golden.py runs the real HF vision module on fixed random patches
// and dumps its inputs and output. This runs the same inputs through our tower
// and compares. The tower is 27 layers of attention and MLP; if any norm,
// activation, rotation or index is wrong the answer diverges immediately, so
// this is a sharp check rather than a smoke test.
#define SPARK27_NO_MAIN
#include "../src/vision.cu"

#include <math.h>
#include <vector>

using namespace spark27;

int main() {
  if (access("out/qwengine.bin", R_OK) != 0 || access("golden/vision.bin", R_OK) != 0) {
    printf("test_vision: SKIP — need out/qwengine.bin and golden/vision.bin\n");
    return 0;
  }
  FILE *f = fopen("golden/vision.bin", "rb");
  int N = 0, GH = 0, GW = 0;
  if (fread(&N, 4, 1, f) != 1 || fread(&GH, 4, 1, f) != 1 || fread(&GW, 4, 1, f) != 1) return 1;
  const int M = N / 4;
  std::vector<float> pix((size_t)N * vis::PATCH_FEAT);
  std::vector<int> pid((size_t)N * 2), bidx((size_t)4 * N);
  std::vector<float> bw((size_t)4 * N), want((size_t)M * vis::OUT);
  if (fread(pix.data(), 4, pix.size(), f) != pix.size()) return 1;
  if (fread(pid.data(), 4, pid.size(), f) != pid.size()) return 1;
  if (fread(bidx.data(), 4, bidx.size(), f) != bidx.size()) return 1;
  if (fread(bw.data(), 4, bw.size(), f) != bw.size()) return 1;
  if (fread(want.data(), 4, want.size(), f) != want.size()) return 1;
  fclose(f);
  printf("golden: %d patches (%dx%d) -> %d merged tokens\n", N, GH, GW, M);

  Model m = load_model("out/qwengine.bin");
  VisionTower T;
  if (!vision_init(T, m, 4096)) {
    printf("test_vision: SKIP — this build has no vision tower "
           "(convert without --no-vision)\n");
    free_model(m);
    return 0;
  }

  float *dpix = nullptr;
  LCHECK(cudaMalloc(&dpix, pix.size() * 4));
  LCHECK(cudaMemcpy(dpix, pix.data(), pix.size() * 4, cudaMemcpyHostToDevice));
  vision_forward(T, dpix, pid.data(), bidx.data(), bw.data(), N, 0);
  LCHECK(cudaDeviceSynchronize());

  std::vector<float> got((size_t)M * vis::OUT);
  LCHECK(cudaMemcpy(got.data(), T.out, got.size() * 4, cudaMemcpyDeviceToHost));

  double se = 0, sr = 0, worst = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = (double)got[i] - want[i];
    se += d * d;
    sr += (double)want[i] * want[i];
    worst = fmax(worst, fabs(d));
  }
  const double rel = sqrt(se / sr);
  printf("  merged embeddings: rel-err %.3e, worst abs %.3e\n", rel, worst);
  printf("  first 6 of token 0: ");
  for (int i = 0; i < 6; ++i) printf("%+.4f/%+.4f ", got[i], want[i]);
  printf("\n");

  // BF16 tensor cores against an fp32 reference: a few 1e-3 is expected,
  // anything larger means a real disagreement, not precision.
  const bool ok = rel < 2e-2;
  cudaFree(dpix);
  free_model(m);
  printf("test_vision: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
