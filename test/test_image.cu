// test/test_image.cu — does our preprocessing feed the tower the same pixels
// the reference does? A silent mismatch here means the model sees a subtly
// different picture, with no error to notice.
#define SPARK27_NO_MAIN
#include "../src/image.cpp"
#include <math.h>
#include <vector>
using namespace spark27;

int main() {
  if (access("golden/image.bin", R_OK) != 0) { printf("test_image: SKIP\n"); return 0; }
  FILE *f = fopen("golden/image.bin", "rb");
  int H, W, GH, GW, N;
  if (fread(&H,4,1,f)!=1||fread(&W,4,1,f)!=1||fread(&GH,4,1,f)!=1||
      fread(&GW,4,1,f)!=1||fread(&N,4,1,f)!=1) return 1;
  std::vector<uint8_t> rgb((size_t)H*W*3);
  std::vector<float> want((size_t)N*vis::PATCH_FEAT);
  if (fread(rgb.data(),1,rgb.size(),f)!=rgb.size()) return 1;
  if (fread(want.data(),4,want.size(),f)!=want.size()) return 1;
  fclose(f);

  ImageTokens t = preprocess_image(rgb.data(), H, W);
  printf("  input %dx%d -> grid %dx%d (%d patches, %d tokens); reference grid %dx%d, %d patches\n",
         W, H, t.grid_w, t.grid_h, t.n_patches, t.n_tokens, GW, GH, N);
  int fail = 0;
  if (t.grid_h != GH || t.grid_w != GW) { printf("  FAIL: grid mismatch\n"); fail = 1; }
  if (t.n_patches != N) { printf("  FAIL: patch count mismatch\n"); fail = 1; }
  if (!fail) {
    double se = 0, sr = 0, worst = 0;
    for (size_t i = 0; i < want.size(); ++i) {
      const double d = t.pixels[i] - want[i];
      se += d*d; sr += (double)want[i]*want[i];
      worst = fmax(worst, fabs(d));
    }
    const double rel = sqrt(se / sr);
    printf("  pixels: rel-err %.4f, worst abs %.4f\n", rel, worst);
    // Our bicubic is not bit-identical to torchvision's, so a small difference
    // is expected; a large one means the layout or normalisation is wrong.
    if (rel > 0.05) { printf("  FAIL: pixels differ too much\n"); fail = 1; }
  }
  printf("test_image: %s\n", fail ? "FAIL" : "PASS");
  return fail;
}
