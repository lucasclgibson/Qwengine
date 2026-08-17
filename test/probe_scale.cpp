// test/probe_scale.cpp — diagnostic: how does block SSE actually vary with the
// chosen E4M3 scale code, on real weights?
//
// Written because widening the scale-search window from [-8,+2] to [-16,+4]
// improved reported rel-err in a way that should not be possible: a scale 4x
// below nominal clips most of a gaussian block. Either the search is finding
// a genuine second optimum, or something is wrong. This prints the curve
// instead of guessing.
#define SPARK27_NO_MAIN
#include "../src/convert.cpp"

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s <shard.safetensors> [tensor]\n", argv[0]); return 2; }
  Shard sh;
  open_shard(sh, argv[1]);
  const char *want = argc > 2 ? argv[2] : nullptr;

  std::string name;
  const json::Value *t = nullptr;
  for (auto &kv : sh.header->obj) {
    if (kv.first == "__metadata__") continue;
    if (json::get(kv.second, "shape")->arr.size() != 2) continue;
    if (want && kv.first.find(want) == std::string::npos) continue;
    name = kv.first; t = kv.second; break;
  }
  if (!t) { fprintf(stderr, "no 2-D tensor found\n"); return 1; }
  int64_t N = (int64_t)json::get(t, "shape")->arr[0]->num;
  int64_t K = (int64_t)json::get(t, "shape")->arr[1]->num;
  int64_t off = (int64_t)json::get(t, "data_offsets")->arr[0]->num;
  const uint16_t *w = (const uint16_t *)(sh.base + sh.data_start + off);
  printf("tensor %s [%lld,%lld]\n\n", name.c_str(), (long long)N, (long long)K);

  // Tensor scale exactly as pack_nvfp4 computes it.
  float absmax = 0;
  for (int64_t i = 0; i < N * K; ++i) {
    float v = fabsf(bf16_to_f32(w[i]));
    if (v > absmax) absmax = v;
  }
  float tscale = absmax / (kE2M1Max * kE4M3Max), inv_t = 1.0f / tscale;
  printf("absmax %.6g  tscale %.6g\n\n", absmax, tscale);

  // Aggregate SSE across many blocks as a function of offset from nominal.
  const int LO = 24, HI = 8;
  std::vector<double> agg((size_t)(LO + HI + 1), 0.0);
  std::vector<int64_t> chosen((size_t)(LO + HI + 1), 0);
  const int64_t NBLK = 200000 < N * K / NVFP4_BLOCK ? 200000 : N * K / NVFP4_BLOCK;
  double total_sig = 0.0;

  for (int64_t b = 0; b < NBLK; ++b) {
    const uint16_t *blk = w + b * NVFP4_BLOCK;
    float bmax = 0;
    for (int i = 0; i < NVFP4_BLOCK; ++i) {
      float v = fabsf(bf16_to_f32(blk[i]));
      if (v > bmax) bmax = v;
    }
    for (int i = 0; i < NVFP4_BLOCK; ++i) {
      double x = bf16_to_f32(blk[i]);
      total_sig += x * x;
    }
    if (!(bmax > 0)) continue;
    int nominal = f32_to_e4m3(bmax * inv_t / kE2M1Max);
    double best = -1; int besto = 0;
    for (int o = -LO; o <= HI; ++o) {
      int c = nominal + o;
      if (c < 1 || c > 0x7E) continue;
      float sdec = e4m3_to_f32((uint8_t)c) * tscale;
      if (!(sdec > 0)) continue;
      float inv_s = 1.0f / sdec;
      double sse = 0;
      for (int i = 0; i < NVFP4_BLOCK; ++i) {
        double x = bf16_to_f32(blk[i]);
        double d = x - (double)e2m1_to_f32(f32_to_e2m1((float)(x * inv_s))) * sdec;
        sse += d * d;
      }
      agg[(size_t)(o + LO)] += sse;
      if (best < 0 || sse < best) { best = sse; besto = o; }
    }
    chosen[(size_t)(besto + LO)]++;
  }

  printf("offset  aggregate rel-err   times chosen as best\n");
  for (int o = -LO; o <= HI; ++o) {
    double rel = sqrt(agg[(size_t)(o + LO)] / total_sig);
    printf("  %+3d      %.4f            %lld%s\n", o, rel,
           (long long)chosen[(size_t)(o + LO)], o == 0 ? "   <- nominal" : "");
  }
  return 0;
}
