// src/main.cu — run the engine.
//
//   build/spark27 <weights.bin> <tok0,tok1,...> [ngen]
//
// Feeds the prompt tokens through the model one at a time (each one updating
// the caches), then generates `ngen` tokens greedily, printing the ids. A
// separate script turns text into ids and ids back into text; that becomes a
// built-in tokenizer later.

#define SPARK27_NO_MAIN
#include "prefill.cu"
#include "spec.cu"

#include <stdlib.h>
#include <string.h>
#include <vector>

using namespace spark27;

static double wall_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "out/qwengine.bin";
  std::vector<int> prompt;
  if (argc > 2) {
    char *s = strdup(argv[2]);
    for (char *t = strtok(s, ","); t; t = strtok(nullptr, ",")) prompt.push_back(atoi(t));
    free(s);
  }
  if (prompt.empty()) prompt.push_back(9707);  // "Hello"
  const int ngen = argc > 3 ? atoi(argv[3]) : 32;
  // Draft depth. 1 disables speculation; 3 measured best on this hardware.
  const int D = argc > 4 ? atoi(argv[4]) : 3;
  const int ctx = (int)prompt.size() + ngen + 8;

  Model m = load_model(path);
  Runtime rt;
  const int CTX = ctx < 512 ? 512 : ctx;
  rt_init(rt, m, CTX);
  MtpHead M;
  if (D > 1) {
    rt_capture_verify(rt, D);
    mtp_init(M, m, CTX);
    mtp_capture(M, rt.embed, rt.s_embed, rt.lm_head, rt.s_lm, rt.hlast, D - 1,
                rt.stream);
  }

  // ---- prefill: read the whole prompt in chunks ----------------------------
  PrefillBuf pb;
  const int PF_CHUNK = 256;
  prefill_init(pb, PF_CHUNK, CTX);
  double t0 = wall_ms();
  int next = prefill(rt, pb, prompt);
  LCHECK(cudaDeviceSynchronize());
  const double prefill_ms = wall_ms() - t0;

  // ---- generate ------------------------------------------------------------
  printf("PROMPT_TOKENS");
  for (int t : prompt) printf(" %d", t);
  printf("\nGENERATED");
  fflush(stdout);

  std::vector<int> out;
  SpecStats stats;
  t0 = wall_ms();
  if (D > 1) {
    while ((int)out.size() < ngen) next = spec_step(rt, M, D, next, out, stats);
    out.resize(ngen);
  } else {
    for (int i = 0; i < ngen; ++i) { out.push_back(next); next = step(rt, next); }
  }
  LCHECK(cudaDeviceSynchronize());
  const double gen_ms = wall_ms() - t0;
  for (int t : out) printf(" %d", t);
  printf("\n");

  printf("\nprefill : %zu tokens in %.0f ms (%.1f tok/s)\n", prompt.size(),
         prefill_ms, prompt.size() * 1000.0 / prefill_ms);
  printf("decode  : %d tokens in %.0f ms (%.2f tok/s, %.1f ms/token)\n", ngen,
         gen_ms, ngen * 1000.0 / gen_ms, gen_ms / ngen);
  if (D > 1)
    printf("spec    : depth %d, mean accept %.2f tok/pass, draft hit %.0f%%\n", D,
           stats.mean_accept(), 100 * stats.draft_hit());

  free_model(m);
  return 0;
}
