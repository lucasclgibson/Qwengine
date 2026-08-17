// test/test_spec.cu — speculation must be invisible.
//
// The whole safety argument for speculative decoding is that the big model has
// the final say at every position, so the tokens committed are EXACTLY the
// tokens plain greedy decoding would have produced. Speculation is allowed to
// change the speed and nothing else.
//
// This runs both paths from the same prompt and demands identical token
// sequences. If they ever diverge, the accept logic or the state rollback is
// wrong — and a subtly wrong rollback would show up as output that is
// plausible but not reproducible, which is the worst kind of bug to chase.
#define SPARK27_NO_MAIN
#include "../src/spec.cu"

#include <vector>
using namespace spark27;

static int g_fail = 0;

static std::vector<int> run_plain(Model &m, const std::vector<int> &prompt, int n,
                                  int ctx, double *ms) {
  Runtime rt;
  rt_init(rt, m, ctx);
  int next = 0;
  for (int t : prompt) next = step(rt, t);
  std::vector<int> out;
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  for (int i = 0; i < n; ++i) { out.push_back(next); next = step(rt, next); }
  clock_gettime(CLOCK_MONOTONIC, &b);
  *ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) * 1e-6;
  return out;
}

static std::vector<int> run_spec(Model &m, const std::vector<int> &prompt, int n,
                                 int ctx, int D, SpecStats &stats, double *ms) {
  Runtime rt;
  rt_init(rt, m, ctx);
  rt_capture_verify(rt, D);
  MtpHead M;
  mtp_init(M, m, ctx);
  mtp_capture(M, rt.embed, rt.s_embed, rt.lm_head, rt.s_lm, rt.hlast, D - 1, rt.stream);
  int next = 0;
  for (int t : prompt) next = step(rt, t);
  std::vector<int> out;
  struct timespec a, b;
  clock_gettime(CLOCK_MONOTONIC, &a);
  while ((int)out.size() < n) next = spec_step(rt, M, D, next, out, stats);
  clock_gettime(CLOCK_MONOTONIC, &b);
  *ms = (b.tv_sec - a.tv_sec) * 1e3 + (b.tv_nsec - a.tv_nsec) * 1e-6;
  out.resize(n);
  return out;
}

int main(int argc, char **argv) {
  if (access("out/qwengine.bin", R_OK) != 0) {
    printf("test_spec: SKIP — no weights\n");
    return 0;
  }
  const int n = argc > 1 ? atoi(argv[1]) : 48;
  const int D = argc > 2 ? atoi(argv[2]) : 4;
  Model m = load_model("out/qwengine.bin");

  const std::vector<std::vector<int>> prompts = {
      {760, 6511, 314, 9338, 369},                 // "The capital of France is"
      {688, 220, 16, 24, 21, 24, 279, 76671},      // an Apollo sentence
  };

  for (size_t pi = 0; pi < prompts.size(); ++pi) {
    const auto &p = prompts[pi];
    const int ctx = (int)p.size() + n + 16;
    double ms_plain = 0, ms_spec = 0;
    SpecStats stats;
    printf("\n=== prompt %zu ===\n", pi);
    std::vector<int> a = run_plain(m, p, n, ctx, &ms_plain);
    std::vector<int> b = run_spec(m, p, n, ctx, D, stats, &ms_spec);

    int firstbad = -1;
    for (int i = 0; i < n; ++i)
      if (a[i] != b[i]) { firstbad = i; break; }

    printf("  plain : %.2f tok/s\n", n * 1000.0 / ms_plain);
    printf("  spec  : %.2f tok/s   (D=%d, mean accept %.2f tok/pass, "
           "draft hit %.0f%%)\n",
           n * 1000.0 / ms_spec, D, stats.mean_accept(), 100 * stats.draft_hit());
    printf("  speedup: %.2fx\n", ms_plain / ms_spec);
    if (firstbad < 0) {
      printf("  IDENTICAL to greedy: yes (%d tokens)\n", n);
    } else {
      printf("  MISMATCH at token %d: greedy=%d spec=%d\n", firstbad, a[firstbad],
             b[firstbad]);
      printf("    greedy:");
      for (int i = 0; i < n && i < firstbad + 6; ++i) printf(" %d", a[i]);
      printf("\n    spec  :");
      for (int i = 0; i < n && i < firstbad + 6; ++i) printf(" %d", b[i]);
      printf("\n");
      ++g_fail;
    }
  }
  free_model(m);
  printf("\ntest_spec: %s\n", g_fail ? "FAIL" : "PASS");
  return g_fail ? 1 : 0;
}
