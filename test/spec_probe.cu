// test/spec_probe.cu — how often is the draft head right?
//
// This is the number the whole speculation design hinges on. Break-even at
// batch 4 needs 1.46 accepted tokens per pass; that requires the first draft
// to be right well over half the time. Measure it before building the
// accept/rollback machinery around it.
//
// Protocol: run ordinary greedy decode. At each step the trunk turns token
// t_i into hidden h_i and predicts t_{i+1}. Feed (h_i, t_{i+1}) to the draft
// head and ask it for t_{i+2}. Compare against what the trunk actually says
// next. That is exactly the question speculation asks.
#define SPARK27_NO_MAIN
#include "../src/model.cu"
#include "../src/mtp.cu"
#include <vector>
using namespace spark27;

int main(int argc, char **argv) {
  const char *path = "out/qwengine.bin";
  std::vector<int> prompt;
  if (argc > 1) { char *s = strdup(argv[1]);
    for (char *t = strtok(s, ","); t; t = strtok(nullptr, ",")) prompt.push_back(atoi(t)); }
  if (prompt.empty()) prompt = {760, 6511, 314, 9338, 369};
  const int ngen = argc > 2 ? atoi(argv[2]) : 64;

  Model m = load_model(path);
  const int ctx = (int)prompt.size() + ngen + 8;
  Runtime rt; rt_init(rt, m, ctx < 512 ? 512 : ctx);
  MtpHead M; mtp_init(M, m, ctx < 512 ? 512 : ctx);

  int next = 0;
  for (size_t i = 0; i < prompt.size(); ++i) next = step(rt, prompt[i]);

  int hits = 0, tries = 0;
  int pending_draft = -1;
  std::vector<int> gen;
  for (int i = 0; i < ngen; ++i) {
    // Score the draft made on the previous iteration against what the trunk
    // actually produced now.
    if (pending_draft >= 0) { ++tries; if (pending_draft == next) ++hits; }
    gen.push_back(next);

    // Draft from (hidden of the token we just consumed, token just produced).
    LCHECK(cudaMemcpyAsync(M.dtok, &next, sizeof(int), cudaMemcpyHostToDevice, rt.stream));
    // MTP_POSTNORM: feed the draft head the hidden state AFTER the trunk's final
    // norm instead of before it. The checkpoint does not say which is right --
    // mtp.norm exists as the head's own read-out norm, which argues for
    // pre-norm, but that is inference. Draft accuracy settles it.
#if MTP_POSTNORM
    mtp_draft(M, rt.embed, rt.s_embed, rt.lm_head, rt.s_lm, rt.hn, M.dtok, M.dtok, rt.stream);
#else
    mtp_draft(M, rt.embed, rt.s_embed, rt.lm_head, rt.s_lm, rt.hlast, M.dtok, M.dtok, rt.stream);
#endif
    LCHECK(cudaMemcpyAsync(&pending_draft, M.dtok, sizeof(int), cudaMemcpyDeviceToHost, rt.stream));
    LCHECK(cudaStreamSynchronize(rt.stream));

    next = step(rt, next);
  }
  printf("\ndraft head: %d/%d correct = %.1f%%\n", hits, tries, 100.0 * hits / tries);
  printf("tokens: ");
  for (int t : gen) printf("%d ", t);
  printf("\n");
  free_model(m);
  return 0;
}
