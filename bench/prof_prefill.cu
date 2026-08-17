// bench/prof_prefill.cu — replay one prefill chunk, for a profiler to watch.
#define SPARK27_NO_MAIN
#include "../src/prefill.cu"
using namespace spark27;
int main(int argc, char **argv) {
  const int T = argc > 1 ? atoi(argv[1]) : 512;
  Model m = load_model("out/qwengine.bin");
  Runtime rt; rt_init(rt, m, 4096); rt_reset(rt);
  PrefillBuf p; prefill_init(p, T, 4096);
  std::vector<int> toks(T, 1000);
  LCHECK(cudaMemcpy(p.dtok, toks.data(), T * sizeof(int), cudaMemcpyHostToDevice));
  for (int i = 0; i < 2; ++i) { rt_reset(rt); prefill_chunk(rt, p, T); }
  LCHECK(cudaDeviceSynchronize());
  for (int i = 0; i < 5; ++i) { rt_reset(rt); prefill_chunk(rt, p, T); }
  LCHECK(cudaDeviceSynchronize());
  free_model(m);
  return 0;
}
