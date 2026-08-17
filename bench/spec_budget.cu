// bench/spec_budget.cu — where does a speculative cycle actually spend its time?
//
// A cycle is: draft D-1 guesses with the small head, then verify all D in one
// batched trunk pass. The tokens/s that comes out is
//
//     accepted_tokens / (draft_time + verify_time)
//
// and every optimisation is either "make one of those two smaller" or "make
// accepted larger". This times the two graphs in isolation, against the
// bandwidth floor each one would hit if it were reading weights and doing
// nothing else. The gap is the compute overhead — the part that is ours to
// remove rather than the memory system's to concede.
#define SPARK27_NO_MAIN
#include "../src/spec.cu"

#include <algorithm>
#include <vector>

using namespace spark27;

// Bytes a single pass over the trunk / the draft head has to stream.
static double trunk_bytes(Model &m) {
  double b = 0;
  for (auto &e : m.table)
    if (e.kind == KIND_NVFP4 && !strstr(e.name, "embed_tokens") &&
        !strstr(e.name, "mtp.") && !strstr(e.name, ".draft"))
      b += e.bytes;
  return b;
}
static double draft_bytes(Model &m) {
  double b = 0;
  for (auto &e : m.table)
    if (e.kind == KIND_NVFP4 && (strstr(e.name, "mtp.") || !strcmp(e.name, "lm_head.weight")))
      b += e.bytes;
  return b;
}

static double time_graph(cudaGraphExec_t g, cudaStream_t st, int reps = 20) {
  for (int i = 0; i < 3; ++i) LCHECK(cudaGraphLaunch(g, st));
  LCHECK(cudaStreamSynchronize(st));
  cudaEvent_t a, b;
  LCHECK(cudaEventCreate(&a)); LCHECK(cudaEventCreate(&b));
  double best = 1e30;
  for (int i = 0; i < 3; ++i) {
    LCHECK(cudaEventRecord(a, st));
    for (int r = 0; r < reps; ++r) LCHECK(cudaGraphLaunch(g, st));
    LCHECK(cudaEventRecord(b, st));
    LCHECK(cudaEventSynchronize(b));
    float ms = 0; LCHECK(cudaEventElapsedTime(&ms, a, b));
    best = std::min(best, (double)ms / reps);
  }
  LCHECK(cudaEventDestroy(a)); LCHECK(cudaEventDestroy(b));
  return best;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "out/qwengine.bin";
  const int DMAX = argc > 2 ? atoi(argv[2]) : 6;
  const double BW = 235.6;              // measured peak, GB/s

  Model m = load_model(path);
  const double tb = trunk_bytes(m) / 1e9, db = draft_bytes(m) / 1e9;
  printf("trunk pass %.3f GB (floor %.2f ms)   draft step %.3f GB (floor %.2f ms)\n\n",
         tb, tb / BW * 1e3, db, db / BW * 1e3);

  Runtime rt;
  rt_init(rt, m, 2048);
  rt_reset(rt);
  MtpHead M;
  mtp_init(M, m, 2048);


  printf("%6s %10s %10s %10s %10s %10s\n", "depth", "verify", "over", "draft/ea",
         "over", "cycle");
  for (int D = 2; D <= DMAX && D <= MAX_B; ++D) {
    // Fresh graphs for this depth.
    rt.vgraph_exec = nullptr; rt.vgraph = nullptr;
    rt_capture_verify(rt, D);
    M.graph_exec = nullptr; M.graph = nullptr;
    mtp_capture(M, rt.embed, rt.s_embed, rt.lm_head, rt.s_lm, rt.hlast, D - 1,
                rt.stream);

    const double v = time_graph(rt.vgraph_exec, rt.stream);
    const double d = time_graph(M.graph_exec, rt.stream) / (D - 1);
    printf("%6d %8.2f ms %+8.2f %8.2f ms %+8.2f %8.2f ms\n", D, v, v - tb / BW * 1e3,
           d, d - db / BW * 1e3, v + (D - 1) * d);
  }
  printf("\n'over' is time above the bandwidth floor: kernels doing arithmetic\n"
         "rather than waiting on memory. That is the part worth attacking.\n");
  free_model(m);
  return 0;
}
