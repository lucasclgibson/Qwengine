// test/test_delta.cu — the chunked delta rule must equal the per-token one.
//
// The DeltaNet recurrence had no test. It is the one part of the model whose
// state carries across tokens, so an error in it does not produce a crash or an
// obviously wrong number: it produces slightly wrong text, later. Before
// replacing the per-token kernel with a chunked one that keeps the state in
// shared memory, both are run over the same random inputs and compared, on both
// the outputs and the final state.
//
// The two do the same arithmetic in the same order, so they should agree to
// floating-point reassociation -- a few 1e-6 relative, not 1e-2.
#define SPARK27_NO_MAIN
#include "../qwen38.h"
#include "../src/mixers.cu"
using namespace q38;

#include <math.h>
#include <stdio.h>
#include <vector>

using namespace spark27;

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 1;} } while(0)

int main() {
  const int T = 96;                       // enough tokens for drift to show
  const int HV = LIN_V_HEADS, HK = LIN_K_HEADS, DK = LIN_HEAD_DIM, DV = LIN_HEAD_DIM;
  const int group = HV / HK;
  const size_t nstate = (size_t)HV * DK * DV;

  // Random but reproducible inputs. Keys and queries are l2-normalised the way
  // the model normalises them, since the recurrence is only stable there.
  std::vector<float> hqkv((size_t)T * LIN_QKV_DIM), hg((size_t)T * HV),
      hb((size_t)T * HV), hs(nstate);
  unsigned r = 12345;
  auto rnd = [&r]() { r = r * 1664525u + 1013904223u; return (float)((r >> 8) & 0xFFFF) / 32768.f - 1.f; };
  for (auto &v : hs) v = rnd() * 0.05f;
  for (int t = 0; t < T; ++t) {
    float *row = hqkv.data() + (size_t)t * LIN_QKV_DIM;
    for (int i = 0; i < LIN_QKV_DIM; ++i) row[i] = rnd();
    for (int hh = 0; hh < HK; ++hh) {           // normalise q and k per head
      for (int off : {0, LIN_Q_DIM}) {
        float *p = row + off + (size_t)hh * DK, s = 0.f;
        for (int i = 0; i < DK; ++i) s += p[i] * p[i];
        s = 1.f / sqrtf(s + 1e-6f);
        for (int i = 0; i < DK; ++i) p[i] *= s;
      }
    }
    for (int hh = 0; hh < HV; ++hh) {
      hg[(size_t)t * HV + hh] = -0.05f - 0.2f * fabsf(rnd());   // decay, so g < 0
      hb[(size_t)t * HV + hh] = 0.5f + 0.4f * fabsf(rnd());
    }
  }

  float *dqkv, *dg, *db, *dS1, *dS2, *dout1, *dout2;
  CK(cudaMalloc(&dqkv, hqkv.size() * 4));  CK(cudaMalloc(&dg, hg.size() * 4));
  CK(cudaMalloc(&db, hb.size() * 4));      CK(cudaMalloc(&dS1, nstate * 4));
  CK(cudaMalloc(&dS2, nstate * 4));
  CK(cudaMalloc(&dout1, (size_t)T * LIN_V_DIM * 4));
  CK(cudaMalloc(&dout2, (size_t)T * LIN_V_DIM * 4));
  CK(cudaMemcpy(dqkv, hqkv.data(), hqkv.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dg, hg.data(), hg.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dS1, hs.data(), nstate * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dS2, hs.data(), nstate * 4, cudaMemcpyHostToDevice));

  // Reference: the shipping per-token kernel, one launch per token.
  for (int t = 0; t < T; ++t) {
    const float *row = dqkv + (size_t)t * LIN_QKV_DIM;
    k_delta_step<<<HV, 128 * DELTA_DKSPLIT>>>(
        dS1, row, row + LIN_Q_DIM, row + LIN_Q_DIM + LIN_K_DIM,
        dg + (size_t)t * HV, db + (size_t)t * HV,
        dout1 + (size_t)t * LIN_V_DIM, DK, DV, group);
  }
  CK(cudaDeviceSynchronize());

  // Chunked: one launch for the whole chunk, state resident in shared.
  const size_t smem = ((size_t)DK * DELTA_DVT + 2 * DK + 2 * DELTA_SPLIT * DELTA_DVT
                       + DELTA_SPLIT) * sizeof(float);
  CK(cudaFuncSetAttribute(k_delta_chunk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)smem));
  k_delta_chunk<<<dim3(HV, DV / DELTA_DVT), DELTA_DVT * DELTA_SPLIT, smem>>>(
      dS2, dqkv, dg, db, dout2, T, DK, DV, group, 0, LIN_Q_DIM,
      LIN_Q_DIM + LIN_K_DIM, LIN_QKV_DIM);
  CK(cudaDeviceSynchronize());
  CK(cudaGetLastError());

  auto cmp = [](const char *what, const std::vector<float> &a,
                const std::vector<float> &b) {
    double se = 0, sr = 0, worst = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - b[i];
      se += d * d; sr += (double)b[i] * b[i];
      worst = fmax(worst, fabs(d));
    }
    const double rel = sqrt(se / (sr > 0 ? sr : 1));
    printf("  %-14s rel-err %.3e, worst abs %.3e\n", what, rel, worst);
    return rel;
  };

  std::vector<float> o1((size_t)T * LIN_V_DIM), o2(o1.size()), s1(nstate), s2(nstate);
  CK(cudaMemcpy(o1.data(), dout1, o1.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(o2.data(), dout2, o2.size() * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(s1.data(), dS1, nstate * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(s2.data(), dS2, nstate * 4, cudaMemcpyDeviceToHost));

  printf("delta rule, %d tokens, %d heads, %dx%d state:\n", T, HV, DK, DV);
  const double ro = cmp("outputs", o2, o1);
  const double rs = cmp("final state", s2, s1);

  // Same operations in the same order: only reassociation should differ. A real
  // disagreement (a wrong index, a missed decay) lands orders of magnitude
  // above this, and the state error compounds over tokens so it is the sharper
  // of the two checks.
  const bool ok = ro < 1e-4 && rs < 1e-4;
  printf("test_delta: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
