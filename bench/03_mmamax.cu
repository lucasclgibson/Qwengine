// bench/03_mmamax.cu — what is the BF16 tensor-core ceiling on GB10?
//
// The prefill GEMM runs at ~22 TFLOP/s and SGLang gets roughly 110 on the same
// box. Before rewriting anything it is worth knowing what the hardware can
// actually do, exactly as bench/02 pinned the memory ceiling for decode. This
// is the same discipline: measure the denominator, then every ratio means
// something.
//
// The loop keeps fragments in registers and issues mma_sync back to back with
// no global traffic and no shared traffic in the timed region, so it measures
// pure tensor-core issue rate -- an upper bound no real GEMM can beat.
//
// It also times a variant that RELOADS fragments from shared each step, which
// is what a real GEMM must do, to separate "the tensor cores are slow" from
// "we cannot feed them".
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

namespace wmma = nvcuda::wmma;

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 1;} } while(0)

// ACC accumulators per warp, fed from registers. ITERS mma per accumulator.
template <int ACC>
__global__ __launch_bounds__(256) void mma_peak(float *sink, int iters) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[ACC];
  wmma::fill_fragment(fa, __float2bfloat16(1.0f));
  wmma::fill_fragment(fb, __float2bfloat16(1.0f));
#pragma unroll
  for (int i = 0; i < ACC; ++i) wmma::fill_fragment(acc[i], 0.0f);

  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < ACC; ++i) wmma::mma_sync(acc[i], fa, fb, acc[i]);
  }
  float s = 0;
#pragma unroll
  for (int i = 0; i < ACC; ++i) s += acc[i].x[0];
  if (threadIdx.x == 1024) sink[0] = s;      // never true, never optimised away
  if (s == 12345.678f) sink[blockIdx.x] = s;
}

// Same, but each step reloads both fragments from shared memory, which is what
// a real GEMM does between MMAs.
template <int ACC>
__global__ __launch_bounds__(256) void mma_shared(float *sink, int iters) {
  __shared__ __nv_bfloat16 buf[2][16 * 24];         // padded stride 24
  const int tid = threadIdx.x;
  for (int i = tid; i < 16 * 24; i += blockDim.x) {
    buf[0][i] = __float2bfloat16(1.0f);
    buf[1][i] = __float2bfloat16(1.0f);
  }
  __syncthreads();
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[ACC];
#pragma unroll
  for (int i = 0; i < ACC; ++i) wmma::fill_fragment(acc[i], 0.0f);

  for (int it = 0; it < iters; ++it) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;
    wmma::load_matrix_sync(fa, &buf[0][0], 24);
    wmma::load_matrix_sync(fb, &buf[1][0], 24);
#pragma unroll
    for (int i = 0; i < ACC; ++i) wmma::mma_sync(acc[i], fa, fb, acc[i]);
  }
  float s = 0;
#pragma unroll
  for (int i = 0; i < ACC; ++i) s += acc[i].x[0];
  if (s == 12345.678f) sink[blockIdx.x] = s;
}

int main() {
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  const int SM = p.multiProcessorCount;
  printf("GB10: %d SMs\n\n", SM);
  float *sink; CK(cudaMalloc(&sink, 1 << 20));

  const int ITERS = 20000;
  const int WARPS = 8;                    // 256 threads
  printf("%-26s %6s %10s %12s\n", "variant", "acc", "blocks/SM", "TFLOP/s");

  auto run = [&](const char *name, int acc, bool shared, int bpsm) {
    const int grid = SM * bpsm;
    std::vector<double> r;
    for (int rep = 0; rep < 5; ++rep) {
      cudaEvent_t a, b;
      cudaEventCreate(&a); cudaEventCreate(&b);
      cudaEventRecord(a);
#define GO(A) do { if (shared) mma_shared<A><<<grid, WARPS*32>>>(sink, ITERS); \
                   else        mma_peak<A><<<grid, WARPS*32>>>(sink, ITERS); } while (0)
      if (acc == 1) GO(1); else if (acc == 2) GO(2);
      else if (acc == 4) GO(4); else GO(8);
#undef GO
      cudaEventRecord(b); cudaEventSynchronize(b);
      float ms = 0; cudaEventElapsedTime(&ms, a, b);
      cudaEventDestroy(a); cudaEventDestroy(b);
      // one mma_sync = 16x16x16 MACs = 2*16*16*16 flops
      const double flops = 2.0 * 16 * 16 * 16 * (double)acc * ITERS *
                           (double)grid * WARPS;
      r.push_back(flops / (ms * 1e-3) / 1e12);
    }
    std::sort(r.begin(), r.end());
    printf("%-26s %6d %10d %12.1f\n", name, acc, bpsm, r.back());
    return r.back();
  };

  double peak = 0;
  for (int acc : {1, 2, 4, 8})
    peak = std::max(peak, run("registers only", acc, false, 2));
  for (int acc : {1, 2, 4, 8})
    run("reload from shared", acc, true, 2);
  for (int acc : {4, 8})
    peak = std::max(peak, run("registers only, 4 blk", acc, false, 4));

  printf("\nBF16 tensor-core peak: %.1f TFLOP/s\n", peak);
  printf("prefill GEMM measured at ~22 TFLOP/s = %.0f%% of it.\n", 100 * 22.0 / peak);
  printf("SGLang's ~110 TFLOP/s equivalent would be %.0f%% of it.\n", 100 * 110.0 / peak);
  cudaFree(sink);
  return 0;
}
