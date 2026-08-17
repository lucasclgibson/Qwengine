// bench/02_readmax.cu — what is the REAL streaming-read ceiling on GB10?
//
// Every performance conclusion in this engine is a ratio against one number:
// the peak rate at which weights can be pulled out of memory. bench/01 measures
// it with a grid-stride uint4 loop and gets ~235.6 GB/s, against a 273 GB/s
// spec figure. If a different access pattern gets meaningfully closer to 273,
// then the decode kernel is not at "98% of peak" and there is headroom nobody
// has been looking for. If nothing beats it, the 235.6 denominator is real and
// the engine's floor is where we think it is.
//
// Variants, all reading the same buffer, all fully coalesced:
//   u4      one uint4 per thread per step (what bench/01 does)
//   u4xN    N independent uint4 loads per step, to raise memory-level
//           parallelism -- more outstanding loads per thread hides latency the
//           grid-stride loop otherwise pays for
//   evict   the same, hinting L2 evict-first, since streamed weights are never
//           re-read and should not displace anything
//   b64     32-BYTE loads (ld.global.v4.b64). sm_121 supports 256-bit vector
//           loads, which is wider than the uint4 this engine uses everywhere.
//           Half the load instructions for the same bytes.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 1;} } while(0)

// The store must be unconditional or the compiler proves the accumulator dead
// and deletes every load with it.
__device__ __forceinline__ void sink_block(unsigned r, unsigned *sink) {
  __shared__ unsigned red[32];
  unsigned lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o; o >>= 1) r ^= __shfl_xor_sync(0xFFFFFFFFu, r, o);
  if (lane == 0) red[warp] = r;
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned t = 0;
    for (unsigned i = 0; i < blockDim.x / 32; ++i) t ^= red[i];
    sink[blockIdx.x] = t;
  }
}

template <int UNROLL, bool EVICT>
__global__ void read_kernel(const uint4 *__restrict__ src, size_t n4,
                            unsigned *__restrict__ sink) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  // Each iteration issues UNROLL independent loads before consuming any of
  // them, so UNROLL requests per thread are in flight at once.
  for (size_t i = tid; i < n4; i += stride * UNROLL) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const size_t idx = i + (size_t)u * stride;
      if (idx < n4) {
        v[u] = __ldg(src + idx);
      } else {
        v[u] = make_uint4(0, 0, 0, 0);
      }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w;
    }
  }
  sink_block(acc.x ^ acc.y ^ acc.z ^ acc.w, sink);
}

// 32-byte loads. sm_121 accepts ld.global.v4.b64 (256 bits); the L2 eviction
// hint is in fact ONLY legal at this width, which is how the width was found.
template <int UNROLL, bool EVICT>
__global__ void read_b64(const uint4 *__restrict__ src, size_t n4,
                         unsigned *__restrict__ sink) {
  unsigned long long a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t n8 = n4 / 2;                       // count of 32-byte units
  for (size_t i = tid; i < n8; i += stride * UNROLL) {
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      const size_t idx = i + (size_t)u * stride;
      if (idx < n8) {
        unsigned long long x, y, z, w;
        const void *p = (const void *)(src + idx * 2);
        if (EVICT)
          asm volatile("ld.global.nc.L2::evict_first.v4.b64 {%0,%1,%2,%3}, [%4];"
                       : "=l"(x), "=l"(y), "=l"(z), "=l"(w) : "l"(p) : "memory");
        else
          asm volatile("ld.global.nc.v4.b64 {%0,%1,%2,%3}, [%4];"
                       : "=l"(x), "=l"(y), "=l"(z), "=l"(w) : "l"(p) : "memory");
        a0 ^= x; a1 ^= y; a2 ^= z; a3 ^= w;
      }
    }
  }
  const unsigned long long a = a0 ^ a1 ^ a2 ^ a3;
  sink_block((unsigned)(a ^ (a >> 32)), sink);
}

struct Cfg { const char *name; int unroll; bool evict; int block; int bpsm; };

int main(int argc, char **argv) {
  const double GB = argc > 1 ? atof(argv[1]) : 8.0;
  const size_t bytes = (size_t)(GB * 1e9) & ~(size_t)0xFFFFF;
  const size_t n4 = bytes / 16;

  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  printf("GB10: %d SMs, buffer %.2f GB\n\n", p.multiProcessorCount, bytes / 1e9);

  uint4 *d = nullptr; unsigned *sink = nullptr;
  CK(cudaMalloc(&d, bytes));
  CK(cudaMemset(d, 0x5A, bytes));
  CK(cudaMalloc(&sink, 1u << 20));

  const Cfg cfgs[] = {
      {"u4       ", 1, false, 256, 16}, {"u4       ", 1, false, 512, 8},
      {"u4x2     ", 2, false, 256, 16}, {"u4x4     ", 4, false, 256, 16},
      {"u4x8     ", 8, false, 256, 16}, {"u4x4     ", 4, false, 512, 8},
      {"u4x4     ", 4, false, 128, 32}, {"u4x8     ", 8, false, 512, 8},
      {"u4x16    ",16, false, 256, 16},
      {"b64x2    ", 2, false, 256, 16}, {"b64x4    ", 4, false, 256, 16},
      {"b64x8    ", 8, false, 256, 16}, {"b64x4 ev ", 4, true,  256, 16},
      {"b64x8 ev ", 8, true,  256, 16}, {"b64x4    ", 4, false, 512,  8},
  };

  printf("%-10s %6s %8s   %10s %10s   %s\n", "variant", "block", "blocks",
         "best GB/s", "median", "% of 273 spec");
  double overall = 0; const char *overall_name = "";
  for (const Cfg &c : cfgs) {
    const int grid = p.multiProcessorCount * c.bpsm;
    std::vector<double> runs;
    for (int rep = 0; rep < 7; ++rep) {
      cudaEvent_t a, b;
      CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
      // warm
      if (c.unroll == 1 && !c.evict) read_kernel<1, false><<<grid, c.block>>>(d, n4, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(a));
#define LAUNCH(U) read_kernel<U, false><<<grid, c.block>>>(d, n4, sink)
#define LAUNCH64(U, E) read_b64<U, E><<<grid, c.block>>>(d, n4, sink)
      if (c.name[2] == '6') {                       // "b64..." variants
        if (c.evict) { switch (c.unroll) { case 2: LAUNCH64(2,true); break;
                        case 4: LAUNCH64(4,true); break; default: LAUNCH64(8,true); } }
        else        { switch (c.unroll) { case 2: LAUNCH64(2,false); break;
                        case 4: LAUNCH64(4,false); break; default: LAUNCH64(8,false); } }
      } else {
        switch (c.unroll) { case 1: LAUNCH(1); break; case 2: LAUNCH(2); break;
          case 4: LAUNCH(4); break; case 8: LAUNCH(8); break; default: LAUNCH(16); }
      }
#undef LAUNCH
#undef LAUNCH64
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));
      CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
      runs.push_back(bytes / (ms * 1e-3) / 1e9);
    }
    std::sort(runs.begin(), runs.end());
    const double best = runs.back(), med = runs[runs.size() / 2];
    printf("%-10s %6d %8d   %10.1f %10.1f   %5.0f%%\n", c.name, c.block, grid,
           best, med, 100.0 * best / 273.0);
    if (best > overall) { overall = best; overall_name = c.name; }
  }
  printf("\nbest read bandwidth: %.1f GB/s (%s)\n", overall, overall_name);
  printf("bench/01 reports 235.6 GB/s; a 27B model streams 14.41 GB per token,\n");
  printf("so the decode floor is %.1f ms/token at this rate (%.1f at 235.6).\n",
         14.41e9 / (overall * 1e9) * 1e3, 14.41e9 / (235.6e9) * 1e3);
  cudaFree(d); cudaFree(sink);
  return 0;
}
