#pragma once
// src/loader.cpp — get the weight file into GPU-readable memory.
//
// The tempting approach on a unified-memory machine is mmap plus
// cudaHostRegister: the GPU can read host pages directly, so why copy? Because
// it is 29% slower. Measured on the real weight file, registered-mmap reads at
// 161.8 GB/s against cudaMalloc's 234.1. Decode re-reads every weight for every
// token, so that 29% would be a permanent tax on the only number that matters.
// Weights live in cudaMalloc memory and we pay a one-time copy at load.
//
// The copy goes through a bounded pinned staging buffer rather than a single
// giant pinned allocation: cudaHostAlloc of 15 GB costs seconds and gains
// nothing, whereas a 256 MB window keeps the DMA engine saturated at a fixed,
// predictable memory cost.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // O_DIRECT
#endif
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../qwen38.h"
#include "format.h"

namespace spark27 {

#define LCHECK(call)                                                           \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "loader: CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
              cudaGetErrorString(_e));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

struct Tensor {
  const uint8_t *dev = nullptr;  // device pointer to this tensor's payload
  int64_t bytes = 0;
  int32_t n = 0, k = 0, kind = 0;
  float scale = 1.0f;
};

struct Model {
  uint8_t *base = nullptr;  // device buffer holding every tensor, contiguous
  int64_t bytes = 0;
  std::vector<TensorEntry> table;
  std::unordered_map<std::string, int> index;
  double load_ms = 0, copy_gbs = 0;

  const Tensor get(const char *name) const {
    auto it = index.find(name);
    if (it == index.end()) {
      fprintf(stderr, "loader: no tensor '%s'\n", name);
      exit(1);
    }
    const TensorEntry &e = table[(size_t)it->second];
    Tensor t;
    t.dev = base + (e.offset - table[0].offset);
    t.bytes = e.bytes;
    t.n = e.n;
    t.k = e.k;
    t.kind = e.kind;
    t.scale = e.tensor_scale;
    return t;
  }
  bool has(const char *name) const { return index.count(name) != 0; }

  // Convenience for the 64-layer body: layer(3, "self_attn.q_proj.weight").
  const Tensor layer(int i, const char *role) const {
    char buf[kNameMax];
    snprintf(buf, sizeof buf, "model.language_model.layers.%d.%s", i, role);
    return get(buf);
  }
};

// MemAvailable from /proc/meminfo: memory the kernel could hand out, including
// reclaimable page cache. Very different from MemFree on a warm machine.
static size_t meminfo(const char *key) {
  FILE *f = fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256], pat[64];
  snprintf(pat, sizeof pat, "%s %%zu kB", key);
  size_t kb = 0;
  while (fgets(line, sizeof line, f))
    if (sscanf(line, pat, &kb) == 1) break;
  fclose(f);
  return kb * 1024;
}

static size_t mem_available() {
  FILE *f = fopen("/proc/meminfo", "r");
  if (!f) return 0;
  char line[256];
  size_t kb = 0;
  while (fgets(line, sizeof line, f))
    if (sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
  fclose(f);
  return kb * 1024;
}

// Make the kernel evict page cache.
//
// On unified memory the GPU allocates from the same pool the kernel caches
// files in. A machine that has been doing any real work sits with almost all
// memory in page cache: MemFree near zero, MemAvailable near total. CUDA looks
// at free, not available, and will not push the kernel to reclaim, so
// cudaMalloc fails on a box with 120 GB going spare.
//
// Faulting in a large anonymous mapping is the ordinary allocation path, which
// *does* trigger reclaim. We touch what we need, then hand it straight back.
static void force_reclaim(size_t need) {
  void *p = mmap(nullptr, need, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) return;
  // One write per page is enough to fault it in.
  const size_t pg = 4096;
  for (size_t off = 0; off < need; off += pg) ((volatile char *)p)[off] = 0;
  munmap(p, need);
}

static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

// Load every tensor into one contiguous device allocation.
inline Model load_model(const char *path) {
  Model m;
  double t0 = now_ms();

  // O_DIRECT: read straight from disk into our pinned buffer, bypassing the
  // kernel's page cache entirely.
  //
  // This matters more than it looks. The machine has unified memory, so the
  // GPU allocates from the same pool the kernel uses to cache files. Reading
  // 15 GB normally leaves 15 GB of cache behind; cudaMemGetInfo reports *free*
  // memory rather than *available*, and CUDA will not push the kernel to
  // reclaim — so the next run can fail to allocate with 120 GB nominally
  // available. Not caching a file we read exactly once and never re-read is
  // both faster and the only way to keep allocation predictable.
  int fd = open(path, O_RDONLY | O_DIRECT);
  bool direct = fd >= 0;
  if (!direct) fd = open(path, O_RDONLY);  // some filesystems refuse O_DIRECT
  if (fd < 0) { fprintf(stderr, "loader: cannot open %s\n", path); exit(1); }

  // Header and table are small and unaligned, so read them through a normal
  // descriptor; only the bulk payload goes through O_DIRECT.
  int mfd = open(path, O_RDONLY);
  if (mfd < 0) { fprintf(stderr, "loader: cannot reopen %s\n", path); exit(1); }
  FileHeader h;
  if (read(mfd, &h, sizeof h) != (ssize_t)sizeof h) {
    fprintf(stderr, "loader: short read on header\n"); exit(1);
  }
  if (h.magic != kMagic) {
    fprintf(stderr, "loader: bad magic 0x%08X — not a qwengine.bin\n", h.magic);
    exit(1);
  }
  if (h.version != kVersion) {
    fprintf(stderr, "loader: version %u, expected %u — reconvert\n", h.version,
            kVersion);
    exit(1);
  }

  m.table.resize((size_t)h.n_tensors);
  const size_t tb = (size_t)h.n_tensors * sizeof(TensorEntry);
  if (read(mfd, m.table.data(), tb) != (ssize_t)tb) {
    fprintf(stderr, "loader: short read on tensor table\n"); exit(1);
  }
  posix_fadvise(mfd, 0, 0, POSIX_FADV_DONTNEED);
  close(mfd);
  for (int i = 0; i < h.n_tensors; ++i) m.index[m.table[(size_t)i].name] = i;

  // Tensor data is contiguous from data_offset to total_bytes.
  m.bytes = h.total_bytes - h.data_offset;
  if (m.bytes <= 0) { fprintf(stderr, "loader: empty payload\n"); exit(1); }

  // Reclaim BEFORE touching CUDA at all. Creating a CUDA context itself needs
  // memory, so on a machine whose RAM is all page cache even cudaMemGetInfo
  // fails with "out of memory". By the time the CUDA API can tell us we are
  // short, it is too late to ask.
  // Headroom matters: the CUDA context, the pinned staging window and the
  // kernel's own slack all come out of the same pool, and the cache starts
  // refilling the moment we stop pushing. Ask for a few GB more than the
  // weights and retry, rather than landing exactly short.
  const size_t need = (size_t)m.bytes + (4ull << 30);
  if (meminfo("MemFree:") < need) {
    const size_t avail = mem_available();
    if (avail < need) {
      fprintf(stderr,
              "loader: need %.2f GB but only %.2f GB is available on this "
              "machine (MemFree %.2f GB).\n",
              need / 1e9, avail / 1e9, meminfo("MemFree:") / 1e9);
      exit(1);
    }
    printf("  page cache holds the memory we need (%.1f GB free of %.1f GB "
           "available) — reclaiming...\n",
           meminfo("MemFree:") / 1e9, avail / 1e9);
    for (int attempt = 0; attempt < 3 && meminfo("MemFree:") < need; ++attempt)
      force_reclaim(need);
    printf("  reclaimed: %.1f GB free\n", meminfo("MemFree:") / 1e9);
  }

  size_t freeb = 0, totalb = 0;
  LCHECK(cudaMemGetInfo(&freeb, &totalb));
  if ((size_t)m.bytes > freeb) {
    fprintf(stderr,
            "loader: need %.2f GB for weights but only %.2f GB free after "
            "reclaim (device total %.2f GB).\n",
            m.bytes / 1e9, freeb / 1e9, totalb / 1e9);
    exit(1);
  }
  LCHECK(cudaMalloc(&m.base, (size_t)m.bytes));

  // Bounded pinned window; read() straight into it so the page cache is not
  // copied twice.
  // O_DIRECT requires the buffer, the file offset and the length to be
  // block-aligned. 256 MB and our 4096-aligned data_offset both are; only the
  // final partial chunk needs care.
  const size_t kAlign = 4096;
  const size_t kStage = 256ull << 20;
  uint8_t *stage_raw = nullptr;
  LCHECK(cudaHostAlloc(&stage_raw, kStage + kAlign, cudaHostAllocDefault));
  uint8_t *stage = (uint8_t *)(((uintptr_t)stage_raw + kAlign - 1) & ~(uintptr_t)(kAlign - 1));

  if (lseek(fd, h.data_offset, SEEK_SET) != h.data_offset) {
    fprintf(stderr, "loader: seek to data\n"); exit(1);
  }
  double tc0 = now_ms();
  int64_t done = 0;
  while (done < m.bytes) {
    const size_t left = (size_t)(m.bytes - done);
    // Round the request up to a block boundary for O_DIRECT; the tail read
    // simply returns fewer bytes at EOF, and we only copy what we need.
    size_t want = left < kStage ? left : kStage;
    size_t req = direct ? ((want + kAlign - 1) & ~(kAlign - 1)) : want;
    if (req > kStage) req = kStage;
    size_t got = 0;
    while (got < req) {
      ssize_t r = read(fd, stage + got, req - got);
      if (r < 0) { fprintf(stderr, "loader: read error at %lld\n", (long long)(done + got)); exit(1); }
      if (r == 0) break;  // EOF: the tail was shorter than the aligned request
      got += (size_t)r;
    }
    const size_t use = got < want ? got : want;
    if (use == 0) { fprintf(stderr, "loader: short read at %lld\n", (long long)done); exit(1); }
    LCHECK(cudaMemcpy(m.base + done, stage, use, cudaMemcpyHostToDevice));
    done += (int64_t)use;
  }
  LCHECK(cudaDeviceSynchronize());
  double tc1 = now_ms();

  posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // belt and braces
  LCHECK(cudaFreeHost(stage_raw));
  close(fd);

  m.copy_gbs = m.bytes / ((tc1 - tc0) * 1e-3) / 1e9;
  m.load_ms = now_ms() - t0;
  return m;
}

inline void free_model(Model &m) {
  if (m.base) LCHECK(cudaFree(m.base));
  m.base = nullptr;
}

}  // namespace spark27

#ifndef SPARK27_NO_MAIN
// The P1 gate: re-run P0's streaming read over the *real* weight pages, not
// over synthetic memory, and check it against 0.85*B. Same access pattern as
// bench/01_bandwidth.cu — every byte read once, coalesced, unconditional store
// so nothing can be optimised away.
__global__ __launch_bounds__(256) void stream_weights(const uint4 *__restrict__ p,
                                                      size_t n4,
                                                      unsigned *__restrict__ sink) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    uint4 v = __ldg(&p[i]);
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
  for (int o = 16; o; o >>= 1) r ^= __shfl_xor_sync(0xFFFFFFFFu, r, o);
  if ((threadIdx.x & 31) == 0) sink[blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5)] = r;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "out/qwengine.bin";
  bool do_bench = false;
  double B = 0.0;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--bench")) do_bench = true;
    else if (!strcmp(argv[i], "--B") && i + 1 < argc) B = atof(argv[++i]);
  }

  spark27::Model m = spark27::load_model(path);
  int nv = 0;
  for (auto &e : m.table) nv += (e.kind == spark27::KIND_NVFP4);
  printf("loaded %s\n", path);
  printf("  tensors   : %zu (%d NVFP4, %zu BF16)\n", m.table.size(), nv,
         m.table.size() - (size_t)nv);
  printf("  device    : %.3f GB at %p\n", m.bytes / 1e9, (void *)m.base);
  printf("  load      : %.0f ms total, copy %.1f GB/s\n", m.load_ms, m.copy_gbs);

  if (do_bench) {
    cudaDeviceProp prop;
    LCHECK(cudaGetDeviceProperties(&prop, 0));
    int block = 256, grid = prop.multiProcessorCount * 32;
    unsigned *sink = nullptr;
    LCHECK(cudaMalloc(&sink, sizeof(unsigned) * grid * (block / 32)));
    size_t n4 = (size_t)m.bytes / sizeof(uint4);
    cudaEvent_t a, b;
    LCHECK(cudaEventCreate(&a)); LCHECK(cudaEventCreate(&b));
    for (int i = 0; i < 2; ++i) stream_weights<<<grid, block>>>((const uint4 *)m.base, n4, sink);
    LCHECK(cudaDeviceSynchronize());
    double best = 0;
    for (int i = 0; i < 7; ++i) {
      LCHECK(cudaEventRecord(a));
      stream_weights<<<grid, block>>>((const uint4 *)m.base, n4, sink);
      LCHECK(cudaEventRecord(b));
      LCHECK(cudaEventSynchronize(b));
      float ms = 0; LCHECK(cudaEventElapsedTime(&ms, a, b));
      double g = m.bytes / (ms * 1e-3) / 1e9;
      if (g > best) best = g;
    }
    printf("\n  streaming real weights: %.1f GB/s\n", best);
    if (B > 0)
      printf("  P1 gate (>= 0.85*B = %.1f): %s\n", 0.85 * B,
             best >= 0.85 * B ? "PASS" : "FAIL");
    printf("  implied raw passes/s at %.2f GB: %.2f\n", m.bytes / 1e9,
           best / (m.bytes / 1e9));
    LCHECK(cudaFree(sink));
  }
  spark27::free_model(m);
  return 0;
}
#endif
