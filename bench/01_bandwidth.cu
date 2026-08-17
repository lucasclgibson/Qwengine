// bench/01_bandwidth.cu — P0 "speed of light" calibration.
//
// Measures sustained streaming-READ bandwidth (call it B) over the allocation
// paths the engine could plausibly use for weights:
//
//   [0] cudaMalloc              — plain device allocation
//   [1] cudaMallocManaged       — unified managed, prefetched to device
//   [2] mmap + cudaHostRegister — the loader path, 4 KiB pages
//   [3] mmap + MADV_HUGEPAGE
//         + cudaHostRegister    — the loader path, 2 MiB pages
//
// Read-only streaming is *the* relevant pattern: decode is weight-bandwidth
// bound and every weight byte is read exactly once per forward pass. So this
// number, not the marketing 273 GB/s, is the denominator for every
// performance gate in the project.
//
// [2] vs [3] exists because the first honest run of this tool put the loader
// path at 71.5% of cudaMalloc — a red P0 gate. THP on this class of machine
// defaults to `madvise`, so a plain anonymous mmap is backed entirely by 4 KiB
// pages: a 12 GiB buffer is 3.1M translations instead of 6.1K. Both variants
// are measured in one process so the comparison shares thermal state.
//
// PORTABILITY: this is a calibration tool, not a one-off hand measurement.
// B is a property of the machine the engine is *running* on, not a constant
// baked in at development time. Every DGX Spark is a GB10/sm_121 with 128 GB
// of LPDDR5x, but units vary with binning, thermals, power cap, driver, and
// whatever else is resident. So: measure at startup, record B alongside every
// bench row, and express gates as fractions of the locally measured B.
// Nothing here may be hard-coded to the development machine.
//
// Build:  nvcc -O3 -arch=sm_121 -o build/01_bandwidth bench/01_bandwidth.cu
// Run:    ./build/01_bandwidth [--gb N] [--reps N] [--file PATH] [--force] [--json]

#include <cuda_runtime.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__, __LINE__,       \
              cudaGetErrorString(_e), #call);                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

enum { NPATH = 6 };

// ---------------------------------------------------------------------------
// Streaming read kernel.
//
// One uint4 (16 B) per thread per step, grid-stride so the access pattern is
// fully coalesced and every byte is touched exactly once. The XOR accumulator
// is cheap enough that we stay bandwidth-bound.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256) void stream_read(const uint4 *__restrict__ src,
                                                   size_t n4,
                                                   unsigned *__restrict__ sink) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
       i += stride) {
    uint4 v = __ldg(&src[i]);
    acc.x ^= v.x;
    acc.y ^= v.y;
    acc.z ^= v.z;
    acc.w ^= v.w;
  }
  unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;

  // Reduce the block to one store. The store MUST be unconditional: any guard
  // the compiler can prove false kills the accumulator and with it every load,
  // and the kernel then "runs" at petabytes/s. (It did, once.)
  // One 4-byte store per block is ~6 KB against a 12 GiB read — irrelevant.
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

struct Result {
  double best_gbs;
  double median_gbs;
  double setup_ms;  // allocation + registration + first-touch
  bool ok;
  char note[160];
};

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

// Launch config derived from the device, never hard-coded.
static void launch_geometry(int dev, int *grid, int *block) {
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, dev));
  *block = 256;
  *grid = p.multiProcessorCount * 32;
}

static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static size_t meminfo_field(const char *key) {
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

// ---------------------------------------------------------------------------
// Machine protection.
//
// This tool streams multi-GiB buffers at full bandwidth back to back, on a
// box that may be doing other work. Three ways it could misbehave, all guarded:
//   - eat memory the rest of the machine needs  -> MemAvailable check in main()
//   - cook the GPU with no pause between paths  -> thermal gate + cooldown here
//   - leak registrations/mappings on an error   -> every path unwinds its own
// Never disable these to make a number look better.
// ---------------------------------------------------------------------------
enum { TEMP_LIMIT_C = 84, TEMP_RESUME_C = 72, COOLDOWN_MAX_S = 120 };

static int gpu_temp_c() {
  FILE *f = popen(
      "nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null",
      "r");
  if (!f) return -1;
  int t = -1;
  if (fscanf(f, "%d", &t) != 1) t = -1;
  pclose(f);
  return t;
}

// Settle between paths so one path's heat doesn't bias the next one's clocks,
// and hard-stop if the GPU is genuinely hot.
static void thermal_gate(bool verbose) {
  int t = gpu_temp_c();
  if (t < 0) return;  // no nvidia-smi: nothing to enforce, don't pretend
  if (t < TEMP_LIMIT_C) {
    struct timespec ts = {0, 400 * 1000 * 1000};  // 400 ms settle
    nanosleep(&ts, nullptr);
    return;
  }
  if (verbose)
    printf("  [thermal] %d C >= %d C, waiting for %d C ...\n", t, TEMP_LIMIT_C,
           TEMP_RESUME_C);
  for (int s = 0; s < COOLDOWN_MAX_S; ++s) {
    struct timespec ts = {1, 0};
    nanosleep(&ts, nullptr);
    t = gpu_temp_c();
    if (t >= 0 && t <= TEMP_RESUME_C) return;
  }
  fprintf(stderr,
          "ABORT: GPU still at %d C after %d s cooldown. Refusing to continue "
          "— fix cooling or reduce --gb rather than pushing through.\n",
          t, COOLDOWN_MAX_S);
  exit(4);
}

static Result time_stream(const void *dptr, size_t bytes, int reps, int dev,
                          double setup_ms) {
  Result r = {};
  r.setup_ms = setup_ms;
  thermal_gate(true);  // settle/cool before every timed path
  int grid, block;
  launch_geometry(dev, &grid, &block);

  unsigned *sink = nullptr;
  CHECK(cudaMalloc(&sink, sizeof(unsigned) * grid));
  CHECK(cudaMemset(sink, 0, sizeof(unsigned) * grid));

  size_t n4 = bytes / sizeof(uint4);
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));

  // Warm up: fault/migrate pages and settle clocks before we trust a number.
  for (int i = 0; i < 2; ++i)
    stream_read<<<grid, block>>>((const uint4 *)dptr, n4, sink);
  CHECK(cudaDeviceSynchronize());

  double *gbs = (double *)malloc(sizeof(double) * reps);
  for (int i = 0; i < reps; ++i) {
    CHECK(cudaEventRecord(t0));
    stream_read<<<grid, block>>>((const uint4 *)dptr, n4, sink);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    gbs[i] = (double)bytes / (ms * 1e-3) / 1e9;  // decimal GB/s
  }

  qsort(gbs, reps, sizeof(double), cmp_double);
  r.best_gbs = gbs[reps - 1];
  r.median_gbs = gbs[reps / 2];
  r.ok = true;

  free(gbs);
  CHECK(cudaEventDestroy(t0));
  CHECK(cudaEventDestroy(t1));
  CHECK(cudaFree(sink));
  return r;
}

// mmap + cudaHostRegister, optionally asking for 2 MiB pages. `path` non-null
// mmaps that file (the true loader path); otherwise anonymous.
static Result run_mmap(size_t bytes, int reps, int dev, const char *path,
                       bool huge) {
  Result r = {};
  double t = now_ms();
  int fd = -1;
  void *p = MAP_FAILED;

  if (path) {
    fd = open(path, O_RDONLY);
    if (fd < 0) {
      snprintf(r.note, sizeof r.note, "open(%s) failed", path);
      return r;
    }
    struct stat st;
    if (fstat(fd, &st) || (size_t)st.st_size < bytes) {
      snprintf(r.note, sizeof r.note, "file smaller than %.2f GiB buffer",
               bytes / (double)(1ull << 30));
      close(fd);
      return r;
    }
    p = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  } else {
    p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
  if (p == MAP_FAILED) {
    snprintf(r.note, sizeof r.note, "mmap failed");
    if (fd >= 0) close(fd);
    return r;
  }

  // MADV_HUGEPAGE must precede first touch, or khugepaged has to collapse
  // 4 KiB pages afterwards and we measure the wrong thing.
  size_t ahp_before = meminfo_field("AnonHugePages:");
  if (huge && madvise(p, bytes, MADV_HUGEPAGE) != 0)
    snprintf(r.note, sizeof r.note, "madvise(MADV_HUGEPAGE) failed; ");

  // Fault the pages in before registering, so registration time is
  // registration and not page-fault time.
  if (path) {
    volatile unsigned char sum = 0;
    for (size_t i = 0; i < bytes; i += 4096) sum ^= ((unsigned char *)p)[i];
    (void)sum;
  } else {
    memset(p, 0x5A, bytes);
  }
  double ahp_gib =
      (double)(meminfo_field("AnonHugePages:") - ahp_before) / (1ull << 30);

  double treg = now_ms();
  cudaError_t e = cudaHostRegister(p, bytes, cudaHostRegisterReadOnly);
  if (e != cudaSuccess)  // not every driver takes ReadOnly
    e = cudaHostRegister(p, bytes, cudaHostRegisterDefault);
  double reg_ms = now_ms() - treg;

  if (e != cudaSuccess) {
    snprintf(r.note, sizeof r.note, "cudaHostRegister failed: %s",
             cudaGetErrorString(e));
    munmap(p, bytes);
    if (fd >= 0) close(fd);
    return r;
  }

  void *dptr = nullptr;
  CHECK(cudaHostGetDevicePointer(&dptr, p, 0));
  r = time_stream(dptr, bytes, reps, dev, now_ms() - t);
  snprintf(r.note, sizeof r.note, "reg %.0f ms (%.1f GB/s), AnonHugePages +%.2f GiB",
           reg_ms, bytes / (reg_ms * 1e-3) / 1e9, ahp_gib);

  CHECK(cudaHostUnregister(p));
  munmap(p, bytes);
  if (fd >= 0) close(fd);
  return r;
}

int main(int argc, char **argv) {
  double want_gb = 12.0;  // P0 asks for >= 12 GB buffers
  int reps = 5;
  const char *path = nullptr;
  bool force = false, json = false;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--gb") && i + 1 < argc) want_gb = atof(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--file") && i + 1 < argc) path = argv[++i];
    else if (!strcmp(argv[i], "--force")) force = true;
    else if (!strcmp(argv[i], "--json")) json = true;
    else {
      fprintf(stderr,
              "usage: %s [--gb N] [--reps N] [--file PATH] [--force] [--json]\n",
              argv[0]);
      return 2;
    }
  }

  int dev = 0;
  CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, dev));
  int rt = 0, drv = 0;
  CHECK(cudaRuntimeGetVersion(&rt));
  CHECK(cudaDriverGetVersion(&drv));

  size_t bytes = (size_t)(want_gb * (1ull << 30));
  bytes &= ~(size_t)(sizeof(uint4) - 1);

  // Guard: on unified memory a 12 GB buffer competes with everything else on
  // the box. Refuse rather than drive the machine into swap.
  size_t avail = meminfo_field("MemAvailable:");
  if (!force && avail && bytes > (size_t)(0.80 * avail)) {
    fprintf(stderr,
            "REFUSING: want %.1f GiB but MemAvailable is only %.1f GiB.\n"
            "  A buffer this size would thrash swap and the resulting B would\n"
            "  be meaningless. Free the machine, lower --gb, or pass --force.\n",
            bytes / (double)(1ull << 30), avail / (double)(1ull << 30));
    return 3;
  }

  if (!json) {
    printf("device        : %s (sm_%d%d), %d SMs\n", prop.name, prop.major,
           prop.minor, prop.multiProcessorCount);
    printf("cuda          : runtime %d.%d, driver %d.%d\n", rt / 1000,
           (rt % 1000) / 10, drv / 1000, (drv % 1000) / 10);
    printf("buffer        : %.2f GiB, %d reps%s\n", bytes / (double)(1ull << 30),
           reps, path ? " (file-backed)" : "");
    printf("MemAvailable  : %.1f GiB\n\n", avail / (double)(1ull << 30));
  }

  const char *names[NPATH] = {"cudaMalloc",         "managed (host-touch)",
                              "mmap+register (4K)", "mmap+register (2M)",
                              "cudaHostAlloc",      "managed (dev-touch)"};
  Result res[NPATH] = {};

  // ---- [0] cudaMalloc -----------------------------------------------------
  {
    double t = now_ms();
    void *p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) {
      snprintf(res[0].note, sizeof res[0].note, "cudaMalloc failed: %s",
               cudaGetErrorString(e));
    } else {
      CHECK(cudaMemset(p, 0x5A, bytes));
      CHECK(cudaDeviceSynchronize());
      res[0] = time_stream(p, bytes, reps, dev, now_ms() - t);
      CHECK(cudaFree(p));
    }
  }

  // ---- [1] cudaMallocManaged ---------------------------------------------
  {
    double t = now_ms();
    void *p = nullptr;
    cudaError_t e = cudaMallocManaged(&p, bytes);
    if (e != cudaSuccess) {
      snprintf(res[1].note, sizeof res[1].note, "cudaMallocManaged failed: %s",
               cudaGetErrorString(e));
    } else {
      memset(p, 0x5A, bytes);  // first-touch on host
      // Prefetch so we measure steady-state reads, not migration faults.
      // CUDA 13 respecified this to take a cudaMemLocation.
#if CUDART_VERSION >= 13000
      cudaMemLocation loc;
      loc.type = cudaMemLocationTypeDevice;
      loc.id = dev;
      cudaMemPrefetchAsync(p, bytes, loc, 0, 0);
#else
      cudaMemPrefetchAsync(p, bytes, dev, 0);
#endif
      CHECK(cudaDeviceSynchronize());
      res[1] = time_stream(p, bytes, reps, dev, now_ms() - t);
      CHECK(cudaFree(p));
    }
  }

  // ---- [2],[3] the loader path, 4 KiB vs 2 MiB pages ----------------------
  res[2] = run_mmap(bytes, reps, dev, path, false);
  res[3] = run_mmap(bytes, reps, dev, path, true);

  // ---- [4] cudaHostAlloc — pinned host memory -----------------------------
  // Isolates "is it host memory?" from "is it how the pages were obtained?".
  {
    double t = now_ms();
    void *p = nullptr;
    cudaError_t e = cudaHostAlloc(&p, bytes, cudaHostAllocMapped);
    if (e != cudaSuccess) {
      snprintf(res[4].note, sizeof res[4].note, "cudaHostAlloc failed: %s",
               cudaGetErrorString(e));
    } else {
      memset(p, 0x5A, bytes);
      void *dptr = nullptr;
      CHECK(cudaHostGetDevicePointer(&dptr, p, 0));
      res[4] = time_stream(dptr, bytes, reps, dev, now_ms() - t);
      CHECK(cudaFreeHost(p));
    }
  }

  // ---- [5] cudaMallocManaged, first touched by the DEVICE ------------------
  // [1] memsets on the host first, which pins the physical pages to a
  // host-preferred placement that a later prefetch may not undo. Here the GPU
  // touches first and we advise device residency, so if placement is what
  // separates cudaMalloc from the host paths, this should reach cudaMalloc.
  {
    double t = now_ms();
    void *p = nullptr;
    cudaError_t e = cudaMallocManaged(&p, bytes);
    if (e != cudaSuccess) {
      snprintf(res[5].note, sizeof res[5].note, "cudaMallocManaged failed: %s",
               cudaGetErrorString(e));
    } else {
#if CUDART_VERSION >= 13000
      cudaMemLocation loc;
      loc.type = cudaMemLocationTypeDevice;
      loc.id = dev;
      cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, loc);
      cudaMemPrefetchAsync(p, bytes, loc, 0, 0);
#else
      cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, dev);
      cudaMemPrefetchAsync(p, bytes, dev, 0);
#endif
      CHECK(cudaMemset(p, 0x5A, bytes));  // device-side first touch
      CHECK(cudaDeviceSynchronize());
      res[5] = time_stream(p, bytes, reps, dev, now_ms() - t);
      CHECK(cudaFree(p));
    }
  }

  // ---- report -------------------------------------------------------------
  double best = 0.0;
  int best_i = -1;
  for (int i = 0; i < NPATH; ++i)
    if (res[i].ok && res[i].best_gbs > best) { best = res[i].best_gbs; best_i = i; }

  // The P0 gate as written is specifically about the registered-mmap path, so
  // it compares only paths [2]/[3] against the best overall. Paths [4]/[5]
  // exist to diagnose *why*, not to quietly redefine the gate.
  double loader = 0.0;
  for (int i = 2; i <= 3; ++i)
    if (res[i].ok && res[i].best_gbs > loader) loader = res[i].best_gbs;

  if (json) {
    printf("{\"kind\":\"calib_bandwidth\",\"gpu\":\"%s\",\"sm\":\"%d%d\",",
           prop.name, prop.major, prop.minor);
    printf("\"cuda_rt\":%d,\"cuda_drv\":%d,\"buf_gib\":%.2f,\"reps\":%d,", rt,
           drv, bytes / (double)(1ull << 30), reps);
    for (int i = 0; i < NPATH; ++i)
      printf("\"%s\":%.2f,", names[i], res[i].ok ? res[i].best_gbs : 0.0);
    printf("\"loader_gbs\":%.2f,\"loader_frac\":%.4f,\"B_gbs\":%.2f}\n", loader,
           best > 0 ? loader / best : 0.0, best);
  } else {
    printf("%-22s %10s %10s %9s   %s\n", "path", "best GB/s", "median",
           "setup ms", "note");
    for (int i = 0; i < NPATH; ++i) {
      if (res[i].ok)
        printf("%-22s %10.1f %10.1f %9.0f   %s%s\n", names[i], res[i].best_gbs,
               res[i].median_gbs, res[i].setup_ms, res[i].note,
               i == best_i ? "  <- best" : "");
      else
        printf("%-22s %10s %10s %9s   %s\n", names[i], "-", "-", "-",
               res[i].note[0] ? res[i].note : "skipped");
    }
    printf("\nB = %.1f GB/s  (%s)\n", best,
           best_i >= 0 ? names[best_i] : "none");
    if (loader > 0 && best > 0) {
      double ratio = loader / best;
      printf("loader path = %.1f%% of best — P0 gate needs >= 90%%: %s\n",
             ratio * 100.0, ratio >= 0.90 ? "PASS" : "FAIL");
    }
    // Model-level consequence, recomputed from the measured B rather than
    // quoted from a spec sheet.
    if (loader > 0) {
      const double w_gb = 15.6;  // NVFP4 language-model weights
      printf("\nimplied raw passes/s at %.1f GB weights (no KV): %.1f"
             "  [loader-path bound]\n", w_gb, loader / w_gb);
    }
  }
  return 0;
}
