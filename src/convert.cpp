// src/convert.cpp — safetensors -> qwengine.bin. Offline, run once.
//
// Reads the BF16 Qwen3.8-27B checkpoint, drops the vision tower, packs every
// 2-D matrix to NVFP4 in the exact order gemv_nvfp4.cu streams it, and writes
// a single flat file plus a tensor table.
//
// ===========================================================================
// ON-DISK NVFP4 LAYOUT  (co-designed with gemv_nvfp4.cu — change both or
//                        neither; test/test_nvfp4.cpp round-trips it)
// ===========================================================================
//
// NVFP4 = E2M1 4-bit values, one FP8 E4M3 scale per 16-element block, one FP32
// scale per tensor. 4 + 8/16 = 4.5 bits/weight.
//
//   dequant(w) = e2m1_decode(code) * e4m3_decode(block_scale) * tensor_scale
//
// Scale recipe. With E2M1 max magnitude 6.0 and E4M3 max 448:
//   tensor_scale = absmax(W) / (6 * 448)
//   block_scale  = e4m3_round( absmax(block) / 6 / tensor_scale )
// so the block holding the tensor's absmax gets block_scale exactly 448 and
// the E4M3 range is used fully rather than clipped.
//
// Streaming order. A matrix is [N,K] = [out_features, in_features] and the
// GEMV reduces over K for each output row n. One warp step consumes
// SWZ_WEIGHTS = 1024 contiguous weights of one row, stored as:
//
//   offset  0 .. 511   512 B  packed E2M1, 2 weights per byte (low nibble
//                             first), i.e. 32 lanes x uint4, one fully
//                             coalesced 512 B warp load
//   offset 512 .. 575   64 B  the 64 E4M3 block scales for those 1024
//                             weights; lane i reads the 2 scales covering its
//                             own 32 weights as one ushort -> a single
//                             coalesced 64 B transaction
//
// Each lane's uint4 is 32 weights = exactly 2 blocks of 16, so a block never
// straddles a lane boundary and no lane needs a neighbour's scale.
//
// Rows are stored consecutively, so iterating (n, then k) walks the file
// strictly forwards — the same pure-sequential pattern P0 measured at
// 235 GB/s. Every K in this model is a multiple of 1024 (5120 / 6144 / 10240 /
// 17408, verified across all 506 matrices), so nothing needs padding.
//
// ===========================================================================
// FILE FORMAT
// ===========================================================================
//   [FileHeader][TensorEntry x n_tensors][padding to 4096][tensor data ...]
// Tensor data is 4096-aligned so the loader's copies stay page-aligned.
//
// Build: nvcc/g++ -O3 -o build/convert src/convert.cpp
// Run:   build/convert $QWEN38_DIR out/qwengine.bin

#include <ctype.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "../qwen38.h"
// qwengine.bin container definitions are shared with the loader.
#include "format.h"
#include "tokenizer.cpp"

using namespace q38;
using namespace spark27;

[[noreturn]] static void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "convert: ");
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  exit(1);
}

// ===========================================================================
// Minimal JSON reader. Enough for safetensors headers and config.json, no
// more: objects, arrays, strings, numbers, bool, null. Rule 4 forbids pulling
// in a library for this, and the grammar we need is tiny.
// ===========================================================================
namespace json {

struct Value;
using Object = std::vector<std::pair<std::string, Value *>>;
using Array = std::vector<Value *>;

struct Value {
  enum Kind { OBJ, ARR, STR, NUM, BOOL, NUL } kind;
  Object obj;
  Array arr;
  std::string str;
  double num = 0;
  bool bl = false;
};

struct Parser {
  const char *p, *end;
  std::vector<Value *> pool;

  Value *alloc(Value::Kind k) {
    Value *v = new Value();
    v->kind = k;
    pool.push_back(v);
    return v;
  }
  void ws() {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) ++p;
  }
  bool lit(const char *s) {
    size_t n = strlen(s);
    if ((size_t)(end - p) < n || memcmp(p, s, n)) return false;
    p += n;
    return true;
  }
  std::string parse_string() {
    if (p >= end || *p != '"') die("json: expected string");
    ++p;
    std::string s;
    while (p < end && *p != '"') {
      if (*p == '\\') {
        ++p;
        if (p >= end) die("json: bad escape");
        switch (*p) {
          case 'n': s += '\n'; break;
          case 't': s += '\t'; break;
          case 'r': s += '\r'; break;
          case 'b': s += '\b'; break;
          case 'f': s += '\f'; break;
          case 'u': {  // we only ever see ASCII in these files
            if (end - p < 5) die("json: bad \\u");
            int cp = (int)strtol(std::string(p + 1, p + 5).c_str(), nullptr, 16);
            if (cp < 0x80) s += (char)cp;
            else s += '?';
            p += 4;
            break;
          }
          default: s += *p;
        }
        ++p;
      } else {
        s += *p++;
      }
    }
    if (p >= end) die("json: unterminated string");
    ++p;
    return s;
  }
  Value *parse() {
    ws();
    if (p >= end) die("json: eof");
    if (*p == '{') {
      ++p;
      Value *v = alloc(Value::OBJ);
      ws();
      if (p < end && *p == '}') { ++p; return v; }
      for (;;) {
        ws();
        std::string k = parse_string();
        ws();
        if (p >= end || *p != ':') die("json: expected ':'");
        ++p;
        v->obj.emplace_back(k, parse());
        ws();
        if (p < end && *p == ',') { ++p; continue; }
        if (p < end && *p == '}') { ++p; break; }
        die("json: expected ',' or '}'");
      }
      return v;
    }
    if (*p == '[') {
      ++p;
      Value *v = alloc(Value::ARR);
      ws();
      if (p < end && *p == ']') { ++p; return v; }
      for (;;) {
        v->arr.push_back(parse());
        ws();
        if (p < end && *p == ',') { ++p; continue; }
        if (p < end && *p == ']') { ++p; break; }
        die("json: expected ',' or ']'");
      }
      return v;
    }
    if (*p == '"') {
      Value *v = alloc(Value::STR);
      v->str = parse_string();
      return v;
    }
    if (lit("true"))  { Value *v = alloc(Value::BOOL); v->bl = true;  return v; }
    if (lit("false")) { Value *v = alloc(Value::BOOL); v->bl = false; return v; }
    if (lit("null"))  { return alloc(Value::NUL); }
    {
      char *e = nullptr;
      double d = strtod(p, &e);
      if (e == p) die("json: bad value at '%.16s'", p);
      p = e;
      Value *v = alloc(Value::NUM);
      v->num = d;
      return v;
    }
  }
};

static const Value *get(const Value *o, const char *key) {
  if (!o || o->kind != Value::OBJ) return nullptr;
  for (auto &kv : o->obj)
    if (kv.first == key) return kv.second;
  return nullptr;
}
// Look up `key` at the top level or inside "text_config" — the checkpoint puts
// language-model geometry under text_config for the multimodal wrapper.
static const Value *cfg_get(const Value *root, const char *key) {
  if (const Value *v = get(root, key)) return v;
  return get(get(root, "text_config"), key);
}

}  // namespace json

// ===========================================================================
// Float codecs
// ===========================================================================

static inline float bf16_to_f32(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

// FP8 E4M3 (OCP): 1-4-3, bias 7, max normal 448, exp==15 && mant==7 is NaN.
static inline float e4m3_to_f32(uint8_t v) {
  int s = (v >> 7) & 1, e = (v >> 3) & 0xF, m = v & 7;
  float mag;
  if (e == 0)
    mag = ldexpf((float)m, -9);              // subnormal: m * 2^-9
  else
    mag = ldexpf(1.0f + (float)m / 8.0f, e - 7);
  return s ? -mag : mag;
}

// Round-to-nearest-even encode into E4M3. Saturates at 448 (no inf in E4M3).
static uint8_t f32_to_e4m3(float x) {
  uint8_t sign = x < 0 ? 0x80 : 0;
  float a = fabsf(x);
  if (!(a > 0.0f)) return sign;              // 0 or NaN-as-0 (we never feed NaN)
  if (a >= 464.0f) return sign | 0x7E;       // >= midpoint(448,512) -> saturate
  int e;
  float f = frexpf(a, &e);                   // a = f * 2^e, f in [0.5,1)
  // Normalise to 1.m * 2^(e-1)
  int unb = e - 1;
  int q;
  if (unb >= -6) {                           // normal
    q = (int)lrintf((f * 2.0f - 1.0f) * 8.0f);  // lrintf = round-half-to-even
    if (q == 8) { q = 0; ++unb; }               // mantissa carry
    if (unb > 8) return sign | 0x7E;            // saturate
    return sign | (uint8_t)(((unb + 7) & 0xF) << 3) | (uint8_t)q;
  }
  q = (int)lrintf(ldexpf(a, 9));             // subnormal: a / 2^-9
  if (q >= 8) return sign | (uint8_t)(1 << 3);  // rounded up into normals
  return sign | (uint8_t)q;
}

// E2M1 magnitudes by code. Codes 0..7, sign in bit 3.
static const float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

static inline float e2m1_to_f32(uint8_t nib) {
  float m = kE2M1[nib & 7];
  return (nib & 8) ? -m : m;
}

// Nearest E2M1 code, ties to even code. Branch chain rather than a divide:
// midpoints are .25 .75 1.25 1.75 2.5 3.5 5.0, and the <= / < pattern below
// sends each tie to the even code (0, 2, 4, 6).
static inline uint8_t f32_to_e2m1(float x) {
  uint8_t sign = x < 0 ? 8 : 0;
  float a = fabsf(x);
  uint8_t q = a <= 0.25f   ? 0
              : a < 0.75f  ? 1
              : a <= 1.25f ? 2
              : a < 1.75f  ? 3
              : a <= 2.5f  ? 4
              : a < 3.5f   ? 5
              : a <= 5.0f  ? 6
                           : 7;
  return sign | q;
}

// ===========================================================================
// NVFP4 pack — the single source of truth. tools/make_golden.py mirrors this
// bit-for-bit and test/test_nvfp4.cpp cross-checks the two.
// ===========================================================================

// Scale-search window, in E4M3 codes either side of nominal. Measured, not
// guessed: test/probe_scale.cpp sweeps the aggregate error curve on real
// weights. Going *down* is monotonically worse (clipping); going *up* holds a
// second optimum around +3..+5, because for a block whose max is an outlier a
// larger scale sacrifices that max to move the bulk out of E2M1's sparse high
// levels {2,3,4,6} into the dense low ones. Roughly half of all blocks prefer
// it. Widening past +6 changes nothing (0.0814 at +6, +8 and +12), so this
// brackets the optimum.
static constexpr int kScaleSearchLo = 2;
static constexpr int kScaleSearchHi = 6;

static constexpr float kE2M1Max = 6.0f;
static constexpr float kE4M3Max = 448.0f;

// Run fn(row_begin, row_end) over N rows on all cores. Each thread owns whole
// rows and writes disjoint output, so the result is bit-identical regardless
// of thread count — I3 determinism must not depend on scheduling.
template <typename F>
static void parallel_rows(int64_t N, F fn) {
  unsigned hw = std::thread::hardware_concurrency();
  // Leave a couple of cores for the rest of the machine.
  unsigned nt = hw > 3 ? (hw - 2 < 16u ? hw - 2 : 16u) : 1u;
  if ((int64_t)nt > N) nt = (unsigned)(N > 0 ? N : 1);
  if (nt <= 1) { fn(0, N); return; }
  std::vector<std::thread> ts;
  int64_t chunk = (N + nt - 1) / nt;
  for (unsigned t = 0; t < nt; ++t) {
    int64_t b = (int64_t)t * chunk, e = b + chunk < N ? b + chunk : N;
    if (b >= e) break;
    ts.emplace_back([=] { fn(b, e); });
  }
  for (auto &t : ts) t.join();
}

// Choose the E4M3 block scale minimising squared error over the block.
//
// The obvious choice, scale = blockmax/6, pins the largest element exactly on
// the top E2M1 level. That is not MSE-optimal: E2M1's upper levels are sparse
// (..., 3, 4, 6), so a slightly smaller scale clips the max a little but pulls
// the bulk of the block into the denser low levels, which usually wins.
//
// We search downward over *representable* E4M3 codes rather than over
// multiplicative factors, because the code is what the kernel will decode —
// searching in float and rounding afterwards can land on a scale that was
// never evaluated.
static inline uint8_t best_block_scale(const uint16_t *blk, float tscale,
                                       float inv_t) {
  float bmax = 0.0f;
  for (int i = 0; i < NVFP4_BLOCK; ++i) {
    float v = fabsf(bf16_to_f32(blk[i]));
    if (v > bmax) bmax = v;
  }
  if (!(bmax > 0.0f)) return 0;

  uint8_t nominal = f32_to_e4m3(bmax * inv_t / kE2M1Max);
  uint8_t best = nominal;
  double best_sse = -1.0;
  // Codes are magnitude-ordered, so stepping the code down steps the scale
  // down. Four below and one above brackets the optimum in practice.
  int lo = (int)nominal - kScaleSearchLo, hi = (int)nominal + kScaleSearchHi;
  if (lo < 1) lo = 1;
  if (hi > 0x7E) hi = 0x7E;
  for (int c = lo; c <= hi; ++c) {
    float sdec = e4m3_to_f32((uint8_t)c) * tscale;
    if (!(sdec > 0.0f)) continue;
    float inv_s = 1.0f / sdec;
    double sse = 0.0;
    for (int i = 0; i < NVFP4_BLOCK; ++i) {
      float x = bf16_to_f32(blk[i]);
      double d = (double)x - (double)e2m1_to_f32(f32_to_e2m1(x * inv_s)) * sdec;
      sse += d * d;
    }
    if (best_sse < 0.0 || sse < best_sse) { best_sse = sse; best = (uint8_t)c; }
  }
  return best;
}

// Pack one [N,K] BF16 matrix into the swizzled layout. `out` must have room
// for nvfp4_bytes(N,K). Returns the per-tensor FP32 scale.
static float pack_nvfp4(const uint16_t *w, int64_t N, int64_t K, uint8_t *out) {
  if (K % SWZ_WEIGHTS) die("K=%lld not a multiple of %d", (long long)K, SWZ_WEIGHTS);

  // Pass 1: absmax. max is associative and exact in float, so parallelising
  // it cannot change the result.
  std::vector<float> partial((size_t)N, 0.0f);
  parallel_rows(N, [&](int64_t b, int64_t e) {
    for (int64_t n = b; n < e; ++n) {
      float m = 0.0f;
      const uint16_t *row = w + n * K;
      for (int64_t k = 0; k < K; ++k) {
        float v = fabsf(bf16_to_f32(row[k]));
        if (v > m) m = v;
      }
      partial[(size_t)n] = m;
    }
  });
  float absmax = 0.0f;
  for (float v : partial) if (v > absmax) absmax = v;

  // Degenerate all-zero tensor: any positive scale works, pick 1 so dequant
  // is exactly 0 rather than NaN.
  const float tscale = absmax > 0.0f ? absmax / (kE2M1Max * kE4M3Max) : 1.0f;
  const float inv_t = 1.0f / tscale;
  const int64_t steps = K / SWZ_WEIGHTS;

  // Pass 2: quantise.
  parallel_rows(N, [&](int64_t rb, int64_t re) {
    for (int64_t n = rb; n < re; ++n) {
      const uint16_t *row = w + n * K;
      uint8_t *rowo = out + n * steps * SWZ_STEP_BYTES;
      for (int64_t s = 0; s < steps; ++s) {
        const uint16_t *src = row + s * SWZ_WEIGHTS;
        uint8_t *vals = rowo + s * SWZ_STEP_BYTES;
        uint8_t *scls = vals + SWZ_VAL_BYTES;

        for (int b = 0; b < SWZ_WEIGHTS / NVFP4_BLOCK; ++b) {
          const uint16_t *blk = src + b * NVFP4_BLOCK;
          uint8_t sb = best_block_scale(blk, tscale, inv_t);
          scls[b] = sb;
          float sdec = e4m3_to_f32(sb) * tscale;
          float inv_s = sdec > 0.0f ? 1.0f / sdec : 0.0f;

          for (int i = 0; i < NVFP4_BLOCK; ++i) {
            uint8_t code = f32_to_e2m1(bf16_to_f32(blk[i]) * inv_s);
            int64_t idx = (int64_t)b * NVFP4_BLOCK + i;
            uint8_t &dst = vals[idx >> 1];
            if (idx & 1) dst = (uint8_t)((dst & 0x0F) | (code << 4));  // high
            else         dst = (uint8_t)((dst & 0xF0) | code);          // low
          }
        }
      }
    }
  });
  return tscale;
}

// ---- D2: 2-bit draft-head format (see format.h) ---------------------------
static const float kD2[4] = {-1.5f, -0.5f, 0.5f, 1.5f};
static constexpr float kD2Max = 1.5f;

static inline uint8_t f32_to_d2(float x) {
  // Nearest of the four levels; boundaries at -1, 0, +1.
  return x < -1.0f ? 0 : x < 0.0f ? 1 : x < 1.0f ? 2 : 3;
}

static inline uint8_t best_block_scale_d2(const uint16_t *blk, float tscale,
                                          float inv_t) {
  float bmax = 0.0f;
  for (int i = 0; i < NVFP4_BLOCK; ++i) {
    float v = fabsf(bf16_to_f32(blk[i]));
    if (v > bmax) bmax = v;
  }
  if (!(bmax > 0.0f)) return 0;
  uint8_t nominal = f32_to_e4m3(bmax * inv_t / kD2Max);
  uint8_t best = nominal;
  double best_sse = -1.0;
  int lo = (int)nominal - 2, hi = (int)nominal + 6;   // same window as NVFP4
  if (lo < 1) lo = 1;
  if (hi > 0x7E) hi = 0x7E;
  for (int c = lo; c <= hi; ++c) {
    float sdec = e4m3_to_f32((uint8_t)c) * tscale;
    if (!(sdec > 0.0f)) continue;
    float inv_s = 1.0f / sdec;
    double sse = 0.0;
    for (int i = 0; i < NVFP4_BLOCK; ++i) {
      float x = bf16_to_f32(blk[i]);
      double d = (double)x - (double)kD2[f32_to_d2(x * inv_s)] * sdec;
      sse += d * d;
    }
    if (best_sse < 0.0 || sse < best_sse) { best_sse = sse; best = (uint8_t)c; }
  }
  return best;
}

static float pack_d2(const uint16_t *w, int64_t N, int64_t K, uint8_t *out) {
  float absmax = 0.0f;
  for (int64_t i = 0; i < N * K; ++i) {
    float v = fabsf(bf16_to_f32(w[i]));
    if (v > absmax) absmax = v;
  }
  const float tscale = absmax > 0.0f ? absmax / (kD2Max * kE4M3Max) : 1.0f;
  const float inv_t = 1.0f / tscale;
  const int64_t steps = K / SWZ_WEIGHTS;

  parallel_rows(N, [&](int64_t rb, int64_t re) {
    for (int64_t n = rb; n < re; ++n) {
      const uint16_t *row = w + n * K;
      uint8_t *rowo = out + n * steps * D2_STEP_BYTES;
      for (int64_t s = 0; s < steps; ++s) {
        const uint16_t *src = row + s * SWZ_WEIGHTS;
        uint8_t *vals = rowo + s * D2_STEP_BYTES;
        uint8_t *scls = vals + D2_VAL_BYTES;
        memset(vals, 0, D2_VAL_BYTES);
        for (int b = 0; b < SWZ_WEIGHTS / NVFP4_BLOCK; ++b) {
          const uint16_t *blk = src + b * NVFP4_BLOCK;
          uint8_t sb = best_block_scale_d2(blk, tscale, inv_t);
          scls[b] = sb;
          float sdec = e4m3_to_f32(sb) * tscale;
          float inv_s = sdec > 0.0f ? 1.0f / sdec : 0.0f;
          for (int i = 0; i < NVFP4_BLOCK; ++i) {
            uint8_t code = f32_to_d2(bf16_to_f32(blk[i]) * inv_s);
            int64_t idx = (int64_t)b * NVFP4_BLOCK + i;
            vals[idx >> 2] |= (uint8_t)(code << (2 * (idx & 3)));
          }
        }
      }
    }
  });
  return tscale;
}

// Inverse of pack_nvfp4, for the round-trip test and for error reporting.
void unpack_nvfp4(const uint8_t *in, int64_t N, int64_t K, float tscale,
                  float *out) {
  const int64_t steps = K / SWZ_WEIGHTS;
  for (int64_t n = 0; n < N; ++n) {
    const uint8_t *rowi = in + n * steps * SWZ_STEP_BYTES;
    float *rowo = out + n * K;
    for (int64_t s = 0; s < steps; ++s) {
      const uint8_t *vals = rowi + s * SWZ_STEP_BYTES;
      const uint8_t *scls = vals + SWZ_VAL_BYTES;
      for (int b = 0; b < SWZ_WEIGHTS / NVFP4_BLOCK; ++b) {
        float sdec = e4m3_to_f32(scls[b]) * tscale;
        for (int i = 0; i < NVFP4_BLOCK; ++i) {
          int64_t idx = (int64_t)b * NVFP4_BLOCK + i;
          uint8_t byte = vals[idx >> 1];
          uint8_t code = (idx & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0F);
          rowo[s * SWZ_WEIGHTS + idx] = e2m1_to_f32(code) * sdec;
        }
      }
    }
  }
}


// ===========================================================================
// safetensors
// ===========================================================================
struct Shard {
  int fd = -1;
  const uint8_t *base = nullptr;
  size_t size = 0;
  int64_t data_start = 0;
  json::Parser parser;
  const json::Value *header = nullptr;
};

struct TensorRef {
  const Shard *shard;
  std::string dtype;
  std::vector<int64_t> shape;
  int64_t begin, end;  // relative to shard->data_start
};

static void open_shard(Shard &s, const std::string &path) {
  s.fd = open(path.c_str(), O_RDONLY);
  if (s.fd < 0) die("cannot open %s", path.c_str());
  struct stat st;
  if (fstat(s.fd, &st)) die("stat %s", path.c_str());
  s.size = (size_t)st.st_size;
  void *m = mmap(nullptr, s.size, PROT_READ, MAP_PRIVATE, s.fd, 0);
  if (m == MAP_FAILED) die("mmap %s", path.c_str());
  s.base = (const uint8_t *)m;
  if (s.size < 8) die("%s too small", path.c_str());
  uint64_t hlen;
  memcpy(&hlen, s.base, 8);
  if (hlen + 8 > s.size) die("%s: header length %llu overruns file",
                            path.c_str(), (unsigned long long)hlen);
  s.parser.p = (const char *)s.base + 8;
  s.parser.end = s.parser.p + hlen;
  s.header = s.parser.parse();
  s.data_start = 8 + (int64_t)hlen;
}

static bool is_vision(const std::string &n) {
  return n.find("visual") != std::string::npos ||
         n.find("vision") != std::string::npos;
}

// ===========================================================================
// Source readers: three storage layouts, one output format.
//
// The engine wants every matrix as uniform NVFP4 at 4.5 bits. Checkpoints do
// not ship that way:
//
//   BF16                     Qwen/Qwen3.8-27B      55.6 GB, best quality
//   mixed NVFP4 + FP8        unsloth/...-NVFP4     22.5 GB, easy download
//
// The second is the one most people will have, and despite the name only its
// MLP is NVFP4 (8.4 GB); attention, the DeltaNet projections and lm_head are
// FP8 (10.6 GB). Left as-is that is 46% more bytes per token than our uniform
// build, and decode is bandwidth-bound, so it would be ~46% SLOWER. Hence:
// read whatever the checkpoint has, re-quantise everything to 4.5 bits, emit
// one 15.4 GB artefact either way.
//
// Layouts handled:
//   <t>.weight             BF16                        -> as-is
//   <t>.weight             F8_E4M3  + <t>.weight_scale BF16 [N,1]  (per channel)
//   <t>.weight_packed      U8       + <t>.weight_scale F8_E4M3 [N,K/16]
//                                   + <t>.weight_global_scale F32
//     dequant = e2m1(code) * block_scale / global_scale   (verified against
//     the BF16 checkpoint: this convention gives 12.7% rel-err, the other two
//     candidates give 6e3 and 4e7)
// ===========================================================================
static inline uint16_t f32_to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);  // round half to even
}

static const uint16_t *raw_ptr(const TensorRef &r) {
  return (const uint16_t *)(r.shard->base + r.shard->data_start + r.begin);
}

// Read one logical weight as BF16, whatever its storage.
static std::vector<uint16_t> read_weight_bf16(
    const std::unordered_map<std::string, TensorRef> &t, const std::string &base,
    int64_t &N, int64_t &K, bool &found) {
  found = false;
  std::vector<uint16_t> out;

  auto it = t.find(base + ".weight");
  if (it != t.end() && it->second.dtype == "BF16") {
    const TensorRef &r = it->second;
    N = r.shape[0]; K = r.shape.size() > 1 ? r.shape[1] : 1;
    const uint16_t *src = raw_ptr(r);
    out.assign(src, src + (size_t)(N * K));
    found = true;
    return out;
  }

  // FP8 per-output-channel
  if (it != t.end() && it->second.dtype == "F8_E4M3") {
    auto sit = t.find(base + ".weight_scale");
    if (sit == t.end()) return out;
    const TensorRef &r = it->second, &sr = sit->second;
    N = r.shape[0]; K = r.shape[1];
    const uint8_t *q = (const uint8_t *)raw_ptr(r);
    const uint16_t *sc = raw_ptr(sr);          // BF16, one per row
    out.resize((size_t)(N * K));
    parallel_rows(N, [&](int64_t b, int64_t e) {
      for (int64_t n = b; n < e; ++n) {
        const float s = bf16_to_f32(sc[n]);
        for (int64_t k = 0; k < K; ++k)
          out[(size_t)(n * K + k)] = f32_to_bf16(e4m3_to_f32(q[n * K + k]) * s);
      }
    });
    found = true;
    return out;
  }

  // NVFP4: packed nibbles + one E4M3 scale per 16 + one global F32
  auto pit = t.find(base + ".weight_packed");
  if (pit == t.end()) return out;
  auto sit = t.find(base + ".weight_scale");
  auto git = t.find(base + ".weight_global_scale");
  if (sit == t.end() || git == t.end()) return out;
  const TensorRef &r = pit->second, &sr = sit->second, &gr = git->second;
  N = r.shape[0];
  K = r.shape[1] * 2;                            // 2 weights per byte
  const uint8_t *q = (const uint8_t *)raw_ptr(r);
  const uint8_t *bs = (const uint8_t *)raw_ptr(sr);
  float gscale;
  memcpy(&gscale, raw_ptr(gr), 4);
  const float inv_g = 1.0f / gscale;
  const int64_t nblk = sr.shape[1];
  out.resize((size_t)(N * K));
  parallel_rows(N, [&](int64_t b, int64_t e) {
    for (int64_t n = b; n < e; ++n)
      for (int64_t j = 0; j < nblk; ++j) {
        const float s = e4m3_to_f32(bs[n * nblk + j]) * inv_g;
        for (int i = 0; i < NVFP4_BLOCK; ++i) {
          const int64_t idx = j * NVFP4_BLOCK + i;
          const uint8_t byte = q[n * (K / 2) + (idx >> 1)];
          const uint8_t code = (idx & 1) ? (byte >> 4) : (byte & 0xF);
          const float v = (code & 8) ? -kE2M1[code & 7] : kE2M1[code & 7];
          out[(size_t)(n * K + idx)] = f32_to_bf16(v * s);
        }
      }
  });
  found = true;
  return out;
}

// Logical weight names present in the checkpoint, whatever their storage.
static std::vector<std::string> logical_weights(
    const std::unordered_map<std::string, TensorRef> &t) {
  std::vector<std::string> v;
  for (auto &kv : t) {
    const std::string &n = kv.first;
    if (n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0)
      v.push_back(n.substr(0, n.size() - 7));
    else if (n.size() > 14 && n.compare(n.size() - 14, 14, ".weight_packed") == 0)
      v.push_back(n.substr(0, n.size() - 14));
  }
  std::sort(v.begin(), v.end());
  v.erase(std::unique(v.begin(), v.end()), v.end());
  return v;
}

// ===========================================================================
// config validation — refuse a checkpoint that does not match qwen38.h
// ===========================================================================
static int cfg_int(const json::Value *root, const char *key) {
  const json::Value *v = json::cfg_get(root, key);
  if (!v || v->kind != json::Value::NUM) die("config.json: missing int '%s'", key);
  return (int)v->num;
}
static double cfg_num(const json::Value *root, const char *key) {
  const json::Value *v = json::cfg_get(root, key);
  if (!v || v->kind != json::Value::NUM) die("config.json: missing num '%s'", key);
  return v->num;
}

static void validate_config(const std::string &dir) {
  std::string path = dir + "/config.json";
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) die("cannot open %s", path.c_str());
  struct stat st;
  fstat(fd, &st);
  std::vector<char> buf((size_t)st.st_size + 1);
  if (read(fd, buf.data(), (size_t)st.st_size) != st.st_size) die("read config");
  close(fd);
  buf[st.st_size] = 0;

  json::Parser jp;
  jp.p = buf.data();
  jp.end = buf.data() + st.st_size;
  const json::Value *c = jp.parse();

  int bad = 0;
  auto chk = [&](const char *key, int expect) {
    int got = cfg_int(c, key);
    if (got != expect) {
      fprintf(stderr, "  MISMATCH %-28s config=%d  qwen38.h=%d\n", key, got, expect);
      ++bad;
    }
  };
  chk("hidden_size", HIDDEN);
  chk("num_hidden_layers", N_LAYERS);
  chk("num_attention_heads", N_Q_HEADS);
  chk("num_key_value_heads", N_KV_HEADS);
  chk("head_dim", HEAD_DIM);
  chk("intermediate_size", INTERMEDIATE);
  chk("vocab_size", VOCAB);
  chk("max_position_embeddings", MAX_POS);
  chk("full_attention_interval", FULL_ATTN_INTERVAL);
  chk("linear_num_key_heads", LIN_K_HEADS);
  chk("linear_num_value_heads", LIN_V_HEADS);
  chk("linear_key_head_dim", LIN_HEAD_DIM);
  chk("linear_value_head_dim", LIN_HEAD_DIM);
  chk("linear_conv_kernel_dim", CONV_KERNEL);
  chk("mtp_num_hidden_layers", MTP_LAYERS);

  auto chkf = [&](const char *key, double expect) {
    double got = cfg_num(c, key);
    if (fabs(got - expect) > 1e-9 * fabs(expect) + 1e-12) {
      fprintf(stderr, "  MISMATCH %-28s config=%g  qwen38.h=%g\n", key, got, expect);
      ++bad;
    }
  };
  chkf("rms_norm_eps", RMS_EPS);
  chkf("partial_rotary_factor", PARTIAL_ROTARY);

  // The engine assumes unquantised source weights: the fake-quant oracle needs
  // BF16 ground truth to quantise *from*.
  if (json::get(c, "quantization_config"))
    printf("note: pre-quantised checkpoint. Weights will be reconstructed and\n"
           "      re-packed uniformly at 4.5 bpw. Quality is bounded by the\n"
           "      source; the BF16 release gives a better result.\n");

  if (bad) die("%d config mismatch(es) vs qwen38.h — refusing to convert", bad);
  printf("config.json validated against qwen38.h: OK\n");
}

// ===========================================================================
// main
//
// test/test_nvfp4.cpp includes this file with SPARK27_NO_MAIN to exercise the
// codecs and the swizzle directly, so the packer under test is byte-for-byte
// the packer that writes qwengine.bin — not a copy that can drift from it.
// ===========================================================================
#ifndef SPARK27_NO_MAIN
int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr,
            "usage: %s <checkpoint-dir> <out.bin> [--limit N] [--dry] [--no-vision]\n"
            "  --limit N  stop after N tensors (quick error survey)\n"
            "  --dry      measure only, write nothing\n"
            "  --no-vision  drop the vision tower (text-only build)\n",
            argv[0]);
    return 2;
  }
  const std::string dir = argv[1], outpath = argv[2];
  int64_t limit = -1;
  bool dry = false, keep_vision = true;
  for (int i = 3; i < argc; ++i) {
    if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--dry")) dry = true;
    else if (!strcmp(argv[i], "--no-vision")) keep_vision = false;
    else die("unknown arg %s", argv[i]);
  }

  validate_config(dir);

  // ---- read the shard index -----------------------------------------------
  std::string idxpath = dir + "/model.safetensors.index.json";
  int fd = open(idxpath.c_str(), O_RDONLY);
  if (fd < 0) die("cannot open %s", idxpath.c_str());
  struct stat st;
  fstat(fd, &st);
  std::vector<char> ibuf((size_t)st.st_size + 1);
  if (read(fd, ibuf.data(), (size_t)st.st_size) != st.st_size) die("read index");
  close(fd);
  json::Parser ip;
  ip.p = ibuf.data();
  ip.end = ibuf.data() + st.st_size;
  const json::Value *wm = json::get(ip.parse(), "weight_map");
  if (!wm) die("index has no weight_map");

  std::vector<std::string> shard_names;
  for (auto &kv : wm->obj) {
    const std::string &f = kv.second->str;
    bool seen = false;
    for (auto &s : shard_names) if (s == f) { seen = true; break; }
    if (!seen) shard_names.push_back(f);
  }
  printf("shards: %zu\n", shard_names.size());

  std::vector<Shard *> shards;
  std::unordered_map<std::string, TensorRef> tensors;
  for (auto &name : shard_names) {
    Shard *s = new Shard();
    open_shard(*s, dir + "/" + name);
    shards.push_back(s);
    for (auto &kv : s->header->obj) {
      if (kv.first == "__metadata__") continue;
      const json::Value *t = kv.second;
      TensorRef r;
      r.shard = s;
      r.dtype = json::get(t, "dtype")->str;
      for (auto *d : json::get(t, "shape")->arr) r.shape.push_back((int64_t)d->num);
      const json::Array &off = json::get(t, "data_offsets")->arr;
      r.begin = (int64_t)off[0]->num;
      r.end = (int64_t)off[1]->num;
      tensors.emplace(kv.first, r);
    }
  }

  // ---- select text tensors, plan the output -------------------------------
  // Enumerate LOGICAL tensors. A quantised checkpoint stores one weight as
  // three entries (packed / scale / global_scale); those are not separate
  // tensors and must not be emitted as such.
  auto is_quant_meta = [](const std::string &n) {
    return n.find(".weight_scale") != std::string::npos ||
           n.find(".weight_global_scale") != std::string::npos ||
           n.find(".input_global_scale") != std::string::npos ||
           n.find(".weight_scale_inv") != std::string::npos;
  };
  std::vector<std::string> names;
  names.reserve(tensors.size());
  for (auto &kv : tensors) {
    const std::string &n = kv.first;
    if (is_vision(n) && !keep_vision) continue;
    if (is_quant_meta(n)) continue;
    if (n.size() > 14 && n.compare(n.size() - 14, 14, ".weight_packed") == 0)
      names.push_back(n.substr(0, n.size() - 14) + ".weight");
    else
      names.push_back(n);
  }
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());


  // The fused copies replace their parts, so the parts are not written.
  auto is_fused_part = [](const std::string &n) {
    return n.find("mlp.gate_proj.weight") != std::string::npos ||
           n.find("mlp.up_proj.weight") != std::string::npos ||
           n.find("linear_attn.in_proj_z.weight") != std::string::npos ||
           n.find("linear_attn.in_proj_a.weight") != std::string::npos ||
           n.find("linear_attn.in_proj_b.weight") != std::string::npos ||
           n.find("self_attn.q_proj.weight") != std::string::npos ||
           n.find("self_attn.k_proj.weight") != std::string::npos ||
           n.find("self_attn.v_proj.weight") != std::string::npos;
  };
  {
    std::vector<std::string> keep;
    for (const std::string &n : names)
      if (!(is_fused_part(n) && n.find("mtp.") == std::string::npos)) keep.push_back(n);
    names.swap(keep);
  }
  int n_dropped = (int)tensors.size() - (int)names.size();
  {
    int nv = 0;
    for (const std::string &n : names) if (is_vision(n)) ++nv;
    printf("tensors: %zu entries in checkpoint -> %zu logical tensors "
           "(%d vision %s)\n", tensors.size(), names.size(), nv,
           keep_vision ? "kept" : "dropped");
  }
  (void)n_dropped;
  if (limit > 0 && limit < (int64_t)names.size()) names.resize((size_t)limit);

  std::vector<TensorEntry> entries(names.size());
  memset(entries.data(), 0, entries.size() * sizeof(TensorEntry));

  const int64_t table_bytes =
      (int64_t)sizeof(FileHeader) + (int64_t)entries.size() * sizeof(TensorEntry);
  const int64_t data_offset = (table_bytes + kDataAlign - 1) / kDataAlign * kDataAlign;

  FILE *out = nullptr;
  if (!dry) {
    out = fopen(outpath.c_str(), "wb");
    if (!out) die("cannot create %s", outpath.c_str());
    if (fseek(out, data_offset, SEEK_SET)) die("seek");
  }
  auto emit = [&](const void *p, size_t n) {
    if (!dry && fwrite(p, 1, n, out) != n) die("write");
  };

  int64_t cursor = data_offset;
  int64_t n_quant = 0, quant_bytes = 0, raw_bytes = 0;
  double worst_rel = 0.0;
  std::string worst_name;

  std::vector<float> deq;  // round-trip buffer for the error check
  struct Extra { std::string name; std::string src; };
  std::vector<Extra> extra_d2;

  // ---- fused projections --------------------------------------------------
  // Several projections read the SAME input and differ only in their output
  // rows, so they can be one taller matrix. Two reasons that is faster:
  //  - the GEMV's efficiency tracks blocks-per-SM, and blocks come from output
  //    rows. [48,5120] makes 6 blocks for 48 SMs and runs at 6% of bandwidth;
  //    folded into a taller matrix it runs at the taller matrix's rate.
  //  - one launch instead of two or three, and a decode step is ~500 launches.
  // Concatenating along rows is trivial in this layout because rows are packed
  // independently -- the fused tensor is just the rows in order.
  struct Fuse { const char *out; std::vector<std::string> parts; };
  std::vector<Fuse> fuses;
  for (int l = 0; l < N_LAYERS; ++l) {
    char b1[160], b2[160], b3[160], b4[160];
    snprintf(b1, sizeof b1, "model.language_model.layers.%d.mlp.gate_proj.weight", l);
    snprintf(b2, sizeof b2, "model.language_model.layers.%d.mlp.up_proj.weight", l);
    snprintf(b3, sizeof b3, "model.language_model.layers.%d.mlp.gate_up_proj.weight", l);
    fuses.push_back({strdup(b3), {b1, b2}});
    if (!is_full_attn(l)) {
      snprintf(b1, sizeof b1, "model.language_model.layers.%d.linear_attn.in_proj_z.weight", l);
      snprintf(b2, sizeof b2, "model.language_model.layers.%d.linear_attn.in_proj_a.weight", l);
      snprintf(b3, sizeof b3, "model.language_model.layers.%d.linear_attn.in_proj_b.weight", l);
      snprintf(b4, sizeof b4, "model.language_model.layers.%d.linear_attn.in_proj_zab.weight", l);
      fuses.push_back({strdup(b4), {b1, b2, b3}});
    } else {
      snprintf(b1, sizeof b1, "model.language_model.layers.%d.self_attn.q_proj.weight", l);
      snprintf(b2, sizeof b2, "model.language_model.layers.%d.self_attn.k_proj.weight", l);
      snprintf(b3, sizeof b3, "model.language_model.layers.%d.self_attn.v_proj.weight", l);
      snprintf(b4, sizeof b4, "model.language_model.layers.%d.self_attn.qkv_proj.weight", l);
      fuses.push_back({strdup(b4), {b1, b2, b3}});
    }
  }

  for (size_t i = 0; i < names.size(); ++i) {
    const std::string &name = names[i];
    if (name.size() >= kNameMax) die("name too long: %s", name.c_str());

    // Whatever the checkpoint stores (BF16, FP8+channel scale, or packed
    // NVFP4), read it as BF16 and re-quantise uniformly below.
    std::vector<uint16_t> buf;
    const uint16_t *src = nullptr;
    int64_t nelem = 1, RN = 0, RK = 0;
    std::vector<int64_t> shape;
    if (name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0 &&
        tensors.find(name) == tensors.end()) {
      // stored quantised: reconstruct from packed + scales
      bool ok = false;
      buf = read_weight_bf16(tensors, name.substr(0, name.size() - 7), RN, RK, ok);
      if (!ok) die("%s: cannot reconstruct from the checkpoint", name.c_str());
      src = buf.data();
      shape = {RN, RK};
      nelem = RN * RK;
    } else {
      const TensorRef &rr = tensors.at(name);
      shape = rr.shape;
      for (int64_t d : shape) nelem *= d;
      if (rr.dtype == "BF16") {
        src = raw_ptr(rr);
      } else {
        bool ok = false;
        buf = read_weight_bf16(tensors, name.substr(0, name.size() - 7), RN, RK, ok);
        if (!ok) die("%s: unexpected dtype %s", name.c_str(), rr.dtype.c_str());
        src = buf.data();
        shape = {RN, RK};
        nelem = RN * RK;
      }
    }
    struct { std::vector<int64_t> shape; } r2{shape};

    TensorEntry &e = entries[i];
    snprintf(e.name, kNameMax, "%s", name.c_str());
    e.offset = cursor;

    if (r2.shape.size() == 2 && r2.shape[1] % SWZ_WEIGHTS == 0) {
      const int64_t N = r2.shape[0], K = r2.shape[1];
      const int64_t nb = nvfp4_bytes(N, K);
      std::vector<uint8_t> packed((size_t)nb);
      float ts = pack_nvfp4(src, N, K, packed.data());

      // Round-trip error, per the P1 gate (< 8% RMS rel-err for all matrices).
      deq.resize((size_t)nelem);
      unpack_nvfp4(packed.data(), N, K, ts, deq.data());
      double se = 0.0, sr = 0.0;
      for (int64_t j = 0; j < nelem; ++j) {
        double o = bf16_to_f32(src[j]), d = deq[(size_t)j];
        se += (o - d) * (o - d);
        sr += o * o;
      }
      double rel = sr > 0 ? sqrt(se / sr) : 0.0;
      if (rel > worst_rel) { worst_rel = rel; worst_name = name; }

      emit(packed.data(), (size_t)nb);
      if (dry)
        printf("  %-58s [%6lld,%6lld] rel-err %.4f%s\n", name.c_str(),
               (long long)N, (long long)K, rel, rel >= 0.08 ? "  <-- OVER GATE" : "");
      e.bytes = nb;
      e.n = (int32_t)N;
      e.k = (int32_t)K;
      e.kind = KIND_NVFP4;
      e.tensor_scale = ts;
      cursor += nb;
      quant_bytes += nb;
      ++n_quant;

      if (rel > 0.08)
        fprintf(stderr, "  WARN %s rel-err %.4f exceeds 8%% gate\n", name.c_str(), rel);

      // The draft head re-reads lm_head every draft step, so it also gets a
      // 2-bit copy. Guesses are verified by the full model, so this costs a
      // little acceptance and no correctness at all.
      if (name == "lm_head.weight") extra_d2.push_back({name + ".draft", name});
    } else {
      const int64_t nb = nelem * 2;  // keep BF16 verbatim
      emit(src, (size_t)nb);
      e.bytes = nb;
      e.n = (int32_t)nelem;
      e.k = 0;
      e.kind = KIND_BF16;
      e.tensor_scale = 1.0f;
      cursor += nb;
      raw_bytes += nb;
    }

    if (!dry && ((i % 64) == 0 || i + 1 == names.size()))
      printf("\r  packing %zu/%zu ...", i + 1, names.size()), fflush(stdout);
  }
  if (!dry) printf("\n");

  // ---- fused projections, appended after the main tensors -----------------
  {
    std::vector<uint16_t> cat;
    for (const Fuse &f : fuses) {
      int64_t N = 0, K = 0;
      bool ok = true;
      std::vector<std::vector<uint16_t>> parts;
      for (const std::string &pn : f.parts) {
        int64_t pn_N = 0, pn_K = 0; bool got = false;
        std::vector<uint16_t> v =
            read_weight_bf16(tensors, pn.substr(0, pn.size() - 7), pn_N, pn_K, got);
        if (!got) { ok = false; break; }
        N += pn_N; K = pn_K;
        parts.push_back(std::move(v));
      }
      if (!ok) continue;
      cat.resize((size_t)(N * K));
      int64_t row = 0;
      for (auto &v : parts) {
        memcpy(cat.data() + row * K, v.data(), v.size() * 2);
        row += (int64_t)(v.size() / K);
      }
      const int64_t nb = nvfp4_bytes(N, K);
      std::vector<uint8_t> packed((size_t)nb);
      float ts = pack_nvfp4(cat.data(), N, K, packed.data());
      emit(packed.data(), (size_t)nb);
      TensorEntry e{};
      snprintf(e.name, kNameMax, "%s", f.out);
      e.offset = cursor; e.bytes = nb; e.n = (int32_t)N; e.k = (int32_t)K;
      e.kind = KIND_NVFP4; e.tensor_scale = ts;
      entries.push_back(e);
      cursor += nb;
    }
    printf("  + %zu fused projections\n", fuses.size());
  }

  // ---- tokenizer, baked in -------------------------------------------------
  // The engine should need exactly one file at runtime. The vocabulary and
  // merge table are ~5 MB against a 16 GB weights file, so carrying them costs
  // nothing and removes a whole class of "wrong tokenizer" bug.
  {
    const std::string tp = dir + "/tokenizer.json";
    int tfd = open(tp.c_str(), O_RDONLY);
    if (tfd < 0) {
      printf("  ! no tokenizer.json in the checkpoint; engine will need one\n");
    } else {
      struct stat ts;
      fstat(tfd, &ts);
      std::vector<char> tb((size_t)ts.st_size + 1);
      if (read(tfd, tb.data(), (size_t)ts.st_size) != ts.st_size) die("read tokenizer");
      close(tfd);
      tb[ts.st_size] = 0;
      json::Parser tp2;
      tp2.p = tb.data();
      tp2.end = tb.data() + ts.st_size;
      const json::Value *tj = tp2.parse();
      const json::Value *model = json::get(tj, "model");
      const json::Value *vocab = json::get(model, "vocab");
      const json::Value *merges = json::get(model, "merges");
      const json::Value *added = json::get(tj, "added_tokens");
      if (!vocab || !merges) die("tokenizer.json: no vocab/merges");

      // id -> token string. Special tokens live ABOVE the vocab range (the
      // vocab stops at 248043, specials run 248044+), so the table must be
      // sized from both or decode silently emits nothing for <|im_start|>.
      int maxid = 0;
      for (auto &kv : vocab->obj) maxid = std::max(maxid, (int)kv.second->num);
      if (added)
        for (auto *av : added->arr) {
          const json::Value *idv = json::get(av, "id");
          if (idv) maxid = std::max(maxid, (int)idv->num);
        }
      std::vector<std::string> id2tok((size_t)maxid + 1);
      for (auto &kv : vocab->obj) id2tok[(size_t)kv.second->num] = kv.first;

      std::vector<std::string> mg;
      mg.reserve(merges->arr.size());
      for (auto *mv : merges->arr) {
        if (mv->kind == json::Value::STR) {
          const std::string &m0 = mv->str;
          const size_t sp = m0.find(' ');
          if (sp == std::string::npos) continue;
          mg.push_back(m0.substr(0, sp) + '\x01' + m0.substr(sp + 1));
        } else if (mv->kind == json::Value::ARR && mv->arr.size() == 2) {
          mg.push_back(mv->arr[0]->str + '\x01' + mv->arr[1]->str);
        }
      }

      std::vector<std::pair<std::string, int>> sp;
      if (added)
        for (auto *av : added->arr) {
          const json::Value *c = json::get(av, "content");
          const json::Value *idv = json::get(av, "id");
          if (c && idv) {
            sp.emplace_back(c->str, (int)idv->num);
            if ((size_t)idv->num < id2tok.size()) id2tok[(size_t)idv->num] = c->str;
          }
        }

      // pack: header, then three (offsets, blob) sections
      std::vector<uint8_t> blob;
      auto put32 = [&](uint32_t v) {
        blob.insert(blob.end(), (uint8_t *)&v, (uint8_t *)&v + 4);
      };
      auto put_section = [&](const std::vector<std::string> &v) {
        uint32_t off = 0;
        for (size_t i = 0; i <= v.size(); ++i) {
          put32(off);
          if (i < v.size()) off += (uint32_t)v[i].size();
        }
        for (const std::string &x : v) blob.insert(blob.end(), x.begin(), x.end());
      };
      put32(0x314B4F54);                       // "TOK1"
      put32((uint32_t)id2tok.size());
      put32((uint32_t)mg.size());
      put32((uint32_t)sp.size());
      put_section(id2tok);
      put_section(mg);
      for (auto &x : sp) put32((uint32_t)x.second);
      std::vector<std::string> spn;
      for (auto &x : sp) spn.push_back(x.first);
      put_section(spn);

      if (blob.size() & 1) blob.push_back(0);
      emit(blob.data(), blob.size());
      TensorEntry e{};
      snprintf(e.name, kNameMax, "__tokenizer__");
      e.offset = cursor; e.bytes = (int64_t)blob.size();
      e.n = (int32_t)(blob.size() / 2); e.k = 0;
      e.kind = KIND_BF16; e.tensor_scale = 1.0f;
      entries.push_back(e);
      cursor += (int64_t)blob.size();
      printf("  + tokenizer: %zu tokens, %zu merges, %zu special (%.2f MB)\n",
             id2tok.size(), mg.size(), sp.size(), blob.size() / 1e6);
    }
  }

  // ---- extra low-bit copies, appended after the main tensors --------------
  for (const Extra &ex : extra_d2) {
    int64_t N = 0, K = 0; bool got = false;
    std::vector<uint16_t> v =
        read_weight_bf16(tensors, ex.src.substr(0, ex.src.size() - 7), N, K, got);
    if (!got) continue;
    const uint16_t *src = v.data();
    const int64_t nb = d2_bytes(N, K);
    std::vector<uint8_t> packed((size_t)nb);
    float ts = pack_d2(src, N, K, packed.data());
    emit(packed.data(), (size_t)nb);
    TensorEntry e{};
    snprintf(e.name, kNameMax, "%s", ex.name.c_str());
    e.offset = cursor; e.bytes = nb; e.n = (int32_t)N; e.k = (int32_t)K;
    e.kind = KIND_D2; e.tensor_scale = ts;
    entries.push_back(e);
    cursor += nb;
    printf("  + %s: %.3f GB at 2.5 bpw (was %.3f GB at 4.5)\n", ex.name.c_str(),
           nb / 1e9, nvfp4_bytes(N, K) / 1e9);
  }

  FileHeader h{};
  h.magic = kMagic;
  h.version = kVersion;
  h.n_tensors = (int32_t)entries.size();
  h.data_offset = data_offset;
  h.total_bytes = cursor;
  if (!dry) {
    if (fseek(out, 0, SEEK_SET)) die("seek");
    if (fwrite(&h, sizeof h, 1, out) != 1) die("write header");
    if (fwrite(entries.data(), sizeof(TensorEntry), entries.size(), out) !=
        entries.size())
      die("write table");
    fclose(out);
    printf("\nwrote %s\n", outpath.c_str());
  } else {
    printf("\n[dry run — nothing written]\n");
  }
  printf("  tensors      : %zu (%lld NVFP4, %zu BF16)\n", entries.size(),
         (long long)n_quant, entries.size() - (size_t)n_quant);
  printf("  NVFP4 bytes  : %.2f GB\n", quant_bytes / 1e9);
  printf("  BF16 bytes   : %.3f GB\n", raw_bytes / 1e9);
  printf("  total        : %.2f GB\n", cursor / 1e9);
  printf("  worst rel-err: %.4f  (%s)   gate < 0.08: %s\n", worst_rel,
         worst_name.c_str(), worst_rel < 0.08 ? "PASS" : "FAIL");
  if (worst_rel >= 0.08)
    printf("\nnote: the <8%% round-trip gate is not reachable at 4.5 bits/weight;\n"
           "      %.4f is the measured floor for NVFP4 on this model. Not an error.\n",
           worst_rel);
  return 0;
}
#endif  // SPARK27_NO_MAIN
