#pragma once
// src/format.h — the qwengine.bin container, shared by writer and reader.
//
// One definition, included by both convert.cpp and loader.cpp, so the two can
// never disagree about the layout.
//
//   [FileHeader][TensorEntry x n_tensors][pad to 4096][tensor data ...]
//
// Tensor data is contiguous and 4096-aligned. NVFP4 payloads are in the
// swizzled order documented at the top of convert.cpp; BF16 payloads are the
// checkpoint's bytes verbatim.
#pragma once
#include <stdint.h>

namespace spark27 {

constexpr uint32_t kMagic = 0x37324B53;  // "SK27"
constexpr uint32_t kVersion = 1;
constexpr int kNameMax = 96;
constexpr int64_t kDataAlign = 4096;

enum : int32_t { KIND_BF16 = 0, KIND_NVFP4 = 1, KIND_D2 = 2 };

// D2: a 2-bit weight format used ONLY for the speculative draft head.
//
// Drafts are checked by the full model, so a wrong guess costs a little speed
// and never correctness. That makes the draft the one place in the engine
// where precision can be traded for bytes with no risk at all — and the draft
// re-reads the 715 MB lm_head on every step, which is what stops deeper
// drafting from paying for itself.
//
// Four levels {-1.5, -0.5, +0.5, +1.5} scaled by one FP8 E4M3 factor per 16
// weights: 2 + 8/16 = 2.5 bits/weight, so lm_head drops 715 MB -> 397 MB.
// Layout mirrors NVFP4's, one warp-step at a time:
//   [ 256 B: 1024 two-bit codes ][ 64 B: their 64 scales ]  = 320 B
// Lane i reads 8 B of codes at i*8 and its two scales at 256 + i*2.
constexpr int D2_VAL_BYTES = 256;
constexpr int D2_SCALE_BYTES = 64;
constexpr int D2_STEP_BYTES = D2_VAL_BYTES + D2_SCALE_BYTES;  // 320

constexpr int64_t d2_bytes(int64_t N, int64_t K) {
  return N * (K / 1024) * D2_STEP_BYTES;
}

struct FileHeader {
  uint32_t magic, version;
  int32_t n_tensors;
  int32_t _pad;
  int64_t data_offset;
  int64_t total_bytes;
};

struct TensorEntry {
  char name[kNameMax];
  int64_t offset;  // from start of file
  int64_t bytes;
  int32_t n, k;    // k == 0 for non-matrix tensors; then n is the element count
  int32_t kind;
  float tensor_scale;  // NVFP4 only
};

static_assert(sizeof(FileHeader) == 32, "FileHeader packing");
static_assert(sizeof(TensorEntry) == kNameMax + 32, "TensorEntry packing");

}  // namespace spark27
