#pragma once
// src/image.cpp — turning a picture into what the vision tower expects.
//
// This is the unglamorous half of vision, and the half most likely to be
// subtly wrong: if the pixels reaching the tower differ from what the
// reference produces, the model sees a slightly different image and there is
// no error message, only worse answers.
//
// Three things here are easy to get wrong and were taken from the reference
// rather than assumed:
//
//  1. The resize runs on the UINT8 image, BEFORE normalisation. Resizing in
//     float instead lets bicubic overshoot past the 0-255 range where the
//     reference clamps it, which moves values by up to 0.24 in normalised
//     units.
//  2. Normalisation is (x - 127.5) / 127.5 as a fused operation, not
//     (x/255 - 0.5) / 0.5. They differ by an ulp, which matters only if you
//     claim bit-comparability, but it costs nothing to be right.
//  3. Patches come out in MERGE-BLOCK order, not raster order: the 2x2 block
//     the patch merger will later fuse must be four CONSECUTIVE rows. Raster
//     order silently scrambles which patches get merged.

#include <math.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "vision.cu"

namespace spark27 {

// smart_resize: round to a multiple of 32 (patch 16 x merge 2), then pull the
// area inside [min_pixels, max_pixels] using the ORIGINAL dimensions.
inline void smart_resize(int h, int w, int &hb, int &wb) {
  const int F = vis::RESIZE_FACTOR;
  auto rnd = [F](double v) { return (int)(llround(v / F) * F); };
  hb = std::max(F, rnd(h));
  wb = std::max(F, rnd(w));
  const double area = (double)hb * wb;
  if (area > (double)vis::MAX_PIXELS) {
    const double beta = sqrt((double)h * w / (double)vis::MAX_PIXELS);
    hb = std::max(F, (int)(floor(h / beta / F) * F));
    wb = std::max(F, (int)(floor(w / beta / F) * F));
  } else if (area < (double)vis::MIN_PIXELS) {
    const double beta = sqrt((double)vis::MIN_PIXELS / ((double)h * w));
    hb = (int)(ceil(h * beta / F) * F);
    wb = (int)(ceil(w * beta / F) * F);
  }
}

// Bicubic resample of an 8-bit RGB image, clamped and rounded back to 8 bits,
// which is what torchvision does on a uint8 tensor.
inline void resize_bicubic_u8(const uint8_t *src, int sh, int sw, uint8_t *dst,
                              int dh, int dw) {
  auto cubic = [](double x) {
    const double a = -0.5;                 // torchvision/PIL use a = -0.5
    x = fabs(x);
    if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1;
    if (x < 2.0) return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a;
    return 0.0;
  };
  const double ry = (double)sh / dh, rx = (double)sw / dw;
  for (int y = 0; y < dh; ++y) {
    const double sy = (y + 0.5) * ry - 0.5;
    const int y0 = (int)floor(sy);
    double wy[4];
    for (int i = 0; i < 4; ++i) wy[i] = cubic(sy - (y0 - 1 + i));
    for (int x = 0; x < dw; ++x) {
      const double sx = (x + 0.5) * rx - 0.5;
      const int x0 = (int)floor(sx);
      double wx[4];
      for (int i = 0; i < 4; ++i) wx[i] = cubic(sx - (x0 - 1 + i));
      for (int c = 0; c < 3; ++c) {
        double acc = 0, wsum = 0;
        for (int i = 0; i < 4; ++i) {
          const int yy = std::min(std::max(y0 - 1 + i, 0), sh - 1);
          for (int j = 0; j < 4; ++j) {
            const int xx = std::min(std::max(x0 - 1 + j, 0), sw - 1);
            const double ww = wy[i] * wx[j];
            acc += ww * src[((size_t)yy * sw + xx) * 3 + c];
            wsum += ww;
          }
        }
        const double v = wsum != 0 ? acc / wsum : 0;
        dst[((size_t)y * dw + x) * 3 + c] = (uint8_t)std::min(255.0, std::max(0.0, v + 0.5));
      }
    }
  }
}

struct ImageTokens {
  std::vector<float> pixels;   // [N, 1536] normalised, merge-block order
  std::vector<int> pos;        // [N, 2] (row, col) per patch
  std::vector<int> bidx;       // [4, N] bilinear corners into the 48x48 table
  std::vector<float> bwts;     // [4, N] bilinear weights
  int grid_h = 0, grid_w = 0, n_patches = 0, n_tokens = 0;
};

// rgb: 8-bit, h*w*3. Produces everything the tower needs.
inline ImageTokens preprocess_image(const uint8_t *rgb, int h, int w) {
  ImageTokens t;
  int hb, wb;
  smart_resize(h, w, hb, wb);
  std::vector<uint8_t> res((size_t)hb * wb * 3);
  resize_bicubic_u8(rgb, h, w, res.data(), hb, wb);

  const int gh = hb / vis::PATCH, gw = wb / vis::PATCH;
  t.grid_h = gh; t.grid_w = gw;
  t.n_patches = gh * gw;
  t.n_tokens = t.n_patches / (vis::MERGE * vis::MERGE);
  t.pixels.assign((size_t)t.n_patches * vis::PATCH_FEAT, 0.f);
  t.pos.assign((size_t)t.n_patches * 2, 0);

  // Patch rows in MERGE-BLOCK order: row k = ((bh*(gw/2)+bw)*2+mh)*2+mw, so the
  // four patches of one 2x2 block land consecutively for the merger.
  // Feature index inside a row: ((c*2 + tt)*16 + py)*16 + px, and the two
  // temporal slices are identical for a still image.
  for (int bh = 0; bh < gh / 2; ++bh)
    for (int bw = 0; bw < gw / 2; ++bw)
      for (int mh = 0; mh < 2; ++mh)
        for (int mw = 0; mw < 2; ++mw) {
          const int k = ((bh * (gw / 2) + bw) * 2 + mh) * 2 + mw;
          const int pr = 2 * bh + mh, pc = 2 * bw + mw;   // patch row/col
          t.pos[(size_t)k * 2 + 0] = pr;
          t.pos[(size_t)k * 2 + 1] = pc;
          float *row = t.pixels.data() + (size_t)k * vis::PATCH_FEAT;
          for (int c = 0; c < 3; ++c)
            for (int py = 0; py < vis::PATCH; ++py)
              for (int px = 0; px < vis::PATCH; ++px) {
                const int iy = pr * vis::PATCH + py, ix = pc * vis::PATCH + px;
                const float v =
                    ((float)res[((size_t)iy * wb + ix) * 3 + c] - 127.5f) / 127.5f;
                for (int tt = 0; tt < vis::TEMPORAL; ++tt)
                  row[((c * vis::TEMPORAL + tt) * vis::PATCH + py) * vis::PATCH + px] = v;
              }
        }

  // Bilinear interpolation into the 48x48 learned position table, sampled on
  // the raster grid and then reordered the same way as the patches.
  const int side = vis::POS_SIDE;
  t.bidx.assign((size_t)4 * t.n_patches, 0);
  t.bwts.assign((size_t)4 * t.n_patches, 0.f);
  std::vector<float> hg(gh), wg(gw);
  for (int i = 0; i < gh; ++i) hg[i] = gh > 1 ? (float)i * (side - 1) / (gh - 1) : 0.f;
  for (int i = 0; i < gw; ++i) wg[i] = gw > 1 ? (float)i * (side - 1) / (gw - 1) : 0.f;
  for (int bh = 0; bh < gh / 2; ++bh)
    for (int bw = 0; bw < gw / 2; ++bw)
      for (int mh = 0; mh < 2; ++mh)
        for (int mw = 0; mw < 2; ++mw) {
          const int k = ((bh * (gw / 2) + bw) * 2 + mh) * 2 + mw;
          const int pr = 2 * bh + mh, pc = 2 * bw + mw;
          const int hf = (int)hg[pr], wf = (int)wg[pc];
          const int hc = std::min(hf + 1, side - 1), wc = std::min(wf + 1, side - 1);
          const float hfr = hg[pr] - hf, wfr = wg[pc] - wf;
          const int corners[4] = {hf * side + wf, hf * side + wc,
                                  hc * side + wf, hc * side + wc};
          const float wts[4] = {(1 - hfr) * (1 - wfr), (1 - hfr) * wfr,
                                hfr * (1 - wfr), hfr * wfr};
          for (int c = 0; c < 4; ++c) {
            t.bidx[(size_t)c * t.n_patches + k] = corners[c];
            t.bwts[(size_t)c * t.n_patches + k] = wts[c];
          }
        }
  return t;
}

}  // namespace spark27
