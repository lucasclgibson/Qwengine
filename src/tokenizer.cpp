#pragma once
// src/tokenizer.cpp — text in, token ids out.
//
// The model does not read text. It reads integers, each standing for a chunk of
// text — usually a word fragment. Turning text into those integers has to match
// the reference EXACTLY: one different id and the model is reading a different
// sentence.
//
// The scheme is byte-level BPE:
//   1. Split the text into rough pieces (words, runs of digits, punctuation)
//      with a fixed rule. This stops merges from spanning word boundaries.
//   2. Map each byte to a printable stand-in character, so any binary input is
//      representable and nothing is ever "unknown".
//   3. Repeatedly glue together the adjacent pair with the lowest merge rank,
//      until no pair in the table remains. The ranks come from the checkpoint.
//
// Special tokens (<|im_start|> and friends) are matched before any of that and
// pass through whole.
//
// The vocabulary and merge table live inside qwengine.bin, so the engine needs
// no other file at runtime.

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "format.h"
#include "unicode_nfc.h"
#include "unicode_tables.h"

namespace spark27 {

// GPT-2's byte<->printable-character mapping. Bytes that are already printable
// ASCII map to themselves; the rest are shifted into an unused block so every
// byte has a unique, printable stand-in.
inline void byte_level_alphabet(uint32_t *b2u, int *u2b_key, int *u2b_val, int &n) {
  n = 0;
  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                           (b >= 0xAE && b <= 0xFF);
    b2u[b] = printable ? (uint32_t)b : (uint32_t)(256 + extra++);
    u2b_key[n] = (int)b2u[b];
    u2b_val[n] = b;
    ++n;
  }
}

inline void utf8_append(std::string &s, uint32_t c) {
  if (c < 0x80) s += (char)c;
  else if (c < 0x800) { s += (char)(0xC0 | (c >> 6)); s += (char)(0x80 | (c & 63)); }
  else if (c < 0x10000) {
    s += (char)(0xE0 | (c >> 12)); s += (char)(0x80 | ((c >> 6) & 63));
    s += (char)(0x80 | (c & 63));
  } else {
    s += (char)(0xF0 | (c >> 18)); s += (char)(0x80 | ((c >> 12) & 63));
    s += (char)(0x80 | ((c >> 6) & 63)); s += (char)(0x80 | (c & 63));
  }
}

// Decode one UTF-8 codepoint; advances i.
inline uint32_t utf8_next(const std::string &s, size_t &i) {
  const unsigned char c = (unsigned char)s[i];
  if (c < 0x80) { ++i; return c; }
  if ((c >> 5) == 6 && i + 1 < s.size()) {
    const uint32_t r = ((c & 31u) << 6) | ((unsigned char)s[i + 1] & 63u);
    i += 2; return r;
  }
  if ((c >> 4) == 14 && i + 2 < s.size()) {
    const uint32_t r = ((c & 15u) << 12) | (((unsigned char)s[i + 1] & 63u) << 6) |
                       ((unsigned char)s[i + 2] & 63u);
    i += 3; return r;
  }
  if ((c >> 3) == 30 && i + 3 < s.size()) {
    const uint32_t r = ((c & 7u) << 18) | (((unsigned char)s[i + 1] & 63u) << 12) |
                       (((unsigned char)s[i + 2] & 63u) << 6) |
                       ((unsigned char)s[i + 3] & 63u);
    i += 4; return r;
  }
  ++i;
  return c;
}

// NFC normalisation.
//
// The reference tokenizer normalises before splitting, so "e + combining
// acute" and the single character "é" must produce identical ids. Without this
// the two forms tokenise differently and the model reads a different string.
// Canonical ordering (sort runs of combining marks by combining class) then
// canonical composition (glue starter+mark pairs), which is NFC for input that
// is already decomposed or already composed — the two cases real text is in.
inline int ccc_of(uint32_t c) {
  int lo = 0, hi = kNCCC - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    if (c < kCCC[mid][0]) hi = mid - 1;
    else if (c > kCCC[mid][0]) lo = mid + 1;
    else return (int)kCCC[mid][1];
  }
  return 0;
}
inline uint32_t compose_pair(uint32_t a, uint32_t b) {
  int lo = 0, hi = kNComp - 1;
  while (lo <= hi) {
    const int mid = (lo + hi) >> 1;
    const uint32_t ka = kComp[mid][0], kb = kComp[mid][1];
    if (a < ka || (a == ka && b < kb)) hi = mid - 1;
    else if (a > ka || (a == ka && b > kb)) lo = mid + 1;
    else return kComp[mid][2];
  }
  return 0;
}

inline std::string nfc(const std::string &in) {
  std::vector<uint32_t> cp;
  for (size_t i = 0; i < in.size();) cp.push_back(utf8_next(in, i));

  // canonical ordering: stable sort each run of non-starters by class
  for (size_t i = 1; i < cp.size(); ++i) {
    const int ci = ccc_of(cp[i]);
    if (!ci) continue;
    size_t j = i;
    while (j > 0) {
      const int cj = ccc_of(cp[j - 1]);
      if (cj == 0 || cj <= ci) break;
      std::swap(cp[j], cp[j - 1]);
      --j;
    }
  }

  // canonical composition
  std::vector<uint32_t> out;
  out.reserve(cp.size());
  size_t starter = (size_t)-1;
  int last_ccc = -1;
  for (size_t i = 0; i < cp.size(); ++i) {
    const int c = ccc_of(cp[i]);
    if (starter != (size_t)-1 && (last_ccc < c || (last_ccc == -1 && c == 0))) {
      const uint32_t comp = compose_pair(out[starter], cp[i]);
      if (comp) { out[starter] = comp; continue; }
    }
    if (c == 0) { starter = out.size(); last_ccc = -1; }
    else last_ccc = c;
    out.push_back(cp[i]);
  }

  std::string r;
  for (uint32_t c : out) utf8_append(r, c);
  return r;
}

struct Tokenizer {
  std::vector<std::string> id2tok;
  std::unordered_map<std::string, int> tok2id;
  // merge rank keyed by the concatenation "left\x01right"
  std::unordered_map<std::string, int> ranks;
  std::vector<std::pair<std::string, int>> specials;   // longest-first
  uint32_t b2u[256];
  std::unordered_map<uint32_t, int> u2b;
  int eos = -1, im_start = -1, im_end = -1;

  bool loaded() const { return !id2tok.empty(); }
  int id_of(const std::string &s) const {
    auto it = tok2id.find(s);
    return it == tok2id.end() ? -1 : it->second;
  }
};

// ---------------------------------------------------------------------------
// The pre-tokenizer.
//
// Mirrors the checkpoint's regex, in order:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)     English contractions
//   [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+  a word, with one optional leading symbol
//   \p{N}                            ONE digit at a time (not runs)
//    ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*   punctuation run, optional leading space
//   \s*[\r\n]+                       newlines
//   \s+(?!\S)                        trailing whitespace
//   \s+                              any other whitespace
// Written by hand because the alternative is a Unicode-aware regex engine, and
// this pattern is fixed for the life of the checkpoint.
// ---------------------------------------------------------------------------
inline std::vector<std::string> pretokenize(const std::string &text) {
  std::vector<std::string> out;
  const size_t n = text.size();
  size_t i = 0;
  auto is_space = [](uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0B ||
           c == 0x0C || c == 0x85 || c == 0xA0 || c == 0x2028 || c == 0x2029 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x3000 || c == 0x1680 || c == 0x205F;
  };
  auto peek = [&](size_t p, uint32_t &c) {
    if (p >= n) return (size_t)0;
    size_t q = p;
    c = utf8_next(text, q);
    return q - p;
  };

  while (i < n) {
    const size_t start = i;
    uint32_t c = 0;
    size_t cl = peek(i, c);

    // contractions
    if (c == '\'' && i + 1 < n) {
      const char a = text[i + 1] | 0x20;
      const char b = (i + 2 < n) ? (text[i + 2] | 0x20) : 0;
      int take = 0;
      if (a == 's' || a == 't' || a == 'm' || a == 'd') take = 2;
      else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') ||
               (a == 'l' && b == 'l')) take = 3;
      if (take) { out.push_back(text.substr(i, take)); i += take; continue; }
    }

    // optional single leading symbol, then a run of letters/marks
    {
      size_t j = i;
      uint32_t c0 = c;
      size_t l0 = cl;
      bool lead = false;
      if (c0 != '\r' && c0 != '\n' && !uc_is_letter(c0) && !uc_is_number(c0)) {
        uint32_t c1 = 0;
        const size_t l1 = peek(j + l0, c1);
        if (l1 && (uc_is_letter(c1) || uc_is_mark(c1))) { lead = true; j += l0; }
      }
      // The run is LETTERS ONLY -- combining marks do not extend it, they only
      // serve as the single optional leading character. This is what the Rust
      // regex engine tokenizers uses actually does; Python's `regex` module
      // disagrees on the same pattern, and the reference is what matters.
      // Verified against the reference on Devanagari: "नमस्ते" splits as
      // [नमस][्त][े], not [नमस्ते].
      if (lead || uc_is_letter(c0)) {
        size_t k = j;
        uint32_t cc = 0;
        size_t lc = peek(k, cc);
        size_t cnt = 0;
        while (lc && uc_is_letter(cc)) { k += lc; ++cnt; lc = peek(k, cc); }
        if (cnt) { out.push_back(text.substr(start, k - start)); i = k; continue; }
      }
    }

    // a single digit
    if (uc_is_number(c)) { out.push_back(text.substr(i, cl)); i += cl; continue; }

    // optional leading space, then a run of symbols, then trailing newlines
    {
      size_t j = i;
      if (c == ' ') {
        uint32_t c1 = 0;
        const size_t l1 = peek(j + cl, c1);
        if (l1 && !is_space(c1) && !uc_is_letter(c1) && !uc_is_mark(c1) && !uc_is_number(c1))
          j += cl;
      }
      uint32_t cc = 0;
      size_t lc = peek(j, cc);
      size_t cnt = 0;
      while (lc && !is_space(cc) && !uc_is_letter(cc) && !uc_is_mark(cc) && !uc_is_number(cc)) {
        j += lc; ++cnt; lc = peek(j, cc);
      }
      if (cnt) {
        while (j < n && (text[j] == '\r' || text[j] == '\n')) ++j;
        out.push_back(text.substr(start, j - start));
        i = j;
        continue;
      }
    }

    // whitespace runs: \s*[\r\n]+ , then \s+(?!\S) , then \s+
    if (is_space(c)) {
      size_t j = i, lastnl = std::string::npos;
      uint32_t cc = c;
      size_t lc = cl;
      while (lc && is_space(cc)) {
        if (cc == '\r' || cc == '\n') lastnl = j + lc;
        j += lc;
        lc = peek(j, cc);
      }
      if (lastnl != std::string::npos) {
        out.push_back(text.substr(start, lastnl - start));
        i = lastnl;
        continue;
      }
      // \s+(?!\S): if more text follows, hold back the final space
      size_t end = j;
      if (j < n && end > start + 1) --end;   // ASCII space assumption is safe here
      out.push_back(text.substr(start, end - start));
      i = end;
      continue;
    }

    out.push_back(text.substr(i, cl ? cl : 1));
    i += cl ? cl : 1;
  }
  return out;
}

// Merge the piece down using the rank table: repeatedly glue the adjacent pair
// with the lowest rank until nothing in the table is left.
inline void bpe_piece(const Tokenizer &tk, const std::string &piece,
                      std::vector<int> &out) {
  // start from single byte-level characters
  std::vector<std::string> sym;
  for (size_t i = 0; i < piece.size();) {
    size_t j = i;
    utf8_next(piece, j);
    sym.push_back(piece.substr(i, j - i));
    i = j;
  }
  if (sym.empty()) return;

  for (;;) {
    int best = INT32_MAX;
    size_t at = 0;
    for (size_t i = 0; i + 1 < sym.size(); ++i) {
      auto it = tk.ranks.find(sym[i] + '\x01' + sym[i + 1]);
      if (it != tk.ranks.end() && it->second < best) { best = it->second; at = i; }
    }
    if (best == INT32_MAX) break;
    sym[at] += sym[at + 1];
    sym.erase(sym.begin() + (long)at + 1);
  }
  for (const std::string &s : sym) {
    const int id = tk.id_of(s);
    if (id >= 0) out.push_back(id);
    else for (size_t k = 0; k < s.size(); ++k) {
      // Should be unreachable: every single byte-level char is in the vocab.
      const int b = tk.id_of(std::string(1, s[k]));
      if (b >= 0) out.push_back(b);
    }
  }
}

inline std::vector<int> tokenizer_encode(const Tokenizer &tk, const std::string &raw) {
  const std::string text = nfc(raw);
  std::vector<int> ids;
  size_t pos = 0;
  while (pos < text.size()) {
    // special tokens win over everything, matched longest-first
    size_t hit = std::string::npos;
    int hit_id = -1, hit_len = 0;
    for (const auto &sp : tk.specials) {
      const size_t p = text.find(sp.first, pos);
      if (p != std::string::npos && (hit == std::string::npos || p < hit ||
                                     (p == hit && (int)sp.first.size() > hit_len))) {
        hit = p; hit_id = sp.second; hit_len = (int)sp.first.size();
      }
    }
    const size_t upto = (hit == std::string::npos) ? text.size() : hit;
    if (upto > pos) {
      const std::string chunk = text.substr(pos, upto - pos);
      for (const std::string &piece : pretokenize(chunk)) {
        std::string mapped;
        for (size_t k = 0; k < piece.size(); ++k)
          utf8_append(mapped, tk.b2u[(unsigned char)piece[k]]);
        bpe_piece(tk, mapped, ids);
      }
    }
    if (hit == std::string::npos) break;
    ids.push_back(hit_id);
    pos = hit + (size_t)hit_len;
  }
  return ids;
}

inline std::string tokenizer_decode(const Tokenizer &tk, const std::vector<int> &ids) {
  std::string bytes;
  for (int id : ids) {
    if (id < 0 || id >= (int)tk.id2tok.size()) continue;
    const std::string &t = tk.id2tok[(size_t)id];
    bool special = false;
    for (const auto &sp : tk.specials)
      if (sp.second == id) { special = true; break; }
    if (special) { bytes += t; continue; }
    for (size_t i = 0; i < t.size();) {
      const uint32_t c = utf8_next(t, i);
      auto it = tk.u2b.find(c);
      bytes += (char)(it == tk.u2b.end() ? '?' : it->second);
    }
  }
  return bytes;
}



// Parse the blob the converter baked into qwengine.bin.
inline bool tokenizer_parse(Tokenizer &tk, const uint8_t *b, size_t n) {
  if (n < 16) return false;
  auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, b + o, 4); return v; };
  if (u32(0) != 0x314B4F54u) return false;          // "TOK1"
  const uint32_t nv = u32(4), nm = u32(8), ns = u32(12);
  size_t p = 16;
  auto read_section = [&](uint32_t count, std::vector<std::string> &out) {
    const size_t off0 = p;
    p += (size_t)(count + 1) * 4;
    const size_t base = p;
    const uint32_t total = u32(off0 + (size_t)count * 4);
    out.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t a = u32(off0 + (size_t)i * 4), c = u32(off0 + (size_t)(i + 1) * 4);
      out[i].assign((const char *)b + base + a, c - a);
    }
    p = base + total;
  };
  read_section(nv, tk.id2tok);
  std::vector<std::string> merges;
  read_section(nm, merges);
  std::vector<uint32_t> sids(ns);
  for (uint32_t i = 0; i < ns; ++i) sids[i] = u32(p + (size_t)i * 4);
  p += (size_t)ns * 4;
  std::vector<std::string> snames;
  read_section(ns, snames);

  tk.tok2id.reserve(tk.id2tok.size() * 2);
  for (size_t i = 0; i < tk.id2tok.size(); ++i)
    if (!tk.id2tok[i].empty()) tk.tok2id.emplace(tk.id2tok[i], (int)i);
  tk.ranks.reserve(merges.size() * 2);
  for (size_t i = 0; i < merges.size(); ++i) tk.ranks.emplace(merges[i], (int)i);
  for (uint32_t i = 0; i < ns; ++i) tk.specials.emplace_back(snames[i], (int)sids[i]);
  // longest first so <|im_start|> never loses to a shorter prefix
  std::sort(tk.specials.begin(), tk.specials.end(),
            [](const std::pair<std::string,int>&a, const std::pair<std::string,int>&b) {
              return a.first.size() > b.first.size();
            });

  int keys[256], vals[256], cnt = 0;
  byte_level_alphabet(tk.b2u, keys, vals, cnt);
  for (int i = 0; i < cnt; ++i) tk.u2b[(uint32_t)keys[i]] = vals[i];

  tk.eos = tk.id_of("<|im_end|>");
  tk.im_start = tk.id_of("<|im_start|>");
  tk.im_end = tk.eos;
  return true;
}

// Render the chat template the checkpoint ships with.
inline std::string chat_template(const std::string &user, bool think = true) {
  std::string s;
  s += "<|im_start|>system\nReasoning effort is set to xhigh. Please think "
       "carefully through the task, validate key assumptions, consider plausible "
       "alternatives, and prioritize correctness, consistency, and clarity in the "
       "final answer.<|im_end|>\n";
  s += "<|im_start|>user\n" + user + "<|im_end|>\n";
  s += "<|im_start|>assistant\n";
  if (think) s += "<think>\n";
  return s;
}

}  // namespace spark27
