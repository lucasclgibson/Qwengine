// test/test_tokenizer.cu — does our tokenizer agree with the reference?
//
// A single wrong id means the model reads a different sentence, so this is
// exact-match or nothing. The corpus deliberately includes the things that
// break hand-written tokenizers: CJK, emoji (including ZWJ sequences), RTL
// scripts, combining marks, full-width forms, whitespace runs, contractions,
// code, and the chat-template special tokens.
#define SPARK27_NO_MAIN
#include "../src/loader.cpp"
#include "../src/tokenizer.cpp"

#include <stdio.h>
#include <vector>

using namespace spark27;

// Minimal JSON array-of-{text,ids} reader for the golden file.
// Reads one string field by name out of the current object.
static std::string json_str_field(const std::string &s, size_t from, const char *key) {
  // Tolerate whitespace after the colon: json.dump writes `"dec": "..."`.
  const std::string k = std::string("\"") + key + "\"";
  size_t i = s.find(k, from);
  if (i == std::string::npos) return "";
  i += k.size();
  while (i < s.size() && (s[i] == ':' || s[i] == ' ')) ++i;
  if (i >= s.size() || s[i] != '"') return "";
  ++i;
  std::string t;
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\') {
      ++i;
      switch (s[i]) {
        case 'n': t += '\n'; break; case 't': t += '\t'; break;
        case 'r': t += '\r'; break; case '"': t += '"'; break;
        case '\\': t += '\\'; break;
        case 'u': {
          const uint32_t cp = (uint32_t)strtol(s.substr(i + 1, 4).c_str(), nullptr, 16);
          i += 4;
          if (cp >= 0xD800 && cp < 0xDC00 && s[i + 1] == '\\' && s[i + 2] == 'u') {
            const uint32_t lo = (uint32_t)strtol(s.substr(i + 3, 4).c_str(), nullptr, 16);
            i += 6;
            utf8_append(t, 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00));
          } else utf8_append(t, cp);
          break;
        }
        default: t += s[i];
      }
      ++i;
    } else t += s[i++];
  }
  return t;
}

static bool load_golden(const char *path, std::vector<std::string> &texts,
                        std::vector<std::vector<int>> &ids,
                        std::vector<std::string> &decs) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  std::string s;
  char buf[65536];
  size_t n;
  while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
  fclose(f);
  size_t i = 0;
  while ((i = s.find("{\"text\":", i)) != std::string::npos) {
    i += 8;
    while (i < s.size() && s[i] != '"') ++i;
    ++i;
    std::string t;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\') {
        ++i;
        switch (s[i]) {
          case 'n': t += '\n'; break; case 't': t += '\t'; break;
          case 'r': t += '\r'; break; case '"': t += '"'; break;
          case '\\': t += '\\'; break;
          case 'u': {
            const uint32_t cp = (uint32_t)strtol(s.substr(i + 1, 4).c_str(), nullptr, 16);
            i += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && s[i + 1] == '\\' && s[i + 2] == 'u') {
              const uint32_t lo = (uint32_t)strtol(s.substr(i + 3, 4).c_str(), nullptr, 16);
              i += 6;
              utf8_append(t, 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00));
            } else utf8_append(t, cp);
            break;
          }
          default: t += s[i];
        }
        ++i;
      } else t += s[i++];
    }
    texts.push_back(t);
    const size_t a = s.find('[', i), b = s.find(']', a);
    std::vector<int> v;
    for (size_t p = a + 1; p < b;) {
      while (p < b && (s[p] == ' ' || s[p] == ',')) ++p;
      if (p >= b) break;
      v.push_back(atoi(s.c_str() + p));
      while (p < b && s[p] != ',') ++p;
    }
    ids.push_back(v);
    // The reference's own decode output. decode(encode(x)) is NFC(x), not x,
    // so comparing against the input text would fail on decomposed input for
    // reasons that have nothing to do with our decoder.
    decs.push_back(json_str_field(s, b, "dec"));
    i = b;
  }
  return !texts.empty();
}

int main() {
  if (access("out/qwengine.bin", R_OK) != 0 || access("golden/tokenizer.json", R_OK) != 0) {
    printf("test_tokenizer: SKIP — need weights and golden/tokenizer.json\n");
    return 0;
  }
  Model m = load_model("out/qwengine.bin");
  if (!m.has("__tokenizer__")) {
    printf("test_tokenizer: SKIP — this build has no baked tokenizer\n");
    free_model(m);
    return 0;
  }
  Tensor t = m.get("__tokenizer__");
  std::vector<uint8_t> blob((size_t)t.bytes);
  LCHECK(cudaMemcpy(blob.data(), t.dev, blob.size(), cudaMemcpyDeviceToHost));
  Tokenizer tk;
  if (!tokenizer_parse(tk, blob.data(), blob.size())) {
    printf("test_tokenizer: FAIL — blob did not parse\n");
    return 1;
  }
  printf("tokenizer: %zu tokens, %zu merges, %zu special\n", tk.id2tok.size(),
         tk.ranks.size(), tk.specials.size());

  std::vector<std::string> texts;
  std::vector<std::vector<int>> want;
  std::vector<std::string> decs;
  if (!load_golden("golden/tokenizer.json", texts, want, decs)) {
    printf("test_tokenizer: FAIL — could not read golden\n");
    return 1;
  }

  int enc_ok = 0, dec_ok = 0, shown = 0, dshown = 0;
  for (size_t i = 0; i < texts.size(); ++i) {
    const std::vector<int> got = tokenizer_encode(tk, texts[i]);
    const bool ok = got == want[i];
    if (ok) ++enc_ok;
    else if (shown++ < 5) {
      printf("  MISMATCH on %-28s\n    want:", ("\"" + texts[i].substr(0, 26) + "\"").c_str());
      for (size_t k = 0; k < want[i].size() && k < 14; ++k) printf(" %d", want[i][k]);
      printf("\n    got :");
      for (size_t k = 0; k < got.size() && k < 14; ++k) printf(" %d", got[k]);
      printf("\n");
    }
    const std::string dg = tokenizer_decode(tk, want[i]);
    if (dg == decs[i]) ++dec_ok;
    else if (dshown++ < 4) {
      printf("  DECODE MISMATCH\n    want: %.70s\n    got : %.70s\n",
             decs[i].c_str(), dg.c_str());
    }
  }
  printf("  encode: %d/%zu exact\n", enc_ok, texts.size());
  printf("  decode: %d/%zu exact\n", dec_ok, texts.size());
  free_model(m);
  const bool pass = enc_ok == (int)texts.size() && dec_ok == (int)texts.size();
  printf("test_tokenizer: %s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
