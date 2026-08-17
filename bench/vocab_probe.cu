// bench/vocab_probe.cu — how much of lm_head is real vocabulary?
//
// lm_head is read once per verify pass AND once per draft step, where it is
// three quarters of the bytes. If its row count is padded well past the real
// vocabulary, the padding is pure bandwidth cost on the hottest path there is.
#define SPARK27_NO_MAIN
#include "../qwen38.h"
#include "../src/loader.cpp"
#include "../src/tokenizer.cpp"
using namespace spark27;
int main() {
  Model m = load_model("out/qwengine.bin");
  Tokenizer tk;
  Tensor b = m.get("__tokenizer__");
  std::vector<uint8_t> blob(b.bytes);
  LCHECK(cudaMemcpy(blob.data(), b.dev, b.bytes, cudaMemcpyDeviceToHost));
  if (!tokenizer_parse(tk, blob.data(), blob.size())) { printf("parse failed\n"); return 1; }
  int maxspecial = 0;
  for (auto &s : tk.specials) maxspecial = std::max(maxspecial, s.second);
  printf("id2tok entries : %zu\n", tk.id2tok.size());
  printf("highest special: %d\n", maxspecial);
  printf("q38::VOCAB constant : %d   (lm_head rows)\n", q38::VOCAB);
  printf("eos            : %d\n", tk.eos);
  const int real = std::max((int)tk.id2tok.size(), maxspecial + 1);
  printf("\nreal vocabulary %d of %d rows -> %.1f%% of lm_head is padding "
         "(%.3f GB of %.3f)\n", real, q38::VOCAB, 100.0 * (q38::VOCAB - real) / q38::VOCAB,
         (q38::VOCAB - real) * 5120 * 0.5625 / 1e9, q38::VOCAB * 5120 * 0.5625 / 1e9);
  free_model(m);
  return 0;
}
