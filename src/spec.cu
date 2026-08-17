#pragma once
// src/spec.cu — speculative decoding: draft cheap, verify once, accept a run.
//
// THE CYCLE
//   1. DRAFT   the small head guesses the next D-1 tokens, one at a time,
//              feeding its own output back in.
//   2. VERIFY  the big model processes [committed, guess1 .. guessD-1] in ONE
//              batched pass. Because the pass is bound by streaming weights,
//              checking D tokens costs barely more than checking one.
//   3. ACCEPT  keep the longest run of guesses the big model agrees with, plus
//              one more token that the big model itself produced. Anything
//              after the first disagreement is thrown away.
//
// The big model has the final say at every position, so the tokens committed
// are EXACTLY the tokens plain greedy decoding would have produced. This buys
// speed and nothing else changes — that invariant is what makes speculation
// safe, and test/test_spec.cu checks it directly.
//
// The one piece of real bookkeeping is the DeltaNet state: it has already
// absorbed the rejected tokens by the time we find out they were wrong, and
// unlike the KV cache it cannot heal by being overwritten. See dn_replay.

#include "model.cu"
#include "mtp.cu"

namespace spark27 {

struct SpecStats {
  long cycles = 0, tokens = 0, drafts = 0, accepted = 0;
  double mean_accept() const { return cycles ? (double)tokens / cycles : 0.0; }
  double draft_hit() const { return drafts ? (double)accepted / drafts : 0.0; }
};

// Run one speculate/verify cycle.
//   last_token : the trunk's most recent prediction, not yet fed in
//   rt.hlast   : hidden state that produced it
// Appends the newly committed tokens to `out` and returns the new last_token.
inline int spec_step(Runtime &rt, MtpHead &M, int D, int last_token,
                     std::vector<int> &out, SpecStats &stats) {
  cudaStream_t st = rt.stream;
  const int nd = D - 1;  // number of guesses

  // ---- 1. draft ------------------------------------------------------------
  // The draft head walks the same token stream, so start it at the trunk's
  // position; rejected guesses just get overwritten in its own cache.
  k_set_pos<<<1, 1, 0, st>>>(M.dpos, rt.pos);
  int drafts[16];
  LCHECK(cudaMemcpyAsync(M.dtok_in, &last_token, sizeof(int), cudaMemcpyHostToDevice, st));
  LCHECK(cudaGraphLaunch(M.graph_exec, st));
  LCHECK(cudaMemcpyAsync(drafts, M.dchain, nd * sizeof(int), cudaMemcpyDeviceToHost, st));
  LCHECK(cudaStreamSynchronize(st));

  // ---- 2. verify -----------------------------------------------------------
  int batch[16];
  batch[0] = last_token;
  for (int d = 0; d < nd; ++d) batch[1 + d] = drafts[d];

  const int pos_before = rt.pos;
  dn_snapshot(rt);
  LCHECK(cudaMemcpyAsync(rt.dtok_in, batch, D * sizeof(int), cudaMemcpyHostToDevice, st));
  LCHECK(cudaGraphLaunch(rt.vgraph_exec, st));
  int outs[16];
  LCHECK(cudaMemcpyAsync(outs, rt.dtok_out, D * sizeof(int), cudaMemcpyDeviceToHost, st));
  LCHECK(cudaStreamSynchronize(st));

  // ---- 3. accept -----------------------------------------------------------
  // outs[j] is what the trunk predicts after batch element j. Guess d matches
  // if drafts[d] == outs[d].
  int k = 0;
  while (k < nd && drafts[k] == outs[k]) ++k;

  // Emit the token we came in with plus the accepted guesses. outs[k] is the
  // trunk's own next prediction: it is carried forward as the next cycle's
  // input and emitted then, exactly as the plain loop emits before stepping.
  // Emitting it here as well would shift the whole stream by one token.
  out.push_back(last_token);
  for (int i = 0; i < k; ++i) out.push_back(drafts[i]);
  const int produced = 1 + k;
  const int consumed = 1 + k;        // positions actually used

  stats.cycles++;
  stats.tokens += produced;
  stats.drafts += nd;
  stats.accepted += k;

  // ---- 4. rewind anything we did not keep ---------------------------------
  // The graph advanced the position by D and stepped the recurrence D times.
  k_set_pos<<<1, 1, 0, st>>>(rt.dpos, pos_before + consumed);
  if (consumed < D) dn_replay(rt, consumed);
  rt.pos = pos_before + consumed;

  // The next cycle continues from batch element k's hidden state.
  copy_vec(rt.hlast, rt.hlast + (size_t)k * HIDDEN, HIDDEN, st);
  LCHECK(cudaStreamSynchronize(st));

  return outs[k];
}

}  // namespace spark27
