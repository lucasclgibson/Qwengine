// src/server.cpp — an OpenAI-compatible HTTP endpoint.
//
//   qwengine serve <weights.bin> [--port 8000] [--ctx 8192] [--depth 3]
//
// Endpoints:
//   GET  /health              liveness
//   GET  /v1/models           model list
//   POST /v1/chat/completions chat, streaming or not
//
// Deliberately a single-threaded, one-request-at-a-time server. The engine is
// single-stream: one decode loop owns the GPU and the KV cache, so accepting a
// second request concurrently would not make it faster, it would corrupt the
// first. Requests queue at the socket instead. Batched multi-slot serving is
// the natural next step and is what would make concurrency worth having.

#define SPARK27_NO_MAIN
#include "prefill.cu"
#include "spec.cu"
#include "image.cpp"
#include "tokenizer.cpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "../third_party/stb_image.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>
#include <vector>

using namespace spark27;

// ---------------------------------------------------------------------------
// Small JSON helpers. Requests are shallow, so a full parser is overkill --
// but string values must be unescaped correctly or a prompt containing a quote
// or a newline breaks.
// ---------------------------------------------------------------------------
static std::string json_escape(const std::string &s) {
  std::string o;
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
  }
  return o;
}

static std::string json_unescape(const std::string &s, size_t i, size_t end) {
  std::string t;
  while (i < end && i < s.size() && s[i] != '"') {
    if (s[i] == '\\' && i + 1 < s.size()) {
      ++i;
      switch (s[i]) {
        case 'n': t += '\n'; break; case 't': t += '\t'; break;
        case 'r': t += '\r'; break; case 'b': t += '\b'; break;
        case 'f': t += '\f'; break;
        case 'u': {
          uint32_t cp = (uint32_t)strtol(s.substr(i + 1, 4).c_str(), nullptr, 16);
          i += 4;
          if (cp >= 0xD800 && cp < 0xDC00 && i + 6 < s.size() && s[i + 1] == '\\') {
            const uint32_t lo = (uint32_t)strtol(s.substr(i + 3, 4).c_str(), nullptr, 16);
            i += 6;
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8_append(t, cp);
          break;
        }
        default: t += s[i];
      }
      ++i;
    } else t += s[i++];
  }
  return t;
}

// Find "key" then return its string value.
static std::string json_get_str(const std::string &s, const char *key, size_t from = 0) {
  const std::string k = std::string("\"") + key + "\"";
  size_t i = s.find(k, from);
  if (i == std::string::npos) return "";
  i += k.size();
  while (i < s.size() && (s[i] == ':' || s[i] == ' ')) ++i;
  if (i >= s.size() || s[i] != '"') return "";
  return json_unescape(s, i + 1, s.size());
}

static long json_get_num(const std::string &s, const char *key, long dflt) {
  const std::string k = std::string("\"") + key + "\"";
  size_t i = s.find(k);
  if (i == std::string::npos) return dflt;
  i += k.size();
  while (i < s.size() && (s[i] == ':' || s[i] == ' ')) ++i;
  if (i >= s.size() || (!isdigit((unsigned char)s[i]) && s[i] != '-')) return dflt;
  return strtol(s.c_str() + i, nullptr, 10);
}

static bool json_get_bool(const std::string &s, const char *key, bool dflt) {
  const std::string k = std::string("\"") + key + "\"";
  size_t i = s.find(k);
  if (i == std::string::npos) return dflt;
  i += k.size();
  while (i < s.size() && (s[i] == ':' || s[i] == ' ')) ++i;
  return s.compare(i, 4, "true") == 0 ? true : (s.compare(i, 5, "false") == 0 ? false : dflt);
}

static const std::string kB64 =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static std::vector<uint8_t> b64_decode(const std::string &in) {
  std::vector<int> rev(256, -1);
  for (int i = 0; i < 64; ++i) rev[(unsigned char)kB64[i]] = i;
  std::vector<uint8_t> out;
  int val = 0, bits = 0;
  for (unsigned char c : in) {
    if (rev[c] < 0) continue;
    val = (val << 6) | rev[c];
    bits += 6;
    if (bits >= 8) { bits -= 8; out.push_back((uint8_t)((val >> bits) & 0xFF)); }
  }
  return out;
}

// Collect every image_url in a message's content, whether content is a plain
// string (none) or the array form the vision API uses.
static std::vector<std::string> extract_images(const std::string &obj) {
  std::vector<std::string> urls;
  size_t p = 0;
  const std::string key = "\"url\"";
  while ((p = obj.find(key, p)) != std::string::npos) {
    p += key.size();
    while (p < obj.size() && (obj[p] == ':' || obj[p] == ' ')) ++p;
    if (p < obj.size() && obj[p] == '"') urls.push_back(json_unescape(obj, p + 1, obj.size()));
    ++p;
  }
  return urls;
}

// Pull out the messages array as (role, content) pairs, in order.
static std::vector<std::pair<std::string, std::string>> parse_messages(
    const std::string &s, std::vector<std::string> &raw_objs) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t i = s.find("\"messages\"");
  if (i == std::string::npos) return out;
  i = s.find('[', i);
  if (i == std::string::npos) return out;
  int depth = 0;
  for (size_t p = i; p < s.size(); ++p) {
    if (s[p] == '[') ++depth;
    else if (s[p] == ']') { if (--depth == 0) break; }
    else if (s[p] == '{') {
      const size_t obj = p;
      int d2 = 0;
      size_t q = p;
      for (; q < s.size(); ++q) {
        if (s[q] == '{') ++d2;
        else if (s[q] == '}') { if (--d2 == 0) break; }
      }
      const std::string o = s.substr(obj, q - obj + 1);
      // content may be a plain string, or the array form used for vision. In
      // the array form take every "text" field as the text.
      std::string role = json_get_str(o, "role"), text;
      const size_t ci = o.find("\"content\"");
      if (ci != std::string::npos) {
        size_t j = ci + 9;
        while (j < o.size() && (o[j] == ':' || o[j] == ' ')) ++j;
        if (j < o.size() && o[j] == '[') {
          size_t tp = j;
          const std::string tk = "\"text\"";
          while ((tp = o.find(tk, tp)) != std::string::npos) {
            tp += tk.size();
            while (tp < o.size() && (o[tp] == ':' || o[tp] == ' ')) ++tp;
            if (tp < o.size() && o[tp] == '"') {
              if (!text.empty()) text += "\n";
              text += json_unescape(o, tp + 1, o.size());
            }
            ++tp;
          }
        } else {
          text = json_get_str(o, "content");
        }
      }
      out.emplace_back(role, text);
      raw_objs.push_back(o);
      p = q;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Chat template. Matches the checkpoint's chat_template.jinja: a system turn,
// then alternating user/assistant turns, then an open assistant turn. `think`
// opens a <think> block, which is the checkpoint's default behaviour.
// ---------------------------------------------------------------------------
static const char *kDefaultSystem =
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer.";

static std::string render_chat(
    const std::vector<std::pair<std::string, std::string>> &msgs, bool think) {
  std::string s;
  bool have_system = false;
  for (const auto &m : msgs)
    if (m.first == "system") have_system = true;
  // Only inject the reasoning system prompt when thinking is wanted. Leaving
  // it in with enable_thinking=false makes the model emit a <think> block
  // anyway, because the prompt itself is what asks for one.
  if (!have_system && think)
    s += std::string("<|im_start|>system\n") + kDefaultSystem + "<|im_end|>\n";
  // An assistant turn must be replayed EXACTLY as it was generated, opener and
  // all. The opener below is part of the prompt the model answered, so leaving
  // it out of the history rewrites what the model was actually shown -- and,
  // less obviously, it moves the point where consecutive prompts stop agreeing
  // to before the first reply. That destroys the prefix cache on every turn of
  // every conversation, which is worth about 10 seconds a message.
  const char *opener = think ? "<think>\n" : "<think>\n\n</think>\n\n";
  for (const auto &m : msgs) {
    s += "<|im_start|>" + m.first + "\n";
    if (m.first == "assistant") s += opener;
    s += m.second + "<|im_end|>\n";
  }
  s += "<|im_start|>assistant\n";
  // Thinking is this checkpoint's default behaviour, so it is not enough to
  // omit the <think> opener -- the model writes one itself. To suppress it,
  // hand it an ALREADY-CLOSED empty think block, which is the convention this
  // model family uses to signal "reasoning done, answer now".
  s += opener;
  return s;
}

// ---------------------------------------------------------------------------
static bool send_all(int fd, const char *b, size_t n) {
  while (n) {
    const ssize_t w = write(fd, b, n);
    if (w <= 0) return false;
    b += w;
    n -= (size_t)w;
  }
  return true;
}
static bool send_str(int fd, const std::string &s) { return send_all(fd, s.data(), s.size()); }

static void send_json(int fd, int code, const std::string &body) {
  char h[256];
  snprintf(h, sizeof h,
           "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\n"
           "Access-Control-Allow-Origin: *\r\nContent-Length: %zu\r\n"
           "Connection: close\r\n\r\n",
           code, code == 200 ? "OK" : "Error", body.size());
  send_str(fd, h);
  send_str(fd, body);
}

struct Engine {
  Model m;
  Runtime rt;
  PrefillBuf pb;
  MtpHead mtp;
  Tokenizer tk;
  VisionTower vt;
  bool have_vision = false;
  float *img_dev = nullptr;      // spliced embeddings [n_img_tokens, HIDDEN]
  int *islot_dev = nullptr;      // where each one goes in the prompt
  int *mpos_dev = nullptr;       // (t,h,w) per prompt token
  int depth = 3, ctx = 8192;
  bool spec = true;
  std::string model_name = MODEL_NAME;

  // PREFIX CACHE.
  //
  // The exact token sequence the engine's state currently reflects: the last
  // request's prompt followed by every token the model actually consumed while
  // answering it. KV cache, DeltaNet recurrence and rt.pos all correspond to
  // having read exactly this.
  //
  // A chat client re-sends the whole conversation on every turn, so turn N's
  // prompt is turn N-1's prompt, plus the reply the model just generated, plus
  // the new user message. That means `cached` is a strict prefix of it and only
  // the new tail has to be read. Without this the engine re-read the entire
  // history every message, and time-to-first-token grew without limit: 30
  // seconds at 5400 tokens of context, against SGLang's 1.3.
  std::vector<int> cached;
  bool cache_valid = false;
};

static double now_s() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

// Decode every image in the request, run the tower, and build the prompt with
// vision placeholders spliced in.
//
// The prompt layout the model expects around a picture is
//   <|vision_start|> <|image_pad|> x M <|vision_end|>
// where M is the number of merged patches. Those placeholder embeddings are
// then overwritten with the tower's output.
struct VisionPrep {
  std::vector<float> embeds;     // [total_tokens, HIDDEN]
  std::vector<int> slots;        // prompt index for each
  std::vector<int> mpos;         // (t,h,w) per prompt token
  int n = 0;
  std::string error;
};

static bool prep_vision(Engine &e, const std::vector<std::string> &raw_objs,
                        const std::vector<std::pair<std::string, std::string>> &msgs,
                        bool think, std::vector<int> &prompt, VisionPrep &vp) {
  // Gather images in message order.
  struct Img { ImageTokens t; };
  std::vector<Img> imgs;
  for (const std::string &o : raw_objs) {
    for (const std::string &url : extract_images(o)) {
      const size_t c = url.find("base64,");
      if (c == std::string::npos) {
        vp.error = "only data: URLs with base64 image data are supported";
        return false;
      }
      const std::vector<uint8_t> raw = b64_decode(url.substr(c + 7));
      int w = 0, h = 0, ch = 0;
      uint8_t *px = stbi_load_from_memory(raw.data(), (int)raw.size(), &w, &h, &ch, 3);
      if (!px) { vp.error = "could not decode image"; return false; }
      Img im;
      im.t = preprocess_image(px, h, w);
      stbi_image_free(px);
      if (im.t.n_patches > e.vt.max_patches) {
        vp.error = "image too large for this build";
        return false;
      }
      imgs.push_back(std::move(im));
    }
  }
  if (imgs.empty()) return true;
  if (!e.have_vision) { vp.error = "this build has no vision tower"; return false; }

  // Build the text with placeholders, then tokenise it.
  const int VSTART = e.tk.id_of("<|vision_start|>");
  const int VEND = e.tk.id_of("<|vision_end|>");
  const int VPAD = e.tk.id_of("<|image_pad|>");
  if (VSTART < 0 || VEND < 0 || VPAD < 0) { vp.error = "vision tokens missing"; return false; }

  std::string text;
  bool have_system = false;
  for (const auto &m : msgs) if (m.first == "system") have_system = true;
  if (!have_system && think)
    text += std::string("<|im_start|>system\n") + kDefaultSystem + "<|im_end|>\n";
  size_t img_i = 0;
  for (size_t mi = 0; mi < msgs.size(); ++mi) {
    text += "<|im_start|>" + msgs[mi].first + "\n";
    const size_t nimg = extract_images(raw_objs[mi]).size();
    for (size_t k = 0; k < nimg; ++k, ++img_i) {
      text += "<|vision_start|>";
      for (int j = 0; j < imgs[img_i].t.n_tokens; ++j) text += "<|image_pad|>";
      text += "<|vision_end|>";
    }
    text += msgs[mi].second + "<|im_end|>\n";
  }
  text += "<|im_start|>assistant\n";
  text += think ? "<think>\n" : "<think>\n\n</think>\n\n";
  prompt = tokenizer_encode(e.tk, text);

  // Run each tower pass and record where its outputs go.
  vp.embeds.clear();
  vp.slots.clear();
  size_t cursor = 0;
  for (auto &im : imgs) {
    float *dpix = nullptr;
    LCHECK(cudaMalloc(&dpix, im.t.pixels.size() * 4));
    LCHECK(cudaMemcpy(dpix, im.t.pixels.data(), im.t.pixels.size() * 4,
                      cudaMemcpyHostToDevice));
    vision_forward(e.vt, dpix, im.t.pos.data(), im.t.bidx.data(), im.t.bwts.data(),
                   im.t.n_patches, 0);
    LCHECK(cudaDeviceSynchronize());
    std::vector<float> got((size_t)im.t.n_tokens * HIDDEN);
    LCHECK(cudaMemcpy(got.data(), e.vt.out, got.size() * 4, cudaMemcpyDeviceToHost));
    cudaFree(dpix);
    vp.embeds.insert(vp.embeds.end(), got.begin(), got.end());
    // find the next run of image_pad in the prompt
    while (cursor < prompt.size() && prompt[cursor] != VPAD) ++cursor;
    for (int j = 0; j < im.t.n_tokens; ++j) vp.slots.push_back((int)cursor++);
  }
  vp.n = (int)vp.slots.size();

  // mrope positions: text advances by one per token; an image block advances
  // time by nothing and gives each patch its own (row, col).
  vp.mpos.assign(prompt.size() * 3, 0);
  int p = 0;
  size_t ii = 0, si = 0;
  for (size_t i = 0; i < prompt.size(); ++i) {
    if (prompt[i] == VPAD && ii < imgs.size()) {
      const int lh = imgs[ii].t.grid_h / 2, lw = imgs[ii].t.grid_w / 2;
      const int k = (int)si;
      vp.mpos[i * 3 + 0] = p;
      vp.mpos[i * 3 + 1] = p + k / lw;
      vp.mpos[i * 3 + 2] = p + k % lw;
      if (++si == (size_t)imgs[ii].t.n_tokens) {
        p += std::max(lh, lw);
        si = 0;
        ++ii;
      }
      continue;
    }
    vp.mpos[i * 3 + 0] = vp.mpos[i * 3 + 1] = vp.mpos[i * 3 + 2] = p++;
  }
  return true;
}

// Record the token sequence the engine's state now reflects: this prompt plus
// everything the model consumed answering it. Every token in `out` was fed --
// spec_step advances the position by exactly the number it commits -- so the
// two concatenated are precisely what the KV cache and the recurrence have
// seen. `next` is deliberately excluded: it has been predicted but not fed.
static void remember(Engine &e, const std::vector<int> &prompt,
                     const std::vector<int> &out) {
  e.cached = prompt;
  e.cached.insert(e.cached.end(), out.begin(), out.end());
  e.cache_valid = true;
}

// Run one completion. Calls `on_token` with each decoded text delta.
template <typename F>
static int generate(Engine &e, const std::vector<int> &prompt, int max_new,
                    F on_token, double *ttft, double *gen_s, SpecStats &stats,
                    const VisionPrep *vp = nullptr) {
  // How much of this prompt is already in the engine's state?
  //
  // Only a WHOLE-CACHE match counts. The KV cache could be truncated to any
  // point, but the DeltaNet recurrence cannot: its state is a running summary
  // with no history to roll back to, so the only position we can resume from is
  // the one the state is already at. A prompt that diverges anywhere -- an
  // edited message, a regeneration, a different chat -- falls back to a full
  // read, which is what used to happen every single time.
  size_t reuse = 0;
  // vp is passed unconditionally by the caller and is only meaningful when it
  // holds images, so test its contents, not the pointer.
  const bool has_images = vp && vp->n > 0;
  if (e.cache_valid && !has_images && !e.cached.empty() && prompt.size() > e.cached.size() &&
      prompt.size() < (size_t)e.ctx &&
      std::equal(e.cached.begin(), e.cached.end(), prompt.begin()))
    reuse = e.cached.size();
  if (reuse)
    printf("  prefix cache: %zu of %zu prompt tokens already read\n", reuse,
           prompt.size());

  if (!reuse) {
    rt_reset(e.rt);
    if (e.spec) LCHECK(cudaMemset(e.mtp.dpos, 0, sizeof(int)));
  }
  e.cache_valid = false;          // set again only on a clean finish
  const double t0 = now_s();
  int next;
  if (vp && vp->n > 0) {
    // A prompt containing images goes through in ONE chunk: the spliced
    // embeddings and the 3-D positions are indexed against the whole prompt,
    // so splitting it would need the offsets re-based per chunk.
    LCHECK(cudaMemcpy(e.img_dev, vp->embeds.data(), vp->embeds.size() * 4,
                      cudaMemcpyHostToDevice));
    LCHECK(cudaMemcpy(e.islot_dev, vp->slots.data(), vp->slots.size() * 4,
                      cudaMemcpyHostToDevice));
    LCHECK(cudaMemcpy(e.mpos_dev, vp->mpos.data(), vp->mpos.size() * 4,
                      cudaMemcpyHostToDevice));
    LCHECK(cudaMemcpyAsync(e.pb.dtok, prompt.data(), prompt.size() * sizeof(int),
                           cudaMemcpyHostToDevice, e.rt.stream));
    prefill_chunk(e.rt, e.pb, (int)prompt.size(), e.img_dev, e.islot_dev, vp->n,
                  e.mpos_dev);
    LCHECK(cudaMemcpy(&next, e.rt.dtok_out, sizeof(int), cudaMemcpyDeviceToHost));
  } else if (reuse) {
    // Continue from where the state already is: read only the new tail.
    const std::vector<int> tail(prompt.begin() + reuse, prompt.end());
    next = prefill(e.rt, e.pb, tail);
  } else {
    next = prefill(e.rt, e.pb, prompt);
  }
  *ttft = now_s() - t0;

  const double t1 = now_s();
  std::vector<int> out;
  int produced = 0;
  size_t emitted = 0;
  while (produced < max_new) {
    if (e.spec) {
      const size_t before = out.size();
      next = spec_step(e.rt, e.mtp, e.depth, next, out, stats);
      produced += (int)(out.size() - before);
    } else {
      out.push_back(next);
      next = step(e.rt, next);
      ++produced;
    }
    // Emit only whole tokens we have not sent, and stop at the end marker.
    while (emitted < out.size()) {
      const int id = out[emitted++];
      if (id == e.tk.eos) {
        *gen_s = now_s() - t1;
        remember(e, prompt, out);
        return (int)emitted - 1;
      }
      on_token(tokenizer_decode(e.tk, {id}));
    }
  }
  *gen_s = now_s() - t1;
  remember(e, prompt, out);
  return produced;
}

// ---------------------------------------------------------------------------
static void handle_chat(Engine &e, int fd, const std::string &body) {
  std::vector<std::string> raw_objs;
  const auto msgs = parse_messages(body, raw_objs);
  if (msgs.empty()) {
    send_json(fd, 400, "{\"error\":{\"message\":\"no messages\",\"type\":\"invalid_request_error\"}}");
    return;
  }
  const bool stream = json_get_bool(body, "stream", false);
  const long max_new = json_get_num(body, "max_tokens", 512);
  // The checkpoint thinks by default; allow turning it off like SGLang does.
  const bool think = json_get_bool(body, "enable_thinking", true);

  std::vector<int> prompt;
  VisionPrep vp;
  if (!prep_vision(e, raw_objs, msgs, think, prompt, vp)) {
    send_json(fd, 400, "{\"error\":{\"message\":\"" + json_escape(vp.error) +
                           "\",\"type\":\"invalid_request_error\"}}");
    return;
  }
  if (prompt.empty()) prompt = tokenizer_encode(e.tk, render_chat(msgs, think));
  if (vp.n > 0 && (int)prompt.size() > e.pb.chunk) {
    send_json(fd, 400,
              "{\"error\":{\"message\":\"prompt with images exceeds the single-chunk "
              "limit; start the server with a larger --imgchunk\",\"type\":"
              "\"invalid_request_error\"}}");
    return;
  }
  if ((int)prompt.size() + max_new + 8 > e.ctx) {
    send_json(fd, 400,
              "{\"error\":{\"message\":\"prompt plus max_tokens exceeds the "
              "context this server was started with\",\"type\":\"invalid_request_error\"}}");
    return;
  }

  const long created = (long)time(nullptr);
  char idbuf[64];
  snprintf(idbuf, sizeof idbuf, "chatcmpl-%ld", created);
  double ttft = 0, gen_s = 0;
  SpecStats stats;

  if (stream) {
    send_str(fd,
             "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
             "Cache-Control: no-cache\r\nConnection: close\r\n"
             "Access-Control-Allow-Origin: *\r\n\r\n");
    {
      char h[512];
      snprintf(h, sizeof h,
               "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,"
               "\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
               "\"finish_reason\":null}]}\n\n",
               idbuf, created, e.model_name.c_str());
      send_str(fd, h);
    }
    bool alive = true;
    const int n = generate(e, prompt, (int)max_new, [&](const std::string &piece) {
      if (!alive) return;
      std::string c = "data: {\"id\":\"" + std::string(idbuf) +
                      "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                      std::to_string(created) + ",\"model\":\"" + e.model_name +
                      "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
                      json_escape(piece) + "\"},\"finish_reason\":null}]}\n\n";
      if (!send_str(fd, c)) alive = false;
    }, &ttft, &gen_s, stats, &vp);
    std::string tail = "data: {\"id\":\"" + std::string(idbuf) +
                       "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                       std::to_string(created) + ",\"model\":\"" + e.model_name +
                       "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                       "data: [DONE]\n\n";
    send_str(fd, tail);
    printf("  %d tok | ttft %.2fs | %.2f tok/s | accept %.2f\n", n, ttft,
           gen_s > 0 ? n / gen_s : 0, stats.mean_accept());
    return;
  }

  std::string full;
  const int n = generate(e, prompt, (int)max_new,
                         [&](const std::string &piece) { full += piece; },
                         &ttft, &gen_s, stats, &vp);
  std::string body_out =
      "{\"id\":\"" + std::string(idbuf) + "\",\"object\":\"chat.completion\",\"created\":" +
      std::to_string(created) + ",\"model\":\"" + e.model_name +
      "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" +
      json_escape(full) + "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":" +
      std::to_string(prompt.size()) + ",\"completion_tokens\":" + std::to_string(n) +
      ",\"total_tokens\":" + std::to_string(prompt.size() + (size_t)n) + "}}";
  send_json(fd, 200, body_out);
  printf("  %d tok | ttft %.2fs | %.2f tok/s | accept %.2f\n", n, ttft,
         gen_s > 0 ? n / gen_s : 0, stats.mean_accept());
}

int main(int argc, char **argv) {
  const char *weights = argc > 1 ? argv[1] : "out/qwengine.bin";
  // Draft depth 4, not 3. Measured on this machine: depth 3 accepts 2.22
  // tokens per cycle and depth 4 accepts 2.61, and the extra verify row costs
  // far less than the extra tokens are worth (23.7 -> 24.9 tok/s). Depth 5
  // accepts 2.86 but the cycle grows faster than that, and depth 6 accepts no
  // more at all -- the draft chain has run out of accuracy by then.
  int port = 8000, ctx = 8192, depth = 4, imgchunk = 2048;
  const char *served = nullptr;
  for (int i = 2; i < argc; ++i) {
    if (!strcmp(argv[i], "--port") && i + 1 < argc) port = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) ctx = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--depth") && i + 1 < argc) depth = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--imgchunk") && i + 1 < argc) imgchunk = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--served-model-name") && i + 1 < argc) served = argv[++i];
    else if (!strcmp(argv[i], "--help")) {
      printf("usage: %s <weights.bin> [--port N] [--ctx N] [--depth N]\n"
             "                      [--served-model-name NAME]\n"
             "  --depth 1 disables speculative decoding\n"
             "  --served-model-name overrides the id reported by /v1/models\n"
             "                      (default %s)\n", argv[0], MODEL_NAME);
      return 0;
    }
  }
  signal(SIGPIPE, SIG_IGN);
  // A server's log is useless if it only appears when the buffer fills, which
  // is what happens the moment stdout is a pipe or a file instead of a tty.
  setlinebuf(stdout);

  Engine e;
  e.ctx = ctx;
  e.depth = depth;
  e.spec = depth > 1;
  if (served) e.model_name = served;
  printf("qwengine: loading %s\n", weights);
  e.m = load_model(weights);
  rt_init(e.rt, e.m, ctx);
  // A prompt with images is prefilled in one chunk, so the chunk must be able
  // to hold it. 2048 covers a 1024x1024 picture plus a paragraph of text.
  prefill_init(e.pb, imgchunk, ctx);
  e.have_vision = vision_init(e.vt, e.m, 4096);
  if (e.have_vision) {
    LCHECK(cudaMalloc(&e.img_dev, (size_t)4096 / 4 * HIDDEN * sizeof(float)));
    LCHECK(cudaMalloc(&e.islot_dev, (size_t)4096 / 4 * sizeof(int)));
    LCHECK(cudaMalloc(&e.mpos_dev, (size_t)imgchunk * 3 * sizeof(int)));
  } else {
    printf("vision: not present in this weights file (text only)\n");
  }
  if (e.spec) {
    rt_capture_verify(e.rt, depth);
    mtp_init(e.mtp, e.m, ctx);
    mtp_capture(e.mtp, e.rt.embed, e.rt.s_embed, e.rt.lm_head, e.rt.s_lm,
                e.rt.hlast, depth - 1, e.rt.stream);
  }
  if (!e.m.has("__tokenizer__")) {
    fprintf(stderr, "this weights file has no tokenizer; reconvert it\n");
    return 1;
  }
  {
    Tensor t = e.m.get("__tokenizer__");
    std::vector<uint8_t> blob((size_t)t.bytes);
    LCHECK(cudaMemcpy(blob.data(), t.dev, blob.size(), cudaMemcpyDeviceToHost));
    if (!tokenizer_parse(e.tk, blob.data(), blob.size())) {
      fprintf(stderr, "tokenizer blob did not parse\n");
      return 1;
    }
  }

  const int srv = socket(AF_INET, SOCK_STREAM, 0);
  int one = 1;
  setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  sockaddr_in a{};
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = INADDR_ANY;
  a.sin_port = htons((uint16_t)port);
  if (bind(srv, (sockaddr *)&a, sizeof a) < 0) { perror("bind"); return 1; }
  if (listen(srv, 16) < 0) { perror("listen"); return 1; }
  printf("qwengine: serving %s\n", e.model_name.c_str());
  printf("qwengine: ready on http://0.0.0.0:%d  (ctx %d, draft depth %d%s)\n",
         port, ctx, depth, e.spec ? "" : ", speculation off");
  printf("  POST /v1/chat/completions   GET /v1/models   GET /health\n");

  for (;;) {
    const int fd = accept(srv, nullptr, nullptr);
    if (fd < 0) continue;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    std::string req;
    char buf[8192];
    ssize_t n;
    size_t need = std::string::npos;
    while ((n = read(fd, buf, sizeof buf)) > 0) {
      req.append(buf, (size_t)n);
      if (need == std::string::npos) {
        const size_t h = req.find("\r\n\r\n");
        if (h != std::string::npos) {
          size_t cl = 0;
          const size_t p = req.find("Content-Length:");
          if (p != std::string::npos) cl = (size_t)strtoul(req.c_str() + p + 15, nullptr, 10);
          need = h + 4 + cl;
        }
      }
      if (need != std::string::npos && req.size() >= need) break;
    }
    const size_t hend = req.find("\r\n\r\n");
    const std::string head = hend == std::string::npos ? req : req.substr(0, hend);
    const std::string body = hend == std::string::npos ? "" : req.substr(hend + 4);

    if (head.compare(0, 8, "OPTIONS ") == 0) {
      send_str(fd, "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                   "Access-Control-Allow-Headers: *\r\nAccess-Control-Allow-Methods: *\r\n"
                   "Content-Length: 0\r\nConnection: close\r\n\r\n");
    } else if (head.find("GET /health") == 0) {
      send_json(fd, 200, "{\"status\":\"ok\"}");
    } else if (head.find("GET /v1/models") == 0) {
      send_json(fd, 200,
                "{\"object\":\"list\",\"data\":[{\"id\":\"" + e.model_name +
                    "\",\"object\":\"model\",\"owned_by\":\"qwen\",\"max_model_len\":" +
                    std::to_string(ctx) + "}]}");
    } else if (head.find("POST /v1/chat/completions") == 0) {
      printf("request: %zu bytes\n", body.size());
      handle_chat(e, fd, body);
    } else {
      send_json(fd, 404, "{\"error\":{\"message\":\"not found\"}}");
    }
    close(fd);
  }
}
