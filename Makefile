# qwengine — hand-rolled inference engine for Qwen3.8-27B on DGX Spark.
#
# ARCH is explicit rather than `native` so a build is reproducible and can be
# produced on any DGX Spark for any other: every GB10 is sm_121. Override on
# the command line if that ever stops being true (`make ARCH=sm_XX`).
ARCH    ?= sm_121
NVCC    ?= nvcc
CXX     ?= g++
BUILD   ?= build

# -MMD -MP makes the compiler write each binary's real include dependencies
# into build/*.d, pulled in at the bottom of this file.
#
# This is not a nicety. Every source here is one translation unit that
# #includes the others, and these dependency lists used to be written by hand.
# Three times a hand-written list was missing a file, the binary silently did
# not rebuild, and a STALE BINARY REPORTED THE OLD BEHAVIOUR of a change that
# had just been made. Twice that produced a measurement that was then acted on.
# Generated dependencies remove the whole class of mistake.
DEPFLAGS  := -MMD -MP
NVCCFLAGS := -O3 -arch=$(ARCH) -std=c++17 -lineinfo -Xcompiler -Wall $(DEPFLAGS)
CXXFLAGS  := -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter -pthread $(DEPFLAGS)
LDFLAGS   := -lcuda

# `make bench` defaults; override e.g. `make bench GB=16`.
GB      ?= 12
REPS    ?= 7

BINS  := $(BUILD)/01_bandwidth $(BUILD)/02_readmax $(BUILD)/convert $(BUILD)/loader $(BUILD)/qwengine $(BUILD)/qwengine-serve
TESTS := $(BUILD)/test_nvfp4 $(BUILD)/test_loader $(BUILD)/test_gemv \
         $(BUILD)/test_spec $(BUILD)/test_gemm_tc $(BUILD)/test_vision $(BUILD)/test_tokenizer $(BUILD)/test_image \
         $(BUILD)/test_delta
TOOLS := $(BUILD)/gemv_shapes $(BUILD)/spec_probe $(BUILD)/spec_budget \
         $(BUILD)/tc_shapes $(BUILD)/vocab_probe $(BUILD)/prof_prefill

.PHONY: all test bench clean

# Tests and tools build by default: the other half of never running a stale
# binary is that `make` alone leaves nothing behind.
all: $(BINS) $(TESTS) $(TOOLS)

$(BUILD):
	@mkdir -p $(BUILD)

# ---- CUDA binaries (each is one self-contained translation unit) ------------
$(BUILD)/01_bandwidth: bench/01_bandwidth.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/02_readmax: bench/02_readmax.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/qwengine: src/main.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/qwengine-serve: src/server.cpp | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -x cu -o $@ $< $(LDFLAGS)
$(BUILD)/loader: src/loader.cpp | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -x cu -o $@ $< $(LDFLAGS)
$(BUILD)/test_loader: test/test_loader.cpp | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -x cu -o $@ $< $(LDFLAGS)
$(BUILD)/test_gemv: test/test_gemv.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/test_spec: test/test_spec.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/test_gemm_tc: test/test_gemm_tc.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/test_vision: test/test_vision.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/test_tokenizer: test/test_tokenizer.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/test_image: test/test_image.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/test_delta: test/test_delta.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/gemv_shapes: bench/gemv_shapes.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/spec_probe: test/spec_probe.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/spec_budget: bench/spec_budget.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/tc_shapes: bench/tc_shapes.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/prof_prefill: bench/prof_prefill.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)
$(BUILD)/vocab_probe: bench/vocab_probe.cu | $(BUILD)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

# ---- host-only --------------------------------------------------------------
# -Wno-unused-function: the test includes convert.cpp whole, so the converter's
# own file-scope helpers are legitimately unused there.
$(BUILD)/convert: src/convert.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $<
$(BUILD)/test_nvfp4: test/test_nvfp4.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Wno-unused-function -o $@ $<

test: all
	@rc=0; for t in $(TESTS); do \
	  echo "=== $$t ==="; $$t || rc=1; \
	done; \
	if [ $$rc -eq 0 ]; then echo "make test: ALL PASS"; else echo "make test: FAILURES"; fi; \
	exit $$rc

bench: all
	@tools/bench_append.sh $(GB) $(REPS) $(FILE)

clean:
	rm -rf $(BUILD)

-include $(BUILD)/*.d
