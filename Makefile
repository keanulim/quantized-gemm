# Build INT8 GEMM test (requires nvcc + CUDA GPU)
#
# Usage:
#   make test
#   make run

NVCC ?= nvcc
CXXFLAGS = -std=c++17 -O2
INCLUDES = -Isrc/reference -Isrc/kernels
DEFINES = -DRUN_INT8_GEMM_TEST

TEST_SRCS = src/kernels/gemm.cu src/reference/gemm.cpp
TEST_BIN = build/test_int8_gemm

.PHONY: test run clean modal-test

test: $(TEST_BIN)

$(TEST_BIN): $(TEST_SRCS)
	@mkdir -p build
	$(NVCC) $(CXXFLAGS) $(INCLUDES) $(DEFINES) $(TEST_SRCS) -o $(TEST_BIN)

run: test
	./$(TEST_BIN)

# Run on Modal cloud GPU (works on Mac without local nvcc)
modal-test:
	modal run modal/test_gemm.py

clean:
	rm -rf build
