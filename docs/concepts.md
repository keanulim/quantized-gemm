# Core concepts (beginner)

## GEMM notation

`C[M×N] = A[M×K] × B[K×N]`

Each output element: `C[i,j] = sum_k A[i,k] * B[k,j]`

## Why quantize?

- **Memory**: INT8 uses 4× less than FP32
- **Speed**: smaller loads, SIMD-friendly integer ops
- **Deployment**: many accelerators (TPU, NPU) expect INT8/FP8

## Quantization formula (symmetric, per-tensor)

```
scale = max(|x|) / 127        # for INT8
q = round(x / scale)        # clamp to [-128, 127]
x ≈ q * scale               # dequantize
```

## INT8 GEMM with scales

If `A ≈ scale_a * A_q` and `B ≈ scale_b * B_q`:

```
C ≈ (scale_a * scale_b) * (A_q @ B_q)   # A_q @ B_q in int32
```

## FP8 vs INT8

- **INT8**: integer; needs explicit scale; very predictable
- **FP8**: float with tiny mantissa; hardware may support natively (H100, etc.)

## Key pitfalls

- Overflow in int32 accumulators (watch K dimension)
- Per-tensor vs per-channel scales (accuracy vs simplicity)
- Layout: row-major vs column-major affects kernel design
