# Paravec Architecture

## 1. Overview

```
 build:   vectors ─► Clusterer ─► Basis Fitter ─► Bit Allocator ─► Quantizer ─► Packed Store
 query:   q ─► route to cells ─► rotate q into cell basis ─► Staged Scan ─► (optional exact rerank) ─► top-k
```

## 2. Components

### 2.1 Clusterer (geometry)
- IVF-style coarse clustering (k-means) into `k` cells. The cell id is the **geometry** that selects basis and allocation.
- Option: a single global basis (k = 1) as the simplest configuration.

### 2.2 Basis Fitter
- Per cell: mean `μ_c`, orthonormal basis `U_c` (top `r` principal modes, `r ≤ d`), eigenvalues `λ_c`.
- Basis storage is bounded: share bases among groups of cells, or keep rank `r ≤ 64` local bases plus one global basis for the rest of the spectrum.

### 2.3 Bit Allocator
- Given a per-vector bit budget `B`, assign bits `b_i` to modes using reverse water-filling (integer greedy version).
- Modes with `b_i = 0` are dropped.

### 2.4 Quantizer
- Uniform scalar quantizer per mode over `±3σ_i` with `2^{b_i}` levels.
- Optional correction scalar per vector: the quantization error norm `||e||²` (one 16-bit value) to de-bias distance estimates.

### 2.5 Packed Store
- **Mode-major blocks** (dimension-major within blocks of 256 or 1,024 vectors): the first modes of many vectors are contiguous, so stage-1 scans read little memory and use SIMD well.
- Header per cell: basis id, bit allocation vector, step sizes.
- Original vectors may be kept on slower storage for exact rerank (optional).

### 2.6 Staged Scan
Stage `s` uses modes `1..m_s` (for example 16, 64, 256):
1. Compute partial distances for all candidates in the probed cells.
2. Keep candidates whose estimate (minus a margin) is below the current `k`-th best estimate.
3. Continue to the next stage with fewer candidates.
4. Final stage: all stored modes (and optionally exact rerank from original vectors).

### 2.7 Spectrum Analyzer (tool)
Given a sample of embeddings, reports: eigenvalue decay curve, effective dimension (e.g. number of modes for 90% / 99% variance), predicted recall vs. bytes for several allocations, and a recommendation on whether Paravec is worthwhile.

## 3. Index modes

| Mode | Description |
|---|---|
| **Flat spectral** | Scan all vectors with staged pruning (good for evaluation and small sets) |
| **IVF spectral** | Probe a few cells, staged scan within cells (the practical mode) |
| **Graph + spectral codes** (v2) | Use spectral codes for distance estimates during graph traversal (HNSW/DiskANN-style) |

## 4. Updates and drift
- **Inserts:** quantize new vectors with the existing basis; track the fraction of variance outside the basis (residual energy).
- **Refresh trigger:** if residual energy or recall on a monitoring query set drifts past a threshold, refit the basis and requantize.
- **Model change:** a new embedding model means a full rebuild.

## 5. Design rules
1. Always report recall at equal **bytes per vector including basis overhead**.
2. Always compare with the best quantization baseline with exact rerank, not just float32.
3. Keep the exact lower-bound property testable (a unit test on unquantized coefficients).
