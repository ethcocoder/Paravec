# Paravec Implementation Plan

Stack for the prototype: Python 3.10+, NumPy, SciPy, scikit-learn (k-means), matplotlib. Baselines through **Faiss** if installable (flat, IVF-PQ, OPQ, scalar quantizer). Later: C++ or Rust with SIMD for speed.

## Repo layout

```
paravec/
  paravec/
    __init__.py
    basis.py         # PCA basis fitting, local bases
    alloc.py         # bit allocation
    quant.py         # quantizer, bounds
    store.py         # packed layout
    search.py        # staged scan, IVF routing
    analyzer.py      # spectrum analyzer CLI
  bench/
    datasets.py
    baselines.py
    run_bench.py
  tests/
```

## Step 0: Spectrum analyzer first (days 1 to 3)
Before building an index, check whether the idea can work. For each candidate dataset:
1. Sample vectors, fit PCA, plot eigenvalue decay (log scale).
2. Compute `m_90`, `m_99`, participation ratio.
3. If the spectrum is nearly flat, stop for that dataset. This is a gate, not a formality.

## Step 1: Basis, allocation, quantizer (days 4 to 7)

```python
import heapq
import numpy as np

def fit_basis(X):
    mu = X.mean(axis=0)
    Xc = X - mu
    C = (Xc.T @ Xc) / len(Xc)
    lam, U = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    return mu, U[:, order], np.maximum(lam[order], 0.0)

def allocate_bits(lam, budget_bits, max_bits=8):
    b = np.zeros(len(lam), dtype=int)
    heap = [(-0.75 * lam[i], i) for i in range(len(lam))]
    heapq.heapify(heap)
    used = 0
    while used < budget_bits and heap:
        _, i = heapq.heappop(heap)
        b[i] += 1
        used += 1
        if b[i] < max_bits:
            heapq.heappush(heap, (-0.75 * lam[i] * 4.0 ** (-b[i]), i))
    return b

def quantize(C, lam, bits):
    sigma = np.sqrt(lam) + 1e-12
    L = (2 ** bits).astype(np.float64)
    step = np.where(bits > 0, 6 * sigma / np.maximum(L, 1), np.inf)
    codes = np.zeros(C.shape, dtype=np.int32)
    recon = np.zeros_like(C)
    m = bits > 0
    codes[:, m] = np.clip(np.floor((C[:, m] + 3 * sigma[m]) / step[m]), 0, L[m] - 1)
    recon[:, m] = -3 * sigma[m] + (codes[:, m] + 0.5) * step[m]
    return codes, recon, step

def estimate_sq_dist(q_spec, recon):
    return ((q_spec[None, :] - recon) ** 2).sum(axis=1)
```

**Tests (required):**
1. Isometry: with all modes unquantized, `||q−x||²` equals the spectral distance (to 1e-6).
2. Lower bound: partial sums over the first `m` exact modes never exceed the full distance.
3. Allocation: the allocated bits sum to the budget; higher-variance modes get at least as many bits.
4. Error bound: the deterministic bound holds for every sampled pair; probabilistic interval coverage is at least `1 − δ` (minus tolerance).

## Step 2: Flat-scan evaluation (days 8 to 11)
- Hold out queries; compute exact neighbors on float32.
- Measure recall@10 vs. bytes per vector for: Paravec (flat spectral, with and without exact rerank of a shortlist), scalar quantization (SQ8, SQ4), PQ/OPQ at the same bytes.
- Memory accounting includes basis overhead.

## Step 3: Staged scan (days 12 to 15)
- Implement stages with exact leading modes (lossless pruning) first, then quantized with margins.
- Measure the fraction of modes read per vector at fixed recall.

## Step 4: IVF and local bases (days 16 to 20)
- k-means cells, probe `nprobe` cells, local or shared bases.
- Compare global vs. local bases at equal total memory.

## Step 5: Drift and updates (days 21 to 22)
- Insert vectors from a shifted distribution; track residual energy and recall; test the refresh trigger.

## Step 6: Fast implementation (later)
Mode-major packed layout, SIMD distance kernels, bit-packing for variable widths. Port to C++/Rust once recall-per-byte results justify it. Keep Python as the reference implementation.

## Step 7: Packaging (later)
Python wheel with an index class (`build`, `add`, `search`), analyzer CLI, benchmark scripts.

## Definition of done for v0.1
- [ ] Spectrum analyzer results for all datasets (including negative ones)
- [ ] All four unit tests pass
- [ ] Recall-vs-bytes curves against SQ and OPQ baselines on at least three datasets
- [ ] Staged scan with measured work reduction
- [ ] Bound coverage report
- [ ] Honest README results

## Pitfalls
- Forgetting basis overhead in the memory accounting.
- Comparing to float32 only (SQ8 already gives 4× with tiny loss).
- Whitening embeddings (amplifies noise modes).
- Tuning on the query set (use separate tuning and test queries).
- Reporting QPS from a Python prototype as if it were a systems result.
