# Paravec Experiments

Goal: find out whether spectral, variance-driven compression with staged search beats the best existing vector quantizers at equal memory. Record negative results.

## Hypotheses

| ID | Hypothesis | Falsified if |
|---|---|---|
| V1 | Real embedding sets have steep enough spectra for large truncation | Effective dimension `m_90` is close to `d` on most datasets |
| V2 | Variance-driven bit allocation beats uniform scalar quantization at equal bytes | No recall gain over SQ at equal bytes |
| V3 | Paravec matches or beats OPQ/PQ (with rerank) at equal bytes | OPQ/PQ win on every dataset |
| V4 | Staged pruning cuts distance work substantially at fixed recall | Fewer than 30% of coefficient reads saved |
| V5 | Local (per-cell) bases beat a global basis at equal total memory | No gain, or basis overhead cancels it |
| V6 | The distance-error bounds hold empirically | Coverage below nominal level |
| V7 | The refresh trigger keeps recall stable under drift | Recall collapses before the trigger fires |

## Datasets (public; check availability and licenses)
- **Classic ANN benchmarks:** SIFT1M (128-d), GIST1M (960-d), GloVe (100/200-d, angular), Deep1M
- **Modern text embeddings:** embeddings you generate yourself from public corpora (for example passages from BEIR or MS MARCO) with one or two public embedding models of different dimensions (384, 768, 1024+), plus any published million-scale embedding sets
- **Benchmark suites:** ANN-Benchmarks and big-ann-benchmarks protocols for fair comparison
- Include at least one dataset likely to have a **flat** spectrum to check the gate honestly

## Baselines (same memory budget)
1. float32 flat (exact, reference)
2. Scalar quantization: SQ8, SQ4
3. PQ and OPQ, with and without exact rerank of a shortlist
4. Binary quantization + rerank
5. **RaBitQ** (if an implementation is available)
6. **LeanVec**-style or PCA + SQ (learned dimensionality reduction + quantization)
7. Matryoshka-style truncation (only for embeddings trained for it)
8. IVF-PQ and HNSW as system-level references

## Metrics
- **Recall@10 (and @1, @100)** vs. **bytes per vector including basis overhead**
- QPS and latency at recall 0.90 / 0.95 / 0.99 (systems claims only for the optimized implementation)
- Fraction of coefficients read per vector (staged scan)
- Build time and memory during build
- Bound coverage (empirical miss rate vs. nominal `δ`)
- Drift: recall vs. fraction of inserted shifted vectors
- Seeds/queries: ≥ 1,000 held-out queries (separate tuning and test sets); report mean and spread across ≥ 3 random seeds for clustering

## Experiments

**Z0: Spectrum study (gate).** Eigenvalue decay, `m_90`, `m_99`, participation ratio for every dataset. Tests V1.

**Z1: Flat-scan recall vs. bytes.** Paravec (allocated bits) vs. SQ, OPQ, binary+rerank at bytes per vector ∈ {16, 32, 64, 128}. Tests V2, V3.

**Z2: Allocation ablation.** Uniform bits vs. greedy water-filling vs. water-filling with caps; effect of centering and (carefully) whitening.

**Z3: Staged pruning.** Exact leading modes (lossless) and quantized modes with margins; plot coefficient reads vs. recall. Tests V4.

**Z4: Local vs. global bases.** k ∈ {64, 256, 1024}; ranks r ∈ {16, 32, 64}; equal total memory including bases. Tests V5.

**Z5: Bound coverage.** Deterministic bound violations (must be zero) and probabilistic coverage across datasets and bit budgets. Tests V6.

**Z6: Drift.** Insert vectors from a shifted subset or a different model version of the same corpus; track residual energy and recall; evaluate the refresh trigger. Tests V7.

**Z7: System-level.** After the optimized implementation exists: QPS at equal recall vs. IVF-PQ and HNSW-based baselines; p99 latency; memory.

## Success criteria for v0.1

| Level | Criterion |
|---|---|
| Minimum | Z0 shows steep spectra on at least two realistic embedding sets; Z1 matches the best SQ/OPQ baseline at equal bytes |
| Good | Z1: ≥ 15% fewer bytes per vector at recall@10 = 0.95 than the best baseline on ≥ 2 datasets; Z3 saves ≥ 40% of coefficient reads |
| Great | Z7: ≥ 1.3× QPS at equal recall vs. strong baselines (including RaBitQ or LeanVec-style) on a modern embedding dataset; bounds validated |

## Stop / pivot conditions
- If Z0 shows flat spectra for modern embeddings, the compression idea fails for them; keep only the **spectrum analyzer** as a diagnostic tool.
- If RaBitQ, LeanVec-style or OPQ+rerank dominate Z1 at equal bytes on every dataset, publish the negative result and ship the analyzer plus the reference implementation of bounds and staged pruning for teaching and benchmarking.
- If local bases (Z4) show no gain, simplify to a single global basis.
- If bounds (Z5) are violated, fix the independence assumption (for example with random rotation of residual modes) before any recall claim.

## Report template

```
Run ID:
Dataset / dimension / size:
Index config (basis scope, k, r, bits per vector, stages):
Baselines and their settings:
Recall@10 at bytes/vector (including basis):
Coefficient reads per vector / QPS (if applicable):
Bound coverage:
Notes:
```
