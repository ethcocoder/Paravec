# Paravec Math Specification

## 1. Spectral basis

Vectors `x ∈ R^d`. Cell mean `μ`, covariance `Σ = U Λ Uᵀ` with `Λ = diag(λ_1 ≥ ... ≥ λ_d)` and orthonormal `U`.

Spectral coefficients: `c = Uᵀ (x − μ)`. Query: `q' = Uᵀ (q − μ)`.

**Isometry (L2):** `||q − x||² = ||q' − c||² = Σ_{i=1}^{d} (q'_i − c_i)²`.

**Lower bound from partial sums:** with exact coefficients,
```
D_m(q, x) = Σ_{i ≤ m} (q'_i − c_i)²  ≤  ||q − x||²
```
because the omitted terms are nonnegative. So a candidate with `D_m > τ` (current `k`-th best true distance) can be discarded with no loss of recall.

For **inner product / cosine:** normalize vectors to unit length first; then `⟨q, x⟩ = 1 − ½||q − x||²`, and the L2 machinery applies.

## 2. Truncation error (why steep spectra help)

Dropping modes `m+1..d` removes distance energy about `Σ_{i>m} λ_i` per vector pair (in expectation, up to the cross-term variance). The **effective dimension** at level `ρ`:
```
m_ρ = min { m : Σ_{i ≤ m} λ_i / Σ_i λ_i ≥ ρ }
```
Steep decay gives small `m_ρ`, so compression is large; a flat spectrum gives `m_ρ ≈ d` and no gain.

## 3. Rate-distortion bit allocation

Model each coefficient as Gaussian with variance `λ_i` and a `b_i`-bit uniform quantizer. High-resolution distortion:
```
D_i(b_i) ≈ λ_i · 4^{−b_i}          (up to a constant factor)
```
Total distortion `Σ D_i` subject to `Σ b_i = B`. Continuous solution (reverse water-filling): choose water level `θ`:
```
b_i = max( 0, ½ log2(λ_i / θ) ),    with θ set so that Σ b_i = B
```
**Integer greedy version:** the gain from adding one bit to mode `i` currently at `b_i` is
```
Δ_i(b_i) = λ_i · 4^{−b_i} · (1 − 1/4) = 0.75 · λ_i · 4^{−b_i}
```
Repeatedly give a bit to the mode with the largest `Δ_i`, up to a cap (for example 8 bits), until the budget is used. Modes whose gain never wins get 0 bits.

## 4. Quantizer

For mode `i` with `L_i = 2^{b_i}` levels over `[−3σ_i, 3σ_i]`, `σ_i = √λ_i`:
```
step_i = 6σ_i / L_i
code_i = clip( floor((c_i + 3σ_i) / step_i), 0, L_i − 1 )
ĉ_i    = −3σ_i + (code_i + 0.5) · step_i
```
Error `e_i = c_i − ĉ_i` satisfies `|e_i| ≤ step_i / 2` inside the range; coefficients beyond `3σ_i` are clipped (rare for roughly Gaussian modes; heavy tails need a larger range or a per-mode escape code). Modes with `b_i = 0` reconstruct to 0 (the cell mean).

## 5. Distance estimate and error bounds

Asymmetric estimate (query exact, database quantized):
```
d̂² = Σ_i (q'_i − ĉ_i)²
```
True distance `d² = Σ (q'_i − c_i)² = Σ ((q'_i − ĉ_i) − e_i)²`:
```
d² − d̂² = −2 ⟨q' − ĉ, e⟩ + ||e||²
```
**Deterministic bound:**
```
| d² − d̂² | ≤ 2 ||q' − ĉ|| · ||e|| + ||e||²,      with ||e||² ≤ Σ_i step_i² / 4  (plus clipping terms)
```
**Probabilistic bound** (treat `e_i` as independent, uniform on `[−step_i/2, step_i/2]`):
```
E[ ||e||² ] = Σ step_i² / 12
Var( ⟨q' − ĉ, e⟩ ) = Σ (q'_i − ĉ_i)² · step_i² / 12
```
so with a Gaussian approximation, `d² ≈ d̂² + E||e||² ± z_δ · 2 √( Σ (q'_i − ĉ_i)² step_i² / 12 )`, where `z_δ` is the normal quantile for the chosen miss probability `δ`. Adding the expected `||e||²` de-biases the estimate; storing each vector's actual `||e||²` removes the bias entirely.

**Empirical check (required):** the fraction of pairs whose true distance falls outside the predicted interval must be ≤ `δ` (coverage test). Independence of errors is an approximation and must be validated, especially after rotation of structured data.

## 6. Staged search with margins

Stage `s` uses modes `1..m_s` and the estimate `d̂²_{m_s}`. Prune candidate `x` if
```
d̂²_{m_s}(q, x) − margin_s  >  τ_s
```
- For **exact (unquantized) leading modes**: `margin = 0` and `d̂²_m` is a true lower bound, so pruning is lossless.
- For **quantized modes**: `margin_s` comes from the bound in §5 (for the first `m_s` modes only); tighter margins prune more but risk missing true neighbors with probability about `δ` per comparison.
- `τ_s`: the `k`-th smallest upper estimate seen so far among candidates (an upper estimate is `d̂² + margin`).

Expected work: `Σ_s (fraction surviving stage s−1) · (m_s − m_{s−1})` coefficient reads per vector, compared with `d` for a full scan.

## 7. Memory per vector

```
bytes/vector = (Σ_i b_i) / 8  +  2  (residual norm, optional)  +  id/pointer overhead
```
Plus amortized basis cost per cell: `(d · r + r) · 4` bytes divided by cell size. Always include it.

## 8. Local bases and sharing

With `k` cells and rank-`r` local bases, basis overhead is `k · d · r · 4` bytes. For `d = 768`, `r = 32`, `k = 1,000` this is about 98 MB, acceptable for billion-scale sets but not for small ones. Options: cluster the cells' bases into `G` groups and share, or fit local bases only for the top-`r` modes and use the global basis for the remainder.

## 9. Spectrum diagnostics (for the analyzer)

- Eigenvalue decay: `λ_i` vs. `i` on a log scale
- Effective dimension `m_90`, `m_99`
- Participation ratio: `(Σ λ_i)² / Σ λ_i²` (low value means a few dominant modes)
- Predicted distortion vs. bytes from the greedy allocation

## 10. Known risks in the math
- Variance is not the same as retrieval relevance; high-variance modes can encode common directions (a shared offset or "popularity" axis) while the discriminating information lies in smaller modes. Centering per cell helps; whitening is risky because it amplifies noise.
- Independence assumptions behind the probabilistic bound can fail; coverage tests are mandatory.
- Cell-local bases do not help when cells are not locally low-dimensional.
