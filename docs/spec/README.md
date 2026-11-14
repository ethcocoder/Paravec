# Paravec

> *para* + *vec*: vector-search indexes that store each embedding as a few dominant spectral modes, with more precision where the information is.

Paravec rotates embeddings into their **principal modes** (the "hidden dimensions" that carry most of the variance), spends bits where the variance is, and searches in **stages**: coarse modes first to throw away most candidates, finer modes only for survivors. Because the rotation is orthonormal, partial distances over the first modes are **lower bounds** on the true distance, which allows safe pruning.

## The problem

- Vector indexes are memory-hungry: 1 billion vectors at 768 dimensions in float32 is about 3 TB.
- Common compression (product quantization, scalar quantization, binary codes) treats all dimensions alike or learns codebooks without a clear error bound on distance estimates.
- Many embedding sets are **anisotropic**: a small number of directions carry most of the variance, which is an unused opportunity if the spectrum is steep.

## Core idea

1. **Spectral basis:** PCA (principal modes) of the embeddings, orthonormal so distances are preserved.
2. **Bit allocation by variance:** reverse water-filling gives more bits to high-variance modes and none to the weakest.
3. **Staged search:** use the first `m` modes for a lower bound or estimate, prune, then refine.
4. **Local bases (geometry):** per-cluster bases/allocations adapt to local structure.
5. **Error-bounded estimates:** deterministic and probabilistic bounds on the distance estimate, validated empirically.

## String theory mapping

| String theory | Paravec |
|---|---|
| Hidden (compact) dimensions determine what is observable | A few dominant modes determine most of the distance; the weak modes are "compactified" away |
| Vibration modes | Spectral coefficients; each mode carries its own energy and precision |
| Compact dimensions set allowed vibrations | **Geometry** = cluster/domain id selects the basis and bit allocation |
| Duality | Original space and spectral space give identical distances; search in the one that is cheaper |

## What is claimed as new

1. A **multi-resolution layout** where each stored vector supports staged distance estimates with provable lower bounds.
2. **Per-cluster bases with variance-driven bit allocation** inside one index.
3. **Validated error bounds** on distance estimates, so recall loss is predictable.
4. An **embedding spectrum analyzer**: a tool that tells you how compressible your embeddings are before you choose an index.

## Honest status

This area is very crowded: **product quantization and OPQ**, scalar quantization, binary quantization with rerank, **ScaNN**, **RaBitQ** (random rotation plus 1-bit codes with error bounds), **LeanVec** (learned dimensionality reduction for inner-product search), Matryoshka-style truncation, and early-termination methods such as **ADSampling**. PCA-plus-quantization is classic transform coding. Paravec's contribution would be an engineering and evaluation combination, and it may only be competitive on some datasets. If the spectrum is flat, the idea does not apply. See `EXPERIMENTS.md` for the gates and stop conditions.

## Documentation map

| File | Purpose |
|---|---|
| `README.md` | This overview |
| `ARCHITECTURE.md` | Components, storage layout, search flow |
| `MATH_SPEC.md` | Basis, bit allocation, bounds, staged search |
| `IMPLEMENTATION_PLAN.md` | Build order and Python skeleton |
| `EXPERIMENTS.md` | Datasets, baselines, success criteria |
| `ROADMAP.md` | Phases, risks, product directions |
