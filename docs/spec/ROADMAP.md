# Paravec Roadmap

## Phase 0: Does the spectrum help? (weeks 1 to 2)
- Spectrum analyzer, datasets, baseline setup (Faiss etc.)
- Deliverable: Z0 results and a go/no-go decision

## Phase 1: Core compression (weeks 3 to 4)
- Basis, bit allocation, quantizer, bound tests
- Deliverable: Z1, Z2, Z5 results in a flat-scan setting

## Phase 2: Staged search (weeks 5 to 6)
- Lossless pruning with exact leading modes, then quantized modes with margins
- Deliverable: Z3 results

## Phase 3: IVF and local bases (weeks 7 to 8)
- Cells, local/shared bases, equal-memory comparisons
- Deliverable: Z4 results

## Phase 4: Robustness (weeks 9 to 10)
- Drift handling, refresh trigger, insert/delete behavior
- Deliverable: Z6 results

## Phase 5: Speed and release (weeks 11 to 16)
- SIMD mode-major kernels in C++/Rust, Python bindings, documentation
- System-level benchmark (Z7), honest write-up
- Deliverable: open-source release

## Product directions (after results)
- **Embedding Spectrum Analyzer:** a standalone tool that tells teams how compressible their embeddings are and which index settings to use (useful even if the index itself is not competitive)
- Faiss-compatible index class or a plugin for vector databases and Postgres vector extensions
- On-device search with strict memory limits
- Shared transform-coding code with Parapack (views, quantization, bounds)

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Modern embeddings are not steep enough | Medium | Z0 gate first; keep the analyzer as a fallback product |
| Strong baselines (RaBitQ, LeanVec, OPQ+rerank, ScaNN) win | High | Honest comparison; focus on bounds, staging and diagnostics |
| Variance does not track relevance | Medium | Compare with learned projections; keep rerank stage |
| Basis overhead cancels gains at small scale | Medium | Shared bases; target large collections |
| Systems performance requires heavy engineering | High | Prove recall-per-byte first; port only if justified |
| Filtering, updates and deletes complicate the layout | Medium | Start with static indexes; add updates after core results |

## Future ideas
1. **Learned spectral basis:** train the rotation for retrieval loss rather than variance (moves toward LeanVec-style methods; compare carefully).
2. **Per-query adaptive depth:** stop refining once the top-k set is stable.
3. **Graph search with spectral codes:** use the staged estimates during HNSW/DiskANN traversal.
4. **Joint compression of vectors and metadata** using Parapack's view selection.
5. **Spectral memory link:** use Paravec-style modes as the key space for Paravault.

## End of this series
Paraphase and Paravault (AI architecture and memory), Parasync, Parascope, Parapack, Paraguard and Paravec (software systems). Next step is choosing which one to prototype first.
