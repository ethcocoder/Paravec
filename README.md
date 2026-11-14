# Paravec

Paravec is a research reference implementation for analyzing embedding spectra, allocating scalar-quantizer bits by variance, and pruning exact flat searches with PCA-prefix lower bounds.

## Install

```powershell
python -m pip install -e .
```

## Analyze an embedding matrix

Save a finite two-dimensional matrix as a NumPy `.npy` file, then run:

```powershell
paravec analyze embeddings.npy --budgets 16 32 64 --output spectrum.json
```

The report includes eigenvalue decay, effective dimensions at 90% and 99% variance, participation ratio, per-mode bit allocations, predicted distortion, and estimated basis overhead.

## Validate a dataset

```powershell
paravec validate embeddings.npy --queries 32 --k 10 --bit-budget 32 --output validation.json
```

Validation uses a deterministic holdout split and checks PCA distance isometry, exact staged-search results against brute force, and deterministic quantization-bound coverage for every held-out query/database pair. The report also includes quantized top-k recall and coefficient-read fraction as dataset-specific diagnostics. Bound coverage uses the measured source-to-reconstruction error, so it checks the deterministic bound implementation; it is not probabilistic coverage evidence or a cross-dataset benchmark.

## Reference API

```python
from paravec import SpectralIndex

index = SpectralIndex(database_vectors)
result = index.search(query_vector, k=10, stages=[16, 64, database_vectors.shape[1]])
print(result.indices, result.squared_distances)
```

`SpectralIndex` keeps exact spectral coefficients and returns exact top-k results using lossless PCA-prefix pruning. `PackedSpectralIndex` applies the scalar quantizer to the stored coefficients; its per-vector distance intervals use measured coefficient reconstruction error.

## Packed index

`PackedSpectralIndex` stores variance-allocated codes in a mode-major variable-bit stream and keeps a per-vector quantization-error norm. It supports approximate lookup with distance intervals, or exact reranking when original vectors are retained:

```python
from paravec import PackedSpectralIndex

index = PackedSpectralIndex(database_vectors, bit_budget_bits_per_vector=64)
index.save("index.npz")
restored_index = PackedSpectralIndex.load("index.npz")
result = restored_index.search(query_vector, k=10)
```

For exact reranking, build with `store_original=True` and call `search(..., exact_rerank=True)`; retaining originals removes most of the storage savings. The analyzer now includes packed code, error-bound, allocation, and basis costs in its estimated bytes and float32/float64 ratios. These numeric estimates exclude Python object and archive overhead and are not measured recall or throughput.

## Evaluate retrieval

On a supplied embedding matrix, compare approximate packed recall and exact-rerank recall against brute-force float64 neighbors:

```powershell
paravec evaluate embeddings.npy --budgets 32 64 128 --queries 32 --k 10 --repeats 3 --output evaluation.json
```

The report compares variance-allocated spectral quantization with uniform original-space scalar quantization at the same nominal code-bit budget. The scalar ranges are fit on each training database split. Results include packed-only and original-backed storage estimates, mean recall@k and standard deviation across seeds, and rerank shortlist fraction for each budget. Repeats use deterministic seeds beginning at `42` by default. Metadata overhead means actual bytes differ even though code bytes are equal; compare against PQ/OPQ at equal total bytes before drawing compression-quality conclusions. No throughput benchmark is performed.

## Status and limitations

This is not yet a production vector database or a demonstrated retrieval win. IVF/local bases, updates, optimized kernels, equal-total-byte PQ/OPQ baselines, and multi-dataset recall-vs-bytes studies remain unimplemented. Predicted distortion is a model estimate, not measured recall. See `docs/spec` for the original design and evaluation gates.

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```