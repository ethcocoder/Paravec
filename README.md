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

## Reference API

```python
from paravec import SpectralIndex

index = SpectralIndex(database_vectors)
result = index.search(query_vector, k=10, stages=[16, 64, database_vectors.shape[1]])
print(result.indices, result.squared_distances)
```

The flat index keeps exact spectral coefficients and returns exact top-k results. Its PCA-prefix bounds are lossless, but this reference path does not compress the stored search coefficients. The scalar quantizer is a separate prototype; its deterministic distance bound uses each vector's measured coefficient reconstruction error.

## Status and limitations

This is not yet a production vector database or a demonstrated compression win. IVF/local bases, packed storage, approximate quantized search, retrieval recall-vs-bytes curves, and comparisons with SQ/PQ/OPQ baselines remain unimplemented. Predicted distortion is a model estimate, not measured recall. See `docs/spec` for the original design and evaluation gates.

Run the test suite with:

```powershell
python -m unittest discover -s tests -v
```