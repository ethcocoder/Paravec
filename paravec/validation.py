import math

import numpy as np

from .alloc import allocate_bits
from .basis import fit_basis
from .quant import quantize
from .search import SpectralIndex


def _squared_distances(query, vectors, max_block_elements: int = 1_000_000) -> np.ndarray:
    distances = np.empty(vectors.shape[0], dtype=np.float64)
    block_size = max(1, max_block_elements // vectors.shape[1])
    for start in range(0, vectors.shape[0], block_size):
        stop = min(start + block_size, vectors.shape[0])
        difference = vectors[start:stop] - query
        distances[start:stop] = np.einsum("ij,ij->i", difference, difference)
    return distances


def validate_dataset(
    vectors,
    *,
    query_count: int = 32,
    k: int = 10,
    bit_budget: int = 32,
    seed: int = 42,
    tolerance: float = 1e-8,
) -> dict:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("vectors must be a two-dimensional finite array")
    if values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("validation needs at least three vectors and one dimension")
    if type(query_count) is not int or query_count < 1:
        raise ValueError("query_count must be a positive integer")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if type(bit_budget) is not int or bit_budget < 0:
        raise ValueError("bit_budget must be a nonnegative integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(values.shape[0])
    database_count = min(values.shape[0] - 1, max(2, int(values.shape[0] * 0.8)))
    database = values[permutation[:database_count]]
    queries = values[permutation[database_count:database_count + query_count]]
    k = min(k, database_count)

    basis = fit_basis(database)
    database_coefficients = basis.transform(database)
    query_coefficients = basis.transform(queries)
    index = SpectralIndex(database)
    max_isometry_error = 0.0
    exact_search_matches = True
    coefficient_reads = 0
    total_coefficient_reads = 0
    quantized_recall = []
    exact_bound_covered = 0
    pair_count = int(database_count * len(queries))
    positive_modes = int(np.count_nonzero(basis.eigenvalues))
    allocated_budget = min(bit_budget, positive_modes * 8)
    bits = allocate_bits(basis.eigenvalues, allocated_budget)
    quantized = quantize(database_coefficients, basis.eigenvalues, bits)

    for query_index, query in enumerate(queries):
        exact_distances = _squared_distances(query, database)
        spectral_distances = _squared_distances(query_coefficients[query_index], database_coefficients)
        max_isometry_error = max(
            max_isometry_error,
            float(np.max(np.abs(exact_distances - spectral_distances))),
        )
        search_result = index.search(query, k=k)
        expected = np.lexsort((np.arange(database_count), exact_distances))[:k]
        exact_search_matches &= np.array_equal(search_result.indices, expected)
        coefficient_reads += search_result.coefficient_reads
        total_coefficient_reads += search_result.total_coefficient_reads

        query_coefficients_value = query_coefficients[query_index]
        estimates = _squared_distances(query_coefficients_value, quantized.reconstructed)
        block_size = max(1, 1_000_000 // values.shape[1])
        for start in range(0, database_count, block_size):
            stop = min(start + block_size, database_count)
            difference = query_coefficients_value - quantized.reconstructed[start:stop]
            errors = exact_distances[start:stop] - estimates[start:stop]
            bounds = (
                2.0 * np.linalg.norm(difference, axis=1) * quantized.error_norm[start:stop]
                + quantized.error_norm[start:stop] ** 2
            )
            exact_bound_covered += int(np.count_nonzero(np.abs(errors) <= bounds + tolerance))
        approximate_topk = np.argsort(estimates, kind="stable")[:k]
        quantized_recall.append(len(set(approximate_topk).intersection(expected)) / k)

    coverage = exact_bound_covered / pair_count if pair_count else 1.0
    passed = max_isometry_error <= tolerance and exact_search_matches and exact_bound_covered == pair_count
    return {
        "validation_version": 1,
        "status": "passed" if passed else "failed",
        "vector_count": int(values.shape[0]),
        "database_count": int(database_count),
        "query_count": int(len(queries)),
        "dimension": int(values.shape[1]),
        "k": int(k),
        "max_isometry_error": max_isometry_error,
        "isometry_passed": bool(max_isometry_error <= tolerance),
        "exact_staged_search_matches_brute_force": bool(exact_search_matches),
        "deterministic_bound_pair_count": pair_count,
        "deterministic_bound_coverage": coverage,
        "deterministic_bound_passed": bool(exact_bound_covered == pair_count),
        "requested_bit_budget": int(bit_budget),
        "allocated_bit_budget": int(allocated_budget),
        "bits_by_mode": bits.tolist(),
        "quantized_recall_at_k_mean": float(np.mean(quantized_recall)),
        "coefficient_reads": int(coefficient_reads),
        "total_coefficient_reads": int(total_coefficient_reads),
        "coefficient_read_fraction": (
            float(coefficient_reads / total_coefficient_reads) if total_coefficient_reads else 0.0
        ),
        "notes": [
            "The dataset is deterministically split into a database and held-out queries.",
            "Deterministic bound coverage uses measured source-to-reconstruction error and validates the bound implementation.",
            "Quantized recall is a dataset diagnostic, not a benchmark or release claim.",
        ],
    }