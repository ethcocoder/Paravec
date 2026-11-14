import math

import numpy as np

from .search import PackedSpectralIndex


MAX_EVALUATION_PAIRS = 5_000_000


def evaluate_retrieval(
    vectors,
    *,
    bit_budgets=(32, 64, 128),
    query_count: int = 32,
    k: int = 10,
    train_fraction: float = 0.8,
    seed: int = 42,
) -> dict:
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("vectors must be a two-dimensional finite array")
    if values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("evaluation needs at least three vectors and one dimension")
    if type(query_count) is not int or query_count < 1:
        raise ValueError("query_count must be a positive integer")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not math.isfinite(train_fraction) or not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0.5 inclusive and 1 exclusive")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    raw_budgets = tuple(bit_budgets)
    if any(
        not isinstance(budget, (int, np.integer)) or isinstance(budget, (bool, np.bool_))
        for budget in raw_budgets
    ):
        raise ValueError("bit_budgets must contain integers")
    budgets = tuple(int(budget) for budget in raw_budgets)
    if not budgets or any(budget < 0 for budget in budgets):
        raise ValueError("bit_budgets must contain nonnegative integers")
    if len(set(budgets)) != len(budgets):
        raise ValueError("bit_budgets must be unique")

    database_count = int(values.shape[0] * train_fraction)
    if query_count > values.shape[0] - database_count:
        raise ValueError("query_count exceeds the vectors left after the database split")
    if k > database_count:
        raise ValueError("k cannot exceed the database size")
    if database_count * query_count * values.shape[1] * len(budgets) > MAX_EVALUATION_PAIRS:
        raise ValueError("evaluation exceeds the work limit; reduce data size, query_count, or dimensions")

    permutation = np.random.default_rng(seed).permutation(values.shape[0])
    database = values[permutation[:database_count]]
    queries = values[permutation[database_count:database_count + query_count]]
    exact_neighbors = []
    for query in queries:
        distances = np.einsum("ij,ij->i", database - query, database - query)
        exact_neighbors.append(np.lexsort((np.arange(database_count), distances))[:k])

    budget_results = []
    for budget in budgets:
        try:
            index = PackedSpectralIndex(
                database,
                bit_budget_bits_per_vector=budget,
                store_original=True,
            )
        except ValueError as error:
            budget_results.append({
                "bit_budget_bits_per_vector": budget,
                "feasible": False,
                "reason": str(error),
            })
            continue

        approximate_recalls = []
        reranked_recalls = []
        shortlist_fractions = []
        for query, expected in zip(queries, exact_neighbors):
            approximate = index.search(query, k=k)
            reranked = index.search(query, k=k, exact_rerank=True)
            expected_set = set(expected.tolist())
            approximate_recalls.append(
                len(expected_set.intersection(approximate.indices.tolist())) / k
            )
            reranked_recalls.append(
                len(expected_set.intersection(reranked.indices.tolist())) / k
            )
            shortlist_fractions.append(reranked.shortlist_size / database_count)

        packed_storage_bytes = index.numeric_storage_bytes - database.nbytes
        raw_float32_bytes = database_count * values.shape[1] * np.dtype(np.float32).itemsize
        raw_float64_bytes = database_count * values.shape[1] * np.dtype(np.float64).itemsize
        budget_results.append({
            "bit_budget_bits_per_vector": budget,
            "feasible": True,
            "allocated_bits_per_vector": index.bits_per_vector,
            "code_bytes": len(index.packed_codes),
            "packed_numeric_storage_bytes": int(packed_storage_bytes),
            "packed_bytes_per_vector": float(packed_storage_bytes / database_count),
            "rerank_storage_bytes": int(index.numeric_storage_bytes),
            "rerank_bytes_per_vector": float(index.numeric_storage_bytes / database_count),
            "compression_ratio_vs_float32_packed_only": float(raw_float32_bytes / packed_storage_bytes),
            "compression_ratio_vs_float64_packed_only": float(raw_float64_bytes / packed_storage_bytes),
            "approximate_recall_at_k_mean": float(np.mean(approximate_recalls)),
            "exact_rerank_recall_at_k_mean": float(np.mean(reranked_recalls)),
            "exact_rerank_recall_at_k_min": float(np.min(reranked_recalls)),
            "mean_rerank_shortlist_fraction": float(np.mean(shortlist_fractions)),
        })

    return {
        "evaluation_version": 1,
        "status": "evaluated",
        "dataset_type": "user-supplied NumPy matrix",
        "vector_count": int(values.shape[0]),
        "database_count": int(database_count),
        "query_count": int(len(queries)),
        "dimension": int(values.shape[1]),
        "k": k,
        "train_fraction": train_fraction,
        "seed": seed,
        "baselines": ["exact float64 brute-force neighbors"],
        "results": budget_results,
        "limitations": [
            "This single seeded split is a dataset diagnostic, not a statistically powered benchmark.",
            "Compression ratios include packed codes, error norms, basis, and allocation metadata but exclude Python and archive overhead.",
            "Rerank storage includes original float64 database vectors; exact reranking therefore does not represent packed-only memory.",
            "No SQ/PQ/OPQ or optimized-kernel baseline is included, and no throughput claim is made.",
        ],
    }