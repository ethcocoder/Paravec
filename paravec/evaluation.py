import math

import numpy as np

from .packed import pack_codes
from .search import PackedSpectralIndex


MAX_EVALUATION_PAIRS = 5_000_000


def _evaluate_uniform_scalar(database, queries, exact_neighbors, budget: int, k: int) -> dict:
    vector_count, dimension = database.shape
    if budget > dimension * 16:
        return {"feasible": False, "reason": "budget exceeds the 16-bit-per-dimension limit"}

    base_width, remainder = divmod(budget, dimension)
    widths = np.full(dimension, base_width, dtype=np.uint8)
    widths[:remainder] += 1
    minimums = database.min(axis=0)
    maximums = database.max(axis=0)
    steps = np.zeros(dimension, dtype=np.float64)
    codes = np.zeros((vector_count, dimension), dtype=np.uint16)
    reconstructed = np.empty_like(database)
    for mode, width_value in enumerate(widths):
        width = int(width_value)
        span = maximums[mode] - minimums[mode]
        if width == 0:
            reconstructed[:, mode] = (minimums[mode] + maximums[mode]) / 2.0
        elif span == 0:
            reconstructed[:, mode] = minimums[mode]
        else:
            levels = 1 << width
            steps[mode] = span / (levels - 1)
            codes[:, mode] = np.clip(
                np.rint((database[:, mode] - minimums[mode]) / steps[mode]),
                0,
                levels - 1,
            ).astype(np.uint16)
            reconstructed[:, mode] = minimums[mode] + codes[:, mode].astype(np.float64) * steps[mode]

    packed_codes = pack_codes(codes, widths)
    error_norm = np.linalg.norm(database - reconstructed, axis=1)
    storage_bytes = (
        len(packed_codes)
        + error_norm.nbytes
        + minimums.nbytes
        + steps.nbytes
        + widths.nbytes
    )
    approximate_recalls = []
    reranked_recalls = []
    shortlist_fractions = []
    for query, expected in zip(queries, exact_neighbors):
        difference = reconstructed - query
        estimated = np.einsum("ij,ij->i", difference, difference)
        margin = 2.0 * np.sqrt(estimated) * error_norm + error_norm**2
        lower = np.maximum(0.0, estimated - margin)
        upper = estimated + margin
        approximate_indices = np.lexsort((np.arange(vector_count), estimated))[:k]
        threshold = float(np.partition(upper, k - 1)[k - 1])
        candidates = np.flatnonzero(lower <= threshold)
        exact_difference = database[candidates] - query
        exact_distances = np.einsum("ij,ij->i", exact_difference, exact_difference)
        order = np.lexsort((candidates, exact_distances))[:k]
        reranked_indices = candidates[order]
        expected_set = set(expected.tolist())
        approximate_recalls.append(len(expected_set.intersection(approximate_indices.tolist())) / k)
        reranked_recalls.append(len(expected_set.intersection(reranked_indices.tolist())) / k)
        shortlist_fractions.append(candidates.size / vector_count)

    return {
        "feasible": True,
        "allocated_bits_per_vector": int(widths.sum()),
        "code_bytes": len(packed_codes),
        "packed_numeric_storage_bytes": int(storage_bytes),
        "packed_bytes_per_vector": float(storage_bytes / vector_count),
        "rerank_storage_bytes": int(storage_bytes + database.nbytes),
        "rerank_bytes_per_vector": float((storage_bytes + database.nbytes) / vector_count),
        "approximate_recall_at_k_mean": float(np.mean(approximate_recalls)),
        "approximate_recall_at_k_min": float(np.min(approximate_recalls)),
        "exact_rerank_recall_at_k_mean": float(np.mean(reranked_recalls)),
        "exact_rerank_recall_at_k_min": float(np.min(reranked_recalls)),
        "mean_rerank_shortlist_fraction": float(np.mean(shortlist_fractions)),
    }


def evaluate_retrieval(
    vectors,
    *,
    bit_budgets=(32, 64, 128),
    query_count: int = 32,
    k: int = 10,
    train_fraction: float = 0.8,
    repeats: int = 1,
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
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise ValueError("repeats must be between 1 and 20")
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
    if database_count * query_count * values.shape[1] * len(budgets) * repeats > MAX_EVALUATION_PAIRS:
        raise ValueError("evaluation exceeds the work limit; reduce data size, query_count, or dimensions")

    records_by_budget = [[] for _ in budgets]
    for repeat_index in range(repeats):
        repeat_seed = seed + repeat_index
        permutation = np.random.default_rng(repeat_seed).permutation(values.shape[0])
        database = values[permutation[:database_count]]
        queries = values[permutation[database_count:database_count + query_count]]
        exact_neighbors = []
        for query in queries:
            distances = np.einsum("ij,ij->i", database - query, database - query)
            exact_neighbors.append(np.lexsort((np.arange(database_count), distances))[:k])

        for budget_index, budget in enumerate(budgets):
            try:
                index = PackedSpectralIndex(
                    database,
                    bit_budget_bits_per_vector=budget,
                    store_original=True,
                )
            except ValueError as error:
                records_by_budget[budget_index].append({
                    "seed": repeat_seed,
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
            scalar = _evaluate_uniform_scalar(database, queries, exact_neighbors, budget, k)
            records_by_budget[budget_index].append({
                "seed": repeat_seed,
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
                "approximate_recall_at_k_min": float(np.min(approximate_recalls)),
                "exact_rerank_recall_at_k_mean": float(np.mean(reranked_recalls)),
                "exact_rerank_recall_at_k_min": float(np.min(reranked_recalls)),
                "mean_rerank_shortlist_fraction": float(np.mean(shortlist_fractions)),
                "uniform_scalar": scalar,
            })

    budget_results = []
    for budget, records in zip(budgets, records_by_budget):
        infeasible = next((record for record in records if not record["feasible"]), None)
        if infeasible is not None:
            budget_results.append({
                "bit_budget_bits_per_vector": budget,
                "feasible": False,
                "reason": infeasible["reason"],
                "runs": records,
            })
            continue
        approximate_means = [record["approximate_recall_at_k_mean"] for record in records]
        reranked_means = [record["exact_rerank_recall_at_k_mean"] for record in records]
        scalar_records = [record["uniform_scalar"] for record in records]
        budget_results.append({
            "bit_budget_bits_per_vector": budget,
            "feasible": True,
            "allocated_bits_per_vector_mean": float(np.mean([record["allocated_bits_per_vector"] for record in records])),
            "code_bytes_mean": float(np.mean([record["code_bytes"] for record in records])),
            "packed_numeric_storage_bytes_mean": float(np.mean([record["packed_numeric_storage_bytes"] for record in records])),
            "packed_bytes_per_vector_mean": float(np.mean([record["packed_bytes_per_vector"] for record in records])),
            "rerank_bytes_per_vector_mean": float(np.mean([record["rerank_bytes_per_vector"] for record in records])),
            "compression_ratio_vs_float32_packed_only_mean": float(np.mean([record["compression_ratio_vs_float32_packed_only"] for record in records])),
            "compression_ratio_vs_float64_packed_only_mean": float(np.mean([record["compression_ratio_vs_float64_packed_only"] for record in records])),
            "approximate_recall_at_k_mean": float(np.mean(approximate_means)),
            "approximate_recall_at_k_std_across_seeds": float(np.std(approximate_means)),
            "approximate_recall_at_k_min": float(np.min([record["approximate_recall_at_k_min"] for record in records])),
            "exact_rerank_recall_at_k_mean": float(np.mean(reranked_means)),
            "exact_rerank_recall_at_k_min": float(np.min([record["exact_rerank_recall_at_k_min"] for record in records])),
            "mean_rerank_shortlist_fraction": float(np.mean([record["mean_rerank_shortlist_fraction"] for record in records])),
            "uniform_scalar": {
                "feasible": all(record["feasible"] for record in scalar_records),
                "allocated_bits_per_vector": float(np.mean([record["allocated_bits_per_vector"] for record in scalar_records])),
                "code_bytes_mean": float(np.mean([record["code_bytes"] for record in scalar_records])),
                "packed_numeric_storage_bytes_mean": float(np.mean([record["packed_numeric_storage_bytes"] for record in scalar_records])),
                "packed_bytes_per_vector_mean": float(np.mean([record["packed_bytes_per_vector"] for record in scalar_records])),
                "rerank_bytes_per_vector_mean": float(np.mean([record["rerank_bytes_per_vector"] for record in scalar_records])),
                "approximate_recall_at_k_mean": float(np.mean([record["approximate_recall_at_k_mean"] for record in scalar_records])),
                "approximate_recall_at_k_std_across_seeds": float(np.std([record["approximate_recall_at_k_mean"] for record in scalar_records])),
                "approximate_recall_at_k_min": float(np.min([record["approximate_recall_at_k_min"] for record in scalar_records])),
                "exact_rerank_recall_at_k_mean": float(np.mean([record["exact_rerank_recall_at_k_mean"] for record in scalar_records])),
                "exact_rerank_recall_at_k_min": float(np.min([record["exact_rerank_recall_at_k_min"] for record in scalar_records])),
                "mean_rerank_shortlist_fraction": float(np.mean([record["mean_rerank_shortlist_fraction"] for record in scalar_records])),
                "runs": scalar_records,
            },
            "runs": records,
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
        "repeats": repeats,
        "seeds": [seed + repeat_index for repeat_index in range(repeats)],
        "baselines": ["exact float64 brute-force neighbors", "uniform scalar quantization at the same nominal bit budget"],
        "results": budget_results,
        "limitations": [
            "Repeated seeded splits show sensitivity to sampled database/query partitions but are not a substitute for independent datasets or a powered benchmark.",
            "Compression ratios include packed codes, error norms, basis, and allocation metadata but exclude Python and archive overhead.",
            "Rerank storage includes original float64 database vectors; exact reranking therefore does not represent packed-only memory.",
            "The scalar baseline uses per-dimension min/max ranges fit on the database; no PQ/OPQ or optimized-kernel baseline is included, and no throughput claim is made.",
        ],
    }