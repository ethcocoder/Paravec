from dataclasses import dataclass

import numpy as np

from .basis import fit_basis


@dataclass(frozen=True)
class SearchResult:
    indices: np.ndarray
    squared_distances: np.ndarray
    coefficient_reads: int
    total_coefficient_reads: int
    survivors_by_stage: tuple[int, ...]


class SpectralIndex:
    """Exact flat index that prunes with PCA-prefix distance lower bounds."""

    def __init__(self, vectors):
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("vectors must be a two-dimensional finite array")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("vectors must not be empty")
        self.basis = fit_basis(values)
        self.coefficients = self.basis.transform(values)
        self.vector_count, self.dimension = values.shape

    def search(self, query, k: int = 10, stages=None) -> SearchResult:
        query_value = np.asarray(query, dtype=np.float64)
        if query_value.shape != (self.dimension,) or not np.all(np.isfinite(query_value)):
            raise ValueError("query must be a finite vector matching index dimension")
        if type(k) is not int or not 1 <= k <= self.vector_count:
            raise ValueError("k must be between 1 and the number of indexed vectors")

        if stages is None:
            stages = sorted(set((min(16, self.dimension), min(64, self.dimension), self.dimension)))
        else:
            stages = list(stages)
        if not stages or stages[-1] != self.dimension:
            raise ValueError("stages must end at the full vector dimension")
        if any(type(mode_count) is not int or mode_count < 1 for mode_count in stages):
            raise ValueError("stages must contain positive integer mode counts")
        if any(left >= right for left, right in zip(stages, stages[1:])) or stages[-1] > self.dimension:
            raise ValueError("stages must be strictly increasing and no larger than the vector dimension")

        query_coefficients = self.basis.transform(query_value[None, :])[0]
        first_modes = stages[0]
        partial_distances = np.sum(
            (self.coefficients[:, :first_modes] - query_coefficients[:first_modes]) ** 2,
            axis=1,
        )
        coefficient_reads = self.vector_count * first_modes
        seed_count = min(self.vector_count, max(k, min(32, 4 * k)))
        seed_indices = np.argpartition(partial_distances, seed_count - 1)[:seed_count]
        seed_distances = np.sum(
            (self.coefficients[seed_indices] - query_coefficients) ** 2,
            axis=1,
        )
        coefficient_reads += seed_count * (self.dimension - first_modes)
        threshold = float(np.partition(seed_distances, k - 1)[k - 1])

        is_seed = np.zeros(self.vector_count, dtype=bool)
        is_seed[seed_indices] = True
        candidates = np.flatnonzero(~is_seed & (partial_distances <= threshold))
        survivors = [int(candidates.size + seed_count)]
        previous_modes = first_modes
        for mode_count in stages[1:]:
            if candidates.size:
                candidate_slice = self.coefficients[candidates, previous_modes:mode_count]
                query_slice = query_coefficients[previous_modes:mode_count]
                partial_distances[candidates] += np.sum((candidate_slice - query_slice) ** 2, axis=1)
                coefficient_reads += candidates.size * (mode_count - previous_modes)
                candidates = candidates[partial_distances[candidates] <= threshold]
            survivors.append(int(candidates.size + seed_count))
            previous_modes = mode_count

        exact_indices = np.concatenate((seed_indices, candidates))
        exact_distances = np.concatenate((seed_distances, partial_distances[candidates]))
        order = np.lexsort((exact_indices, exact_distances))[:k]
        return SearchResult(
            indices=exact_indices[order],
            squared_distances=exact_distances[order],
            coefficient_reads=int(coefficient_reads),
            total_coefficient_reads=int(self.vector_count * self.dimension),
            survivors_by_stage=tuple(survivors),
        )