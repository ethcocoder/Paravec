from dataclasses import dataclass

import numpy as np

from .alloc import allocate_bits
from .basis import SpectralBasis, fit_basis
from .packed import pack_codes, unpack_codes, unpack_codes_range
from .quant import quantize


MAX_QUERY_BLOCK_ELEMENTS = 4_000_000


@dataclass(frozen=True)
class SearchResult:
    indices: np.ndarray
    squared_distances: np.ndarray
    coefficient_reads: int
    total_coefficient_reads: int
    survivors_by_stage: tuple[int, ...]


@dataclass(frozen=True)
class PackedSearchResult:
    indices: np.ndarray
    squared_distances: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    shortlist_size: int
    exact: bool


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


class PackedSpectralIndex:
    """Bit-packed scalar codes with conservative bounds and optional exact reranking."""

    def __init__(self, vectors, bit_budget_bits_per_vector: int = 64, max_bits: int = 8, store_original: bool = False):
        values = np.asarray(vectors, dtype=np.float64)
        if values.ndim != 2 or not np.all(np.isfinite(values)):
            raise ValueError("vectors must be a two-dimensional finite array")
        if values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("vectors must not be empty")
        self.basis = fit_basis(values)
        coefficients = self.basis.transform(values)
        self.bits = allocate_bits(self.basis.eigenvalues, bit_budget_bits_per_vector, max_bits)
        quantized = quantize(coefficients, self.basis.eigenvalues, self.bits)
        self.packed_codes = pack_codes(quantized.codes, self.bits)
        self.error_norm = quantized.error_norm.astype(np.float64, copy=True)
        self.vector_count, self.dimension = values.shape
        self.store_original = bool(store_original)
        self.original = values.copy() if self.store_original else None
        self.bit_budget_bits_per_vector = int(bit_budget_bits_per_vector)
        self.bits_per_vector = int(self.bits.sum())
        self._update_storage_stats()

    def _update_storage_stats(self) -> None:
        self.numeric_storage_bytes = (
            len(self.packed_codes)
            + self.error_norm.nbytes
            + self.basis.mean.nbytes
            + self.basis.components.nbytes
            + self.basis.eigenvalues.nbytes
            + self.bits.nbytes
        )
        if self.original is not None:
            self.numeric_storage_bytes += self.original.nbytes

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            format_version=np.asarray(1, dtype=np.uint16),
            vector_count=np.asarray(self.vector_count, dtype=np.int64),
            dimension=np.asarray(self.dimension, dtype=np.int64),
            bit_budget_bits_per_vector=np.asarray(self.bit_budget_bits_per_vector, dtype=np.int64),
            has_original=np.asarray(self.original is not None, dtype=np.bool_),
            mean=self.basis.mean,
            components=self.basis.components,
            eigenvalues=self.basis.eigenvalues,
            bits=self.bits,
            packed_codes=np.frombuffer(self.packed_codes, dtype=np.uint8),
            error_norm=self.error_norm,
            original=self.original if self.original is not None else np.empty((0, self.dimension)),
        )

    @classmethod
    def load(cls, path) -> "PackedSpectralIndex":
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "format_version", "vector_count", "dimension", "bit_budget_bits_per_vector",
                "has_original", "mean", "components", "eigenvalues", "bits", "packed_codes",
                "error_norm", "original",
            }
            if not required.issubset(archive.files):
                raise ValueError("packed index is missing required fields")
            if int(archive["format_version"]) != 1:
                raise ValueError("unsupported packed index format version")
            vector_count = int(archive["vector_count"])
            dimension = int(archive["dimension"])
            budget = int(archive["bit_budget_bits_per_vector"])
            has_original = bool(archive["has_original"])
            mean = np.array(archive["mean"], dtype=np.float64, copy=True)
            components = np.array(archive["components"], dtype=np.float64, copy=True)
            eigenvalues = np.array(archive["eigenvalues"], dtype=np.float64, copy=True)
            bits = np.array(archive["bits"], copy=True)
            packed_codes = np.array(archive["packed_codes"], dtype=np.uint8, copy=True).tobytes()
            error_norm = np.array(archive["error_norm"], dtype=np.float64, copy=True)
            original = np.array(archive["original"], dtype=np.float64, copy=True)

        if vector_count < 1 or dimension < 1 or budget < 0:
            raise ValueError("invalid packed index dimensions or bit budget")
        if mean.shape != (dimension,) or components.shape != (dimension, dimension):
            raise ValueError("packed index basis shape does not match dimension")
        if eigenvalues.shape != (dimension,) or not np.all(np.isfinite(eigenvalues)) or np.any(eigenvalues < 0):
            raise ValueError("packed index eigenvalues are invalid")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(components)):
            raise ValueError("packed index basis contains non-finite values")
        if not np.allclose(components.T @ components, np.eye(dimension), rtol=1e-6, atol=1e-6):
            raise ValueError("packed index components are not orthonormal")
        if bits.shape != (dimension,) or not np.issubdtype(bits.dtype, np.integer):
            raise ValueError("packed index bit allocation is invalid")
        if np.any(bits < 0) or np.any(bits > 16) or int(bits.sum()) > budget:
            raise ValueError("packed index bit allocation exceeds its declared budget")
        if error_norm.shape != (vector_count,) or not np.all(np.isfinite(error_norm)) or np.any(error_norm < 0):
            raise ValueError("packed index error norms are invalid")
        if has_original:
            if original.shape != (vector_count, dimension) or not np.all(np.isfinite(original)):
                raise ValueError("packed index original vectors are invalid")
        elif original.size != 0:
            raise ValueError("packed index unexpectedly contains original vectors")
        unpack_codes(packed_codes, vector_count, bits)

        instance = cls.__new__(cls)
        instance.basis = SpectralBasis(mean, components, eigenvalues)
        instance.bits = bits.astype(np.int16, copy=False)
        instance.packed_codes = packed_codes
        instance.error_norm = error_norm
        instance.vector_count = vector_count
        instance.dimension = dimension
        instance.store_original = has_original
        instance.original = original if has_original else None
        instance.bit_budget_bits_per_vector = budget
        instance.bits_per_vector = int(instance.bits.sum())
        instance._update_storage_stats()
        return instance

    @property
    def bytes_per_vector(self) -> float:
        return self.numeric_storage_bytes / self.vector_count

    @property
    def compression_ratio(self) -> float:
        baseline_bytes = self.vector_count * self.dimension * np.dtype(np.float64).itemsize
        return baseline_bytes / self.numeric_storage_bytes

    def _decode_coefficients(self, start: int, stop: int) -> np.ndarray:
        codes = unpack_codes_range(self.packed_codes, self.vector_count, self.bits, start, stop)
        reconstructed = np.zeros((stop - start, self.dimension), dtype=np.float64)
        active = (self.bits > 0) & (self.basis.eigenvalues > 0)
        for mode in np.flatnonzero(active):
            sigma = np.sqrt(self.basis.eigenvalues[mode])
            step = 6.0 * sigma / (1 << int(self.bits[mode]))
            reconstructed[:, mode] = -3.0 * sigma + (codes[:, mode].astype(np.float64) + 0.5) * step
        return reconstructed

    def search(
        self,
        query,
        k: int = 10,
        *,
        exact_rerank: bool = False,
        block_size: int | None = None,
    ) -> PackedSearchResult:
        query_value = np.asarray(query, dtype=np.float64)
        if query_value.shape != (self.dimension,) or not np.all(np.isfinite(query_value)):
            raise ValueError("query must be a finite vector matching index dimension")
        if type(k) is not int or not 1 <= k <= self.vector_count:
            raise ValueError("k must be between 1 and the number of indexed vectors")
        if block_size is None:
            block_size = min(4096, max(1, MAX_QUERY_BLOCK_ELEMENTS // self.dimension))
        elif type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if block_size > 1_000_000 or block_size * self.dimension > MAX_QUERY_BLOCK_ELEMENTS:
            raise ValueError("block_size exceeds the per-query working-memory limit")
        if exact_rerank and self.original is None:
            raise ValueError("exact_rerank requires store_original=True")

        query_coefficients = self.basis.transform(query_value[None, :])[0]
        if exact_rerank:
            best_upper = np.full(k, np.inf, dtype=np.float64)
            for start in range(0, self.vector_count, block_size):
                stop = min(start + block_size, self.vector_count)
                reconstructed = self._decode_coefficients(start, stop)
                differences = reconstructed - query_coefficients
                estimated = np.einsum("ij,ij->i", differences, differences)
                errors = self.error_norm[start:stop]
                bounds = 2.0 * np.sqrt(estimated) * errors + errors**2
                upper = estimated + bounds
                best_upper = np.partition(np.concatenate((best_upper, upper)), k - 1)[:k]
            threshold = float(np.max(best_upper))

            best_indices = np.empty(0, dtype=np.int64)
            best_distances = np.empty(0, dtype=np.float64)
            best_lower = np.empty(0, dtype=np.float64)
            best_upper_bounds = np.empty(0, dtype=np.float64)
            shortlist_size = 0
            for start in range(0, self.vector_count, block_size):
                stop = min(start + block_size, self.vector_count)
                reconstructed = self._decode_coefficients(start, stop)
                differences = reconstructed - query_coefficients
                estimated = np.einsum("ij,ij->i", differences, differences)
                errors = self.error_norm[start:stop]
                bounds = 2.0 * np.sqrt(estimated) * errors + errors**2
                lower = np.maximum(0.0, estimated - bounds)
                upper = estimated + bounds
                local_indices = np.flatnonzero(lower <= threshold)
                shortlist_size += int(local_indices.size)
                if local_indices.size == 0:
                    continue
                indices = local_indices.astype(np.int64) + start
                exact_differences = self.original[indices] - query_value
                exact_distances = np.einsum("ij,ij->i", exact_differences, exact_differences)
                combined_indices = np.concatenate((best_indices, indices))
                combined_distances = np.concatenate((best_distances, exact_distances))
                combined_lower = np.concatenate((best_lower, lower[local_indices]))
                combined_upper = np.concatenate((best_upper_bounds, upper[local_indices]))
                order = np.lexsort((combined_indices, combined_distances))[:k]
                best_indices = combined_indices[order]
                best_distances = combined_distances[order]
                best_lower = combined_lower[order]
                best_upper_bounds = combined_upper[order]
            return PackedSearchResult(
                indices=best_indices,
                squared_distances=best_distances,
                lower_bounds=best_lower,
                upper_bounds=best_upper_bounds,
                shortlist_size=shortlist_size,
                exact=True,
            )

        best_indices = np.empty(0, dtype=np.int64)
        best_distances = np.empty(0, dtype=np.float64)
        best_lower = np.empty(0, dtype=np.float64)
        best_upper = np.empty(0, dtype=np.float64)
        for start in range(0, self.vector_count, block_size):
            stop = min(start + block_size, self.vector_count)
            reconstructed = self._decode_coefficients(start, stop)
            differences = reconstructed - query_coefficients
            estimated = np.einsum("ij,ij->i", differences, differences)
            errors = self.error_norm[start:stop]
            bounds = 2.0 * np.sqrt(estimated) * errors + errors**2
            lower = np.maximum(0.0, estimated - bounds)
            upper = estimated + bounds
            indices = np.arange(start, stop, dtype=np.int64)
            combined_indices = np.concatenate((best_indices, indices))
            combined_distances = np.concatenate((best_distances, estimated))
            combined_lower = np.concatenate((best_lower, lower))
            combined_upper = np.concatenate((best_upper, upper))
            order = np.lexsort((combined_indices, combined_distances))[:k]
            best_indices = combined_indices[order]
            best_distances = combined_distances[order]
            best_lower = combined_lower[order]
            best_upper = combined_upper[order]
        return PackedSearchResult(
            indices=best_indices,
            squared_distances=best_distances,
            lower_bounds=best_lower,
            upper_bounds=best_upper,
            shortlist_size=int(self.vector_count),
            exact=False,
        )