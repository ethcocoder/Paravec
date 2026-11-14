from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralBasis:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray

    @property
    def dimension(self) -> int:
        return int(self.mean.size)

    def transform(self, vectors) -> np.ndarray:
        values = _as_matrix(vectors, "vectors")
        if values.shape[1] != self.dimension:
            raise ValueError("vector dimension does not match the basis")
        return (values - self.mean) @ self.components

    def inverse_transform(self, coefficients) -> np.ndarray:
        values = _as_matrix(coefficients, "coefficients")
        if values.shape[1] != self.dimension:
            raise ValueError("coefficient dimension does not match the basis")
        return values @ self.components.T + self.mean


def _as_matrix(values, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a two-dimensional finite array")
    return matrix


def fit_basis(vectors) -> SpectralBasis:
    values = _as_matrix(vectors, "vectors")
    if values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("vectors must contain at least two rows and one dimension")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / values.shape[0]
    eigenvalues, components = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    return SpectralBasis(
        mean=mean,
        components=components[:, order],
        eigenvalues=np.maximum(eigenvalues[order], 0.0),
    )


def effective_dimensions(eigenvalues, levels=(0.9, 0.99)) -> dict[float, int]:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("eigenvalues must be a one-dimensional nonnegative finite array")
    if any(not 0.0 < float(level) <= 1.0 for level in levels):
        raise ValueError("variance levels must be in (0, 1]")
    total = float(values.sum())
    if total == 0.0:
        return {float(level): 0 for level in levels}
    cumulative = np.cumsum(values) / total
    return {
        float(level): int(np.searchsorted(cumulative, level, side="left") + 1)
        for level in levels
    }


def participation_ratio(eigenvalues) -> float:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("eigenvalues must be a one-dimensional nonnegative finite array")
    squared_sum = float(np.dot(values, values))
    if squared_sum == 0.0:
        return 0.0
    return float(values.sum() ** 2 / squared_sum)