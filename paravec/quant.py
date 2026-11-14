from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuantizedCoefficients:
    codes: np.ndarray
    reconstructed: np.ndarray
    steps: np.ndarray
    error_norm: np.ndarray


def quantize(coefficients, eigenvalues, bits) -> QuantizedCoefficients:
    values = np.asarray(coefficients, dtype=np.float64)
    variances = np.asarray(eigenvalues, dtype=np.float64)
    allocation = np.asarray(bits)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("coefficients must be a two-dimensional finite array")
    if variances.shape != (values.shape[1],) or not np.all(np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError("eigenvalues must match coefficient dimensions and be finite and nonnegative")
    if allocation.shape != variances.shape or not np.issubdtype(allocation.dtype, np.integer):
        raise ValueError("bits must be an integer vector matching coefficient dimensions")
    if np.any(allocation < 0) or np.any(allocation > 16):
        raise ValueError("each mode must use between 0 and 16 bits")

    steps = np.zeros(variances.size, dtype=np.float64)
    active = (allocation > 0) & (variances > 0)
    steps[active] = 6.0 * np.sqrt(variances[active]) / np.exp2(allocation[active])
    codes = np.zeros(values.shape, dtype=np.uint16)
    reconstructed = np.zeros_like(values)
    for index in np.flatnonzero(active):
        levels = 1 << int(allocation[index])
        codes[:, index] = np.clip(
            np.floor((values[:, index] + 3.0 * np.sqrt(variances[index])) / steps[index]),
            0,
            levels - 1,
        ).astype(np.uint16)
        reconstructed[:, index] = (
            -3.0 * np.sqrt(variances[index]) + (codes[:, index].astype(np.float64) + 0.5) * steps[index]
        )
    error_norm = np.linalg.norm(values - reconstructed, axis=1)
    return QuantizedCoefficients(codes, reconstructed, steps, error_norm)


def squared_distance_error_bound(query, reconstruction, error_norm: float) -> float:
    query_values = np.asarray(query, dtype=np.float64)
    reconstructed_values = np.asarray(reconstruction, dtype=np.float64)
    if query_values.ndim != 1 or reconstructed_values.shape != query_values.shape:
        raise ValueError("query and reconstruction must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(query_values)) or not np.all(np.isfinite(reconstructed_values)):
        raise ValueError("query and reconstruction must be finite")
    if not np.isfinite(error_norm) or error_norm < 0:
        raise ValueError("error_norm must be finite and nonnegative")
    estimate_norm = float(np.linalg.norm(query_values - reconstructed_values))
    return 2.0 * estimate_norm * float(error_norm) + float(error_norm) ** 2