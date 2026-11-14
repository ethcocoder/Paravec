import heapq

import numpy as np


def allocate_bits(eigenvalues, budget_bits: int, max_bits: int = 8) -> np.ndarray:
    values = np.asarray(eigenvalues, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("eigenvalues must be a one-dimensional nonnegative finite array")
    if type(max_bits) is not int or not 1 <= max_bits <= 16:
        raise ValueError("max_bits must be between 1 and 16")
    positive_modes = np.flatnonzero(values > 0)
    if type(budget_bits) is not int or not 0 <= budget_bits <= len(positive_modes) * max_bits:
        raise ValueError("budget_bits must fit within the positive-variance mode capacity")

    allocation = np.zeros(values.size, dtype=np.int16)
    heap = [(-0.75 * values[index], int(index)) for index in positive_modes]
    heapq.heapify(heap)
    for _ in range(budget_bits):
        negative_gain, index = heapq.heappop(heap)
        allocation[index] += 1
        if allocation[index] < max_bits:
            next_gain = negative_gain / 4.0
            heapq.heappush(heap, (next_gain, index))
    return allocation


def predicted_distortion(eigenvalues, bits) -> float:
    values = np.asarray(eigenvalues, dtype=np.float64)
    allocation = np.asarray(bits)
    if values.ndim != 1 or allocation.shape != values.shape:
        raise ValueError("eigenvalues and bits must be one-dimensional arrays of equal length")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("eigenvalues must be finite and nonnegative")
    if not np.issubdtype(allocation.dtype, np.integer) or np.any(allocation < 0):
        raise ValueError("bits must be nonnegative integers")
    return float(np.sum(values * np.power(4.0, -allocation)))