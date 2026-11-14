import numpy as np

from .alloc import allocate_bits, predicted_distortion
from .basis import effective_dimensions, fit_basis, participation_ratio


def analyze_spectrum(vectors, bit_budgets=(16, 32, 64, 128)) -> dict:
    values = np.asarray(vectors, dtype=np.float64)
    basis = fit_basis(values)
    eigenvalues = basis.eigenvalues
    total_variance = float(eigenvalues.sum())
    if total_variance > 0:
        explained = eigenvalues / total_variance
    else:
        explained = np.zeros_like(eigenvalues)
    basis_bytes = basis.mean.nbytes + basis.components.nbytes + eigenvalues.nbytes
    allocations = []
    for budget in bit_budgets:
        if type(budget) is not int or budget < 0:
            raise ValueError("bit budgets must be nonnegative integers")
        capacity = int(np.count_nonzero(eigenvalues) * 8)
        if budget > capacity:
            allocations.append({
                "budget_bits_per_vector": budget,
                "feasible": False,
                "reason": f"budget exceeds the {capacity}-bit positive-mode capacity",
            })
            continue
        bits = allocate_bits(eigenvalues, budget)
        packed_code_bytes = (values.shape[0] * budget + 7) // 8
        quantization_bound_bytes = values.shape[0] * np.dtype(np.float64).itemsize
        allocation_bytes = bits.nbytes
        packed_index_bytes = (
            packed_code_bytes + quantization_bound_bytes + basis_bytes + allocation_bytes
        )
        float32_baseline_bytes = values.shape[0] * values.shape[1] * np.dtype(np.float32).itemsize
        float64_baseline_bytes = values.shape[0] * values.shape[1] * np.dtype(np.float64).itemsize
        allocations.append({
            "budget_bits_per_vector": int(budget),
            "feasible": True,
            "estimated_bytes_per_vector": float(budget) / 8.0,
            "basis_bytes_per_vector": float(basis_bytes) / values.shape[0],
            "packed_code_bytes": int(packed_code_bytes),
            "error_bound_bytes": int(quantization_bound_bytes),
            "allocation_metadata_bytes": int(allocation_bytes),
            "estimated_packed_index_bytes": int(packed_index_bytes),
            "estimated_index_bytes_per_vector": float(packed_index_bytes) / values.shape[0],
            "estimated_compression_ratio_vs_float32": float(float32_baseline_bytes / packed_index_bytes),
            "estimated_compression_ratio_vs_float64": float(float64_baseline_bytes / packed_index_bytes),
            "predicted_distortion": predicted_distortion(eigenvalues, bits),
            "bits_by_mode": bits.tolist(),
        })
    return {
        "vector_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "total_variance": total_variance,
        "effective_dimensions": effective_dimensions(eigenvalues),
        "participation_ratio": participation_ratio(eigenvalues),
        "explained_variance_by_mode": np.cumsum(explained).tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "allocations": allocations,
        "recommendation": (
            "spectrum is concentrated; evaluate compression against retrieval baselines"
            if effective_dimensions(eigenvalues).get(0.9, 0) < values.shape[1] * 0.8
            else "spectrum is relatively flat; spectral truncation may offer limited benefit"
        ),
    }