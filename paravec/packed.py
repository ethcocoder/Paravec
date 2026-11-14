import numpy as np


def _validate_bits(bits) -> np.ndarray:
    widths = np.asarray(bits)
    if widths.ndim != 1 or not np.issubdtype(widths.dtype, np.integer):
        raise ValueError("bits must be a one-dimensional integer array")
    if np.any(widths < 0) or np.any(widths > 16):
        raise ValueError("each packed mode must use between 0 and 16 bits")
    return widths.astype(np.uint8, copy=False)


def pack_codes(codes, bits) -> bytes:
    values = np.asarray(codes)
    widths = _validate_bits(bits)
    if values.ndim != 2 or values.shape[1] != widths.size:
        raise ValueError("codes must be a matrix with one column per bit width")
    if not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
        raise ValueError("codes must be nonnegative integers")

    vector_count = values.shape[0]
    total_bits = vector_count * int(widths.sum())
    packed = np.zeros((total_bits + 7) // 8, dtype=np.uint8)
    vector_offsets = np.arange(vector_count, dtype=np.uint64)
    offset = 0
    for mode, width_value in enumerate(widths):
        width = int(width_value)
        if width == 0:
            if np.any(values[:, mode] != 0):
                raise ValueError("zero-bit modes must contain only zero codes")
            continue
        if np.any(values[:, mode] >= (1 << width)):
            raise ValueError("code value does not fit its allocated bit width")
        positions = offset + vector_offsets * width
        mode_values = values[:, mode].astype(np.uint32, copy=False)
        for bit_index in range(width):
            bit_positions = positions + bit_index
            byte_indices = (bit_positions >> 3).astype(np.intp)
            bit_offsets = (bit_positions & 7).astype(np.uint8)
            contributions = ((mode_values >> bit_index) & 1).astype(np.uint8) << bit_offsets
            np.bitwise_or.at(packed, byte_indices, contributions)
        offset += vector_count * width
    return packed.tobytes()


def unpack_codes(data: bytes | bytearray | memoryview, vector_count: int, bits) -> np.ndarray:
    widths = _validate_bits(bits)
    if type(vector_count) is not int or vector_count < 0:
        raise ValueError("vector_count must be a nonnegative integer")
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("packed codes must be bytes-like")
    total_bits = vector_count * int(widths.sum())
    expected_length = (total_bits + 7) // 8
    if len(data) != expected_length:
        raise ValueError("packed code length does not match vector count and bit widths")

    packed = np.frombuffer(data, dtype=np.uint8)
    codes = np.zeros((vector_count, widths.size), dtype=np.uint16)
    vector_offsets = np.arange(vector_count, dtype=np.uint64)
    offset = 0
    for mode, width_value in enumerate(widths):
        width = int(width_value)
        if width == 0:
            continue
        positions = offset + vector_offsets * width
        values = np.zeros(vector_count, dtype=np.uint32)
        for bit_index in range(width):
            bit_positions = positions + bit_index
            byte_indices = (bit_positions >> 3).astype(np.intp)
            bit_offsets = (bit_positions & 7).astype(np.uint8)
            values |= (((packed[byte_indices] >> bit_offsets) & 1).astype(np.uint32) << bit_index)
        codes[:, mode] = values.astype(np.uint16)
        offset += vector_count * width
    return codes