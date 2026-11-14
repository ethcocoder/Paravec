import unittest

import numpy as np

from paravec.packed import pack_codes, unpack_codes, unpack_codes_range


class PackedCodeTests(unittest.TestCase):
    def test_round_trip_mixed_width_mode_major_codes(self):
        bits = np.array([0, 1, 3, 5, 8, 16], dtype=np.int16)
        rng = np.random.default_rng(42)
        codes = np.zeros((37, len(bits)), dtype=np.uint16)
        for mode, width in enumerate(bits):
            if width:
                codes[:, mode] = rng.integers(0, 1 << int(width), size=len(codes), dtype=np.uint16)

        packed = pack_codes(codes, bits)
        restored = unpack_codes(packed, len(codes), bits)

        np.testing.assert_array_equal(restored, codes)
        np.testing.assert_array_equal(unpack_codes_range(packed, len(codes), bits, 7, 29), codes[7:29])
        self.assertEqual(len(packed), (len(codes) * int(bits.sum()) + 7) // 8)

    def test_rejects_codes_outside_allocated_width(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            pack_codes(np.array([[0, 4]], dtype=np.uint16), [0, 2])

    def test_rejects_incorrect_payload_length(self):
        with self.assertRaisesRegex(ValueError, "length"):
            unpack_codes(b"\x00", 5, [2])


if __name__ == "__main__":
    unittest.main()