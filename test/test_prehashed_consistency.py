import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "fastsketchlsh_ext"
if str(EXT_DIR) not in sys.path:
    sys.path.insert(0, str(EXT_DIR))

from FastSketchLSH import FastSimilaritySketch


def hash_int32_py(x: int) -> int:
    z = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF
    return z


class TestPrehashedConsistency(unittest.TestCase):
    def test_sketch_prehashed_matches_sketch_for_uint32(self) -> None:
        tokens = np.array(
            [
                0,
                1,
                2,
                3,
                7,
                11,
                17,
                23,
                31,
                63,
                127,
                255,
                511,
                1023,
                65535,
                2**31 - 1,
            ],
            dtype=np.uint32,
        )

        manual_hashes = np.array([hash_int32_py(int(v)) for v in tokens], dtype=np.uint64)

        sketcher = FastSimilaritySketch(size=128, seed=77787)
        expected = sketcher(tokens)
        actual = sketcher(manual_hashes, prehashed=True)

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
