import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT_DIR = ROOT / "fastsketchlsh_ext"
if str(EXT_DIR) not in sys.path:
    sys.path.insert(0, str(EXT_DIR))

from FastSketchLSH import FastSimilaritySketch, LSH


def expected_flags_from_csr(
    flat: np.ndarray,
    indptr: np.ndarray,
    self_id_start: int,
) -> np.ndarray:
    """Recompute duplicate flags from CSR output using self-id semantics."""
    batch = int(indptr.shape[0]) - 1
    expected = np.zeros(batch, dtype=np.uint8)
    for row in range(batch):
        start = int(indptr[row])
        end = int(indptr[row + 1])
        self_id = self_id_start + row
        # Duplicate if any candidate id differs from this row's self id.
        expected[row] = 1 if any(int(flat[idx]) != self_id for idx in range(start, end)) else 0
    return expected


class TestLSHDuplicateFlagsFastPath(unittest.TestCase):
    def _make_sketches(self, token_sets: list[list[str]]) -> np.ndarray:
        sketcher = FastSimilaritySketch(sketch_size=128, seed=42)
        return np.asarray(sketcher.sketch_batch(token_sets, num_threads=1), dtype=np.uint64)

    def test_batch_query_duplicate_flags_matches_csr(self) -> None:
        token_sets = [
            ["a", "b", "c", "d"],
            ["a", "b", "c", "d"],  # intentional near-duplicate
            ["x", "y", "z"],
            ["x", "y", "z"],       # intentional near-duplicate
            ["unique", "tokens", "here"],
        ]
        sketches = self._make_sketches(token_sets)

        lsh = LSH(num_perm=128, num_bands=8, num_threads=1)
        lsh.build_from_batch(sketches)

        flat, indptr = lsh.batch_query_csr(sketches)
        expected = expected_flags_from_csr(flat, indptr, self_id_start=0)
        actual = np.asarray(lsh.batch_query_duplicate_flags(sketches, self_id_start=0), dtype=np.uint8)

        np.testing.assert_array_equal(actual, expected)

    def test_batch_query_duplicate_flags_nonzero_self_id_start(self) -> None:
        warmup_sets = [
            ["preamble", "one"],
            ["preamble", "two"],
        ]
        target_sets = [
            ["cat", "dog", "mouse"],
            ["cat", "dog", "mouse"],  # intentional near-duplicate
            ["tree", "river", "mountain"],
        ]

        warmup_sketches = self._make_sketches(warmup_sets)
        target_sketches = self._make_sketches(target_sets)

        lsh = LSH(num_perm=128, num_bands=8, num_threads=1)
        lsh.build_from_batch(warmup_sketches)
        lsh.build_from_batch(target_sketches)

        self_id_start = len(warmup_sets)
        flat, indptr = lsh.batch_query_csr(target_sketches)
        expected = expected_flags_from_csr(flat, indptr, self_id_start=self_id_start)
        actual = np.asarray(
            lsh.batch_query_duplicate_flags(target_sketches, self_id_start=self_id_start),
            dtype=np.uint8,
        )

        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
