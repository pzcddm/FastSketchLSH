from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np

from comparison.deduplicator import Deduplicator
from FastSketchLSH import FastSimilaritySketch, LSH  # type: ignore


class FastSketchDeduplicator(Deduplicator):
    """Deduplicator backed by FastSimilaritySketch and banded LSH."""

    def __init__(
        self,
        bands: int,
        rows: int,
        threshold: float = 0.8,
        seed: int = 42,
        sketch_threads: int = 0,
        lsh_threads: int = 0,
    ) -> None:
        super().__init__("FastSketchDeduplicator")
        self.bands = bands
        self.rows = rows
        self.threshold = threshold
        self.seed = seed
        self.sketch_threads = sketch_threads
        self.lsh_threads = lsh_threads
        self.num_perm = bands * rows
        self._sketcher = FastSimilaritySketch(sketch_size=self.num_perm, seed=self.seed)
        self._lsh = self._new_lsh()
        self._last_sketch_threads_label = self._format_threads(self.sketch_threads)
        self._last_lsh_threads_label = self._format_threads(self.lsh_threads)

    def _new_lsh(self) -> LSH:  # type: ignore[valid-type]
        lsh = LSH(num_perm=self.num_perm, num_bands=self.bands, num_threads=self.lsh_threads)  # type: ignore[arg-type]
        self._last_lsh_threads_label = self._format_threads(self.lsh_threads)
        return lsh

    @staticmethod
    def _format_threads(value: int) -> str:
        return str(value) if value > 0 else "auto"

    @staticmethod
    def _encode_tokens(token_sets: Sequence[Sequence[str]]) -> List[List[bytes]]:
        return [[str(token).encode("utf-8") for token in tokens] for tokens in token_sets]

    @staticmethod
    def _stringify_tokens(token_sets: Sequence[Sequence[str]]) -> list:
        """
        Convert tokens to strings, preserving NumPy arrays for optimal performance.
        """
        return [[str(token) for token in tokens] for tokens in token_sets]

    def sketch(self, token_sets: Sequence[Sequence[str]]) -> np.ndarray:
        self.reset_timings()
        sketch_start = time.perf_counter()
        sketches = self._sketcher.sketch_batch(token_sets, num_threads=self.sketch_threads)
        sketch_time = time.perf_counter() - sketch_start
        self._last_sketch_threads_label = self._format_threads(self.sketch_threads)
        
        self.timings["sketch"] = sketch_time
        return sketches

    def deduplicate(self, sketches: np.ndarray) -> dict[str, List[int]]:
        self._lsh = self._new_lsh()
        build_start = time.perf_counter()
        self._lsh.build_from_batch(sketches)
        self.timings["build"] = time.perf_counter() - build_start

        query_start = time.perf_counter()
        flat, indptr = self._lsh.batch_query_csr(sketches)
        self.timings["query"] = time.perf_counter() - query_start
        _ = flat
        B = int(sketches.shape[0])
        flags = [1 if int(indptr[i + 1] - indptr[i]) > 1 else 0 for i in range(B)]

        # The single-row and Python-list batch query paths remain available for
        # experimentation, but they are commented out because the CSR routine above
        # is typically the most efficient. In practice the difference between these
        # approaches is small, so users may enable them if they prefer.
        # single_start = time.perf_counter()
        # single_flags = [1 if len(self._lsh.query_candidates(row)) > 1 else 0 for row in sketches]
        # self.timings["query_single_np"] = time.perf_counter() - single_start
        #
        # list_start = time.perf_counter()
        # list_flags = [1 if len(row) > 1 else 0 for row in self._lsh.batch_query(sketches)]
        # self.timings["query_batch_list"] = time.perf_counter() - list_start

        return {"query": flags}

    @property
    def thread_info(self) -> Dict[str, str]:
        return {"sketch": self._last_sketch_threads_label, "lsh": self._last_lsh_threads_label}

    @property
    def resolved_threads(self) -> int:
        return getattr(self._lsh, "num_threads", 0)
