from __future__ import annotations

import time
from typing import Dict, List, Sequence

import numpy as np

from .deduplicator import Deduplicator
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
        self._sketcher = FastSimilaritySketch(size=self.num_perm, seed=self.seed)
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

    def sketch(
        self,
        token_sets: Sequence[Sequence[str]] | tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        self.reset_timings()
        sketch_start = time.perf_counter()
        if (
            isinstance(token_sets, tuple)
            and len(token_sets) == 2
            and isinstance(token_sets[0], np.ndarray)
            and isinstance(token_sets[1], np.ndarray)
        ):
            # Pre-hashed fast path: caller provides a CSR matrix of uint64 token hashes.
            data = np.asarray(token_sets[0], dtype=np.uint64)
            indptr = np.asarray(token_sets[1], dtype=np.uint64)
            sketches = self._sketcher.batch_csr(
                data,
                indptr,
                prehashed=True,
                num_threads=self.sketch_threads,
            )
        else:
            # sketch_batch has thread-aware dispatch: single-thread uses a fused
            # chunked path; multi-thread falls through to the ptrs/lengths path
            # where sketch_batch_flat_bytes parallelizes both hashing and sketching.
            sketches = self._sketcher.batch(token_sets, num_threads=self.sketch_threads)
        sketch_time = time.perf_counter() - sketch_start
        self._last_sketch_threads_label = self._format_threads(self.sketch_threads)
        self.timings["sketch"] = sketch_time
        return sketches

    def deduplicate(self, sketches: np.ndarray) -> dict[str, List[int]]:
        self._lsh = self._new_lsh()
        # The true one-shot path is row-serial today, so prefer it for the
        # single-thread benchmark where it removes the second full LSH pass.
        if self.lsh_threads == 1 and hasattr(self._lsh, "insert_and_query_duplicates"):
            build_start = time.perf_counter()
            flag_bytes = self._lsh.insert_and_query_duplicates(sketches)
            self.timings["build"] = time.perf_counter() - build_start
            self.timings["query"] = 0.0
            flags = np.asarray(flag_bytes, dtype=np.uint8).tolist()
            return {"query": flags}

        build_start = time.perf_counter()
        self._lsh.insert(sketches)
        self.timings["build"] = time.perf_counter() - build_start

        # Prefer the duplicate-flag fast path when available. It avoids
        # materializing full candidate lists and mirrors rensa's one-shot style.
        if hasattr(self._lsh, "duplicates"):
            query_start = time.perf_counter()
            flag_bytes = self._lsh.duplicates(sketches, self_start=0)
            self.timings["query"] = time.perf_counter() - query_start
            flags = np.asarray(flag_bytes, dtype=np.uint8).tolist()
        else:
            query_start = time.perf_counter()
            flat, indptr = self._lsh.query(sketches, format="csr")
            self.timings["query"] = time.perf_counter() - query_start
            _ = flat
            B = int(sketches.shape[0])
            flags = [1 if int(indptr[i + 1] - indptr[i]) > 1 else 0 for i in range(B)]

        # The single-row and Python-list batch query paths remain available for
        # experimentation, but they are commented out because the CSR routine above
        # is typically the most efficient. In practice the difference between these
        # approaches is small, so users may enable them if they prefer.
        # single_start = time.perf_counter()
        # flags = [1 if len(self._lsh.query(row)) > 1 else 0 for row in sketches]
        # self.timings["query_single_np"] = time.perf_counter() - single_start
        #
        # list_start = time.perf_counter()
        # flags = [1 if len(row) > 1 else 0 for row in self._lsh.query(sketches)]
        # self.timings["query_batch_list"] = time.perf_counter() - list_start

        return {"query": flags}

    @property
    def thread_info(self) -> Dict[str, str]:
        return {"sketch": self._last_sketch_threads_label, "lsh": self._last_lsh_threads_label}

    @property
    def resolved_threads(self) -> int:
        return getattr(self._lsh, "num_threads", 0)
