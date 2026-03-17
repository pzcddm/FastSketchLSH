from __future__ import annotations

import os
import time
from typing import Any, List, Sequence

from .deduplicator import Deduplicator
from rensa import RMinHash, RMinHashLSH  # type: ignore


class RensaDeduplicator(Deduplicator):
    """Deduplicator using the rensa RMinHashLSH implementation."""

    def __init__(
        self,
        bands: int,
        rows: int,
        threshold: float = 0.8,
        seed: int = 42,
        num_threads: int = 0,
    ) -> None:
        super().__init__("RensaDeduplicator")
        self.bands = bands
        self.rows = rows
        self.threshold = threshold
        self.seed = seed
        self.num_perm = bands * rows
        self.num_threads = num_threads
        # Rensa uses Rayon internally; control via env var.
        if num_threads > 0:
            os.environ["RAYON_NUM_THREADS"] = str(num_threads)
        self._lsh = self._new_lsh()

    def _new_lsh(self) -> RMinHashLSH:  # type: ignore[valid-type]
        return RMinHashLSH(threshold=self.threshold, num_perm=self.num_perm, num_bands=self.bands)  # type: ignore[arg-type]

    @staticmethod
    def _stringify_tokens(token_sets: Sequence[Sequence[str]]) -> List[List[str]]:
        return [[str(token) for token in tokens] for tokens in token_sets]

    def sketch(self, token_sets: Sequence[Sequence[str]]) -> Any:
        self.reset_timings()
        sketch_start = time.perf_counter()
        # Use rensa's matrix API when available to avoid Python-level
        # per-document object construction and update loops.
        #
        # NOTE: Rensa also exposes a "rho" sketch path
        # (digest_matrix_from_token_sets_rho) which aggressively samples a
        # small subset of tokens per document instead of hashing all of them.
        # This trades accuracy for speed and is therefore excluded from our
        # benchmarks — comparing it against standard MinHash would not be an
        # apples-to-apples comparison.
        if hasattr(RMinHash, "digest_matrix_from_token_sets"):
            sketches = RMinHash.digest_matrix_from_token_sets(  # type: ignore[attr-defined]
                token_sets,
                num_perm=self.num_perm,
                seed=self.seed,
            )
        else:
            minhashes: List[RMinHash] = []  # type: ignore[var-annotated]
            for tokens in token_sets:
                m = RMinHash(num_perm=self.num_perm, seed=self.seed)  # type: ignore[call-arg]
                m.update(tokens)
                minhashes.append(m)
            sketches = minhashes
        self.timings["sketch"] = time.perf_counter() - sketch_start
        return sketches

    def deduplicate(self, sketches: Any) -> dict[str, List[int]]:
        self._lsh = self._new_lsh()
        # Matrix one-shot path: build and duplicate-flag query are fused in rust.
        if not isinstance(sketches, list) and hasattr(self._lsh, "query_duplicate_flags_matrix_one_shot"):
            build_start = time.perf_counter()
            raw_flags = self._lsh.query_duplicate_flags_matrix_one_shot(sketches)  # type: ignore[attr-defined]
            self.timings["build"] = time.perf_counter() - build_start
            self.timings["query"] = 0.0
            return {"rensa": [1 if bool(v) else 0 for v in raw_flags]}

        if not isinstance(sketches, list) and hasattr(self._lsh, "insert_matrix_and_query_duplicate_flags"):
            build_start = time.perf_counter()
            raw_flags = self._lsh.insert_matrix_and_query_duplicate_flags(sketches, start_key=0)  # type: ignore[attr-defined]
            self.timings["build"] = time.perf_counter() - build_start
            self.timings["query"] = 0.0
            return {"rensa": [1 if bool(v) else 0 for v in raw_flags]}

        if (
            not isinstance(sketches, list)
            and hasattr(self._lsh, "insert_matrix")
            and hasattr(self._lsh, "query_duplicate_flags_matrix")
        ):
            build_start = time.perf_counter()
            self._lsh.insert_matrix(sketches, start_key=0)  # type: ignore[attr-defined]
            self.timings["build"] = time.perf_counter() - build_start

            query_start = time.perf_counter()
            raw_flags = self._lsh.query_duplicate_flags_matrix(sketches)  # type: ignore[attr-defined]
            self.timings["query"] = time.perf_counter() - query_start
            return {"rensa": [1 if bool(v) else 0 for v in raw_flags]}

        build_start = time.perf_counter()
        for idx, m in enumerate(sketches):
            self._lsh.insert(idx, m)
        self.timings["build"] = time.perf_counter() - build_start

        query_start = time.perf_counter()
        flags = [1 if len(self._lsh.query(m)) > 1 else 0 for m in sketches]
        self.timings["query"] = time.perf_counter() - query_start
        return {"rensa": flags}
