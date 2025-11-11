from __future__ import annotations

import time
from typing import List, Sequence

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
    ) -> None:
        super().__init__("RensaDeduplicator")
        self.bands = bands
        self.rows = rows
        self.threshold = threshold
        self.seed = seed
        self.num_perm = bands * rows
        self._lsh = self._new_lsh()

    def _new_lsh(self) -> RMinHashLSH:  # type: ignore[valid-type]
        return RMinHashLSH(threshold=self.threshold, num_perm=self.num_perm, num_bands=self.bands)  # type: ignore[arg-type]

    @staticmethod
    def _stringify_tokens(token_sets: Sequence[Sequence[str]]) -> List[List[str]]:
        return [[str(token) for token in tokens] for tokens in token_sets]

    def sketch(self, token_sets: Sequence[Sequence[str]]) -> List[RMinHash]:
        self.reset_timings()
        # token_sets = self._stringify_tokens(token_sets)
        minhash_start = time.perf_counter()
        minhashes: List[RMinHash] = []  # type: ignore[var-annotated]
        for tokens in token_sets:
            m = RMinHash(num_perm=self.num_perm, seed=self.seed)  # type: ignore[call-arg]
            m.update(tokens)
            minhashes.append(m)
        self.timings["sketch"] = time.perf_counter() - minhash_start
        return minhashes

    def deduplicate(self, sketches: Sequence[RMinHash]) -> dict[str, List[int]]:
        self._lsh = self._new_lsh()
        build_start = time.perf_counter()
        for idx, m in enumerate(sketches):
            self._lsh.insert(idx, m)
        self.timings["build"] = time.perf_counter() - build_start

        query_start = time.perf_counter()
        flags = [1 if len(self._lsh.query(m)) > 1 else 0 for m in sketches]
        self.timings["query"] = time.perf_counter() - query_start
        return {"rensa": flags}
