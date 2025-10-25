from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence


class Deduplicator(ABC):
    """Minimal interface for sketching token sets and running LSH-style deduplication."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.timings: Dict[str, float] = {}

    def reset_timings(self) -> None:
        self.timings.clear()

    @abstractmethod
    def sketch(self, token_sets: Sequence[Sequence[str]]) -> Any:
        """Build engine-specific sketches from tokenised documents."""

    @abstractmethod
    def deduplicate(self, sketches: Any) -> Dict[str, List[int]]:
        """Run deduplication queries and return duplicate flags keyed by query mode."""

    def save_sketches(self, path: Path, sketches: Any) -> None:
        raise NotImplementedError(f"{self.name} does not support saving sketches.")

    def load_sketches(self, path: Path) -> Any:
        raise NotImplementedError(f"{self.name} does not support loading sketches.")

