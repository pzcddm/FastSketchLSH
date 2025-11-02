from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from datasets import load_dataset  # type: ignore

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def _extract_text(record: dict) -> str:
    """Best-effort extraction of a representative text field from a dataset record."""
    for key in ("text", "content", "document", "body", "raw"):
        if key in record and isinstance(record[key], str) and record[key].strip():
            return record[key]
    parts: List[str] = []
    for value in record.values():
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return " \n ".join(parts)


def _tokenize(text: str) -> List[str]:
    """Tokenise text to a lowercase whitespace token list (keeps duplicates)."""
    return [token for token in text.lower().split() if token]


def _generate_ngrams(tokens: List[str], n: int) -> List[str]:
    """Convert tokens into contiguous n-gram strings; fallback to tokens if too short."""
    if n <= 1 or len(tokens) < n:
        return tokens[:]
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


@dataclass
class PreprocessResult:
    texts: List[str]
    token_sets: List[List[str]]
    elapsed_seconds: float


class DatasetPreprocessor:
    """Load and tokenize HuggingFace datasets with optional caching and mirroring."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        seed: int = 12345,
        cache_dir: Optional[Path] = None,
        use_mirror: bool = False,
        hf_endpoint: str = DEFAULT_HF_ENDPOINT,
        download_mode: str = "reuse_cache_if_exists",
        processed_dir: Optional[Path] = None,
        ngram_size: int = 3,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.seed = seed
        self.cache_dir = cache_dir
        self.use_mirror = use_mirror
        self.hf_endpoint = hf_endpoint
        self.download_mode = download_mode
        self.processed_dir = processed_dir
        self.ngram_size = max(1, ngram_size)

    def _prepare_environment(self) -> None:
        if self.use_mirror:
            os.environ["HF_ENDPOINT"] = self.hf_endpoint
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        if self.processed_dir is not None:
            self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _processed_path(self) -> Optional[Path]:
        if self.processed_dir is None:
            return None
        safe_dataset = self.dataset_name.replace("/", "__")
        filename = f"{safe_dataset}__{self.split}__ng{self.ngram_size}.pkl"
        return self.processed_dir / filename

    def load_and_tokenize(self) -> PreprocessResult:
        """Load dataset records, shuffle, and return tokenised documents."""
        import time

        self._prepare_environment()

        processed_path = self._processed_path()
        if processed_path is not None and processed_path.exists():
            print(f"Loading from Pickle: {processed_path.name}...")
            cache_start = time.perf_counter()
            with processed_path.open("rb") as fh:
                token_sets = pickle.load(fh)
            cache_time = time.perf_counter() - cache_start
            print(f"✓ Loaded {len(token_sets):,} items in {cache_time:.1f}s")
            return PreprocessResult(
                texts=[""] * len(token_sets),
                token_sets=token_sets,
                elapsed_seconds=cache_time,
            )

        # Load from HuggingFace
        load_kwargs = {}
        if self.cache_dir is not None:
            load_kwargs["cache_dir"] = str(self.cache_dir)
            load_kwargs["download_mode"] = self.download_mode
        start = time.perf_counter()
        dataset = load_dataset(self.dataset_name, **load_kwargs)  # type: ignore[arg-type]
        records = list[Any](dataset[self.split])  # type: ignore[index]
        random.Random(self.seed).shuffle(records)
        texts = [_extract_text(rec) for rec in records]
        token_sets = [_generate_ngrams(_tokenize(text), self.ngram_size) for text in texts]
        elapsed = time.perf_counter() - start

        # Save pickle for future loads
        if processed_path is not None:
            print(f"Saving to Pickle: {processed_path.name}...")
            with processed_path.open("wb") as fh:
                pickle.dump(token_sets, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✓ Saved pickle")

        return PreprocessResult(texts=texts, token_sets=token_sets, elapsed_seconds=elapsed)
