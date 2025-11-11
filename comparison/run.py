"""
Compare duplicate-flag outputs across deduplication engines on a HuggingFace dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enum import Enum

from comparison.deduplicator import Deduplicator
from comparison.fastsketch_deduplicator import FastSketchDeduplicator
from comparison.rensa_deduplicator import RensaDeduplicator
from comparison.util import DEFAULT_HF_ENDPOINT, DatasetPreprocessor


class DatasetChoice(str, Enum):
    PINECONE = "pinecone/core-2020-05-10-deduplication"
    SHUYUEJ = "shuyuej/pretraining-dataset"
    BOOKCORPUSOPEN = "lucadiliello/bookcorpusopen"
    BOOKS3 = "P1ayer-1/books-3-textbooks"


def parse_dataset(value: str) -> DatasetChoice:
    normalized = value.strip()
    for choice in DatasetChoice:
        if normalized == choice.value:
            return choice
        if normalized.upper() == choice.name:
            return choice
    raise argparse.ArgumentTypeError(
        f"Unknown dataset '{value}'. Use one of: "
        + ", ".join(choice.name for choice in DatasetChoice)
        + " or their full HuggingFace IDs."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare deduplication engines on a HuggingFace dataset."
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=("fastsketch", "rensa"),
        default="fastsketch",
        help="Deduplication backend to execute.",
    )
    parser.add_argument(
        "--dataset",
        type=parse_dataset,
        default=DatasetChoice.PINECONE,
        help="Dataset to load from HuggingFace hub (use enum name, e.g. PINECONE, or full ID).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train).",
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=8,
        help="Number of LSH bands.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=16,
        help="Number of rows per band.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Target Jaccard threshold for the deduplicator.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Threads to use for both sketching and LSH (overrides per-stage flags).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for dataset shuffling.",
    )
    parser.add_argument(
        "--sketch-threads",
        type=int,
        default=0,
        help="[fastsketch] Threads for FastSimilaritySketch (<=0 lets library decide).",
    )
    parser.add_argument(
        "--lsh-threads",
        type=int,
        default=0,
        help="[fastsketch] Threads for FastSketch LSH build/query (<=0 lets library decide).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="",
        help="Cache directory for HuggingFace dataset downloads (default: project_root/data/huggingface_cache).",
    )
    parser.add_argument(
        "--save-artifacts",
        type=str,
        default="",
        help="Optional path to save deduplicator artifacts (supported by fastsketch).",
    )
    parser.add_argument(
        "--load-artifacts",
        type=str,
        default="",
        help="Optional path to load deduplicator artifacts (supported by fastsketch).",
    )
    parser.add_argument(
        "--save-fastsketch",
        type=str,
        dest="save_artifacts",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--load-fastsketch",
        type=str,
        dest="load_artifacts",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--use-hf-mirror",
        action="store_true",
        help="Whether to force HuggingFace requests through the mirror endpoint.",
    )
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default=DEFAULT_HF_ENDPOINT,
        help="Custom HuggingFace endpoint (implies --use-hf-mirror when different from default).",
    )
    return parser.parse_args()


def resolve_cache_dir(args: argparse.Namespace, project_root: Path) -> Optional[Path]:
    if args.cache_dir:
        return Path(args.cache_dir)
    return project_root / "data" / "huggingface_cache"


def create_deduplicator(args: argparse.Namespace) -> Deduplicator:
    if args.engine == "fastsketch":
        return FastSketchDeduplicator(
            bands=args.bands,
            rows=args.rows,
            threshold=args.threshold,
            seed=42,
            sketch_threads=args.sketch_threads,
            lsh_threads=args.lsh_threads,
        )
    if args.engine == "rensa":
        return RensaDeduplicator(
            bands=args.bands,
            rows=args.rows,
            threshold=args.threshold,
            seed=42,
        )
    raise SystemExit(f"Unsupported engine: {args.engine}")


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        args.sketch_threads = args.threads
        args.lsh_threads = args.threads
    dataset_choice: DatasetChoice = args.dataset
    dataset_name = dataset_choice.value
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    cache_dir = resolve_cache_dir(args, project_root)
    use_mirror = args.use_hf_mirror or (args.hf_endpoint and args.hf_endpoint != DEFAULT_HF_ENDPOINT)
    hf_endpoint = args.hf_endpoint or DEFAULT_HF_ENDPOINT

    load_path = Path(args.load_artifacts) if args.load_artifacts else None
    save_path = Path(args.save_artifacts) if args.save_artifacts else None
    processed_dir = project_root / "comparison" / "processed_ds"

    deduper = create_deduplicator(args)

    if args.engine != "fastsketch" and load_path is not None:
        raise SystemExit("--load-artifacts is only supported for the fastsketch engine.")
    if args.engine != "fastsketch" and save_path is not None:
        print("Warning: save-artifacts is ignored for engines without serialization support.")
        save_path = None

    sketches = None
    dataset_load_time: Optional[float] = None
    if load_path is not None:
        if not load_path.exists():
            raise SystemExit(f"Artifacts not found: {load_path}")
        print(f"Loading precomputed sketches from {load_path}")
        try:
            sketches = deduper.load_sketches(load_path)
        except NotImplementedError:
            raise SystemExit("This engine does not support loading precomputed sketches.")
    else:
        print(f"Preparing dataset '{dataset_name}' (split='{args.split}')")
        preprocessor = DatasetPreprocessor(
            dataset_name=dataset_name,
            split=args.split,
            seed=args.seed,
            cache_dir=cache_dir,
            use_mirror=use_mirror,
            hf_endpoint=hf_endpoint,
            processed_dir=processed_dir,
        )
        preprocess_result = preprocessor.load_and_tokenize()
        print(
            f"Loaded {len(preprocess_result.texts)} records "
            f"in {preprocess_result.elapsed_seconds:.3f}s"
        )
        dataset_load_time = preprocess_result.elapsed_seconds
        sketches = deduper.sketch(preprocess_result.token_sets)
        if save_path is not None:
            try:
                deduper.save_sketches(save_path, sketches)
                print(f"Saved sketches to {save_path}")
            except NotImplementedError:
                print("Warning: this engine does not support saving sketches; skipping.")

    if sketches is None:
        raise SystemExit("No sketches available for deduplication.")

    duplicate_flags = deduper.deduplicate(sketches)

    print(f"bands={args.bands}, rows={args.rows}, num_perm={args.bands * args.rows}")
    for flags in duplicate_flags.values():
        print(f"reported_duplicates={sum(flags)}")

    thread_info = getattr(deduper, "thread_info", None)
    if isinstance(thread_info, dict) and thread_info:
        sketch_threads = thread_info.get("sketch")
        lsh_threads = thread_info.get("lsh")
        if sketch_threads is not None or lsh_threads is not None:
            sketch_label = sketch_threads if sketch_threads is not None else "n/a"
            lsh_label = lsh_threads if lsh_threads is not None else "n/a"
            print(f"threads: sketch={sketch_label}, lsh={lsh_label}")

    timing_items = []
    for key, value in deduper.timings.items():
        timing_items.append(f"{key}={value:.3f}")
    if timing_items:
        print(f"  {deduper.name}: {', '.join(timing_items)}")


if __name__ == "__main__":
    main()
