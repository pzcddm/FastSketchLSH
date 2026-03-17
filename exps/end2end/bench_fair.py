"""
Fair benchmark: FastSketchLSH vs Rensa (standard MinHash, no rho).

Rensa uses Rayon internally with a thread pool that initializes once per
process. To benchmark at different thread counts, each (dataset, threads)
combination runs in a separate subprocess.

Usage:
  python -m exps.end2end.bench_fair --datasets PINECONE SHUYUEJ --threads 1 8
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import List, Optional

# ── Fixed benchmark parameters ────────────────────────────────────────────
BANDS = 8
ROWS = 16
NUM_PERM = BANDS * ROWS  # 128
THRESHOLD = 0.8
SEED = 42

SENTINEL = "@@BENCH_RESULT@@"


# ── Data classes ──────────────────────────────────────────────────────────


@dataclass
class EngineResult:
    engine: str
    sketch: float
    build: float
    query: float
    total: float
    duplicates: int
    error: str = ""


@dataclass
class BenchResult:
    dataset: str
    dataset_name: str
    num_docs: int
    threads: int
    engines: List[EngineResult] = field(default_factory=list)


# ── Dataset enum ──────────────────────────────────────────────────────────


class DatasetChoice(str, Enum):
    PINECONE = "pinecone/core-2020-05-10-deduplication"
    SHUYUEJ = "shuyuej/pretraining-dataset"
    BOOKCORPUSOPEN = "lucadiliello/bookcorpusopen"
    BOOKS3 = "P1ayer-1/books-3-textbooks"


def _parse_dataset(value: str) -> DatasetChoice:
    normalized = value.strip()
    for choice in DatasetChoice:
        if normalized == choice.value or normalized.upper() == choice.name:
            return choice
    raise argparse.ArgumentTypeError(
        f"Unknown dataset '{value}'. Use one of: "
        + ", ".join(c.name for c in DatasetChoice)
        + " or their full HuggingFace IDs."
    )


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fair benchmark: FastSketchLSH vs Rensa (standard MinHash, no rho).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        type=_parse_dataset,
        nargs="+",
        default=[DatasetChoice.PINECONE, DatasetChoice.SHUYUEJ],
        help="Datasets to benchmark (default: PINECONE SHUYUEJ).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 8],
        help="Thread counts to test (default: 1 8).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/Volumes/ZhencanDisk/hf_cache",
        help="HuggingFace download cache directory.",
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="exps/end2end/processed_ds",
        help="Pickle cache directory.",
    )
    parser.add_argument(
        "--engine",
        choices=("fastsketch", "rensa", "both"),
        default="both",
        help="Engine(s) to benchmark (default: both).",
    )
    # Internal: subprocess worker mode
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker_dataset", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_worker_threads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker_engine", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--_worker_processed_dir", type=str, help=argparse.SUPPRESS)
    return parser.parse_args()


# ── Ensure pickle caches exist ────────────────────────────────────────────


def _ensure_pickle_caches(
    datasets: List[DatasetChoice],
    cache_dir: str,
    processed_dir: str,
) -> None:
    """Pre-download and tokenize datasets if pickle caches are missing."""
    from .util import DatasetPreprocessor

    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    cache_path = Path(cache_dir) if cache_dir else None

    for ds in datasets:
        safe_name = ds.value.replace("/", "__")
        pkl = proc_path / f"{safe_name}__train__ng3.pkl"
        if pkl.exists():
            print(f"  {ds.name}: cached ({pkl.name})")
            continue
        print(f"  {ds.name}: downloading and tokenizing...")
        preprocessor = DatasetPreprocessor(
            dataset_name=ds.value,
            split="train",
            seed=12345,
            cache_dir=cache_path,
            processed_dir=proc_path,
            prepare_prehashed_for_fastsketch=False,
        )
        preprocessor.load_and_tokenize()


# ── Worker-side engine runners ────────────────────────────────────────────


def _run_fastsketch(token_sets: list, threads: int) -> EngineResult:
    from .fastsketch_deduplicator import FastSketchDeduplicator

    fs = FastSketchDeduplicator(
        bands=BANDS,
        rows=ROWS,
        threshold=THRESHOLD,
        seed=SEED,
        sketch_threads=threads,
        lsh_threads=threads,
    )
    sketches = fs.sketch(token_sets)
    flags_dict = fs.deduplicate(sketches)
    flags = list(flags_dict.values())[0]
    return EngineResult(
        engine="FastSketchLSH",
        sketch=fs.timings.get("sketch", 0.0),
        build=fs.timings.get("build", 0.0),
        query=fs.timings.get("query", 0.0),
        total=sum(fs.timings.get(k, 0.0) for k in ("sketch", "build", "query")),
        duplicates=sum(flags),
    )


def _run_rensa(token_sets: list, threads: int) -> EngineResult:
    try:
        from .rensa_deduplicator import RensaDeduplicator
    except ImportError:
        return EngineResult(
            engine="Rensa",
            sketch=0,
            build=0,
            query=0,
            total=0,
            duplicates=0,
            error="rensa not installed",
        )
    rensa = RensaDeduplicator(
        bands=BANDS,
        rows=ROWS,
        threshold=THRESHOLD,
        seed=SEED,
        num_threads=threads,
    )
    sketches = rensa.sketch(token_sets)
    flags_dict = rensa.deduplicate(sketches)
    flags = list(flags_dict.values())[0]
    return EngineResult(
        engine="Rensa",
        sketch=rensa.timings.get("sketch", 0.0),
        build=rensa.timings.get("build", 0.0),
        query=rensa.timings.get("query", 0.0),
        total=sum(rensa.timings.get(k, 0.0) for k in ("sketch", "build", "query")),
        duplicates=sum(flags),
    )


# ── Worker subprocess entry ──────────────────────────────────────────────


def _worker_main(args: argparse.Namespace) -> None:
    """Run inside a subprocess: load data, run engines, emit JSON result."""
    import pickle

    dataset_choice = _parse_dataset(args._worker_dataset)
    threads = args._worker_threads
    engine = args._worker_engine
    processed_dir = Path(args._worker_processed_dir)

    # Load token_sets from pickle cache
    safe_name = dataset_choice.value.replace("/", "__")
    pkl_path = processed_dir / f"{safe_name}__train__ng3.pkl"
    if not pkl_path.exists():
        result = BenchResult(
            dataset=dataset_choice.name,
            dataset_name=dataset_choice.value,
            num_docs=0,
            threads=threads,
        )
        print(SENTINEL)
        print(json.dumps(asdict(result)))
        print(SENTINEL)
        return

    with pkl_path.open("rb") as fh:
        token_sets = pickle.load(fh)

    result = BenchResult(
        dataset=dataset_choice.name,
        dataset_name=dataset_choice.value,
        num_docs=len(token_sets),
        threads=threads,
    )

    if engine in ("fastsketch", "both"):
        result.engines.append(_run_fastsketch(token_sets, threads))
    if engine in ("rensa", "both"):
        result.engines.append(_run_rensa(token_sets, threads))

    print(SENTINEL)
    print(json.dumps(asdict(result)))
    print(SENTINEL)


# ── Main-side subprocess launcher ─────────────────────────────────────────


def _run_worker(
    dataset: DatasetChoice,
    threads: int,
    engine: str,
    processed_dir: str,
) -> Optional[BenchResult]:
    """Spawn a subprocess for one (dataset, threads) pair and parse result."""
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)

    cmd = [
        sys.executable,
        "-m",
        "exps.end2end.bench_fair",
        "--_worker",
        "--_worker_dataset",
        dataset.name,
        "--_worker_threads",
        str(threads),
        "--_worker_engine",
        engine,
        "--_worker_processed_dir",
        processed_dir,
    ]

    print(f"  Running {engine} on {dataset.name} threads={threads}...", flush=True)
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[2]),
    )

    if proc.returncode != 0:
        print(f"  [ERROR] Worker failed (exit {proc.returncode}):")
        for line in proc.stderr.strip().splitlines()[-10:]:
            print(f"    {line}")
        return None

    # Parse JSON between sentinels
    stdout = proc.stdout
    start = stdout.find(SENTINEL)
    end = stdout.find(SENTINEL, start + len(SENTINEL))
    if start < 0 or end < 0:
        print("  [ERROR] No result sentinel found in worker output")
        if proc.stderr.strip():
            for line in proc.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")
        return None

    json_str = stdout[start + len(SENTINEL) : end].strip()
    data = json.loads(json_str)
    engines = [EngineResult(**e) for e in data.pop("engines", [])]
    return BenchResult(**data, engines=engines)


# ── Output formatting ────────────────────────────────────────────────────

_COL_ENGINE = 16
_COL_NUM = 9
_COL_DUP = 14


def _format_table(result: BenchResult) -> str:
    """Format a single (dataset, threads) result as a plain-text table."""
    lines = [f"  threads={result.threads}"]
    lines.append(
        f"  {'Engine':<{_COL_ENGINE}}"
        f"{'Sketch':>{_COL_NUM}}"
        f"{'Build':>{_COL_NUM}}"
        f"{'Query':>{_COL_NUM}}"
        f"{'Total':>{_COL_NUM}}"
        f"{'Duplicates':>{_COL_DUP}}"
    )

    for e in result.engines:
        if e.error:
            lines.append(f"  {e.engine:<{_COL_ENGINE}}  [skipped: {e.error}]")
        else:
            lines.append(
                f"  {e.engine:<{_COL_ENGINE}}"
                f"{e.sketch:{_COL_NUM}.3f}"
                f"{e.build:{_COL_NUM}.3f}"
                f"{e.query:{_COL_NUM}.3f}"
                f"{e.total:{_COL_NUM}.3f}"
                f"{e.duplicates:{_COL_DUP}}"
            )

    valid = [e for e in result.engines if not e.error]
    if len(valid) == 2:
        fs = next((e for e in valid if e.engine == "FastSketchLSH"), valid[0])
        rn = next((e for e in valid if e.engine == "Rensa"), valid[1])
        sketch_sp = rn.sketch / fs.sketch if fs.sketch > 0 else float("inf")
        total_sp = rn.total / fs.total if fs.total > 0 else float("inf")
        lines.append(
            f"  {'Speedup':<{_COL_ENGINE}}"
            f"{sketch_sp:{_COL_NUM - 1}.2f}x"
            f"{'':>{_COL_NUM}}"
            f"{'':>{_COL_NUM}}"
            f"{total_sp:{_COL_NUM - 1}.2f}x"
        )

    return "\n".join(lines)


def _format_summary(
    results: List[BenchResult],
    thread_counts: List[int],
) -> str:
    """Format the summary table across all datasets and thread counts."""
    lines = ["=== Summary ===", ""]
    grp_w = 19

    # Header row 1: dataset label + thread labels
    h1 = f"  {'Dataset':<12}"
    for t in thread_counts:
        h1 += f"{'threads=' + str(t):<{grp_w}}"
    lines.append(h1)

    # Header row 2: sub-column labels
    h2 = f"  {'':12}"
    for _ in thread_counts:
        h2 += f"{'Sketch':>9}{'Total':>8}  "
    lines.append(h2)

    # Organize results by dataset, preserving insertion order
    by_dataset: dict[str, dict[int, BenchResult]] = {}
    ds_order: list[str] = []
    for r in results:
        if r.dataset not in by_dataset:
            by_dataset[r.dataset] = {}
            ds_order.append(r.dataset)
        by_dataset[r.dataset][r.threads] = r

    for ds_name in ds_order:
        thread_map = by_dataset[ds_name]
        row = f"  {ds_name:<12}"
        for t in thread_counts:
            r = thread_map.get(t)
            if r is None:
                row += f"{'n/a':>9}{'n/a':>8}  "
                continue
            valid = [e for e in r.engines if not e.error]
            if len(valid) < 2:
                row += f"{'n/a':>9}{'n/a':>8}  "
                continue
            fs = next((e for e in valid if e.engine == "FastSketchLSH"), valid[0])
            rn = next((e for e in valid if e.engine == "Rensa"), valid[1])
            sketch_sp = rn.sketch / fs.sketch if fs.sketch > 0 else float("inf")
            total_sp = rn.total / fs.total if fs.total > 0 else float("inf")
            row += f"{sketch_sp:>8.2f}x{total_sp:>7.2f}x  "
        lines.append(row)

    return "\n".join(lines)


def _save_json(results: List[BenchResult], output_dir: Path) -> Path:
    """Save structured results as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"fair_benchmark_{date.today().isoformat()}.json"
    path = output_dir / filename
    payload = {
        "parameters": {
            "bands": BANDS,
            "rows": ROWS,
            "num_perm": NUM_PERM,
            "threshold": THRESHOLD,
            "seed": SEED,
        },
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ── Main orchestrator ─────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    # Worker subprocess entry
    if args._worker:
        _worker_main(args)
        return

    datasets: List[DatasetChoice] = args.datasets
    thread_counts: List[int] = args.threads
    engine: str = args.engine
    processed_dir: str = args.processed_dir

    print("=== Fair Benchmark: FastSketchLSH vs Rensa (standard MinHash, no rho) ===")
    print(
        f"Parameters: bands={BANDS}, rows={ROWS}, "
        f"num_perm={NUM_PERM}, threshold={THRESHOLD}"
    )
    print()

    # Step 1: Ensure pickle caches exist
    print("Checking dataset caches...")
    _ensure_pickle_caches(datasets, args.cache_dir, processed_dir)
    print()

    # Step 2: Run benchmarks via subprocess isolation
    all_results: List[BenchResult] = []

    for ds in datasets:
        num_docs = 0
        ds_results: List[BenchResult] = []

        for threads in thread_counts:
            result = _run_worker(ds, threads, engine, processed_dir)
            if result is not None:
                num_docs = result.num_docs
                ds_results.append(result)
                all_results.append(result)

        if ds_results:
            print()
            print(f"--- {ds.name} ({num_docs:,} docs) ---")
            print()
            for r in ds_results:
                print(_format_table(r))
                print()

    # Step 3: Summary (only meaningful when both engines ran)
    has_comparison = any(
        len([e for e in r.engines if not e.error]) == 2 for r in all_results
    )
    if has_comparison and len(all_results) > 1:
        print()
        print(_format_summary(all_results, thread_counts))

    # Step 4: Save JSON
    if all_results:
        results_dir = Path(__file__).resolve().parent / "results"
        json_path = _save_json(all_results, results_dir)
        print()
        print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
