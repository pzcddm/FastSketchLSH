# BOOKS3 Server Experiment Guide

## Goal

Run a fair `fastsketch` vs `rensa` benchmark on `BOOKS3` with the same settings, then compute speedup.

## Relevant Entrypoints

- `exps/end2end/run.py` (main benchmark entrypoint)
- `exps/end2end/util.py` (dataset preprocess/cache)
- `exps/end2end/fastsketch_deduplicator.py`
- `exps/end2end/rensa_deduplicator.py`

## Recommended Server Specs

- Linux x86_64 or ARM64
- Python 3.11+ (3.12 recommended)
- RAM: 64 GB minimum (96+ GB preferred for BOOKS3 safety margin)
- Disk: 150+ GB free

## Environment Setup

```bash
cd /path/to/FastSketchLSH
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e fastsketchlsh_ext
python -m pip install rensa==0.4.0 xxhash
```

## Common Benchmark Setting

Use this exact setting for both engines:

- `dataset=BOOKS3` (`P1ayer-1/books-3-textbooks`)
- `num_perm=128`, `bands=8`, `rows=16`
- `threads=1`

Fairness rule:

- For strict end-to-end comparison, do not pass `--use-prehashed-csr`.
- Only enable `--use-prehashed-csr` when you explicitly want to measure FastSketch with reusable pre-hashed input.

## Option A: Standard Runner (with processed cache)

This uses the normal `run.py` flow and writes `processed_ds/*.pkl`.
By default it does not enable the FastSketch pre-hashed CSR path.

```bash
cd /path/to/FastSketchLSH
source .venv/bin/activate

RUN_DIR="exps/end2end/results/books3_server_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

# FastSketch
python -m exps.end2end.run \
  --engine fastsketch \
  --dataset BOOKS3 \
  --bands 8 \
  --rows 16 \
  --threads 1 | tee "$RUN_DIR/books3_fastsketch.log"

# Rensa
python -m exps.end2end.run \
  --engine rensa \
  --dataset BOOKS3 \
  --bands 8 \
  --rows 16 \
  --threads 1 | tee "$RUN_DIR/books3_rensa.log"
```

If you want to benchmark FastSketch with pre-hashed CSR explicitly, add:

```bash
--use-prehashed-csr
```

### Cache Health Check (important for BOOKS3)

`BOOKS3` cache pickle is large. If interrupted/crashed during write, it may become corrupted.

```bash
python - <<'PY'
import pickle
from pathlib import Path
p = Path("exps/end2end/processed_ds/P1ayer-1__books-3-textbooks__train__ng3.pkl")
print("exists:", p.exists(), "size_bytes:", p.stat().st_size if p.exists() else 0)
if p.exists():
    with p.open("rb") as f:
        obj = pickle.load(f)
    print("ok, rows:", len(obj))
PY
```

If this throws `EOFError`, move the broken file away before rerunning:

```bash
mv exps/end2end/processed_ds/P1ayer-1__books-3-textbooks__train__ng3.pkl \
   exps/end2end/processed_ds/P1ayer-1__books-3-textbooks__train__ng3.pkl.corrupt.$(date +%Y%m%d%H%M%S)
```

## Option B: Robust No-Pickle Runner (recommended on BOOKS3)

This avoids writing the huge processed pickle file entirely.
It runs both engines in one process on the same tokenized data.
This is not the strict fairness path because it manually builds pre-hashed CSR for FastSketch.

```bash
cd /path/to/FastSketchLSH
source .venv/bin/activate

RUN_DIR="exps/end2end/results/books3_server_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

python - <<'PY' | tee "$RUN_DIR/books3_inmemory_compare.log"
from __future__ import annotations
import gc
from pathlib import Path

from exps.end2end.util import DatasetPreprocessor
from exps.end2end.fastsketch_deduplicator import FastSketchDeduplicator
from exps.end2end.rensa_deduplicator import RensaDeduplicator

DATASET = "P1ayer-1/books-3-textbooks"
BANDS = 8
ROWS = 16
THREADS = 1

prep = DatasetPreprocessor(
    dataset_name=DATASET,
    split="train",
    seed=12345,
    cache_dir=Path("data/huggingface_cache"),
    processed_dir=None,  # key: disable huge pickle write/read
    prepare_prehashed_for_fastsketch=False,
)

res = prep.load_and_tokenize()
print(f"docs={len(res.token_sets)}")

rensa = RensaDeduplicator(bands=BANDS, rows=ROWS, threshold=0.8, seed=42)
r_sk = rensa.sketch(res.token_sets)
r_flags = rensa.deduplicate(r_sk)
r_total = sum(rensa.timings.values())
print(
    f"rensa: sketch={rensa.timings.get('sketch',0.0):.3f}, "
    f"build={rensa.timings.get('build',0.0):.3f}, "
    f"query={rensa.timings.get('query',0.0):.3f}, "
    f"total={r_total:.3f}, "
    f"dups={sum(next(iter(r_flags.values())))}"
)

data, indptr = prep._build_prehashed_csr(res.token_sets)
del r_sk, r_flags
res.token_sets = []
gc.collect()

fast = FastSketchDeduplicator(
    bands=BANDS,
    rows=ROWS,
    threshold=0.8,
    seed=42,
    sketch_threads=THREADS,
    lsh_threads=THREADS,
)
f_sk = fast.sketch((data, indptr))
f_flags = fast.deduplicate(f_sk)
f_total = sum(fast.timings.values())
print(
    f"fastsketch: sketch={fast.timings.get('sketch',0.0):.3f}, "
    f"build={fast.timings.get('build',0.0):.3f}, "
    f"query={fast.timings.get('query',0.0):.3f}, "
    f"total={f_total:.3f}, "
    f"dups={sum(next(iter(f_flags.values())))}"
)

print(f"speedup_fastsketch_vs_rensa={r_total/f_total:.3f}x")
PY
```

## Multi-Run (Median) Recommendation

Run each engine 3 times and use median total time to reduce noise.

```bash
for i in 1 2 3; do
  python -m exps.end2end.run --engine fastsketch --dataset BOOKS3 --bands 8 --rows 16 --threads 1 \
    | tee "exps/end2end/results/books3_fastsketch_run${i}.log"
done

for i in 1 2 3; do
  python -m exps.end2end.run --engine rensa --dataset BOOKS3 --bands 8 --rows 16 --threads 1 \
    | tee "exps/end2end/results/books3_rensa_run${i}.log"
done
```

## Log Parsing (quick)

```bash
rg "Deduplicator:" exps/end2end/results/books3_*run*.log
```

This prints lines like:

- `FastSketchDeduplicator: sketch=..., build=..., query=...`
- `RensaDeduplicator: sketch=..., build=..., query=...`

Compute `total = sketch + build + query` for each run, then compare medians.

## Notes

- `BOOKS3` is heavy; prefer `tmux`/`screen` and keep the job isolated.
- If using Option A and jobs are interrupted often, cache corruption is likely; use Option B.
- Keep the same thread setting for fairness (`threads=1` for both engines).
- Use Option A without `--use-prehashed-csr` when your goal is a strict end-to-end benchmark against `rensa`.
