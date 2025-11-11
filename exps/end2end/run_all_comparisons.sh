#!/usr/bin/env bash
# Purpose: Execute both Rensa and FastSketch deduplication engines on a curated
#          set of datasets with identical LSH parameters, capture per-stage
#          timings (sketch, build, query), and emit a Markdown-ready summary
#          that can be pasted into README.md.
# Usage:   bash run_all_comparisons.sh
# Notes:   - FastSketch runs with a single thread (`--threads 1`) to stay
#            comparable with Rensa.
#          - Dataset preparation (downloads, tokenisation) may take significant
#            time on first execution; subsequent runs use cached artefacts.

set -euo pipefail

EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${EXPERIMENT_DIR}/../.." && pwd)"

if [[ ! -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
  echo "Missing Python virtual environment (.venv)." >&2
  echo "Create it with 'python3 -m venv .venv' and install dependencies first." >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/.venv/bin/activate"

OUTPUT_DIR="${EXPERIMENT_DIR}/results"
mkdir -p "${OUTPUT_DIR}"
RESULTS_FILE="${OUTPUT_DIR}/comparison_timings.jsonl"
: >"${RESULTS_FILE}"

DATASETS=(
  "PINECONE"
  "SHUYUEJ"
  "BOOKS3"
  "BOOKCORPUSOPEN"
)

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

parse_timings() {
  local log_file="$1"
  local dataset="$2"
  local engine="$3"

  python3 - <<'PY' "${log_file}" "${dataset}" "${engine}" "${RESULTS_FILE}"
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
dataset = sys.argv[2]
engine = sys.argv[3]
out_path = Path(sys.argv[4])

log_text = log_path.read_text()
pattern = re.compile(r"^\s*(FastSketchDeduplicator|RensaDeduplicator):\s+(.*)$", re.MULTILINE)
matches = list(pattern.finditer(log_text))
if not matches:
    raise SystemExit(f"Could not locate timing line for {engine} on {dataset}")
timing_line = matches[-1].group(2)
pairs = dict(re.findall(r"([a-zA-Z_]+)=([0-9.]+)", timing_line))

def extract_float(key_options):
    for key in key_options:
        if key in pairs:
            return float(pairs[key])
    raise KeyError(f"No timing key from {key_options} found in line: {timing_line}")

sketch = extract_float(("sketch",))
build = extract_float(("build",))
query = extract_float((
    "query",
    "query_batch",
    "query_csr",
    "query_batch_list",
    "query_single_np",
))
total = sketch + build + query

record = {
    "dataset": dataset,
    "engine": engine,
    "sketch": sketch,
    "build": build,
    "query": query,
    "total": total,
}

with out_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\n")
PY
}

summarise_results() {
  python3 - <<'PY' "${RESULTS_FILE}"
import json
import sys
from collections import defaultdict

path = sys.argv[1]
records = defaultdict(dict)

with open(path, "r", encoding="utf-8") as fh:
    for line in fh:
        entry = json.loads(line)
        records[entry["dataset"]][entry["engine"]] = entry

header = (
    "| Dataset | Engine | Sketch (s) | Build (s) | Query (s) | Total (s) | "
    "FastSketch Sketch Speedup | FastSketch Total Speedup |\n"
    "|---------|--------|------------|-----------|-----------|-----------|--------------------------|------------------------|"
)
print(header)

for dataset in sorted(records.keys()):
    engines = records[dataset]
    rensa = engines.get("rensa")
    fast = engines.get("fastsketch")
    for engine_name in ("rensa", "fastsketch"):
        entry = engines.get(engine_name)
        if entry is None:
            continue
        sketch = f"{entry['sketch']:.3f}"
        build = f"{entry['build']:.3f}"
        query = f"{entry['query']:.3f}"
        total = f"{entry['total']:.3f}"
        if engine_name == "fastsketch" and rensa and fast:
            sketch_speedup = rensa["sketch"] / fast["sketch"] if fast["sketch"] else float("inf")
            total_speedup = rensa["total"] / fast["total"] if fast["total"] else float("inf")
            sketch_speedup_str = f"{sketch_speedup:.2f}×"
            total_speedup_str = f"{total_speedup:.2f}×"
        else:
            sketch_speedup_str = "-"
            total_speedup_str = "-"
        print(
            f"| {dataset} | {engine_name} | {sketch} | {build} | {query} | {total} | "
            f"{sketch_speedup_str} | {total_speedup_str} |"
        )
PY
}

for dataset in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "Dataset: ${dataset}"
  for engine in rensa fastsketch; do
    echo "--> Running ${engine}"
    log_file="${TMP_DIR}/${dataset}_${engine}.log"
    if [[ "${engine}" == "fastsketch" ]]; then
      python3 -m exps.end2end.run \
        --engine "${engine}" \
        --dataset "${dataset}" \
        --threads 1 | tee "${log_file}"
    else
      python3 -m exps.end2end.run \
        --engine "${engine}" \
        --dataset "${dataset}" \
        --threads 1 | tee "${log_file}"
    fi
    parse_timings "${log_file}" "${dataset}" "${engine}"
  done
done

echo "============================================================"
echo "Summary (paste into README.md if desired):"
summarise_results