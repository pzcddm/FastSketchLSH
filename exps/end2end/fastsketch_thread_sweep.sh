#!/usr/bin/env bash
# Purpose: Benchmark FastSketch with varying thread counts (1, 2, 4, 8) on a
#          specified dataset, capture per-stage timings, calculate total
#          execution time (sketch + build + query), and generate a line chart
#          visualising total time versus thread count.
# Usage:   bash fastsketch_thread_sweep.sh [DATASET_ENUM]
# Example: bash fastsketch_thread_sweep.sh PINECONE
# Notes:   - The dataset must correspond to a value accepted by
#            exps/end2end/run.py (e.g. PINECONE, SHUYUEJ, BOOKS3).

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

DATASET="${1:-PINECONE}"
THREAD_COUNTS=(1 2 4 8)

OUTPUT_DIR_REL="exps/end2end/results"
OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR_REL}"
mkdir -p "${OUTPUT_DIR}"
DATA_FILE_BASENAME="fastsketch_thread_timings_${DATASET}.jsonl"
FIGURE_BASENAME="fastsketch_thread_scaling_${DATASET}.png"
DATA_FILE="${OUTPUT_DIR}/${DATA_FILE_BASENAME}"
FIGURE_PATH="${OUTPUT_DIR}/${FIGURE_BASENAME}"
: >"${DATA_FILE}"

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

parse_fastsketch_timings() {
  local log_file="$1"
  local dataset="$2"
  local threads="$3"

  python3 - <<'PY' "${log_file}" "${dataset}" "${threads}" "${DATA_FILE}"
import json
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
dataset = sys.argv[2]
threads = int(sys.argv[3])
out_path = Path(sys.argv[4])

log_text = log_path.read_text()
pattern = re.compile(r"^\s*FastSketchDeduplicator:\s+(.*)$", re.MULTILINE)
matches = list(pattern.finditer(log_text))
if not matches:
    raise SystemExit(f"Could not locate FastSketch timing line for {dataset} @ threads={threads}")
timing_line = matches[-1].group(1)
pairs = dict(re.findall(r"([a-zA-Z_]+)=([0-9.]+)", timing_line))

def extract_float(key_options):
    for key in key_options:
        if key in pairs:
            return float(pairs[key])
    raise KeyError(f"Missing timing value ({key_options}) in: {timing_line}")

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
    "threads": threads,
    "sketch": sketch,
    "build": build,
    "query": query,
    "total": total,
}

with out_path.open("a", encoding="utf-8") as fh:
    fh.write(json.dumps(record) + "\n")
PY
}

generate_plot() {
  python3 - <<'PY' "${DATA_FILE}" "${FIGURE_PATH}"
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

data_path = Path(sys.argv[1])
figure_path = Path(sys.argv[2])

records = []
with data_path.open("r", encoding="utf-8") as fh:
    for line in fh:
        records.append(json.loads(line))

if not records:
    raise SystemExit("No timing records found; run the sweep before generating a plot.")

records.sort(key=lambda item: item["threads"])

threads = [item["threads"] for item in records]
totals = [item["total"] for item in records]
dataset_name = records[0]["dataset"]

plt.figure(figsize=(6, 4))
plt.plot(threads, totals, marker="o", linestyle="-", color="#1f77b4", label="Total time")
plt.title(f"FastSketch Total Time vs Threads ({dataset_name})")
plt.xlabel("Threads")
plt.ylabel("Total time (s)")
plt.xticks(threads)
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(figure_path, dpi=200)
PY
}

echo "============================================================"
echo "Dataset: ${DATASET}"
for threads in "${THREAD_COUNTS[@]}"; do
  echo "--> Running FastSketch with ${threads} thread(s)"
  log_file="${TMP_DIR}/${DATASET}_fastsketch_${threads}.log"
  python3 -m exps.end2end.run \
    --engine fastsketch \
    --dataset "${DATASET}" \
    --threads "${threads}" | tee "${log_file}"
  parse_fastsketch_timings "${log_file}" "${DATASET}" "${threads}"
done

echo "Generating plot: ${OUTPUT_DIR_REL}/${FIGURE_BASENAME}"
generate_plot

echo "Timing data written to ${OUTPUT_DIR_REL}/${DATA_FILE_BASENAME}"
echo "Done."

