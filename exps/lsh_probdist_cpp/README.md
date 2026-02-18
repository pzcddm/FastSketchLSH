# LSH Probability Curves (C++ Extension)

This folder contains a C++-extension-based version of the probability-curve
experiment originally implemented in the Python prototype.

## What this experiment checks

- Whether FastSketchLSH with `sketch_size=256` and `sketch_size=4096` still
  follows the MinHash-LSH theoretical collision curve.
- Whether `Pr[estimate >= theta]` from FastSketch simulation aligns with the
  k-mins theoretical acceptance curve.

## Script

- `display_lsh_probdist_cpp.py`

## Output figures

- `figures/lsh_collision_cpp_k256_k4096.png`
- `figures/kmins_vs_fastsketch_cpp_k256_k4096.png`

## Run

```bash
python3 exps/lsh_probdist_cpp/display_lsh_probdist_cpp.py
```

The script supports configurable trial counts:

```bash
python3 exps/lsh_probdist_cpp/display_lsh_probdist_cpp.py \
  --trials-core 200 \
  --trials-tail 40 \
  --num-j 50
```
