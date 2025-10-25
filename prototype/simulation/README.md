## Simulation Overview

This folder contains small, self-contained scripts to visualize and compare the behavior of the sketching and LSH components in this project. The simulations generate figures under `prototype/simulation/figures/` and reuse implementations from `prototype/src/` and helpers in `prototype/simulation/util.py`.

### What’s included

- **display_jaccard_estimate_histograms.py**: Runs Monte Carlo simulations to show the distribution of Jaccard similarity estimates for `FastSimilaritySketch` and `KMinSketch` at a target Jaccard (default 0.5). Saves `combined_fast_and_kmins_hist.png`.

- **display_lsh_probdist.py**: Produces two probability-curves figures:
  - LSH collision probability: theoretical curve `1 - (1 - J^r)^b` vs. simulated FastSketch LSH collisions.
  - Acceptance probability: theoretical k-mins acceptance vs. simulated FastSketch acceptance `Pr[estimate ≥ θ]` across Jaccard values.
  Saves `kmins_vs_fastsketch_in_lsh_probdist.png` and `kmins_vs_fastsketch_probdist.png`.

### Quick start (from project root)

Use Python 3 to run scripts. Figures will be saved in `prototype/simulation/figures/`.

```python
# Generate the side-by-side histograms figure
from prototype.simulation import display_jaccard_estimate_histograms as hist
hist.main()

# Generate the LSH curve and probability distribution comparison figures
from prototype.simulation import display_lsh_probdist as lsh
lsh.main()
```

Alternatively, you can run them as modules (from the project root):

```
python3 -m prototype.simulation.display_jaccard_estimate_histograms
python3 -m prototype.simulation.display_lsh_probdist
```

### Notes

- The scripts set random seeds for reproducibility.
- The LSH/probability curves use more trials near the threshold region and can take longer to run. You can reduce runtime by adjusting parameters at the top of `display_lsh_probdist.py`.


### Conclusions

- **Variance comparison (k = 128)**: In `combined_fast_and_kmins_hist.png`, the variance of FastSketch is **0.001447** while KMinSketch is **0.002175**. **FastSketch has lower variance**, indicating **higher estimation accuracy** at the same sketch size.
- **Probability distributions**: The probability distribution results show **FastSketch and k-mins are almost identical**, confirming comparable behavior across Jaccard values.

