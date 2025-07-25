# InvestigateDocDuplicate

InvestigateDocDuplicate is a research project focused on exploring and implementing algorithms for deduplication and similarity search. It leverages techniques such as MinHash, Fast Similarity Sketching, and more to do Jaccard similarity estimation.

## Project Structure

- **src/**: Canonical implementations of core algorithms, including:
  - `fast_sketch.py` (FastSimilaritySketch)
  - `kmins_sketch.py` (KMinSketch)
- **simulation/**: Scripts for running reproducible experiments and visualizations. These import algorithms from `src/` and use shared utilities from `simulation/util.py` for set generation and Jaccard calculations.
- **test/**: Unit and comparison tests for the algorithms. All tests import from `src/` and `simulation/util.py`. Each test file sets up `sys.path` for compatibility when run as a script or with pytest.
- **records/**: Directory for storing experimental results and data records, such as CSV files from large-scale comparisons.

## Requirements

- Python 3.11.9
- Required packages: `numpy`, `mmh3`, `matplotlib`

Install dependencies with:
```bash
pip install numpy mmh3 matplotlib
```

## Usage

### Running Simulations

Navigate to the simulation directory and run:
```bash
python simulation/simulate_fast_sketch.py
python simulation/simulate_kmins_sketch.py
```

### Running Tests

You can run tests from the project root or the test directory. Each test file sets up the import path automatically:
```bash
python test/test_fast_sketch.py
python test/test_kmins_sketch.py
python test/test_comparison.py
```
Or use pytest:
```bash
pytest test/
```

## Algorithms Overview

This project implements several algorithms for efficient similarity calculations:

- **Fast Similarity Sketch**: Generates sketch representations from sets to enable quick estimation of similarity. See `src/fast_sketch.py`.
- **K-Min Sketch**: Applies a method based on selecting the minimum hashes to approximate set similarity. See `src/kmins_sketch.py`.

All algorithms are designed for linear time and space efficiency, using robust hashing techniques.

## Utilities and Reproducibility

- **simulation/util.py** provides:
  - Efficient, reproducible set generation with a specified Jaccard similarity (`generate_interval_sets_with_jaccard`).
  - Canonical Jaccard estimation and calculation functions.
- All simulations and tests use these utilities for consistency and reproducibility.
- The project structure ensures that experiments are modular and results are reproducible.

## Experimental Results

- Results from large-scale comparisons (e.g., accuracy and speed of different sketch sizes and set sizes) are saved in `records/sketch_comparison_results.csv`.
- The CSV contains columns:
  - `k`: Sketch size
  - `n`: Set size
  - `true_jaccard`: True Jaccard similarity used in the experiment
  - `fast_avg_error`: Average error of FastSimilaritySketch
  - `kmins_avg_error`: Average error of KMinSketch
  - `fast_avg_time`: Average time (seconds) for FastSimilaritySketch
  - `kmins_avg_time`: Average time (seconds) for KMinSketch
  - `speedup_ratio`: Ratio of FastSketch time to KMinSketch time
- You can analyze this file directly or import it into a spreadsheet for further analysis.

## Contributing

Contributions and improvements are welcome. Please follow the coding style and run the tests before submitting any changes.

## License

This project is provided as-is for research and educational purposes.

## Acknowledgments

- The Fast Similarity Sketch algorithm is inspired by the work presented in [Fast Similarity Sketch Paper](https://arxiv.org/abs/1704.04370).
