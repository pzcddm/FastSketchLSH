# Installation
You can install FastHashSketch using `pip`. It's available in all platforms:
```bash
pip install .
```
 
## TODO

- [ ] Return NumPy ndarray when input is NumPy ndarray for single-set `sketch` overloads (np.uint32/np.int32 inputs).

# Usage Example
```python
from FastSketchLSH import FastSimilaritySketch

def estimate_jaccard(sketch1, sketch2):
    if len(sketch1) != len(sketch2):
        raise ValueError("Sketches must have the same length to compare.")
    matches = sum(1 for i in range(len(sketch1)) if sketch1[i] == sketch2[i])
    return matches / len(sketch1)

if __name__ == '__main__':
    t = 256
    A = set(range(0, 1000))
    B = set(range(500, 1500))
    sketcher = FastSimilaritySketch(sketch_size=t)
    S_A = sketcher.sketch(A)
    S_B = sketcher.sketch(B)
    est_j = estimate_jaccard(S_A, S_B)
    print(f"Estimated Jaccard: {est_j:.4f}")

```