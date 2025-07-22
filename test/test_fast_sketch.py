import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mmh3
import heapq

from src.fast_sketch import FastSimilaritySketch
from simulation.util import estimate_jaccard, actual_jaccard


if __name__ == '__main__':
    t = 256
    A = set(range(0, 1000))
    B = set(range(500, 1500))

    print(f"|A| = {len(A)}, |B| = {len(B)}")
    true_j = actual_jaccard(A, B)
    print(f"True Jaccard: {true_j:.4f}")

    sketcher = FastSimilaritySketch(sketch_size=t)
    print("Generating sketches...")
    S_A = sketcher.sketch(A)
    S_B = sketcher.sketch(B)
    print("Done.")

    est_j = estimate_jaccard(S_A, S_B)
    print(f"Estimated Jaccard: {est_j:.4f}")
    print(f"Error: {abs(true_j - est_j):.4f}")
