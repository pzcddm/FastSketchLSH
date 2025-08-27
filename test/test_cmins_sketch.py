import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from typing import Iterable, List
from src.cmins_sketch import CMinHashSketch
from simulation.util import estimate_jaccard, actual_jaccard

if __name__ == '__main__':
    t = 256
    A = set(range(0, 100000))
    B = set(range(50000, 150000))

    print(f"|A| = {len(A)}, |B| = {len(B)}")
    true_j = actual_jaccard(A, B)
    print(f"True Jaccard: {true_j:.4f}")

    sketcher_1 = CMinHashSketch()
    sketcher_2 = CMinHashSketch()
    print("Generating sketches...")
    S_A = sketcher_1.sketch(A)
    S_B = sketcher_2.sketch(B)
    print("Done.")

    est_j = estimate_jaccard(S_A, S_B)
    print(f"Estimated Jaccard: {est_j:.4f}")
    print(f"Error: {abs(true_j - est_j):.4f}")
