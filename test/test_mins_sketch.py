import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from FastSketchLSH import CMinHashSketch
from simulation.util import estimate_jaccard, actual_jaccard


if __name__ == '__main__':
    t = 256
    A = set(range(0, 1000))
    B = set(range(500, 1500))

    print(f"|A| = {len(A)}, |B| = {len(B)}")
    true_j = actual_jaccard(A, B)
    print(f"True Jaccard: {true_j:.4f}")

    A_list = [str(x) for x in A]
    B_list = [str(x) for x in B]
    sketcher = CMinHashSketch()
    print("Generating sketches...")
    S_A = sketcher.sketch(A_list)
    S_B = sketcher.sketch(B_list)
    print("Done.")

    est_j = estimate_jaccard(S_A, S_B)
    print(f"Estimated Jaccard: {est_j:.4f}")
    print(f"Error: {abs(true_j - est_j):.4f}")
