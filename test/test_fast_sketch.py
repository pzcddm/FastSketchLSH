import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mmh3
import heapq

from FastSketchLSH import FastSimilaritySketch
from prototype.simulation.util import estimate_jaccard, actual_jaccard


if __name__ == '__main__':
    t = 256
    A = set(range(0, 1000))
    B = set(range(500, 1500))

    print(f"|A| = {len(A)}, |B| = {len(B)}")
    true_j = actual_jaccard(A, B)
    print(f"True Jaccard: {true_j:.4f}")

    # Integer path via SIMD implementation
    sketcher_simd = FastSimilaritySketch(sketch_size=t, seed=77787)
    print("[SIMD/int] Generating sketches...")
    S_A = sketcher_simd.sketch(A)
    S_B = sketcher_simd.sketch(B)
    print("[SIMD/int] Done.")

    est_j = estimate_jaccard(S_A, S_B)
    print(f"[SIMD/int] Estimated Jaccard: {est_j:.4f}")
    print(f"[SIMD/int] Error: {abs(true_j - est_j):.4f}")

    # String path via scalar implementation (UTF-8 encoded in binding)
    A_str = [str(x) for x in A]
    B_str = [str(x) for x in B]
    sketcher_str = FastSimilaritySketch(sketch_size=t, seed=77787)
    print("[scalar/str] Generating sketches...")
    S_A_str = sketcher_str.sketch(A_str)
    S_B_str = sketcher_str.sketch(B_str)
    print("[scalar/str] Done.")

    est_j_str = estimate_jaccard(S_A_str, S_B_str)
    print(f"[scalar/str] Estimated Jaccard: {est_j_str:.4f}")
    print(f"[scalar/str] Error: {abs(true_j - est_j_str):.4f}")

    # Bytes path via scalar implementation (pre-encoded strings)
    A_bytes = [s.encode('utf-8') for s in A_str]
    B_bytes = [s.encode('utf-8') for s in B_str]
    sketcher_bytes = FastSimilaritySketch(sketch_size=t, seed=2)
    print("[scalar/bytes] Generating sketches...")
    S_A_bytes = sketcher_bytes.sketch(A_bytes)
    S_B_bytes = sketcher_bytes.sketch(B_bytes)
    print("[scalar/bytes] Done.")

    est_j_bytes = estimate_jaccard(S_A_bytes, S_B_bytes)
    print(f"[scalar/bytes] Estimated Jaccard: {est_j_bytes:.4f}")
    print(f"[scalar/bytes] Error: {abs(true_j - est_j_bytes):.4f}")
