from typing import Iterable, List

from rensa import CMinHash as RensaCMinHash


class CMinHashSketch:
    """
    Rensa CMinHash wrapper expecting pre-stringified items.

    Time Complexity: O(k * n); Space: O(k)
    """

    def __init__(self, num_perm: int = 128, seed: int = 42) -> None:
        self.num_perm = num_perm
        self.seed = seed

    def sketch(self, items: Iterable[str]) -> List[int]:
        hasher = RensaCMinHash(num_perm=self.num_perm, seed=self.seed)
        hasher.update(items)
        return hasher.digest()


