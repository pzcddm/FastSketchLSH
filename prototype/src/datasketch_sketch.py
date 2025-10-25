from typing import Iterable, List

from datasketch import MinHash


class DatasketchMinHashSketch:
    """
    Datasketch-based MinHash wrapper expecting pre-stringified items.

    Time Complexity: O(k * n); Space: O(k)
    """

    def __init__(self, num_perm: int = 128, random_seed: int = 42) -> None:
        self.num_perm = num_perm
        self.random_seed = random_seed

    def sketch(self, items: Iterable[str]) -> List[int]:
        hasher = MinHash(num_perm=self.num_perm, seed=self.random_seed)
        for item in items:
            hasher.update(item.encode("utf-8"))
        return list(hasher.hashvalues)


