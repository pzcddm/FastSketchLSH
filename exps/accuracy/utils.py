import time
from collections import defaultdict


class Timer:
    """Timer class for measuring execution time of code blocks."""

    def __init__(self):
        self.elapsed_times = {}
        self._start_times = {}
        self._current_timer = None

    def __call__(self, name):
        """Enable the Timer instance to be used as a context manager."""
        self._current_timer = name
        return self

    def __enter__(self):
        self._start_times[self._current_timer] = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_times[self._current_timer] = time.perf_counter() - self._start_times[self._current_timer]

    def reset(self):
        """Reset the timer."""
        self.elapsed_times = {}
        self._start_times = {}


class UnionFind:
    """Union-Find data structure for tracking clusters of duplicates."""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        """Find the root of x (with path compression)."""
        if x not in self.parent:
            self.parent[x] = x
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union the sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            self.parent[root_x] = root_y