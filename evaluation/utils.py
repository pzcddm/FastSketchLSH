import time
from collections import defaultdict


class Timer:
    """计时器类，用于测量代码块的执行时间"""

    def __init__(self):
        self.elapsed_times = {}
        self._start_times = {}
        self._current_timer = None

    def __call__(self, name):
        """使Timer实例可以作为上下文管理器使用"""
        self._current_timer = name
        return self

    def __enter__(self):
        self._start_times[self._current_timer] = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_times[self._current_timer] = time.perf_counter() - self._start_times[self._current_timer]

    def reset(self):
        """重置计时器"""
        self.elapsed_times = {}
        self._start_times = {}


class UnionFind:
    """并查集数据结构，用于跟踪重复项的聚类"""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        """查找x的根节点（带路径压缩）"""
        if x not in self.parent:
            self.parent[x] = x
            return x

        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """合并x和y所在的集合"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            self.parent[root_x] = root_y