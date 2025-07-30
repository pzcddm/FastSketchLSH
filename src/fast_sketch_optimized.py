import mmh3
import numpy as np
from typing import Iterable, List

class FastSimilaritySketchNP:
    """
    NumPy-accelerated Fast Similarity Sketch (Algorithm 1)
    保持与原算法一致的语义，提升速度：
    - 仅对每个元素做一次 mmh3（seed=0）预哈希
    - 后续各轮种子用 NumPy 向量化的 64bit 混洗函数生成
    - 分桶最小值用 np.minimum.at 做 segment-min
    期望时间仍为 O(n + t log t)
    实际表现发现并没有好 反而差了一点
    """

    def __init__(self, sketch_size: int, random_seed: int = 42):
        if not isinstance(sketch_size, int) or sketch_size <= 0:
            raise ValueError("Sketch size (t) must be a positive integer.")
        self.t = sketch_size
        # 统一使用 uint64，避免溢出；使用 Generator 提高速度与可复现性
        self.rng = np.random.default_rng(random_seed)
        self.seeds = self.rng.integers(
            low=0, high=2**64, size=2 * self.t, dtype=np.uint64
        )
        # 是否是 2 的幂，便于用位运算替代取模
        self._t_is_pow2 = (self.t & (self.t - 1)) == 0
        if self._t_is_pow2:
            self._t_mask = np.uint64(self.t - 1)

    @staticmethod
    def _mix64(x: np.ndarray) -> np.ndarray:
        """
        SplitMix64 风格的 64-bit 混洗，纯 NumPy 可广播版本。
        参考 Sebastiano Vigna 的常用混洗序列。输入输出均为 uint64。
        """
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
        x = (x ^ (x >> np.uint64(31)))
        return x.astype(np.uint64, copy=False)

    @staticmethod
    def _to_uint64_max() -> np.uint64:
        return np.uint64(np.iinfo(np.uint64).max)

    def _bucket(self, h: np.ndarray) -> np.ndarray:
        # b = h % t；若 t 为 2 的幂，用按位与更快
        if self._t_is_pow2:
            return (h & self._t_mask).astype(np.int64, copy=False)
        else:
            return (h % self.t).astype(np.int64, copy=False)

    def sketch(self, A: Iterable) -> List[int]:
        A_list = list(A)
        n = len(A_list)
        if n == 0:
            # 与原实现一致：返回 t 个 "inf" 替代值（这里用全 0），或你也可按需定义
            return [0] * self.t

        # 1) 预哈希（一次性）：seed=0
        base_hashes = np.empty(n, dtype=np.uint64)
        for idx, a in enumerate(A_list):
            # mmh3 返回 (low, high)，取 [0] 与原代码一致
            base_hashes[idx] = np.uint64(
                mmh3.hash64(str(a).encode("utf-8"), seed=0, signed=False)[0]
            )

        # 结果容器
        filled = np.zeros(self.t, dtype=bool)
        best_hash = np.full(self.t, self._to_uint64_max(), dtype=np.uint64)

        # 2) 前半阶段：i = 0..t-1，按 i 递增
        for i in range(self.t):
            if filled.all():
                break

            # h_i = mix64(base ^ seed_i)  —— 完全向量化
            h = self._mix64(base_hashes ^ self.seeds[i])

            # 只对“尚未填的桶”进行分桶最小值
            b = self._bucket(h)
            # 筛掉那些映射到“已填桶”的元素，减少 np.minimum.at 的工作量
            mask = ~filled[b]
            if not mask.any():
                continue

            b2 = b[mask]
            h2 = h[mask]

            # 分桶最小值（同一 i 内对每个桶取 hash 最小）
            tmp_min = np.full(self.t, self._to_uint64_max(), dtype=np.uint64)
            # segment-min：对同一桶索引进行最小归约
            np.minimum.at(tmp_min, b2, h2)

            # 只更新还未填、且当轮确实命中的桶
            upd_mask = (~filled) & (tmp_min != self._to_uint64_max())
            if upd_mask.any():
                best_hash[upd_mask] = tmp_min[upd_mask]
                filled[upd_mask] = True

        # 3) 后半阶段：对仍未填的桶 b，用固定 i=t+b 一次性补齐（可批量）
        if not filled.all():
            remaining_b = np.flatnonzero(~filled)
            # 批量计算 h = mix64(base ^ seed_{t + b})，然后对列做 min
            rem_seeds = self.seeds[self.t + remaining_b]  # shape = (R,)
            # 广播到 (n, R)
            h_rem = self._mix64(base_hashes[:, None] ^ rem_seeds[None, :])
            # 每个剩余桶的最小 hash
            mins = h_rem.min(axis=0).astype(np.uint64, copy=False)
            best_hash[remaining_b] = mins
            filled[remaining_b] = True

        # 返回与原实现一致的“只保留 hash 值”的列表
        return best_hash.tolist()
