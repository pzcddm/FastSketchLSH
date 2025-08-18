import os
import pickle
import time
import argparse
import random
from collections import defaultdict
from typing import List, Set, Dict, Any, Tuple

import pandas as pd
from datasets import load_dataset, Features, Sequence, Value
from tqdm import tqdm

# 你的原有导入
from datasketch import MinHash, MinHashLSH
from datasketch.lsh import _optimal_param
from rensa import RMinHash, RMinHashLSH


class Timer:
    """计时器类，用于记录各个算法的执行时间"""

    def __init__(self):
        self.elapsed_times = {}
        self._start_time = None
        self._current_name = None

    def __call__(self, name):
        self._current_name = name
        return self

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self._start_time
        self.elapsed_times[self._current_name] = elapsed


class UnionFind:
    """并查集数据结构，用于管理重复文档的聚类"""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py


def _recall(row):
    """计算召回率"""
    labelled_dups = set(row["duplicates"])
    LEN_LABELLED_DUPLICATES = len(labelled_dups)
    if LEN_LABELLED_DUPLICATES == 0:
        return 1
    dups = set(row["predictions"])
    return len(dups & labelled_dups) / LEN_LABELLED_DUPLICATES


def _precision(row):
    """计算精确率"""
    labelled_dups = set(row["duplicates"])
    dups = set(row["predictions"])
    LEN_DUPLICATES = len(dups)
    if LEN_DUPLICATES == 0:
        return 0
    return len(dups & labelled_dups) / LEN_DUPLICATES


def classify_in_paper(record):
    """分类预测结果：TP, TN, FP, FN"""
    duplicates = set(record["duplicates"])
    predictions = set(record["predictions"])

    LEN_PREDICTIONS = len(predictions)
    LEN_DUPLICATES = len(duplicates)

    if LEN_PREDICTIONS == 0:
        if LEN_DUPLICATES == 0:
            return "TN"
        if LEN_DUPLICATES > 0:
            return "FN"

    if LEN_PREDICTIONS > 0:
        if LEN_DUPLICATES > 0 and duplicates.issubset(predictions):
            return "TP"
        if LEN_DUPLICATES == 0 or not duplicates.issubset(predictions):
            return "FP"

    raise ValueError(f"This should not happen {duplicates} {predictions}")


def inverse(label: str) -> str:
    """反转分类标签"""
    return {"TP": "TN", "FN": "FP", "FP": "FN", "TN": "TP"}[label]


def calculate_metrics(uf: UnionFind, truth: list, id2core_id: dict, labels: dict, name: str, elapsed_time: float):
    """计算并返回算法的各项指标"""
    # 从UnionFind构建聚类
    id2cluster = defaultdict(set)
    for idx in range(len(truth)):
        cluster_id = uf.find(idx)
        id2cluster[cluster_id].add(idx)

    # 生成预测结果
    predictions = {}
    for x in truth:
        idx = x["id"]
        core_id = id2core_id[idx]
        cluster_id = uf.find(idx)
        neighbors = id2cluster[cluster_id]
        predictions[core_id] = {
            id2core_id[neighbor] for neighbor in neighbors
            if neighbor != idx and neighbor in id2core_id
        }

    # 创建DataFrame进行评估
    df = (
        pd.Series(labels)
        .to_frame("duplicates")
        .reset_index()
        .merge(pd.Series(predictions).to_frame("predictions").reset_index(), on="index")
    )

    # 计算准确度
    df["Correct"] = df.apply(
        lambda row: set(row["duplicates"]) == set(row["predictions"]), axis=1
    ).astype(int)

    # 计算召回率和精确率
    recalls = df.apply(_recall, axis=1)
    precisions = df.apply(_precision, axis=1)

    # 分类结果
    df["Class"] = df.apply(classify_in_paper, axis=1)
    df["Class_"] = df.apply(lambda row: inverse(row["Class"]), axis=1)

    # 计算各类别的精确率和召回率
    metrics = {}
    for col in ["Class", "Class_"]:
        label_counts = df[col].value_counts().to_dict()
        tp = label_counts.get("TP", 0)
        fp = label_counts.get("FP", 0)
        fn = label_counts.get("FN", 0)

        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        metrics[f"{col}_precision"] = precision
        metrics[f"{col}_recall"] = recall

    return {
        "name": name,
        "precision_duplicates": metrics.get("Class_precision", 0),
        "recall_duplicates": metrics.get("Class_recall", 0),
        "precision_non_duplicates": metrics.get("Class__precision", 0),
        "recall_non_duplicates": metrics.get("Class__recall", 0),
        "macro_f1": (metrics.get("Class_precision", 0) + metrics.get("Class__precision", 0)) / 2,
        "accuracy": df["Correct"].mean(),
        "time": elapsed_time,
        "total_predictions": len(predictions),
        "mean_recall": recalls.mean(),
        "mean_precision": precisions.mean()
    }


def convert_results_to_unionfind(kept_indices: Set[int], all_indices: Set[int],
                                 candidate_sets: List[Set[int]]) -> UnionFind:
    """将去重结果转换为UnionFind格式"""
    uf = UnionFind()

    # 初始化所有索引
    for idx in all_indices:
        uf.find(idx)

    # 将候选集中的文档进行合并
    for i, candidates in enumerate(candidate_sets):
        if len(candidates) > 1:  # 只有当候选集有多个元素时才合并
            candidates_list = list(candidates)
            # 将候选集中的所有元素与第一个元素合并
            for j in range(1, len(candidates_list)):
                uf.union(candidates_list[0], candidates_list[j])

    return uf


def print_performance_comparison(results_table: List[Dict]):
    """打印性能对比"""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    # 找到基准算法（Datasketch）
    baseline = next(r for r in results_table if r["name"] == "Datasketch")

    for result in results_table:
        if result["name"] != "Datasketch":
            speedup = baseline["time"] / result["time"]
            accuracy_diff = result["accuracy"] - baseline["accuracy"]

            print(f"\n{result['name']} vs Datasketch:")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Accuracy difference: {accuracy_diff:+.4f}")
            print(f"  Mean precision difference: {result['mean_precision'] - baseline['mean_precision']:+.4f}")
            print(f"  Mean recall difference: {result['mean_recall'] - baseline['mean_recall']:+.4f}")


def create_comparison_table(results_table: List[Dict]) -> pd.DataFrame:
    """创建格式化的对比表格"""
    data = []
    for result in results_table:
        data.append([
            result["name"],
            f"{result['precision_duplicates']:.4f}",
            f"{result['recall_duplicates']:.4f}",
            f"{result['precision_non_duplicates']:.4f}",
            f"{result['recall_non_duplicates']:.4f}",
            f"{result['macro_f1']:.4f}",
            f"{result['accuracy']:.4f}",
            f"{result['time']:.2f}s"
        ])

    df = pd.DataFrame(data, columns=[
        "Algorithm",
        "Precision (Duplicates)",
        "Recall (Duplicates)",
        "Precision (Non Duplicates)",
        "Recall (Non Duplicates)",
        "Macro F1 score",
        "Accuracy",
        "Time"
    ])

    return df

def run_evaluation_benchmark(args):
    """运行完整的评估基准测试"""
    timer = Timer()

    print("Loading dataset...")
    # 加载数据集
    ds = load_dataset("pinecone/core-2020-05-10-deduplication", split="train")

    # 采样数据
    if args.ratio < 1.0:
        total_size = len(ds)
        sample_size = int(total_size * args.ratio)
        indices = random.sample(range(total_size), sample_size)
        ds = ds.select(indices)

    # 准备文本数据
    texts = []
    truth = []
    for idx, record in enumerate(ds):
        text = " ".join((record.get("processed_title", ""),
                         record.get("processed_abstract", ""))).lower()
        texts.append(text)
        truth.append({
            "core_id": record["core_id"],
            "id": idx,
            "duplicates": record.get("labelled_duplicates", [])
        })

    # 构建token sets
    token_sets = [list({tok for tok in text.lower().split() if tok}) for text in texts]

    # 构建标签映射
    id2core_id = {x["id"]: int(x["core_id"]) for x in truth}
    labels = {
        int(x["core_id"]): set(map(int, x["duplicates"])) if x["duplicates"] else set()
        for x in truth
    }

    # 计算LSH参数
    bands, rows = _optimal_param(args.lsh_threshold, args.num_perm, 0.5, 0.5)
    num_perm = bands * rows

    print(f"\nLSH Parameters:")
    print(f"  Threshold: {args.lsh_threshold}")
    print(f"  Bands: {bands}, Rows: {rows}")
    print(f"  Effective num_perm: {num_perm}")

    results_table = []

    # 1. 运行Datasketch LSH
    print("\nRunning Datasketch LSH...")
    with timer("Datasketch"):
        ds_res = run_datasketch_lsh_with_candidates(
            token_sets, args.lsh_threshold, num_perm, bands, rows
        )

    # 转换为UnionFind格式
    ds_uf = convert_results_to_unionfind(
        ds_res["kept_indices"],
        set(range(len(token_sets))),
        ds_res["candidate_sets"]
    )

    # 计算指标
    ds_metrics = calculate_metrics(
        ds_uf, truth, id2core_id, labels,
        "Datasketch", timer.elapsed_times["Datasketch"]
    )
    results_table.append(ds_metrics)

    # 2. 运行FastSketch LSH
    print("\nRunning FastSketch LSH...")
    with timer("FastSketch"):
        fs_res = run_fastsketch_lsh_with_candidates(
            token_sets, args.lsh_threshold, num_perm, bands, args.seed
        )

    fs_uf = convert_results_to_unionfind(
        fs_res["kept_indices"],
        set(range(len(token_sets))),
        fs_res["candidate_sets"]
    )

    fs_metrics = calculate_metrics(
        fs_uf, truth, id2core_id, labels,
        "FastSketch", timer.elapsed_times["FastSketch"]
    )
    results_table.append(fs_metrics)

    # 3. 运行Rensa LSH
    print("\nRunning Rensa LSH...")
    with timer("Rensa"):
        rs_res = run_rensa_lsh_with_candidates(
            token_sets, args.lsh_threshold, num_perm, bands, args.seed
        )

    rs_uf = convert_results_to_unionfind(
        rs_res["kept_indices"],
        set(range(len(token_sets))),
        rs_res["candidate_sets"]
    )

    rs_metrics = calculate_metrics(
        rs_uf, truth, id2core_id, labels,
        "Rensa", timer.elapsed_times["Rensa"]
    )
    results_table.append(rs_metrics)

    # 打印结果表格
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    df_results = pd.DataFrame(results_table)

    # 格式化输出
    print("\nDetailed Metrics:")
    for _, row in df_results.iterrows():
        print(f"\n{row['name']}:")
        print(f"  Precision (Duplicates): {row['precision_duplicates']:.4f}")
        print(f"  Recall (Duplicates): {row['recall_duplicates']:.4f}")
        print(f"  Precision (Non-Duplicates): {row['precision_non_duplicates']:.4f}")
        print(f"  Recall (Non-Duplicates): {row['recall_non_duplicates']:.4f}")
        print(f"  Macro F1 Score: {row['macro_f1']:.4f}")
        print(f"  Accuracy: {row['accuracy']:.4f}")
        print(f"  Mean Precision: {row['mean_precision']:.4f}")
        print(f"  Mean Recall: {row['mean_recall']:.4f}")
        print(f"  Time: {row['time']:.2f}s")

    # 创建对比表格
    comparison_df = df_results[[
        'name', 'mean_precision', 'mean_recall', 'accuracy', 'time'
    ]].round(4)

    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(comparison_df.to_string(index=False))

    # 保存结果
    if args.save_results:
        output_file = f"evaluation_results_{args.lsh_threshold}_{args.num_perm}.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    # 在打印结果表格后添加
    print_performance_comparison(results_table)
    # 创建并打印格式化表格
    comparison_table = create_comparison_table(results_table)
    print("\n" + "=" * 80)
    print("FORMATTED RESULTS TABLE (Markdown Format)")
    print("=" * 80)
    print(comparison_table.to_markdown(index=False))

# 修改原有的运行函数，添加候选集返回
def run_datasketch_lsh_with_candidates(
        token_sets: List[List[str]],
        threshold: float,
        num_perm: int,
        bands: int,
        rows: int
) -> Dict[str, Any]:
    """运行Datasketch LSH并返回候选集"""
    n = len(token_sets)

    # Phase1: Build MinHash
    start1 = time.perf_counter()
    minhashes = []
    for tokens in token_sets:
        m = MinHash(num_perm=num_perm)
        for tok in tokens:
            m.update(tok.encode("utf-8"))
        minhashes.append(m)
    phase1_time = time.perf_counter() - start1

    # Phase2: Insert LSH Index
    start2 = time.perf_counter()
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm, params=(bands, rows))
    for idx, m in enumerate(minhashes):
        lsh.insert(idx, m)
    phase2_time = time.perf_counter() - start2

    # Phase3: Query
    start3 = time.perf_counter()
    candidate_sets = [set(lsh.query(m)) for m in minhashes]

    # 去重处理
    to_remove = set()
    for i in range(n):
        candidates = candidate_sets[i]
        other_candidates = [c for c in candidates if c != i]

        if other_candidates:
            min_id = min([i] + other_candidates)
            to_remove.update([c for c in [i] + other_candidates if c != min_id])

    kept_indices = set(range(n)) - to_remove
    phase3_time = time.perf_counter() - start3

    return {
        "total_time": phase1_time + phase2_time + phase3_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": phase3_time,
        "kept_indices": kept_indices,
        "removed_count": len(to_remove),
        "kept_count": len(kept_indices),
        "candidate_sets": candidate_sets
    }


def run_fastsketch_lsh_with_candidates(
        token_sets: List[List[str]],
        threshold: float,
        num_perm: int,
        bands: int,
        random_seed: int = 42
) -> Dict[str, Any]:
    """运行FastSketch LSH并返回候选集"""
    from src.fast_sketch_lsh import FastSketchLSH
    n = len(token_sets)

    # 阶段1: 插入
    start1 = time.perf_counter()
    lsh = FastSketchLSH(threshold=threshold, sketch_size=num_perm,
                        bands=bands, random_seed=random_seed)
    for idx, tokens in enumerate(token_sets):
        lsh.insert(idx, tokens)
    phase1_time = time.perf_counter() - start1

    # 阶段2: 查询
    start2 = time.perf_counter()
    candidate_sets = [set(lsh.query(tokens)) for tokens in token_sets]
    phase2_time = time.perf_counter() - start2

    # 阶段3: 去重处理
    start3 = time.perf_counter()
    to_remove = set()
    for i in range(n):
        candidates = candidate_sets[i]
        other_candidates = [c for c in candidates if c != i]

        if other_candidates:
            min_id = min([i] + other_candidates)
            to_remove.update([c for c in [i] + other_candidates if c != min_id])

    kept_indices = set(range(n)) - to_remove
    phase3_time = time.perf_counter() - start3

    return {
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": phase3_time,
        "total_time": phase1_time + phase2_time + phase3_time,
        "kept_indices": kept_indices,
        "kept_count": len(kept_indices),
        "removed_count": len(to_remove),
        "candidate_sets": candidate_sets
    }

def run_rensa_lsh_with_candidates(
        token_sets: List[List[str]],
        threshold: float,
        num_perm: int,
        bands: int,
        random_seed: int = 42
) -> Dict[str, Any]:
    """运行Rensa LSH并返回候选集"""
    n = len(token_sets)

    # Phase1: Build MinHash
    start1 = time.perf_counter()
    minhashes = []
    for tokens in token_sets:
        m = RMinHash(num_perm=num_perm, seed=random_seed)
        m.update(tokens)
        minhashes.append(m)
    phase1_time = time.perf_counter() - start1

    # Phase2: Insert LSH Index
    start2 = time.perf_counter()
    lsh = RMinHashLSH(threshold=threshold, num_perm=num_perm, num_bands=bands)
    for idx, m in enumerate(minhashes):
        lsh.insert(idx, m)
    phase2_time = time.perf_counter() - start2

    # Phase3: Query
    start3 = time.perf_counter()
    candidate_sets = [set(lsh.query(m)) for m in minhashes]

    # 去重处理
    to_remove = set()
    for i in range(n):
        candidates = candidate_sets[i]
        other_candidates = [c for c in candidates if c != i]

        if other_candidates:
            min_id = min([i] + other_candidates)
            to_remove.update([c for c in [i] + other_candidates if c != min_id])

    kept_indices = set(range(n)) - to_remove
    phase3_time = time.perf_counter() - start3

    return {
        "total_time": phase1_time + phase2_time + phase3_time,
        "phase1_time": phase1_time,
        "phase2_time": phase2_time,
        "phase3_time": phase3_time,
        "kept_indices": kept_indices,
        "removed_count": len(to_remove),
        "kept_count": len(kept_indices),
        "candidate_sets": candidate_sets
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark LSH deduplication with evaluation metrics"
    )
    parser.add_argument(
        "--num_perm", type=int, default=128,
        help="Number of permutations for MinHash."
    )
    parser.add_argument(
        "--lsh_threshold", type=float, default=0.8,
        help="LSH threshold parameter for candidate generation."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--ratio", type=float, default=1.0,
        help="Fraction of the dataset to use (0 < ratio <= 1)."
    )
    parser.add_argument(
        "--save_results", action="store_true",
        help="Save results to CSV file."
    )

    args = parser.parse_args()
    run_evaluation_benchmark(args)
