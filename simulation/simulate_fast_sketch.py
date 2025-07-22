import os
import mmh3
import numpy as np
import matplotlib.pyplot as plt
import random
from util import generate_interval_sets_with_jaccard, estimate_jaccard
from src.fast_sketch import FastSimilaritySketch

# Set fixed random seeds for reproducibility
RANDOM_SEED = 42
NUMPY_SEED = 42

# 添加这两行来解决中文显示和负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# Remove the FastSimilaritySketch class definition from this file

# --- 模拟主程序 ---
if __name__ == '__main__':
    # Set seeds for reproducibility
    np.random.seed(NUMPY_SEED)
    random.seed(RANDOM_SEED)
    
    # --- 模拟参数 ---
    sketch_size_t = 256
    num_simulations = 400  # 模拟次数，越多曲线越平滑
    target_jaccard = 0.5    # Target Jaccard similarity

    print(f"开始模拟 {num_simulations} 次...")
    print(f"目标 Jaccard 值: {target_jaccard:.4f}")

    # Create one sketcher instance with fixed seed for reproducibility
    sketcher = FastSimilaritySketch(sketch_size=sketch_size_t, random_seed=RANDOM_SEED)

    results = []
    actual_jaccards = []
    
    for i in range(num_simulations):
        if (i + 1) % 100 == 0:
            print(f"  ...已完成 {i+1}/{num_simulations} 次模拟")
        # Use interval-based set generation for reproducibility and efficiency
        set_A, set_B, actual_j = generate_interval_sets_with_jaccard(
            target_jaccard=target_jaccard,
            set_size=10000,
            start_id=i * 100000  # ensure no overlap between different simulations
        )
        sketch_A = sketcher.sketch(set_A)
        sketch_B = sketcher.sketch(set_B)
        estimated_j = estimate_jaccard(sketch_A, sketch_B)
        results.append(estimated_j)
        actual_jaccards.append(actual_j)
    
    avg_actual_jaccard = np.mean(actual_jaccards)
    print(f"平均实际 Jaccard 值: {avg_actual_jaccard:.4f}")
    print("mean, var:", np.mean(results), np.var(results, ddof=1))
    print("theoretical var:", avg_actual_jaccard*(1-avg_actual_jaccard)/sketch_size_t)
    print("模拟完成。")

    # --- 绘制概率分布直方图 ---
    plt.figure(figsize=(12, 6))
    # 'density=True' 会将Y轴归一化为概率密度
    plt.hist(results, bins=50, density=True, alpha=0.75, label=f'模拟分布 (t={sketch_size_t})')
    
    # 画出平均真实的 Jaccard 值作为参考
    plt.axvline(avg_actual_jaccard, color='r', linestyle='--', linewidth=2, label=f'平均真实 Jaccard 值 = {avg_actual_jaccard:.3f}')
    
    plt.title('FastSketch 估计值的概率分布 (模拟)')
    plt.xlabel('估计的 Jaccard 相似度')
    plt.xlim(avg_actual_jaccard-0.1, avg_actual_jaccard+0.1)
    plt.ylabel('概率密度')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Save figure to the 'figures' folder inside simulation, with filename including parameters
    figures_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'figures'))
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    avg_set_size = np.mean([len(set_A), len(set_B)])
    figure_path = os.path.join(figures_dir, f'fast_jaccard_histogram_t{sketch_size_t}_jaccard{target_jaccard:.2f}_size{int(avg_set_size)}.png')
    plt.savefig(figure_path)
    print(f"Figure saved to: {figure_path}")
    
    plt.show()

    # 报告 P(估计值 > 0.4)
    results_array = np.array(results)
    prob_greater_than_0_4 = np.sum(results_array > 0.4) / num_simulations
    print(f"\n根据模拟，估计值 > 0.4 的概率约为: {prob_greater_than_0_4:.6f}")