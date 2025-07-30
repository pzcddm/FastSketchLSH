import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# 测试不同规模的数组加法
array_sizes = [100, 200, 300, 400, 500, 600, 1000, 2000]  # 测试不同规模的数据
numpy_times = []

for size in array_sizes:
    # 生成随机数组
    a = np.random.rand(size)
    b = np.random.rand(size)

    # 计时开始
    start = time.time()
    _ = a + b  # 执行加法
    elapsed = time.time() - start
    print(f"{elapsed:.12f}")  # 输出如：0.12345678
    numpy_times.append(time.time() - start)

# 绘制性能曲线
plt.figure(figsize=(10, 6))
plt.plot(array_sizes, numpy_times, marker='o', label='NumPy Addition')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size (log scale)')
plt.ylabel('Execution Time (seconds, log scale)')
plt.title('NumPy Array Addition Performance')
plt.legend()
plt.show()
