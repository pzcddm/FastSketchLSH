# 比datasketch好两个数量级? 用 Fast Similarity Sketch 优化大规模文本去重（附 C++ 源码）

项目地址：<https://github.com/pzcddm/FastSketchLSH>

论文背景：**Fast Similarity Sketching**（arXiv:1704.04370v4，FOCS’17 扩展版）

最近在优化大规模去重流水线，基于之前的知识, 我们team做了一个基于Fast Similarity Sketching 的Python 去重包fastsketchlsh, 并且我们先后对比了 `datasketch`、`rensa` 和我们自己做的 fastsketchlsh。  
这篇blog就把过程里的关键问题讲透：传统 k-mins 为什么慢、FastSketch 为啥快、以及它和 LSH 搭配时为什么在工程上可用。

---

## 先说结论

- 经典 **k-mins / MinHash** 用来估计 **Jaccard**，再配 **banding LSH** 做候选召回，是业界非常常见的近似去重路线。
- `datasketch` 很常用，但它是纯 Python 实现，规模一大通常会慢；`rensa` 是 k-mins 路线里的 SOTA 工程实现（Rust），速度明显更好。
- 在 sketch 这一核心阶段，FastSimilaritySketch 相比 `datasketch` 甚至能到 **200x** 。
- Fast Similarity Sketching 的关键理论是：在保持 alignment + Chernoff 级别集中界的前提下，把 sketch 构造期望复杂度降到：

$$
O(n + k\log k)
$$

其中 `k` 是 sketch size。

---

## 1. 大规模去重：常见的几条路线

做“去重”前先想清楚：你到底要去的是哪一种“重复”。

### 1.1 精确去重（Exactly Match）

最常见也最简单：

- 统一规范化（大小写、空白、标点、HTML 清洗等）后做哈希（MD5/SHA/xxhash）。
- 适用：完全重复、或你能把数据规整成“完全相同”的形式。

优点：快、准、实现简单。  
缺点：对“近似重复”（少量增删改、顺序变化、模板文本）不鲁棒。

### 1.2 近似去重（本文主要讲: 集合相似度 Jaccard）

文本（或 token 序列） -> shingle / n-gram -> 看成集合 -> **Jaccard 相似度**：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

一个“两个集合很相近”的直观例子：

- 设 `A={a,b,c,d,e,f,g,h,i,j}`（10 个元素）
- 设 `B={a,b,c,d,e,f,g,h,i,x}`（只把 `j` 换成 `x`）
- 交集 `|A∩B|=9`，并集 `|A∪B|=11`

所以：

$$
J(A,B)=\frac{9}{11}\approx 0.818
$$

这就是“很像但不完全相同”的典型情况。

然后用 **sketch（压缩签名）** 去近似估计相似度，再用 **LSH** 做候选召回。

这条路线的优点是：理论清晰、工程成熟、对“局部修改”的鲁棒性强；缺点是：当 token 集很大时，传统 MinHash 的计算量会飙升。

### 1.3 其他路线

- SimHash（Hamming 距离）/ LSH on Hamming：在文本去重里常见问题是 FP 或 FN 偏高，阈值非常敏感。
- Embedding + ANN（Faiss/HNSW）：语义效果更强，但向量化、索引构建、检索的整体成本都更高，很多去重任务性价比并不占优。

---

## 2. 为什么 k-mins / MinHash + LSH 这么流行？

因为它解决了一个“又准又好用”的点：**Jaccard 的无偏估计 + LSH 的候选召回**。本文统一用 `k` 表示 sketch size（论文原文记为 `t`，即这里 `k=t`）。

### 2.1 k-mins / MinHash 的核心性质（和 Jaccard 的关系）

经典 MinHash 做法：用 `k` 个独立哈希函数（或 `k` 个随机置换），每一维取最小哈希值：

$$
\text{sig}_i(A)=\min_{a\in A} h_i(a)
$$

然后用“逐维相等比例”估计 Jaccard：

$$
\hat{J}(A,B)=\frac{1}{k}\sum_{i=1}^k \mathbf{1}[\text{sig}_i(A)=\text{sig}_i(B)]
$$

关键结论：

$$
\Pr[\text{sig}_i(A)=\text{sig}_i(B)] = J(A,B)
$$

直觉上：把 `A ∪ B` 按哈希值排序，最小元素落在交集 `A ∩ B` 的概率就是 Jaccard。

一个最小例子（帮助理解 `kmins` 在做什么）：

- 设 `k=4`，集合 `A` 有 `n=5` 个 token：`{a,b,c,d,e}`。
- 用 4 个哈希函数分别扫这 5 个 token，每个函数只保留最小值：
  - `h1(A)` 最小值是 `5`
  - `h2(A)` 最小值是 `3`
  - `h3(A)` 最小值是 `4`
  - `h4(A)` 最小值是 `2`
- 则 `S(A)=[5,3,4,2]`。
- 若 `S(B)=[5,9,4,2]`，则 4 维里有 3 维相等，估计 `\hat{J}(A,B)=3/4=0.75`。

这个例子里你会直接看到：为了得到 4 维签名，我们确实做了“4 个哈希函数 x 5 个元素”的扫描。

### 2.2 为什么能和 LSH 结合：banding 概率曲线

把长度为 `k` 的签名切成 `b` 个 band，每个 band `r` 行（`k = b * r`）。如果两个集合在任意一个 band 完全相同，就作为候选对。

经典结论：若单行相等概率是 `s`（MinHash 情况下 `s = J`），候选概率为：

$$
P(\text{candidate}) = 1-(1-s^r)^b
$$

这就是常说的 S-curve，可通过调 `b,r` 在阈值附近做“软门槛”。

---

## 3. 为什么很多工程里 datasketch / 传统 MinHash 仍然慢？

这里按工程视角 + 算法复杂度直接说结论（不展开底层运行时细节）：

1. **`datasketch` 纯 Python 实现，工程上更慢**  
在大规模循环和哈希更新上，纯 Python 的吞吐很难拉满。

2. **k-mins 本身复杂度不低**  
经典 MinHash 复杂度是：

$$
O(k\cdot n)
$$

当每条文本 token 很多、或 n-gram 数量很大时，CPU 压力会明显上升。

3. **`rensa` 用 Rust 做到了很快，但仍然是 k-mins 路线**  
`rensa` 是当前 k-mins 路线的 SOTA 工程基线；但算法工作量结构仍然继承了 k-mins。

所以实际痛点是：`datasketch` 慢，`rensa` 快但仍受 k-mins 复杂度结构约束，我们希望继续降复杂度。

### 3.1 一个关键问题：能不能不做 `k` 次全量扫描？

k-mins 的工作方式，本质上是“每个元素都过 `k` 个哈希维度”：

$$
\text{work}_{\text{kmins}} \sim k\cdot n
$$

所以会自然出现一个问题：**能不能只用一个（或期望常数个）哈希轮次，完成同样长度为 `k` 的 sketch？**

Fast Similarity Sketching 的回答是：不再固定做 `k` 次全扫描，而是按轮次更新全部 `k` 个桶，桶填满就提前停止。  
如果记轮次数为 `R`，则总工作量是：

$$
O(R\cdot n)
$$

而论文给出：

$$
\mathbb{E}[R]=O\!\left(1+\frac{k\log k}{n}\right)
$$

于是得到：

$$
\mathbb{E}[\text{work}] = O(n+k\log k)
$$

带数字感受一下这两种工作量（仅看量级）：

- 设 `n=10,000`、`k=256`。
- k-mins：大致是 `k*n=2,560,000` 次“元素-哈希维度”处理。
- FastSketch：是 `R*n`。如果 `R=2`，就是约 `20,000` 次；如果 `R=3`，约 `30,000` 次。

所以 FastSketch 的关键不是“每一步更神奇”，而是把“必须做 `k` 轮”改成“做到桶填满就停”。

这就是从 `O(k\cdot n)` 降到 `O(n+k\log k)` 的核心思想，也是 FastSketchLSH 引入 FastSketch 的根本原因。

---

## 4. Fast Similarity Sketching：更快且保留 MinHash 级别性质

本仓库的 FastSketchLSH，核心思想来自论文 **Fast Similarity Sketching（arXiv:1704.04370v4）**，它正面回答了上面这个问题：在保持 MinHash 级别统计性质的同时，把“`k` 次全扫”改成“少量轮次 + 提前停止”。

### 4.0 先看 FastSketch 伪代码（工程版）

```text
Input : set A, sketch size k, regular rounds 2k   # paper notation: 2t
Output: sketch S[1..k]

Initialize S[i] = EMPTY for i = 1..k

# regular rounds
for r = 1..2k:
  for each token a in A:
    h   = hash(r, a)
    bin = 1 + (h mod k)
    key = (r, h)              # 保证早轮次优先、同轮按哈希最小
    S[bin] = min(S[bin], key)
  if all bins filled:
    break

# fallback rounds: only for empty bins
while exists empty bin:
  r = r + 1
  for each token a in A:
    h   = hash(r, a)
    bin = 1 + (h mod k)
    if S[bin] is EMPTY:
      key = (r, h)
      S[bin] = min(S[bin], key)

return S
```

注意：上面这段是**工程实现写法**（把轮次 `r` 作为哈希输入的一部分），更接近论文里说的 practical implementation。  
如果严格按论文 Algorithm 1 的原始定义，会写成 `2t` 个哈希函数：前 `t` 个把元素映射到区间 `[i, i+1)`，后 `t` 个映射到 `[i-t, i-t+1)` 做补位。两种写法目标一致：快速填满桶，并保持无偏估计性质。

### 4.1 先把符号讲清楚：什么是 `S(A)[i]`、alignment、Chernoff-style concentration

先给几个前置定义（这几个定义后面都会用到）：

- `A,B`：两条文本各自 tokenize 后得到的集合。
- `k`：sketch size（签名维度）。
- `S(A), S(B)`：`A,B` 的 `k` 维 sketch 向量。
- `S(A)[i], S(B)[i]`：第 `i` 维（第 `i` 个桶）上的 sketch 值。直观上它是“该维度里保留下来的最小 key/最小哈希值”，不是原始 token。

论文把“第 `i` 维是否匹配”定义为：

$$
X_i=\mathbf{1}[S(A)[i]=S(B)[i]]
$$

并令总匹配维度数：

$$
X=\sum_{i=1}^{k} X_i
$$

有了这两个量，就能解释两个核心性质：

1. **alignment（对齐）**  
   对两个集合来说，第 `i` 维比较的是“同一套随机机制下的同一维实验”，而不是乱序对比。  
   这保证了逐维比较有意义，并满足：

$$
\mathbb{E}[X_i]=J(A,B)
$$

下面给两条不靠“夹逼”的证明思路（博客里可并列放）：

**路线 A：按轮次分解（第一轮、第二轮、...、兜底轮）**

固定一个坐标 `i`，令并集 `U=A\cup B`，交集大小记为 `c=|A\cap B|`，并集大小记为 `m=|U|`，则 `J=c/m`。

- 定义 `E_r`：坐标 `i` 在第 `r` 轮（`r=1,\dots,2k`，论文写作 `2t`）首次命中。
- 定义 `E_\star`：前 `2k` 轮都未命中，最后由兜底步骤填充。
- 事件族 `\{E_1,\dots,E_{2k},E_\star\}` 两两互斥且完备：

$$
\sum_{r=1}^{2k}\Pr[E_r]+\Pr[E_\star]=1
$$

在任一事件 `E`（`E_r` 或 `E_\star`）下，坐标 `i` 的 winning 元素记为 `W_i`。  
由于哈希/分桶过程对 `U` 中每个元素是同分布、与“是否属于交集”无关，`W_i` 落到交集的条件概率恒为：

$$
\Pr[X_i=1\mid E]=\Pr[W_i\in A\cap B\mid E]=\frac{c}{m}=J
$$

按全概率公式：

$$
\Pr[X_i=1]
=\sum_{r=1}^{2k}\Pr[E_r]\Pr[X_i=1\mid E_r]+\Pr[E_\star]\Pr[X_i=1\mid E_\star]
=\left(\sum_{r=1}^{2k}\Pr[E_r]+\Pr[E_\star]\right)J
=J
$$

因此：

$$
\mathbb{E}[X_i]=\Pr[X_i=1]=J
$$

轮次上限 `2k`（论文记作 `2t`）只影响“首次命中发生在第几轮”的分布，不改变期望。

**路线 B：排列对称性**

同样固定坐标 `i`，令 `W_i` 为最终写入该坐标的元素。  
FastSketch 的规则是 label-oblivious 的：只看哈希值/键值，不看元素身份。对 `U` 上任意置换 $\sigma$，都有分布不变性：

$$
W_i \overset{d}{=} \sigma(W_i)
$$

所以 `U` 中每个元素成为 winner 的概率相同。设该概率为 `p`，则

$$
\sum_{x\in U}\Pr[W_i=x]=1 \Rightarrow mp=1 \Rightarrow p=\frac{1}{m}
$$

于是

$$
\Pr[X_i=1]
=\Pr[W_i\in A\cap B]
=\sum_{x\in A\cap B}\Pr[W_i=x]
=c\cdot \frac{1}{m}
=\frac{c}{m}
=J
$$

同样得到 $\mathbb{E}[X_i]=J$。  
这里把它作为直觉解释更合适：严格证明要结合算法构造细节。关键点是补位机制没有破坏均匀性，所以不会因为“空桶修复”引入偏差。

2. **Chernoff-style concentration（Chernoff 风格集中）**  
   直观上，`X` 不只是“期望正确”，还会高概率贴近 `kJ`，偏差概率随 `k` 指数下降。可写成典型形式：

$$
\Pr\big(|X-kJ|>\epsilon kJ\big)\le 2e^{-c\epsilon^2 kJ},\quad 0<\epsilon\le 1
$$

其中 `epsilon` 是**相对误差阈值**：例如 `epsilon=0.1` 表示“偏离超过 `10% × kJ`”。  
`c>0` 是常数（与 `k, J, epsilon` 无关）。

这对 LSH 很关键：你按阈值做召回时，结果更稳定，不容易因为采样噪声大幅抖动。

### 4.2 核心算法（Similarity-S）

下面统一用 `k` 表示 sketch size：

- 每一轮只用一组哈希（可理解为一个 seed 下的哈希映射），但这一轮会同时尝试更新全部 `k` 个桶。
- 维护 `k` 个桶，每个桶保留一个最小 key。
- 反复做“随机分桶 + 取最小”，直到所有桶被填满。
- 若仍有空桶，用额外轮次补空，保证每个维度都定义良好。

一个直观例子（为什么它常常很快）：

- 设 `k=256`，一条文本有 `n=1200` 个 token。
- 第 1 轮里，1200 个 token 会被随机映射到 256 个桶并更新最小 key。
- 当 `n` 相对 `k` 足够大时，绝大多数桶会在第一轮就被命中并填满，此时直接停止，`R≈1`。
- 这也和我们在仓库里的经验一致：当 token 超过约 700 时，很多样本第一轮就能填满全部桶。

### 4.3 关键复杂度结论（Lemma 1）

论文给出的期望复杂度：

$$
O(n + k\log k)
$$

和传统 MinHash 对比：

$$
O(k\cdot n) \rightarrow O(n + k\log k)
$$

本质是把 `n` 从乘法项变成加法项，大集合时收益非常大。

这里的 `k\log k` 直觉来自 Coupon Collector：要把 `k` 个桶基本填满，期望命中次数量级是 `\Theta(k\log k)`。  
可以把它想成“随机往 `k` 个桶里扔球”：前面很快就能命中大部分桶，但最后几个空桶会越来越难补齐，因此总次数会多出一个 `\log k` 因子。

### 4.4 为什么还能估计 Jaccard 并有 Chernoff 集中

上一节已经给了一个工程直觉版结论：单维有

$$
\mathbb{E}[X_i]=J
$$

从而

$$
\mathbb{E}[X/k]=J
$$

更严格的概率尾界（Chernoff-style concentration）在论文里用更完整的工具来证明；工程上你可以理解为：`X` 会稳定集中在 `kJ` 附近，这就是 FastSketch 做 LSH 阈值筛时表现稳定的根本原因。

---

## 5. FastSketchLSH 这个 repo 具体做了什么？

FastSketchLSH 是面向大规模去重的研究/工程实现集合：

- **C++ 扩展**（`fastsketchlsh_ext/`）
  - `FastSimilaritySketch`：FastSketch 签名生成
  - `LSH`：高吞吐 band-parallel LSH（batch build / batch query）
- **Python 原型与对照实现**（`prototype/src/`）
  - `kmins_sketch.py`：经典 k-mins
  - `fast_sketch.py`：FastSketch 原型
- **模拟实验**（`prototype/simulation/`）
  - FastSketch vs k-mins 估计分布/方差
  - LSH 碰撞概率曲线对比
- **去重对比脚本**（`comparison/`）
  - 同一数据集下跑 fastsketch / rensa
- **基准结果**（`evaluation/results.md`、README）

---

## 6. 我们的 FastSketch 实现原理（工程版本）

结合 C++ 实现（`fastsketchlsh_ext/include/fastsketch.h` + `fastsketchlsh_ext/cpp/fastsketch.cpp`），核心流程可理解为：

1. 预哈希：先把 token 映射到 64-bit 基值。
2. 轮次 `i`：每轮用一个 seed 对 token 计算 hash。
3. 分桶：用 hash 低位映射到 `k` 个桶。
4. 每桶取最小 key：保证更早轮次优先。
5. 提前停止：桶全部填满就停止。
6. 补空：如果仍有空桶，用额外轮次补齐。

这和论文 Similarity-S 的目标一致，但写法是工程优化版。  
再强调一次：论文 Algorithm 1 原始写法是“`2t` 个哈希函数 + 前后半区间映射”；这里实现的是“轮次入哈希”的 practical 版本，代码路径更直接。

另外补一个工程实测结论：**当 token 数超过约 700 时，约 99% 的样本会在第一轮就填满所有桶**。这也是中长文本上吞吐很高的直接原因。

---

## 7. 为什么 FastSketch 也能和 LSH 配合得很好？

LSH 需要的核心是“逐维匹配概率稳定 + 总体匹配率可集中”。论文证明 FastSketch 具备与 MinHash 同级别的这两点，因此 banding 行为会非常接近经典 k-mins。

$$
p_{\text{band}}=\Pr[\text{一个 band 的 }r\text{ 行全相等}]
$$

若把 band 内 `r` 行当作独立，有：

$$
p_{\text{band}}\approx J^r
$$

但 FastSketch 的行与行并非严格独立。论文给了可用的界：对一个 band 里的 `r` 个坐标，

$$
\prod_{h=0}^{r-1}\max\!\left(0,\frac{kJ-h}{k-h}\right)
\le p_{\text{band}} \le J^r
$$

例如常用参数 `k=128,r=8,J=0.7` 时，上界 `J^r=0.0576`，下界约 `0.0522`，两者已经很接近。

在工程里我们据此用近似：

$$
P(\text{candidate}) \approx 1-(1-p_{\text{band}})^b \approx 1-(1-J^r)^b
$$

虽然论文为了理论完备性在 LSH 部分使用了 Tensoring 结构来处理相关性，但我们的实验表明：在常见 sketch size（如 `k=128/256`）下，这种弱相关性对 banding 的 S-curve 影响通常可以忽略。

“band 间近似独立”可这样理解：

1. 每个 band 使用不重叠的坐标组；
2. band key 进入独立哈希表；
3. 仓库模拟曲线几乎贴着经典理论曲线，说明残余相关性在这些参数下很弱、可忽略。

所以这里是“工程上的近似独立”，不是严格数学独立；但对召回判别已经足够好用。  
仓库模拟脚本：`prototype/simulation/display_lsh_probdist.py`。

### 7.1 我们自己模拟的 S-curve：k-mins LSH 和 FastSketch LSH 几乎重叠

下面这张图是我们在仓库里模拟得到的（`k=128, b=16, r=8, \theta=0.7`）：

![KminsLSH&FastSketchLSH_Distribution](https://i-blog.csdnimg.cn/direct/a071252bf09d467180231f665880d2c6.png#pic_center)


图里的红线（k-mins LSH 理论曲线）和蓝线（FastSketch LSH 模拟）几乎重叠，说明两者在 LSH 判别行为上基本一致。  
（对应模拟脚本：`prototype/simulation/display_lsh_probdist.py`，原始输出图在 `prototype/simulation/figures/`）

如果按“`0.7` 附近开始判定相似”这个工程目标来理解，这条曲线可以直观看成：

- `J=0.9`：候选概率几乎是 `1`（理论值约 `0.9999`，基本百分百判为像）
- `J=0.4`：候选概率接近 `0`（理论值约 `0.01`，基本不会判为像）
- `J≈0.7`：处在拐点附近（理论值约 `0.61`），是“是否相似”的主要分界区域

也就是说，FastSketch 在保留 k-mins/MinHash 这套 LSH 判别逻辑的同时，把 sketch 计算做得更快。

### 7.2 为什么我们没完全照搬论文那套 LSH 结构？

论文在 LSH 部分还给了更“理论优化”的方案（中间 sketch + 采样 + tensoring），目标是把查询复杂度做到更强的上界。  
这套做法理论上更完整，但工程实现链路更复杂、常数开销也更大。

本仓库选择了更直接的经典 banding 路线：

- 结构简单，易维护，容易和现有去重流水线集成；
- 在我们的数据与参数下，S-curve 几乎贴合经典 k-mins 理论；
- 结合 FastSketch 的快速建签名，端到端吞吐更有优势。

---

## 8. 结果：速度与准确性

### 8.0 Setting

下面表格为了公平起见我们按单线程看结果（用于和其他实现做公平对比）。即便在这个setting下，FastSketchLSH 也有明显 speedup。  
同时 FastSketchLSH 本身支持多线程：`FastSimilaritySketch.sketch_batch(..., num_threads=...)` 和 `LSH(..., num_threads=...)`，在 OpenMP 可用时还能继续提速。

### 8.1 估计分布/方差（simulation）

`prototype/simulation/README.md` 示例（k=128，目标 J=0.5）：

- FastSketch 方差约 `0.001447`
- KMinSketch 方差约 `0.002175`

同样 `k` 下，FastSketch 更集中。

一个很稳定的经验是：`k` 越大、文档（token set）越大，FastSketch 的相对优势通常越明显。
### 8.2 去重流水线时间（evaluation）

单线程对比示例：

| Dataset | Engine | Sketch(s) | Build(s) | Query(s) | Total(s) | Sketch Speedup | Total Speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| BOOKCORPUSOPEN | rensa | 198.545 | 0.026 | 0.018 | 198.589 | - | - |
| BOOKCORPUSOPEN | fastsketchlsh | 55.280 | 0.039 | 0.031 | 55.350 | 3.59× | 3.59× |
| BOOKS3 | rensa | 95.915 | 0.005 | 0.003 | 95.923 | - | - |
| BOOKS3 | fastsketchlsh | 28.440 | 0.008 | 0.007 | 28.455 | 3.37× | 3.37× |
| PINECONE | rensa | 3.929 | 0.141 | 0.153 | 4.223 | - | - |
| PINECONE | fastsketchlsh | 1.521 | 0.249 | 0.396 | 2.166 | 2.58× | 1.95× |
| SHUYUEJ | rensa | 3.749 | 0.037 | 0.044 | 3.830 | - | - |
| SHUYUEJ | fastsketchlsh | 1.132 | 0.093 | 0.121 | 1.346 | 3.31× | 2.85× |

对应总耗时 speedup：

- `BOOKCORPUSOPEN`: `3.59x`
- `BOOKS3`: `3.37x`
- `PINECONE`: `1.95x`
- `SHUYUEJ`: `2.85x`
---



## 9. 快速上手：安装与最小示例

### 9.1 安装

先确保 Python 版本在 `3.11+`。

**推荐：直接 PyPI 安装**

```bash
pip install fastsketchlsh
```

**如果你要从源码构建（用于实验复现）**

```bash
git clone https://github.com/pzcddm/FastSketchLSH.git
cd FastSketchLSH/fastsketchlsh_ext
pip install .
cd ..
pip install -r requirements.txt
```

### 9.2 生成 sketch 并估计 Jaccard

```python
from FastSketchLSH import FastSimilaritySketch, estimate_jaccard

list_a = [f"a-{i}" for i in range(16_000)]
list_b = [f"a-{i}" for i in range(8_000)] + [f"b-{i}" for i in range(8_000)]

sketcher = FastSimilaritySketch(sketch_size=256)
sig_a = sketcher.sketch(list_a)
sig_b = sketcher.sketch(list_b)

estimated = estimate_jaccard(sig_a, sig_b)
print(f"Estimated Jaccard similarity: {estimated:.4f}")
```

### 9.3 LSH：batch build + batch query

```python
from __future__ import annotations

from datasets import load_dataset
from FastSketchLSH import FastSimilaritySketch, LSH

def tokenize(text: str) -> list[str]:
    return sorted({token for token in text.lower().split() if token})

dataset = load_dataset("lucadiliello/bookcorpusopen", split="train[:2048]")
texts = [row["text"] for row in dataset if row.get("text")]
token_sets = [tokenize(text) for text in texts]

sketcher = FastSimilaritySketch(sketch_size=128, seed=42)
sketch_matrix = sketcher.sketch_batch(token_sets)

lsh = LSH(num_perm=128, num_bands=16)
lsh.build_from_batch(sketch_matrix)

doc_idx = 0
candidates = lsh.query_candidates(sketch_matrix[doc_idx])
print(f"Candidates for {doc_idx}:", candidates)

dup_flags = [1 if len(lsh.query_candidates(row)) > 1 else 0 for row in sketch_matrix]
print("Duplicate flags:", dup_flags)
print("Total duplicates detected:", sum(dup_flags))
```

### 9.4 复现端到端对比（fastsketch vs rensa）

```bash
python -m exps.end2end.run --engine fastsketch --dataset PINECONE
python -m exps.end2end.run --engine rensa --dataset PINECONE

# 批量跑 README 里的对比表
bash exps/end2end/run_all_comparisons.sh
```

---

## 10. 参数建议与常见坑

1. **sketch_size（k）**：常用 `128/256/512`。k 越大越稳，但计算和内存也更高。
2. **LSH bands/rows**：满足 `k = b * r`，按目标阈值调 S-curve。
3. **tokenization 很关键**：分词、停用词、ngram 直接影响 Jaccard 语义。
4. **LSH 只做候选召回**：虽然在目标Jaccard = 0.8/0.9 的时候一般来说已经很准了 但是如果设置的阈值较低的话通常还要做二阶段验证（精确 Jaccard 或其他相似度）。
5. **经验规律**：在我们的实验里，`k` 越大、文档越长（token 越多），FastSketch 相对传统 `O(k·n)` 路线的速度优势通常越大；小 `k`、小文档下差距会缩小。
---

## 11. 总结

- k-mins / MinHash + LSH 是经典去重主线，但复杂度瓶颈真实存在。
- `datasketch` 纯 Python，常见场景会慢；在 sketch 阶段，FastSimilaritySketch 对它通常有一个数量级以上优势（常见 200x+）。
- `rensa` 是 k-mins 路线 SOTA；端到端对比里 FastSketchLSH 仍有稳定加速（约 1.95x–3.59x）。
- Fast Similarity Sketching 把复杂度做到 `O(n + k log k)`，并保持 MinHash 级别理论性质。
- FastSketchLSH 给了可复现、可落地的工程实现与实验结果.

---

## 支持项目

如果这篇文章对你有帮助，欢迎到仓库给我们点一个 Star，这对我们后续持续优化和维护真的真的非常重要：  
<https://github.com/pzcddm/FastSketchLSH>
---
## 致谢

感谢团队的持续投入与协作：

- Zhencan Peng
- Yixuan Cao
- Fangcao Jian
- Huitong Jin
## 参考

- Fast Similarity Sketching（arXiv:1704.04370v4）  
  <https://arxiv.org/abs/1704.04370>
- FastSketchLSH（GitHub）  
  <https://github.com/pzcddm/FastSketchLSH>
- datasketch（GitHub）  
  <https://github.com/ekzhu/datasketch>
- rensa（GitHub）  
  <https://github.com/beowolx/rensa>
