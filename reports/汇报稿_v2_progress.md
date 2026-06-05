# Re=40 PINN v2 阶段汇报稿（约 5 分钟，学术口吻）

> 配套幻灯片：`Reconstruction of Flow Fields using PINNs - v2 progress.pptx`，
> 重点讲 slide 24–27 这四张新页。

---

## 0. 引言（约 30 秒）

老师好。这次汇报承接上一次 Run A / Run B 的结果，主要报告自上次 baseline
以来在 Re=40 单时刻重建上完成的 v2 系列实验。
这一阶段的核心目标，是检验您之前提出的 **(Y) 一致性判据** 在我们当前
的 PINN setup 下是否能被同时满足，并定位影响它的关键变量是 **权重** 还是
**容量**。

判据的形式回顾如下：从纯数据拟合的 Phase 1 切换到 "数据 + PDE" 的
Phase 2 之后，要同时达到

  - mae(P2) / mae(P1) ≤ 2×（流场不能漂离 CFD 太多），
  - PDE residual 下降 ≥ 5×（PDE 项必须真的被压下去）。

只有这两条 **同时** 成立，才能说明 CFD 自身与 NS 在我们这个 setup 下是
相容的，且 PINN 在两阶段切换下找到了 **同一个极小点**。

---

## 1. 实验矩阵（约 1 分钟，对应 slide 24）

在固定参考 vtk、固定 box (-8,12)×(-8,8)、固定 hard BCs + Fourier features
的前提下，我系统性地跑了六组 v2 实验，全部以 Phase 1 的纯数据拟合作为
基线，再切到不同形式的 Phase 2，结果如下：

  - Phase 1 baseline（data-only，全部 CFD 点）：mae_u = 0.106；
  - 严格一致性（Phase 2 完全不用 data）：mae_u 漂到 0.339，3.2× drift；
  - 弱 anchor（500 点，prio = 0.5）：1.64× drift；
  - 全 CFD anchor + data_priority sweep（0.5 / 1.0 / 2.0 / 5.0）：
    drift 从 1.48× 单调降到 1.11×，PDE drop 从 1.57× 单调降到 0.95×；
  - scale-up（width=64 / depth=4，prio = 2.0）：drift 1.15×，PDE drop 5.97×。

可以看到，**只有最后一行同时跨过 (Y) 的两道门槛**。下面我把这张表展开
成三个分析点。

---

## 2. 失败模式 1：去掉 anchor 之后的 trivial attractor（约 1 分钟，slide 25）

第一个对照实验是 **严格一致性**：网络保持小容量 width=32 / depth=3，
Phase 1 拟合全部 CFD 点，Phase 2 把数据项完全关掉，只剩 PDE residual
加上 hard BCs。

结果是：PDE residual 在 BFGS 下被压到接近零（满足弱形式的 (X) 判据），
但 mae_u 从 0.106 漂到 0.339，3.2 倍 drift，尾流回流结构整体消失，速度场
退化为接近均匀来流。

物理上这其实是个 **trivial attractor**：均匀场 (u, v) ≈ (1, 0) 在除壁面
之外的整个域上都精确满足不可压 NS。Hard BCs 只把壁面零速度强行写入网络
结构，却并没有排除这个均匀解。所以一旦数据 anchor 被去掉，BFGS 找到的
不是"与 CFD 一致的 NS 解"，而是"另一个合法的 NS 极小"。

结论：在这种 capacity 下，纯 PDE 模式在 (Y) 判据下 **fail**。data anchor
必须保留，或者必须从 capacity 上动手。

---

## 3. 失败模式 2：权重 sweep 给出严格单调的 trade-off（约 1 分钟，slide 26）

下一步检查：在保留 anchor 的前提下，仅仅调 anchor 的权重 (data_priority)
能否找到 sweet spot？

实验设计：仍然是小网络 32/3，Phase 2 始终保留全部 7456 个 CFD anchor，
只扫 data_priority = 0.5, 1.0, 2.0, 5.0。结果如下：

  - data_priority 越大 → mae drift 单调下降：1.48× → 1.33× → 1.21× → 1.11×；
  - data_priority 越大 → PDE drop 单调下降：1.57× → 1.45× → 1.17× → 0.95×。

也就是说，**两个目标在权重维度上是严格单调反向的**，没有任何中间点能让
(Y) 的两半同时达标。

物理解读：在 width=32 / depth=3 这种小容量下，"匹配 CFD 的极小"和
"低 PDE residual 的极小"在参数空间里位于 **不同的 sheet** 上。调权重
只是让优化器沿着这两个极小之间的 Pareto 前沿滑动，并不能让两个极小合并。

由此得到一个比较干净的推论：**问题的杠杆不在 weight，而在 capacity。**

---

## 4. capacity 实验：(Y) 在同一协议下自动成立（约 1 分钟，slide 27）

基于上面这条推论，我做了 capacity 对照。**实验协议完全不动**——同样的
P1 / P2 两阶段、同样的 data_priority = 2.0、同样保留全部 CFD anchor、
同样 n_f = 20k——只把网络放大到 width=64 / depth=4。

结果如下：

  - Phase 1 mae_u = 0.0186，相比小网络的 0.106 改善 5.7×，说明大网络
    在纯数据拟合阶段就明显更接近真实场；
  - Phase 2 mae_u = 0.0213，rel_l2_u = 3.2%；
  - mae P2/P1 = 1.15×，(Y) 要求 ≤ 2×，**通过**；
  - PDE residual 0.084 → 0.014，下降 5.97×，(Y) 要求 ≥ 5×，**通过**。

也就是说，**两个条件在同一个 protocol、同一个 data_priority、同一个
随机种子下同时满足**。

物理解读：在 width=64 这一档，CFD-matching 的极小和 low-PDE-residual
的极小在数值上基本是 **同一个点**。前一节看到的"两个 sheet"在大容量
下合并了。这同时也回应了上一阶段的悬而未决的问题——CFD 与 NS 在我们
这个 setup 下并非互相矛盾，只是小网络无法表达让两者同时满足的解。

---

## 5. 小结与下一步（约 30 秒）

总结这一阶段的两条主要结论：

1. 在小容量下（32/3），无论是关掉 data anchor 还是连续调 anchor 权重，
   都无法同时跨过 (Y) 的两道门槛；这一档的 PINN 必然在
   "匹配 CFD"与"满足 PDE"之间做单调取舍。
2. 把网络放大到 64/4，在协议完全不变的情况下 (Y) 自动成立。
   这说明 CFD 与 NS 是相容的，瓶颈是模型容量。

下一步计划分两步走：

  - 把网络再放大到 width=96 / depth=5，配合更长的 BFGS，把 (Y) 的数字
    做到 publication-quality；
  - 在此基础上启动 Re curriculum 与 Re sweep，进入项目原定的多 Reynolds
    数流场重建与转捩区分析任务。

汇报到这里，请老师指点。

---

## 6. Q&A 备忘（不读出，备查）

**Q：为什么不直接从 96/5 开始，要做小网络对照？**
> 小网络的"两 sheet"结构正是说明 (Y) 是 capacity-bound 的关键证据。
> 如果直接上 96/5 看到 (Y) 通过，无法区分到底是"weight 调好了"还是
> "capacity 够了"。

**Q：scale-up 的 5.97× PDE drop 会不会只是因为大网更容易把 PDE 项
压下去，与 CFD 一致性无关？**
> 不会。Phase 1 是 data-only、PDE 项关闭；Phase 2 才打开 PDE。
> 如果 CFD 和 NS 不一致，Phase 2 必须靠抬高 data loss 来压 PDE，
> 但实际 mae 仅从 0.0186 升到 0.0213（1.15×），data 几乎没退化。
> 所以两个目标确实是在同一个极小上达成的。

**Q：trivial attractor 这套分析对更高 Re 还成立吗？**
> 在稳态意义上仍成立——均匀流是任意 Re 下不可压 NS 的解。
> 当 Re 超过临界值进入 vortex shedding 之后，问题变成非稳态，
> 时间维度上的 IC + 弱时间一致性会进一步约束解空间，
> 那时再评估 (Y) 需要重新设计判据。这是 Re sweep 阶段要正面回答的问题。
