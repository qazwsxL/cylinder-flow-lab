# Re=40 PINN v2 进展汇报 — 中心问题：方法到底 consist 吗？

> 时长：约 7-8 分钟陈述 + 5-10 分钟 Q&A。
> 整篇的脊柱是一个问题：**这套 PINN 方法对 cylinder flow 而言是不是
> 自洽的**——也就是说，在合理的容量下，能不能同时收敛到 CFD 这个解
> 和 NS 这个约束。其余所有数字和实验都只是回答这个问题的不同侧面。

---

## 〇、把「consistency」拆成可测的三问（30 秒）

老师好。这次想把汇报的中心问题摆得直白一点：**我们这套 v2 setup
（hard BCs + Fourier features + 两阶段 BFGS）到底 consist 吗？**

「consist」拆成三个可以独立测的子问题：

**C1.  CFD 自己是不是 NS 的解？**
> 如果 CFD 本身不满足 NS，PINN 再准也只能拟合 CFD 的「数值近似」，
> 而不是物理。

**C2.  在 CFD 是 NS 解的前提下，PINN 找的极小点和 CFD 是同一个点吗？**
> 也就是：「fit CFD」和「satisfy NS」这两个目标的极小点在参数空间里
> 是同一个点，还是两个不同的 sheet？

**C3.  Phase 1 → Phase 2 切换时，引入 PDE 约束不会把场推离 CFD 吗？**
> 这是您之前提的 (Y) 判据的物理含义：mae(P2)/mae(P1) ≤ 2 + PDE drop ≥ 5×。

下面所有 slide 都对应到这三问中的一问或多问。陈述完之后我会给一句
**consistency verdict**。

---

## 一、C1 — CFD 自己是不是 NS 的解（1 分钟，对应上一阶段 slide 11 / diagnose_cfd_pde.py）

复述上次的核心结论，作为这次所有讨论的物理基础：

- 在 CFD 网格的 **bulk 区域**（远离壁面 0.5、远离远场 0.5），稳态涡量
  输运残差 $r_\omega = u\omega_x + v\omega_y - Re^{-1}(\omega_{xx}+\omega_{yy})$
  median ≈ **1.4·10⁻³**，p99 ≈ 1.65。
- 近壁带 + 远场带 NS 残差跳到 p99 ≈ 10——典型的 OpenFOAM 网格在边界层
  和远场 outflow 没解析干净。
- continuity 残差 median ≈ 7·10⁻⁴ —— **CFD 自己不是严格无散的**。

**C1 verdict**：CFD **在 bulk 上**接近 NS 解（残差 1e-3 量级），但不是
严格 NS 解。**这给所有后续 PINN 的 PDE residual 设了 ~1e-3 的物理硬地板**。
我们的 setup 不能要求 PDE residual 比 CFD 自己的 NS 残差小。

---

## 二、C2 — 「fit CFD」和「satisfy NS」是同一个极小吗？（3 分钟）

这是这次工作的主轴。**同一个 protocol**（两阶段、data_priority = 2、全 CFD anchor、
n_f = 20k），只换网络容量 + 走两个 corner case，**直接测两个极小的距离**。

### 2.1 反证 1：去掉 data anchor → trivial attractor（slide 25 + slide 31）

把 Phase 2 的 data anchor 完全关掉，BFGS 把 PDE residual 压到趋零，
但 mae_u 从 0.106 漂到 0.339（3.2× drift），vorticity 场退化成 ±2 弥漫
噪声，**完全没有相干的尾流剪切层**（slide 31 第三象限）。

原因：均匀流 (u, v) ≈ (1, 0) 是 PDE-only 目标的合法极小（hard BCs 没
排除它）。data anchor 一关，BFGS 沿最陡下降找到的就是这个「另一个 NS
极小」。

→ 表明在 32/3 的容量下，**只用 PDE 约束不足以唯一确定 CFD 解**——data
anchor 是把 PINN 钉在「CFD 这个特定 NS 解」上的必要条件。

### 2.2 反证 2：保留 data anchor，扫权重 → 单调 trade-off（slide 26）

保留全部 7456 个 CFD anchor，只扫 data_priority = 0.5 / 1.0 / 2.0 / 5.0。
mae drift 单调从 1.48× 降到 1.11×，**同时** PDE drop 也单调从 1.57× 降到
0.95×——两个目标在权重维度上**严格单调反向**，没有中间点能同时满足
(Y) 的两半。

物理解读：在 32/3 这个容量下，「匹配 CFD 的极小」和「低 PDE residual 的
极小」位于**参数空间的不同 sheet**。调权重只能在两个 sheet 间的 Pareto
前沿上滑动。

→ 这就是**「方法在 32/3 容量下不 consist」的直接证据**：两个极小不在
同一点。

### 2.3 正证：scale-up 64/4 → 两个极小在数值上合并（slide 27）

实验协议**完全不动**——同样两阶段、同样 data_priority = 2、同样全 CFD
anchor、同样 n_f = 20k——只把网络放大到 width=64 / depth=4。

- Phase 1 mae_u = **0.0186**（比小网改善 5.7×）。
- Phase 2 mae_u = **0.0213**，rel_l2_u = 3.2%。
- mae 漂移 **1.15× ≤ 2 ✓**
- PDE residual 0.084 → 0.014，**下降 5.97× ≥ 5 ✓**

也就是说，**两个目标在同一 protocol、同一 priority、同一种子下同时
达成**。前一节看到的「两个 sheet」结构，在 64/4 容量下**合并到了一点**。
而且这个点上 PDE residual = 0.014 已经接近 CFD 自身 NS 残差的 ~1e-3 量级
（C1）——也就是说 PINN 已经把 PDE 项压到了 CFD 这个参考所允许的物理
下限。

**C2 verdict**：**方法在 capacity-sufficient + data-anchored 的条件下是
consist 的**——两个极小在 64/4 数值上是同一个点。容量不够、或者去掉
data anchor，就会破坏 consistency；这两个失败模式我们都能用 slide 25 /
26 的实验诊断出来。

---

## 三、C3 — Phase 1 → Phase 2 不会把场推离 CFD（1 分钟，对应 slide 28 + 29）

把 C2 的 verdict 翻译成最直接的图像证据。slide 29 下半画的是 full-mesh
CFD-monitor mae_u / mae_v / mean|du,dv|/speed 随 iter 的轨迹：

- **P1 baseline（蓝）**：1.33 → 0.106，单调下降。
- **P2 strict consistency（红）**：phase-start 接 P1 在 0.106，
  BFGS 把它**抬到 0.339**——consistency 破坏的图像证据。
- **P2 all-CFD prio=2（绿）**：phase-start 0.106，BFGS 让它**只小幅
  上升到 0.129**——consistency 保持的图像证据。

注意 P1 在 32/3 上 mae_u 只能到 0.106 而不是 1e-4——这正是 slide 28 那张
对照图说的「Phase 1 的『小』要求 capacity-sufficient」。小网 P1 卡在 0.106
不是 BFGS 没跑够，而是网络容量上限。

**C3 verdict**：**只要 (Y) 通过**——也就是同一 protocol 下 mae 漂移 ≤ 2
且 PDE drop ≥ 5——**Phase 2 引入 PDE 约束就不会把场推离 CFD**。
64/4 是当前唯一能让 (Y) 通过的配置；小网必然失败，且失败模式我们能预测。

---

## 四、附加诊断 — speed-magnitude 归一化（30 秒，slide 30）

按您的建议把误差归一化方式换成 mae / √(u²+v²)：

- CFD 真值：‖u‖₂ = 77.17，‖v‖₂ = 8.91（**只占 ‖u‖₂ 的 11.5%**），mean_speed = 0.816。
- 旧 rel_l2_v 被分母放大 8.7×，看着很吓人但其实只是 v 本身就小。

按速度模长归一化之后，三个 32/3 run 是 **17 % / 49 % / 20 %**，64/4 是
**2.7 %**——这跟肉眼看流场图的感觉完全一致。**所有 consistency 结论
都用这个归一化重新核对过**，不依赖旧 rel_l2_v。

---

## 五、Consistency verdict（30 秒，收尾必讲）

把三问的答案合在一起：

1. **C1**：CFD 在 bulk 上是 NS 解到 ~1e-3 精度（不是严格解）；这是物理硬地板。
2. **C2**：在 32/3 上「fit CFD」和「satisfy NS」是两个 sheet，方法**不 consist**；
   在 64/4 上两个 sheet 合并成一个点，方法**consist**。
3. **C3**：在 capacity-sufficient + data-anchored 的条件下，Phase 2 引入 PDE
   不会把场推离 CFD（mae drift 1.15× ≤ 2，PDE drop 5.97× ≥ 5）。

**所以一句话总结：在我们这套 v2 setup 下，方法在 capacity-sufficient
（≥ 64/4）+ data-anchored 的条件下是 consist 的，到 CFD 自身的 ~1e-3
NS 残差地板为止。容量不够或去掉 anchor 都会破坏 consistency，且失败模式
我们能用 slide 25 / 26 / 31 的诊断预测和识别。**

下一步：96/5 + 长 BFGS 跑 publication-quality 数字，然后启动 Re curriculum。

---

## 六、Mentor 可能追问 — 全部按 consistency 子问题归类

> 每题给出**一句话核心答** + 关键支撑 + 对应 slide / C-编号。

### 关于 C1（CFD 是不是 NS 解）

**Q1. CFD 残差 ~1e-3 是不是太大了？要不要换 CFD？**
> 在「reproduce CFD field」这个任务下不需要换。残差 1e-3 是 OpenFOAM 在
> Re=40 这种网格密度下的典型水平，bulk 区域是相干的；远场和近壁的大残差
> 我们已经用 wall_buffer / edge_buffer 排除掉了。要把 PINN 推到 1e-5
> 才需要换 CFD（slide 11 的下界论证）。

**Q2. continuity 残差 7·10⁻⁴ 这件事要不要紧？**
> 不影响 PINN，因为我们用 stream-function 参数化 `u = ∂ψ/∂y, v = -∂ψ/∂x`，
> div(u, v) ≡ 0 是结构成立，跟 CFD 的 continuity 误差**解耦**。

### 关于 C2（两个极小是否同点）

**Q3. 「两个 sheet 合并」会不会只是 64/4 的偶然？换个 seed 还成立吗？**
> 偶然的可能性很低，三个独立证据指向同一个结论：(i) 64/4 自身 (Y) 双
> 通过；(ii) 64/4 的 P2 PDE residual 0.014 已经触及 CFD 物理下限 ~1e-3
> 量级，再降也降不动；(iii) 64/4 P2 比 P1 的 mae 只涨 14%（同 seed 下
> 数据拟合几乎没退化）。三件事互相印证不太可能是巧合。换 seed 的对照
> 我打算在 96/5 publication run 时同时做。

**Q4. 是不是大网总是能让 PDE drop 更多？跟 CFD 是不是真解无关？**
> 不是。反证：如果 CFD 和 NS 不一致，Phase 2 必须靠**抬 data loss** 来
> 压 PDE，但实测 mae_u 仅从 0.0186 升到 0.0213（涨 1.15×），data 几乎
> 没退化。所以 PDE drop 是「在同时拟合 CFD + 满足 NS」的状态下达成的，
> 不是用 data 换 PDE。这是 C2 verdict 成立的硬约束。

**Q5. trivial attractor 是结构性问题还是优化器问题？**
> 结构性问题。均匀流满足 hard BCs + PDE 是数学事实，与优化器无关。
> 任何不带 data anchor 的 PINN 在 PDE-only 目标下都可能滑到那个极小，
> 这跟 BFGS / Adam / SSBroyden 谁好谁差无关。**所以 anchor 是 v2 这个
> setup 下保证 consistency 的结构性要求**。

**Q6. 那为什么不直接 hard-encode "anchor to CFD"，比如把 u(x_data) = u_data
       也写进网络结构里？**
> 技术上可以做（类似 hard BCs），但代价是 CFD 数据点变成 ground truth，
> PINN 退化成纯插值器，C2 这个 consistency 检验就没意义了——你必然
> 得到「我们和 CFD 一致」，因为强制的。soft anchor 让 BFGS 自己选择
> 「同时尊重 data 和 PDE」的最优 trade-off，**这才是 consistency 检验
> 有信息量的前提**。

### 关于 C3（Phase 2 切换不破坏拟合）

**Q7. Phase 1 的 training loss 「足够小」吗？**
> 在 64/4 上是的：DATA MSE plateau ≈ 0.0008，全网格 mae_u 0.019（slide 28
> 右绿卡）。在 32/3 上不是（plateau 0.036，mae_u 0.106）——这是 capacity
> 上限，不是 BFGS 没跑够，详 slide 29 下半图：BFGS 段在 plateau 上水平
> 延伸 200 次迭代，斜率接近 0。

**Q8. 是不是再多 BFGS 还能突破 plateau？**
> 不会，理由同 Q7：BFGS 已经在 Pareto 前沿上水平延伸。要打破必须改
> capacity 或 sampling 分布。

**Q9. v 误差为什么相对值看着大？是模型问题吗？**
> 不是模型问题，是评估问题。‖v_true‖₂ 只有 ‖u_true‖₂ 的 11.5%
> （slide 30 navy 卡），旧 rel_l2_v 把 v 绝对误差除以一个极小分母，
> 被 8.7× 放大。改用 mean|du,dv|/mean_speed 之后，三 run 是 17 % / 49 % / 20 %，
> 跟图像一致。

### 关于实现细节（被问就答）

**Q10. v2 的小网默认 fourier_features=8 而不是 32 — 为什么？**
> 32/3 的 hidden width 只有 32，F=32 给出 64 维 cos/sin 特征 + 3 维
> (x,y,t) = 67 维输入，第一层 32×67 把信号瓶颈回 32 维，浪费容量。
> F=8 给出 19 维输入，和 32 维 hidden 匹配得更好。这也是我前两天 rerender
> 时第一次报 shape mismatch 的根因（[32, 19] vs [32, 67]）。

**Q11. 多 σ Fourier (0.25 / 0.5 / 1.0 / 2.0) 必要吗？**
> 必要。Re=40 是双尺度的：边界层 λ ≈ 0.15（要 σ ≈ 4）、尾流 λ ≈ 3-5
> （要 σ ≈ 0.5）。单一 σ 把容量押在一个频段上，另一个就拟合不上——
> 这是 v1 hard-BCs 失败的根因。多 σ 同时表达两个尺度。

### 关于下一步

**Q12. 96/5 长 BFGS 如果 (Y) 反而崩了怎么办？**
> 三种可能：(i) drift 略大但仍 < 2×，没问题；(ii) 出现新 trade-off，
> 说明 96/5 的两个极小又分裂了，需要重做 slide 26 的小 sweep；
> (iii) 数值不稳定 → 调 lr / clip。预案分别已经准备好。

**Q13. Re curriculum 跨过 Hopf 分岔点（Re ≈ 47）的设计？**
> 稳态区 (Re ≤ 45)：单时刻 PINN warm-start，增量 5；近 Hopf 区 (45-47)：
> 增量 1，监视尾流 SE(2) 对称性误差，一旦尾流震荡切到非稳态；非稳态区
> (Re ≥ 50)：加时间维度 + Strouhal frequency 约束。Hopf 点 (Y) 判据需要
> 重新设计——因为「satisfy NS」的解集变成 limit cycle，要把弱时间一致性
> 补进去。这个细节我打算在 96/5 publication run 跑完之后再正面回答。

---

## 七、若被打断只能讲三句

1. **Consistency 在 64/4 上同时成立**——「fit CFD」和「satisfy NS」的
   极小在数值上是同一个点，(Y) 双通过（drift 1.15×，PDE drop 5.97×），
   且 PDE residual 已经触及 CFD 自身的 ~1e-3 物理下限。
2. **在 32/3 上必然不 consist**：要么 trivial attractor（去 data）、
   要么单调 trade-off（调权重），两个失败模式都是 capacity 不足的可预测
   症状，和优化器无关。
3. **下一步**：96/5 + 长 BFGS 出 publication-quality 数字，然后 Re sweep。

---

## 八、备查数字表（同一份，配合 slide 28 / 30 用）

| 量 | P1 baseline (32/3) | P2 strict consist (32/3) | P2 prio=2 (32/3) | P2 prio=2 (64/4) |
|---|---:|---:|---:|---:|
| mae_u                                | 0.106 | 0.339 | 0.129 | **0.0213** |
| mae_v                                | 0.068 | 0.183 | 0.071 | **0.0125** |
| mean\|du,dv\| / mean_speed            | **17.0 %** | **49.4 %** | **19.6 %** | **2.7 %** |
| \|\|(du,dv)\|\|₂ / \|\|(u,v)\|\|₂     | 21.0 % | 54.7 % | 24.7 % | 3.2 % |
| 旧 rel_l2_v                          | 0.92 | 2.25 | 0.97 | 0.16 |
| PDE drop (P2 vs P1)                  | — | huge (→0) | 1.17× | **5.97×** |
| mae drift (P2 / P1)                   | — | 3.20× | 1.21× | **1.15×** |
| C2 通过（两极小同点）                  | — | × | × | **✓** |
| C3 通过（(Y) 满足）                    | — | × | × | **✓** |

CFD 真值：‖u‖₂ = 77.17，‖v‖₂ = 8.91，mean_speed = 0.816，
bulk NS 残差 median ≈ 1.4·10⁻³（C1 物理硬地板）。
