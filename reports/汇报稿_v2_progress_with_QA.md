# Re=40 PINN v2 进展汇报 + Mentor Q&A 备稿

> 时长：约 7-8 分钟正式陈述，预留 5-10 分钟 Q&A。
> 学术口吻；陈述按 slide 24→25→26→27→28→29→30→31 顺序展开。
> 配套 PPT：`Reconstruction of Flow Fields using PINNs - v2 progress.pptx`。

---

## 一、陈述（约 8 分钟）

### 0. 开场（30 秒）

老师好。这次汇报承接上一次的 Run A / Run B 结果，主要报告 v2 系列实验
（slide 24 起）和这一轮新加的四张诊断页（slide 28-31）。
核心目标是回答您提出的 **(Y) 一致性判据** 在我们当前 setup 下是否能成立，
以及瓶颈到底在哪里。

> (Y) 判据回顾：从 Phase 1（data-only）切到 Phase 2（data + PDE）后，
> 须同时满足  **mae(P2)/mae(P1) ≤ 2×**  且  **PDE residual 下降 ≥ 5×**。

### 1. 实验矩阵（1 分钟，slide 24）

固定参考 vtk、固定 box (-8,12)×(-8,8)、固定 hard BCs + Fourier features
的前提下做了六组对照：

- P1 baseline（data-only，全 CFD 点）：mae_u = 0.106 ——后面所有比较的基线。
- 严格一致性（P2 完全去 data）：mae_u 漂到 0.339，3.2× drift。
- 全 CFD anchor + data_priority sweep（0.5 / 1.0 / 2.0 / 5.0）：drift 单调
  从 1.48× 降到 1.11×，PDE drop 单调从 1.57× 降到 0.95×。
- scale-up（width=64 / depth=4，prio = 2）：drift 1.15×，PDE drop 5.97×。

**只有最后一行同时跨过 (Y) 的两道门槛。** 下面把每个失败模式和这个成功结果
分别讲清楚。

### 2. 失败模式 1：trivial attractor（1 分钟，slide 25 + 31）

完全去掉 data anchor 之后，PDE residual 被 BFGS 压到趋近零，但 mae_u 从
0.106 漂到 0.339，**尾流回流结构整体消失**。slide 31 的 vorticity 对比图
是这一点的直接图像证据 —— 第三象限那张 (P2 strict consistency) 在
**和 CFD 真值同一个 ±8 colorbar 上**显示出弥漫的 ±2 量级噪声，
**完全没有相干的剪切层结构**。

物理上这是 **trivial attractor**：均匀来流 (u, v) ≈ (1, 0) 在除壁面之外的
整个域上都精确满足不可压 Navier-Stokes。Hard BCs 只把壁面零速度强制写入
模型结构，并没有把这个均匀解从可行解集里剔除。所以一旦数据 anchor 被去掉，
BFGS 找到的并不是「与 CFD 一致的 NS 解」，而是「另一个合法的 NS 极小」。

→ 结论：在这个 capacity 下，纯 PDE 模式在 (Y) 判据下 **失败**。

### 3. 失败模式 2：权重 sweep 给出单调 trade-off（1 分钟，slide 26）

保留全部 7456 个 CFD anchor，只扫 data_priority。结果是 mae drift 和 PDE drop
**双方向严格单调反向**，没有任何中间点能同时跨过 (Y) 的两半。

物理解读：在 width=32 / depth=3 这个容量下，「匹配 CFD 的极小」和「低 PDE
residual 的极小」位于参数空间的**不同 sheet**上。调权重只是让优化器沿着
这两个极小之间的 Pareto 前沿滑动，不能让两个极小合并。

→ 推论：**问题的杠杆不在 weight，而在 capacity。**

### 4. 验证：capacity 实验（1 分钟，slide 27）

实验协议完全不动 —— 同样两阶段、同样 data_priority = 2、同样全 CFD anchor、
同样 n_f = 20k —— 只把网络放大到 width=64 / depth=4。

- Phase 1 mae_u = 0.0186（比小网络改善 5.7×，说明大网络在纯数据拟合阶段
  就明显更逼近真实场，已经为 (Y) 留出了足够的空间）。
- Phase 2 mae_u = 0.0213，rel_l2_u = 3.2%。
- **mae P2/P1 = 1.15× ≤ 2 ✓**
- **PDE residual 0.084 → 0.014，下降 5.97× ≥ 5 ✓**

也就是说，**(Y) 的两半在同一个 protocol、同一个 data_priority 下同时被满足**。
在 width=64 这一档，CFD-matching 的极小和 low-PDE-residual 的极小数值上
基本是**同一个点**。前面那个「两个 sheet」的结构，在大容量下合并了。

### 5. 诊断 1：Phase 1 到底有多小（1 分钟，slide 28）

这是回应您上次问的「Phase 1 应该很小才对」。把数字摆出来：

- 小网 32/3：training-time DATA MSE plateau ≈ 0.036（对应 rmse ≈ 0.19），
  全网格 mae_u = **0.106**，speed-normalized 误差 17.0%。
- 大网 64/4：training-time DATA MSE plateau ≈ 0.0008，
  全网格 mae_u = **0.0186**，speed-normalized 误差 2.9%。

→ 「Phase 1 应该很小」**在大网络上才真正成立**。
小网上 P1 卡在 0.036 不是 BFGS 没跑够，而是 32/3 的网络容量上限。
这反过来也是「scale-up 是必须的」最直接的证据。

### 6. 诊断 2：loss 折线 + full-mesh error 折线（1 分钟，slide 29）

slide 29 上半画的是训练 loss（DATA + PDE，只画进 loss 的项 —— strict
consistency 没有 DATA、baseline 没有 PDE，明示标注 OFF）；下半画的是
full-mesh CFD-monitor mae_u / mae_v / mean|du,dv|·speed⁻¹ 随 iter 变化。

下半三张图最重要：

- **P1 baseline（蓝）**：从初始 1.33 单调下降到 0.10。
- **P2 strict consistency（红）**：phase-start 在 0.10（接 P1 ckpt），
  BFGS 把它**抬升到 0.34**——这就是 trivial-attractor 漂移的图像证据。
- **P2 all-CFD prio=2（绿）**：phase-start 0.10，BFGS 略涨到 0.13；
  小幅 trade-off 而不是漂走。

### 7. 诊断 3：误差归一化（1 分钟，slide 30）

这是回应您「v 误差太大就别比 rel_error，直接除根号 u²+v²」的建议。

CFD 真值统计（7456 cells）：‖u_true‖₂ = 77.17，‖v_true‖₂ = 8.91
（只占 ‖u‖₂ 的 11.5%），mean_speed = 0.816。

也就是说，旧的 rel_l2_v 被 1/0.115 ≈ 8.7× 放大；slide 30 表格里旧 rel_l2_v
看着是 0.92 / 2.25 / 0.97，但真正按速度模长归一化之后是 17 % / 49 % / 20 %，
和肉眼看流场的感觉直接一致。

### 8. 总结与下一步（30 秒）

两条主要结论：

1. 小网络（32/3）下 (Y) 必然不可达；trivial attractor 和单调 trade-off
   都是 capacity-bound 的症状。
2. 大网络（64/4）同一 protocol 下 (Y) 自动成立，证明 CFD 与 NS 相容，
   瓶颈是模型容量。

下一步：
- 把网络再放大到 96/5、配合更长 BFGS，跑出 publication-quality 的 (Y) 数字。
- 然后启动 Re curriculum（5 → 100），进入项目原定的多 Reynolds 数 sweep。

---

## 二、Mentor 可能追问的问题 + 应答

> 下面 12 题按发生概率（基于 mentor 历次反馈的关切方向）排序。
> 每题给出**一句话核心答**+**关键支撑数字 / slide**。

### Q1. Phase 1 的 training loss 到底是不是「足够小」？

> 在 32/3 上 **不是**（DATA MSE plateau 0.036，全网格 mae_u 0.106），
> 在 64/4 上 **是**（DATA MSE plateau ≈ 0.0008，全网格 mae_u 0.019）。
> 「Phase 1 应该很小」是 capacity-sufficient 时的命题；32/3 这个网就不够。
> 详 slide 28 + slide 29 下半图。

### Q2. v 的相对误差为什么一直那么大？是模型问题还是评估问题？

> 是**评估问题**。‖v_true‖₂ 只有 ‖u_true‖₂ 的 11.5%（slide 30 navy 卡）。
> 旧 rel_l2_v 把 v 的 absolute error 除以一个极小分母，结果被放大 8.7×。
> 改用 `mean|du,dv| / mean_speed` 之后三个 run 的相对误差是 17%/49%/20%，
> 与流场图像一致。

### Q3. 是不是 BFGS 还没跑够？再多迭代会不会突破 plateau？

> 不会。slide 29 左下 (DATA loss) 蓝色 BFGS 段已经在 plateau 上水平延伸了
> ~ 200 次迭代，斜率接近 0；继续跑只会让 plateau 又精确一点。这是
> Pareto 前沿（data ↔ PDE）的位置，不是优化器的弱项。要打破它必须改 capacity
> 或 sampling 分布，不是加迭代。

### Q4. trivial attractor 是巧合，还是 BFGS 一定会跑到那？

> **必然**。均匀流 (1, 0) 在 hard BCs 下满足所有约束（incompressibility +
> 远场 + 壁面零滑移），所以是 PDE-only 目标函数的一个合法极小点。
> 数据 anchor 是唯一把 PINN 推离这个极小的力；anchor 一关，BFGS 自然
> 沿着最陡下降方向找最近的合法 PDE 极小，而那个极小不一定是 CFD 解。
> slide 25 + slide 31 第三象限的弥漫噪声场是图像证据。

### Q5. 是不是只要 data_priority 调对，小网也能满足 (Y)？

> 不能。slide 26 的 sweep 是直接反证：在 32/3 上，priority 从 0.5 → 5
> 把 drift 从 1.48× 单调降到 1.11×，但同时把 PDE drop 从 1.57× 单调
> 降到 0.95×。**两个目标在权重维度上严格单调反向**——任何中间值都过不了
> (Y) 的两道门槛。证明此处的杠杆是 capacity 而不是权重。

### Q6. 那 64/4 的成功是不是只是因为大网容易降 PDE，跟 CFD 一致性无关？

> 不是。如果 CFD 与 NS 不一致，Phase 2 必须靠**抬高 data loss** 来压 PDE。
> 实测 mae_u 从 P1 的 0.0186 升到 P2 的 0.0213，只涨 1.15×，data 几乎
> 没退化。所以 PDE drop 5.97× 是真的在「同时拟合 CFD 数据 + 满足 NS」
> 的状态下达成的，不是用 data 换 PDE。

### Q7. 为什么 v2 的小网 (32/3) 默认 fourier_features=8 而不是 32？

> 32/3 的 hidden width 只有 32，如果 Fourier 投影出 64 维输入特征，
> 第一层就是一个 32×64 的矩阵，相当于把信号压回 32 维 —— 信息被瓶颈了。
> F=8 给出 16 维 cos/sin 特征 + 3 维 (x,y,t) = 19 维输入，和 32 维 hidden
> 匹配得更好。这也是为什么我在 rerender 脚本里要显式传 F=8（否则
> load_state_dict 会因 [32, 19] vs [32, 67] 报形状错）。

### Q8. 多 σ Fourier (0.25 / 0.5 / 1.0 / 2.0) 必要吗？

> 必要。Re=40 的特征长度是双尺度的 —— 边界层 λ ≈ 0.15（需要 σ ≈ 4）、
> 尾流回流 λ ≈ 3-5（需要 σ ≈ 0.5）。单一 σ 把容量全部押在一个频段上，
> 另一个频段就拟合不上 —— 这是 v1 hard-BCs 失败的根因。多 σ 把 8 个
> Fourier 行平均分到 4 个频段，每个频段各 2 行，两个尺度同时表达。

### Q9. CFD 自己有多干净？64/4 PINN 的 5.97× drop 是绝对 PDE residual
        到多少？

> 上一阶段 `diagnose_cfd_pde.py` 测过：CFD 在 bulk 区域的 NS 涡量输运
> 残差 median ≈ 1.4·10⁻³，p99 ≈ 1.65（远场和边界层除外）。所以**所有
> PINN 的 PDE residual 物理下限就在 10⁻³ 量级**——再低就不是「逼近 NS」，
> 是「逼近 CFD 网格上的 NS 数值近似」。我们的 64/4 P2 PDE residual 是
> 0.014，处在和 CFD 自身相同的量级，是合理范围。

### Q10. 那 1e-4 ~ 1e-5 这个绝对速度误差目标还能达到吗？

> 在当前这份 CFD 上仍然不行。CFD 自己的 NS 残差 median 1.4·10⁻³ 是
> 物理硬地板，PINN 在同时满足 PDE 的约束下，data MAE 不会比 10⁻³ 更小。
> 要做到 1e-4 量级必须二选一：(a) 把 CFD 网格在尾流和近壁加密 5-10×，
> 让 CFD 自己的 NS 残差降到 10⁻⁵；或 (b) 放弃 PDE 约束、做纯插值器
> （但那样就不是 PINN 了）。

### Q11. 接下来 96/5 + 长 BFGS 如果 (Y) 反而崩了怎么办？

> 三种可能：(i) 长 BFGS 让 P2 找到一个新的全局最优，drift 略大但仍在
> (Y) 之内 —— 没问题；(ii) 出现新的 trade-off，PDE drop 上去但 drift
> 超 2× —— 说明 96/5 的两个极小又分裂了，需要重新调 data_priority；
> (iii) 数值不稳定 / NaN —— 改 lr / clip / dt。预案分别是：(i) 继续；
> (ii) 在 96/5 上重做一遍 slide 26 那样的小 sweep；(iii) 工程调参。

### Q12. Re curriculum 的设计？尤其是越过 Re ≈ 47 的 Hopf 分岔点之后？

> 稳态区 (Re ≤ 45)：单时刻 PINN 直接 warm-start，每档 Re 增量 5；
> 接近 Hopf 分岔点 (Re 45-47)：增量降到 1，监视尾流对称性的 SE(2) 误差，
> 一旦尾流开始震荡就切换到非稳态网络；非稳态区 (Re ≥ 50)：加入时间维度，
> Strouhal frequency 作为额外约束。Hopf 点附近 (Y) 判据需要重新设计，
> 因为「satisfy NS」的解集变成 limit cycle，要把弱时间一致性补进去。
> 这个细节我打算在跑完 96/5 publication-quality 之后再正面回答。

---

## 三、若被打断到不能讲完，最关键的三句话

如果 mentor 中途要去别的会，必须最后只能讲三句的话：

1. **(Y) 在 64/4 上同时成立**——drift 1.15×，PDE drop 5.97×，同一 protocol。
2. 32/3 上 (Y) **永远不可达**：要么 trivial attractor（去 data），要么
   单调 trade-off（调权重），证明是 capacity 限制。
3. 下一步 96/5 + 长 BFGS 出 publication-quality 数字，然后启动 Re sweep。

---

## 四、备查：本次新加的数字一览

| 量 | P1 baseline (32/3) | P2 strict consist (32/3) | P2 prio=2 (32/3) | P2 prio=2 (64/4) |
|---|---:|---:|---:|---:|
| mae_u                                 | 0.106 | 0.339 | 0.129 | **0.0213** |
| mae_v                                 | 0.068 | 0.183 | 0.071 | **0.0125** |
| mean\|du,dv\| / mean_speed             | **17.0 %** | **49.4 %** | **19.6 %** | **2.7 %** |
| \|\|(du,dv)\|\|₂ / \|\|(u,v)\|\|₂      | 21.0 % | 54.7 % | 24.7 % | 3.2 % |
| 旧 rel_l2_v（参考用）                  | 0.92 | 2.25 | 0.97 | 0.16 |
| PDE drop (P2 vs P1)                   | — | huge (→ 0) | 1.17× | **5.97×** |
| mae drift (P2 / P1)                    | — | 3.20× | 1.21× | **1.15×** |
| (Y) 判据是否通过                        | — | × | × | **✓** |

CFD 真值参数：‖u‖₂ = 77.17，‖v‖₂ = 8.91，mean_speed = 0.816。
