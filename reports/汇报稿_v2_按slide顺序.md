# Re=40 PINN v2 — 5/25 后进展汇报（按 slide 顺序，每图详解）

> 时长：约 6-7 分钟。按 PPT 顺序走 slide 28 → 29 → 30 → 31，每张图先说"画的是什么"，
> 再说"看到什么"，最后说"对 consistency 意味着什么"。
> Consistency 判据按老师确认的强版本：**PDE residual 下降 ≥ 5× 且
> mae(P2)/mae(P1) ≤ 2** —— 两个都满足才算。

---

## 开场（30 秒）

老师好。上次汇报到 slide 27 scale-up 之后，你给的四条 follow-up 我都做完了，
对应 slide 28-31。每张 slide 我详细讲一下每个 panel 画的是什么、为什么这么画、
看出什么结论。最后我会用你确认的 (Y) 判据把四个 run 一字排开。

---

## SLIDE 28 — Phase 1 Baseline：How Small Is Small?

> 这一张回应你的 follow-up #1："Phase 1 的 training loss 应该很小才对"。

### 左红卡：Small net (width=32, depth=3, 2.8 k params)

四组数字：

1. **training-time DATA MSE plateau = 0.036**（对应 rmse ≈ 0.19）。这是
   Phase 1 BFGS 收敛后训练目标本身的值。
2. **full-mesh mae_u = 0.106**——把 P1 ckpt 加载、在全部 7456 个 CFD cell 上
   评估，速度 u 分量的 mean absolute error。
3. **full-mesh mae_v = 0.068**——同上但 v 分量。
4. **normalized speed-vector error = 17.0%**——`mean|du,dv| / mean_speed`，
   也就是你建议的用 √(u²+v²) 归一化的相对误差。

→ **小网 Phase 1 没做到"很小"**。mae_u 0.106 在速度 ~0.8 量级下是 13% 误差。

### 右绿卡：Scale-up net (width=64, depth=4, 14.9 k params)

完全同样的训练 protocol，只把网络放大：

1. training DATA MSE plateau = **0.0008**（rmse ≈ 0.028）—— 比小网降 **45×**。
2. **full-mesh mae_u = 0.0186** —— 比小网降 **5.7×**。
3. full-mesh mae_v = 0.0108。
4. normalized speed-vector error = **2.9%**。

→ **同 protocol 下，大网的 Phase 1 才真正"很小"**。

### 黄底 Finding 卡（三条要读出来）

1. **"P1 应该很小"只在容量足够时才成立**。32/3 的 plateau 0.036 不是 BFGS
   没跑够（slide 29 左下蓝线斜率接近 0、已经水平延伸 200+ 次迭代），
   是 **网络容量上限**。
2. **这正是 slide 27 scale-up 打破的瓶颈**——同 protocol 同 priority 下，
   把网络从 32/3 加到 64/4，Phase 1 立刻改善 5.7×，speed-norm 误差从 17% 降到 2.9%。
3. **后面所有诊断比较都用 32/3** 作为 baseline，因为**两个失败模式在小网上
   最干净**（slide 29 / 31 都能看到）。

---

## SLIDE 29 — Loss Curves：Three Runs Side by Side

> 这一张回应你的 follow-up #2："把三个 run 的 loss 折线画出来对比"。
> 一共 5 个子图（2 上 + 3 下），先讲上面两个，再讲下面三个。

### 上半 · 左子图：DATA loss

x 轴 = iteration（Adam epoch 拼 BFGS call，offset 之后并在一起）；
y 轴 = 训练时 DATA MSE，log scale。

三条线全部都画出来了，区分方式：

- **蓝（P1 baseline）**：从 ~1.5 单调降到 ~0.036。**实线 + 虚线**表示这是
  optimizer 真正在压的项。
- **绿（P2 all-CFD prio=2）**：phase-start 接 P1 ckpt 在 ~0.1，BFGS 把它
  小幅压到 ~0.06。**实线 + 虚线**——它在 loss 里。
- **红（P2 strict consistency）**：**点线** —— 因为 strict consistency
  Phase 2 是 `--pde-only` 模式，DATA 项不在 loss 里、但 BFGS log 每步都还在
  monitor 它。值从 0.1 慢慢**抬升到 ~0.2**——这就是 trivial-attractor
  漂移的第一处图像证据：anchor 一关，data 监控值就开始涨。

### 上半 · 右子图：PDE residual loss

同样的 x 轴。

- **红（P2 strict consistency）**：从 ~1 单调降到 ~0.1。PDE 是 strict
  consistency 唯一的优化目标，BFGS 把它压得很狠。
- **绿（P2 all-CFD prio=2）**：PDE 和 DATA 共同优化，PDE 从 ~0.5 降到 ~0.1。
  BFGS 后期斜率明显比红线缓——因为还要兼顾 data 项。
- **蓝（P1 baseline）**：**没有线**，因为 `--data-only` 模式下 PDE 项**完全
  没被计算**（不是值为零，是根本没跑），所以也没法 monitor。
  我在子图角上加了一行注释说明这件事。

→ 上半两张图合起来告诉我们：**strict consistency 把 PDE 压得最低
（满足 PDE drop 5× 那一半），同时 DATA monitor 在涨（违反 mae 不漂的那一半）**。
单看 loss 没法判定 consistency，必须看全网格的 CFD-monitor。

### 下半 · 三个子图：full-mesh CFD-monitor 误差轨迹

这是真正反映"场和 CFD 有多近"的图。Y 轴 log scale，每条曲线是每 100 个
Adam epoch + post-Adam + post-BFGS 抓的快照。

**左下：mae_u 随 iter**
- 蓝 P1：1.33 → 0.30 (Adam 结束) → **0.106 (BFGS 结束)**。单调下降。
- 红 P2 strict：phase-start = 0.106（接 P1）→ Adam 把它推到 0.27 → BFGS 把它
  推到 **0.339**。**单调上升**——anchor 一关，BFGS 直接把场推离 CFD。
- 绿 P2 prio=2：phase-start = 0.106 → 0.17 → **0.129**。小幅涨然后稳住——
  trade-off 但没漂走。

**中下：mae_v 随 iter**——形状和 mae_u 完全一样，三条曲线相对位置不变；
只是数值小一档（v 本身就小）。

**右下：mean|du,dv|/speed 随 iter** —— 这是用 √(u²+v²) 归一化之后的"统一
误差"，三条线收敛到 **17% / 49% / 20%**，跟肉眼看流场图的感觉一致。

### 副标题里那行斜体（要读出来）

> "上半：训练 loss 中真正进入 objective 的项（strict consistency 没 DATA、
> baseline 没 PDE）；下半：全网格 CFD monitor —— 注意 strict consistency
> 一旦关掉 data anchor、mae_u 就从 0.10 涨到 0.34。"

→ **slide 29 的关键 takeaway：strict consistency 在 loss 层面看像在收敛
（PDE 在降），在 field 层面看是在漂走（mae 在涨）。两个层面必须一起看，
单看其中一个会被骗。**

---

## SLIDE 30 — Velocity Error：√(u²+v²) Normalization, not rel_l2_v

> 这一张回应你的 follow-up #3："v 误差被放大是评估问题，用 √(u²+v²) 归一化"。

### 副标题里要读出来的关键事实

CFD 真值的两个分量量级**严重不对称**：

```
‖u_true‖₂ = 77.17
‖v_true‖₂ =  8.91     ← 只占 ‖u‖₂ 的 11.5%
mean_speed = 0.816
```

所以 `rel_l2_v = ‖error_v‖₂ / ‖v_true‖₂` 把 v 的绝对误差除以一个**8.7×
偏小的分母**，看起来很吓人但其实只是分母小。

### 左侧 4 联 bar chart（按从左到右逐个讲）

1. **mae_u** —— P1 0.106 / strict 0.339 / prio=2 0.129。strict 比 P1 涨 3.20×。
2. **mae_v** —— P1 0.068 / strict 0.183 / prio=2 0.071。形状和 mae_u 一致。
3. **mean|du,dv| / mean_speed** —— P1 **17.0%** / strict **49.4%** / prio=2 **19.6%**。
   这是用速度模长归一化的相对误差，跟图像感觉一致。
4. **‖(du,dv)‖₂ / ‖(u,v)‖₂** —— P1 21.0% / strict 54.7% / prio=2 24.7%。
   这是 vector L2 形式，比第 3 个稍大一档但定性结论一致。

### 右侧 navy 卡：Final-iteration metrics

把每个 run 的精确数字列出来，包括**旧 rel_l2_v** 作对比：

| run | mae_u | mae_v | speed-norm | 旧 rel_l2_v |
|---|---:|---:|---:|---:|
| P1 baseline       | 0.106 | 0.068 | **17.0%** | 0.92  |
| P2 strict consist | 0.339 | 0.183 | **49.4%** | 2.25  |
| P2 all-CFD prio=2 | 0.129 | 0.071 | **19.6%** | 0.97  |

旧 rel_l2_v 让 strict 看着才 "2.25"、prio=2 看着 "和 P1 差不多 0.97"——
**会高估这两个 run 的质量**。新归一化让 strict 49% 的漂移**直接可见**。

### 底部黄底 Reading 卡（要读出来）

> "Speed-normalized error 把三个 run 放在 0–1 尺度上可比：17% / 49% / 20%。
> strict consistency 的 ≈ 50% 漂移就是 trivial-attractor 失败。
> 旧 rel_l2_v 0.9 / 2.2 / 1.0 看着吓人但只是因为 ‖v_true‖₂ 太小——
> 数字在不同 run 之间不直接可比。"

→ **slide 30 关键 takeaway：所有 consistency 判据后面都用新归一化重新核对过，
不再依赖旧 rel_l2_v。** 这也是为什么 strict consistency 的 49% 漂移
比之前看的更严重。

---

## SLIDE 31 — Vorticity：CFD truth vs Three PINN Runs（matched colorbar）

> 这一张回应你的 follow-up #4："vorticity 要在同 colorbar 下重画"。
> 我在 Oscar 上重跑了三个 ckpt 的 inference，用 vmin/vmax = ±8 渲染
> （和 CFD 真值的 peak ±7.5 同档），slide 31 现在是真正的 apples-to-apples
> 四宫格。

### 左上：CFD truth

- 颜色范围 ±8。
- 看到两条**对称的剪切层**沿尾流伸出去（红正涡量在上、蓝负涡量在下），
  cylinder 后面紧贴着的是回流区的反号小涡。
- |ω|_peak ≈ 7.5。这是物理上应该长成的样子。

### 右上：P1 baseline (data-only)

- 同 ±8 colorbar 下**几乎全白**。实际 ω 范围只到 ±0.3。
- 意味着：**小网根本没把 wake 的高梯度区表达出来**。
- 但前面 mae_u 0.106 还是凑合的——因为**速度场本身平滑，u/v 不太需要高频
  就能拟合个大概**；vorticity 是导数，对高频敏感，立刻暴露 capacity 不足。

### 左下：P2 strict consistency (no data)

- 满屏 ω ≈ ±2 弥漫噪声 + **斜条纹**（Fourier feature 残留）。
- **没有任何相干的尾流剪切层结构**。Cylinder 后面也是一片噪声。
- 物理含义：trivial attractor 漂走之后场退化成 "uniform-flow + 数值噪声"，
  PDE residual 趋零是真的，但跟 CFD 的解完全是另一个东西。

### 右下：P2 all-CFD anchor, prio=2

- Cylinder 附近有清晰的**双极结构**（接近 CFD 尾流前段）。
- 远场仍很弱（远不到 CFD 的 ±7.5 量级）。
- 比 strict 干净一档，比 P1 多一档结构，**但仍然不足以达到 CFD vorticity 的
  量级**——还是 capacity 限制。

### 右侧四个解释卡

navy / navy / red / green 四张卡，每张对应一个 panel 用一两句话总结物理含义。
按你之前批改的版本写好了。

→ **slide 31 关键 takeaway：vorticity 图是 consistency 判据的物理验证**。
单看 mae_u 0.10、0.34、0.13 可能觉得"差距没那么大"；vorticity 一拉出来
立刻看出三个 run **完全是三种不同的失败模式**。其中 strict consistency
最有迷惑性——单看 PDE residual 它"最 consistent"（PDE drop 最大），
看 vorticity 就知道它和 CFD 解根本是两个东西。

---

## 用 (Y) 判据收尾（30 秒）

最后把四个 run 用老师确认的强判据（**PDE drop ≥ 5× 且 mae 漂移 ≤ 2×**）
一字排开：

| run | 网络 | PDE drop | mae 漂移 | (Y) 通过？ |
|---|---|---:|---:|:-:|
| P2 strict consistency | 32/3 | huge (→0) ✓ | 3.20× ✗ | **× 不 consistent**（trivial attractor） |
| P2 all-CFD prio=2     | 32/3 | 1.17× ✗     | 1.21× ✓ | **× 不 consistent**（PDE 没真降下来） |
| P2 prio=2 scale-up    | 64/4 | **5.97× ✓** | **1.15× ✓** | **✓ consistent** |

只有 64/4 这一行**同时**满足两半。前两个失败模式在小网上都是**结构性的**，
不是调参能救（slide 26 单调 trade-off 已经反证）—— 杠杆是 capacity。

---

## 下一步：Sifan Wang 的 pseudo-time stepping（30 秒）

你最近给我看的 *When PINNs Go Wrong: Pseudo-Time Stepping Against Spurious
Solutions*（arXiv:2604.23528）——你那张 Re=60 vortex shedding 的 PPT
（FEM vs PINN field 三行对比 + ablation bar chart + convergence + error
over period）我看了。

接下来计划两件事：

1. **96/5 + 长 BFGS** 跑 publication-quality 的 Re=40 (Y) 数字
   （顺手给 capacity-scan 加第三个点：32/3 → 64/4 → 96/5）。
2. **把 pseudo-time stepping 用到 Re=60 vortex shedding** —— 我们已经有
   Re=60 的 vtk reference (`Re60/`)。Re=60 跨过了 Hopf 分岔，单时刻 PINN
   本身就会撞上 spurious-solution 问题，正好是这篇论文的目标场景。
   测试结果**按你那张 PPT 的风格画**：FEM/PINN/error 三行 + ablation
   bar chart + convergence + error over period。

汇报到这里，请你给意见。

---

## 备查：数字一张表

| 量 | P1 baseline (32/3) | P2 strict consist (32/3) | P2 prio=2 (32/3) | P2 prio=2 (64/4) |
|---|---:|---:|---:|---:|
| mae_u                          | 0.106 | 0.339 | 0.129 | **0.0213** |
| mae_v                          | 0.068 | 0.183 | 0.071 | **0.0125** |
| mean\|du,dv\| / mean_speed      | 17.0 % | 49.4 % | 19.6 % | **2.7 %** |
| PDE drop (P2 vs P1)            | — | huge (→0) | 1.17× | **5.97×** |
| mae 漂移 (P2 / P1)              | — | 3.20× | 1.21× | **1.15×** |
| (Y) 判据是否通过                 | — | × | × | **✓** |

CFD 真值：‖u‖₂ = 77.17，‖v‖₂ = 8.91，mean_speed = 0.816。
