# v2 三个 run 的诊断图 + 数据校对

按 mentor 这次给的三条要求做的事：

> 1. 比较 phase1 的 training loss，应该很小才对。如果 v 的误差过大就不要
>    比较 rel_error，直接除根号 u²+v²。
> 2. 把 phase1 和 phase2 的 loss 折线图画出来，三个 baseline：
>    P1 baseline，strict consistency (P2, no data) 和 all-CFD anchor (P2, prio=2.0)。
> 3. 比较 vorticity、mae_u、mae_v 等数据的图。

所有图都在 `reports/figs_v2/` 下。

---

## 0. CFD 真值的统计（一次性）

在我们用的训练 box (-8,12)×(-8,8) 里、扣掉圆柱内部，共 7456 个 CFD 单元：

```
mean speed = mean(√(u²+v²))   = 0.8162
RMS  speed = √(mean(u²+v²))   = 0.8997
||u_true||₂ = 77.17
||v_true||₂ =  8.91     ← 只有 ||u||₂ 的 11.5%
```

mentor 提的事在这里数字化了：因为 `||v_true||₂` 太小，`rel_l2_v =
||error_v||₂ / ||v_true||₂` 会被 1/0.115 ≈ 8.7 倍放大，从而失真。
正确的做法是把误差按速度模长 `√(u²+v²)` 归一化。下面两个量都可以用：

- 逐点：`mean(√(eu²+ev²)) / mean_speed`
- L2 ：`||(eu,ev)||₂ / ||(u,v)||₂ = √(N·(rmse_u²+rmse_v²)) / ||u,v||₂`

---

## 1. Phase 1 training loss 到底有没有"很小"？

**结论**：在 32/3 小网络上 P1 training loss **不够小**。
mentor 的直觉对——P1 应该把 data fit 做得很彻底，结果 mae_u 还有 0.10。

来源：`logs/pinn_re40_alldata_p2_quick_2766394.out` 第 `[ALLDATA-P2] Phase 1`
段（与 consist_quick 和 pdew_sweep 的 P1 是同一个 protocol，应该等价）。

| 阶段 | training-time DATA loss | full-mesh mae_u | full-mesh mae_v | mean|du,dv| / speed |
|---|---:|---:|---:|---:|
| phase-start (随机权重)    | —      | 1.329 | 2.242 | 3.450 |
| post-Adam (300 ep)        | 0.267  | 0.285 | 0.172 | 0.443 |
| post-BFGS (~200 calls)    | 0.0358 | **0.106** | **0.068** | **0.170** |

也就是 BFGS 把 training data MSE 从 0.27 → 0.036（7.5× 改善），对应的全网格
mae_u 从 0.28 → 0.106。但 mae_u = 0.10 在物理上不算小（速度 ~ 0.8，相对误差 ~12%），
说明 32/3 这一档网络容量对 CFD 数据的拟合就有结构性上限——这正是
slide 27 scale-up 实验里要打破的瓶颈。

对比 64/4 大网络的同一 protocol：

| 阶段 | full-mesh mae_u | full-mesh mae_v | mean|du,dv| / speed |
|---|---:|---:|---:|
| post-BFGS (64/4) | **0.0186** | **0.0108** | **0.0288** |

→ 5.7× 改善。所以"P1 should be very small"这件事在大网络上才真正成立。

---

## 2. Loss 折线图

### 2.1 训练 loss（DATA + PDE，仅画进 loss 的项）

文件：[fig01_loss_curves.png](figs_v2/fig01_loss_curves.png)

左图 DATA loss、右图 PDE loss，x 轴 = Adam epoch 拼接 BFGS call。
每条曲线分别画三个 run：

- **P1 baseline (data-only)** ：DATA 从 ~0.27 → 0.036；PDE 项不入 loss，未画。
- **P2 strict consistency**   ：PDE 从 ~0.2 → ~0.05；DATA 项不入 loss，未画
   （但 DATA 监控会漂——见 fig01b）。
- **P2 all-CFD anchor prio=2** ：DATA 和 PDE 同时活跃，DATA 略升、PDE 略降。

### 2.2 full-mesh velocity error vs iteration

文件：[fig01b_mae_curves.png](figs_v2/fig01b_mae_curves.png)

这张更直接反映"流场到底有没有靠近 CFD"：mae_u、mae_v、和 mean|du,dv|/speed
随 iter 的变化。

关键现象：
- P1 baseline （蓝）：从 1.33 → 0.10，单调下降。
- P2 strict consistency（红）：phase-start 时已经在 0.10（接 P1 的 ckpt），
  但 BFGS 把 mae_u **抬到了 0.34**——这就是"trivial attractor"漂移的图像证据。
- P2 all-CFD prio=2（绿）：phase-start 也是 0.10，BFGS 让它略涨到 0.13；
  小幅 trade-off 而不是漂走。

---

## 3. mae_u / mae_v / 速度归一化误差 + vorticity 对比

### 3.1 数值对比 (bar chart)

文件：[fig02_metric_bars.png](figs_v2/fig02_metric_bars.png)

四个面板，全部基于训练 log 里 `[post-BFGS] CFD monitor` 那行实测的全网格指标
（不是 training-time DATA，避免之前 Run A 那种 8× 误读）：

| run | mae_u | mae_v | mean|du,dv|/speed | ||(du,dv)||₂/||(u,v)||₂ | （旧 rel_l2_v 仅作对比） |
|---|---:|---:|---:|---:|---:|
| P1 baseline (data-only)      | 0.106 | 0.068 | **0.170** | **0.210** | 0.915 |
| P2 strict consistency (no data) | 0.339 | 0.183 | **0.494** | **0.547** | 2.250 |
| P2 all-CFD anchor, prio=2.0  | 0.129 | 0.071 | **0.196** | **0.247** | 0.966 |

- 旧 `rel_l2_v` ≈ 0.9–2.2 看着很吓人，但其实只是因为 `||v_true||₂` 才 8.9 这么小。
- 用 `√(u²+v²)` 归一化之后：P1 ≈ 17%，prio=2 ≈ 20%，strict consist ≈ 49%。
  数字立刻和肉眼看流场的感觉对上。

### 3.2 Vorticity 对比

文件：[fig03_vorticity_panel.png](figs_v2/fig03_vorticity_panel.png)
（CFD 真值的近尾流放大见 [fig03b_cfd_vorticity_zoom.png](figs_v2/fig03b_cfd_vorticity_zoom.png)）

**重要 caveat**：PINN 的三张子图是直接拿训练时生成的 `viz_v2.png` 拼回来的，
它们各自的 colorbar 是 matplotlib auto-scale，**和 CFD truth 的 ±8 colorbar 不一致**。
要做精确比对需要在 GPU 节点上 rerun 重新渲染（脚本已写好，见下一节）。

不过即使不匹配 colorbar，定性结论已经很清楚：

- **CFD truth**：尾流里两条对称的剪切层 + 中间近壁正负涡量带，|ω|_peak ≈ 7.5。
- **P1 baseline**：vorticity 全场只有 ±0.06 量级——小网络根本没有把高梯度
  解出来；u/v 拟合还行（mae_u 0.10）只是因为速度本身平滑。
- **P2 strict consistency**：vorticity 满屏 ±7.5 噪声，没有任何相干的尾流结构
  ——这就是 trivial attractor 的图像证据；PDE residual 趋近 0，但 ω 场
  完全错乱。
- **P2 all-CFD prio=2**：vorticity 在 ±20 量级，有靠近圆柱的局部结构，
  比 strict consist 好得多。

### 3.3 匹配 colorbar 的 rerender 脚本

写了一个 `reports/rerender_vorticity_matched.py`，在 GPU 节点 / 装好 torch
的机器上跑：

```bash
cd ~/cylinder-flow-lab
python reports/rerender_vorticity_matched.py
```

会把三个 run 的 PINN vorticity 都画在 `vmin/vmax = ±8` 下，输出在
`reports/figs_v2/pinn_vorticity_matched/`。届时再把它们替换进 fig03 就是
严格的 apples-to-apples 比对。

---

## 4. 关键发现复盘

1. **Phase 1 在小网络上不够小**（mae_u 0.10），所以即便 Phase 2 完全
   不引入新误差也只能从这个基线往下做。大网络下 Phase 1 就能拉到 0.019，
   才有空间满足 (Y) 判据。
2. **mentor 提的"v 误差大别看 rel_error"**：用 `mean|du,dv|/speed` 后
   三个 run 的相对误差是 17% / 49% / 20%，符合直觉。
3. **trivial attractor 是真实的失败模式**：strict consistency 不仅 mae_u 漂到
   0.34，连 vorticity 场都完全错乱——这是 slide 25 的硬证据。
4. **prio=2 在小网络下只是"少漂一点"**：mae_u 从 0.106 → 0.129（1.21×），
   但 vorticity 仍然不对——所以 scale-up 是必须的。
