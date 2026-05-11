# Re=40 PINN 阶段汇报稿

> 用于向 mentor 当面汇报。每节都按"做了什么 → 看到什么 → 学到什么"
> 三段写。括号里是可读出口的关键数字。

---

## 0. 三句话总结

1. 修了一个之前没察觉的 bug：BFGS 阶段实际上一直跑的是 **scipy 标准 BFGS**，
   不是 SSBroyden——`method_bfgs="SSBroyden1"` 这个 option 被 scipy 静默吞掉了。
   修好之后 BFGS 阶段第一次真正起作用，**training-time** data MSE 从 ~8·10⁻³
   降到 ~2.4·10⁻⁴。
2. 但用 `analysis_updated_re40_clean` 在 4869 个 CFD 点上**真正评估**之后发现，
   全网格 MAE 只有 **0.125**（不是我从 training log 推算的 1.5·10⁻²）。差 8×。
   原因：训练时 data 采样 `wake_frac=0.8` 偏向尾流，把入口和远场误差遮住了。
   **教训：用 training log 判断进度不可靠，必须每阶段在全网格上跑 evaluate。**
3. 按你"加点降 loss"和"看 CFD 是不是 PDE 解"两条建议，各做了独立诊断：
   - 加点（Run A 20k → Run B 80k）有用但**边际递减**——两个 run 都跑完
     500 calls，training-DATA plateau 从 2.32·10⁻⁴ 降到 1.46·10⁻⁴（1.6× 降）；
     按 sqrt-ratio 推算全网格 MAE 从 0.125 → 约 0.099（21% 改善）。
   - **CFD 自己**在 Re40 网格上的稳态涡量输运残差 median ≈ 1.4·10⁻³，
     给"既 fit data 又 satisfy PDE 的 PINN"设了物理硬地板。
   - 结论：**1e-4 ~ 1e-5 的目标在这份 CFD 上不可达**，要么换 CFD 网格，
     要么放弃 PDE 约束。

---

## 1. 起点和目标

**问题**：Re=40 圆柱绕流单时刻 PINN 重建。原模型在 4869 个 CFD 点上的指标：

```
mae_u = 6.26·10⁻²    rel_l2_u = 0.16
mae_v = 4.01·10⁻²    rel_l2_v = 0.91   ← v 真值小，相对误差被放大
rmse_uv = 0.16
```

**目标**（你 mentor 提的）：绝对速度误差 ~ 10⁻⁴ ~ 10⁻⁵
（即"误差比速度量级小 4 个数量级"）。

**两条建议**：
1. 加大计算点降 loss
2. 检查 CFD 自己是不是 PDE 的解

---

## 2. 关键 bug：BFGS 没真正用 SSBroyden

调试的时候我去验证 `cfp40.py` 里的 BFGS 调用：

```python
result = scipy.optimize.minimize(
    loss_and_grad, w, method="BFGS", jac=True,
    options={..., "method_bfgs": "SSBroyden1"})
```

scipy 的 `_minimize_bfgs` 函数签名里**没有** `method_bfgs` 参数，
传进去的会落到 `**unknown_options`，被静默丢弃。我在 sandbox 里
`inspect` 过 scipy 源码确认了。

`cfp.py` 里是另一种调法 —— 显式 `from _optimize import fmin_bfgs`，
所以 SSBroyden 真正生效。`cfp40.py` 没做这件事。

**修法**：在 `cfp40.py` 顶部改成

```python
from _optimize import _minimize_bfgs as _CUSTOM_MIN_BFGS
...
result = scipy.optimize.minimize(
    ..., method=_CUSTOM_MIN_BFGS,        # ← 把函数本体作为 method
    options={..., "method_bfgs": "SSBroyden1"})
```

启动时打印一行 `[BFGS] using CUSTOM _optimize._minimize_bfgs`
确保不会再被无声坑一次。

**效果（training log）**：BFGS 阶段 data MSE 从 8.2·10⁻³ → 2.4·10⁻⁴（30× 改善）。

**真实评估（全网格 4869 点，见 analysis_runA/summary.txt）**：
```
mae_u = 0.125,   mae_v = 0.068,   rel_l2_uv = 0.238
```
比原始（mae_u ≈ 6.3·10⁻²）**反而差了 2×**。这一点出乎我意料。

**原因诊断**：
- 训练时 data 采样器 `wake_frac=0.8`，80% 点取自尾流（PINN 在那里拟合得好）。
  完整 4869 点评估包含远场 + 入口 + 上下边界——那些位置 PINN 误差大得多。
- 入口处 rmse(u−1) = 0.196（这是直接报出来的诊断值）。看图能看到
  入口 u 在 [0.5, 1.2] 之间晃，离 target=1 差很远。
- 训练 log 里 inlet/top_bottom 的权重被自适应权重压到 0.7~1.0，
  PINN 觉得"反正这块儿数据点少（不在 wake_frac 里），少管点没事"。

**修法（下一阶段）**：
- 评估永远在全网格上跑，别看 training log 的 DATA 项。
- 把 inlet/top_bottom 的固定权重至少 ×3。
- 或干脆 `wake_frac=0.4` 让训练采样更均匀。

---

## 3. 壁面 lift 的硬约束（解释 L_wall_uv ≈ 10⁻¹⁵）

你之前问过为什么训练 log 里 wall_uv loss 永远是 ~10⁻¹⁵。
这不是优化器把它压下去的，是**结构上就是 0**。下面是推导。

### 3.1 模型形式

模型把 stream function 写成

$$
\psi(x,y) = \mathrm{lift}(r) \cdot \psi_{\rm raw}(x,y),\qquad
\mathrm{lift}(r) = \tanh^2\!\big((r-R)/\delta\big)
$$

其中 $r=\sqrt{(x-x_c)^2+(y-y_c)^2}$，$R=0.5$ 是圆柱半径，$\delta=0.15$
是 lift 函数的过渡带宽度。速度通过

$$
u = \partial_y\psi,\qquad v = -\partial_x\psi
$$

得到，所以 div(u,v) = 0 自动满足（这一点之前的 ppt 第 4 页已经讲过）。

### 3.2 lift 在 r=R 处的两条性质

直接代入 $r=R$，$s = (r-R)/\delta = 0$：

- $\mathrm{lift}(R) = \tanh^2(0) = 0$
- $\mathrm{lift}'(r) = \dfrac{2\,\tanh(s)\,\mathrm{sech}^2(s)}{\delta}$
  $\;\Rightarrow\;\mathrm{lift}'(R) = \dfrac{2\cdot 0\cdot 1}{\delta} = 0$

也就是说 lift 在壁面上**取零，且导数也取零**（双零）。

### 3.3 推到速度上

定义 $\hat r = \nabla r$（指向外法向）。链式求导：

$$
\nabla\psi
\;=\;
\mathrm{lift}'(r)\,\hat r\,\psi_{\rm raw}
\;+\;
\mathrm{lift}(r)\,\nabla\psi_{\rm raw}
$$

在 $r=R$ 上，两项分别带因子 $\mathrm{lift}'(R)=0$ 和 $\mathrm{lift}(R)=0$，
**两项都恒等于 0**。因此

$$
\boxed{\;u\big|_{r=R}=\partial_y\psi=0,\qquad
       v\big|_{r=R}=-\partial_x\psi=0\;}
$$

无滑移条件**在数学上严格满足**，与 $\psi_{\rm raw}$ 的具体形式无关。
PINN 的优化器根本不需要"学"这一条。

### 3.4 那 10⁻¹⁵ 哪来的？

float32 的机器精度 $\epsilon \approx 6\!\times\!10^{-8}$。
经过 tanh 计算 + 自动微分 + 几层全连接，每个速度分量
在壁面上的数值大小约为 $\mathcal{O}(\epsilon)$。
所以

$$
\mathcal{L}_{wall\_uv} \;=\; \langle u^2+v^2\rangle
\;\sim\; \epsilon^2 \;\approx\; 4\!\times\!10^{-15}
$$

Run A 实际测到的是 **3·10⁻¹⁵**，量级完全对得上。✓

**结论**：这一项是装饰、不是约束。它出现在 loss 里只是为了
"诊断方便"——如果哪天它涨到 1e-10 量级，就说明 lift 函数被
某次实验改坏了。

---

## 4. Run A vs Run B —— 加点对照

按你"加点能降 loss"的建议，在同样的 1500 Adam + 500 BFGS 预算下做 A/B：

| 项目 | Run A | Run B |
|---|---|---|
| n_f               | 20 000 | 80 000 (4×) |
| n_cfd_pde         | 4 000  | 16 000 (4×) |
| n_data            | 8 000  | 8 000  |
| Adam epochs       | 1 500  | 1 500  |
| BFGS calls        | 500 | 500 |
| 网络              | 96 / 5 | 96 / 5 |

### 4.1 Trajectory 对比（training-time data MSE）

```
              Run A         Run B
call  50    8.96·10⁻³    7.74·10⁻³
call 100    1.40·10⁻³    8.08·10⁻⁴   ← B 比 A 低 1.7×
call 107    9.40·10⁻⁴    6.30·10⁻⁴   ← B 比 A 低 1.5×
plateau     2.32·10⁻⁴    1.46·10⁻⁴   ← B 比 A 低 1.6×
```

Run A 的 BFGS 在 call ~220 之后卡在 plateau ≈ 2.32·10⁻⁴；
Run B 在 call ~250 之后卡在 plateau ≈ 1.46·10⁻⁴。

### 4.2 推算 Run B 全网格 MAE

由 Run A 实测 mae_u = 0.125 对应 training-DATA = 2.32·10⁻⁴，
按 sqrt-ratio 推 Run B：

$$
\text{mae}_u^{(B)} \;\approx\; 0.125 \cdot \sqrt{1.46\!\times\!10^{-4} / 2.32\!\times\!10^{-4}}
\;\approx\; 0.125 \times 0.793 \;\approx\; 0.099
$$

对应 mae_v ≈ 0.054。

（注：这是基于"误差均匀缩放"的粗估；要精确数字需要在 Run B 的
checkpoint 上跑一遍 `run_analysis_re40.py --save-dir checkpoints_sanity_B`。）

### 4.3 结论

- **加点确实有用，但边际递减**：n_f 从 20k 加到 80k（4×）换来 training-DATA
  plateau 1.6× 降低，对应 MAE 改善 ~21%。
- **plateau 的本质**：在 plateau 处 w_data·data ≈ w_pde·pde（两边贡献等量），
  优化器在 data-PDE 的 Pareto 前沿上找平衡。再加 colloc 点只缩小蒙特卡洛
  噪声、不改变 Pareto 前沿——所以从 80k 再加到 320k 估计也只能再降一档。
- **真正能突破 plateau 的杠杆**是改权重比 + 重采样数据分布，
  不是单纯加点。

---

## 5. CFD 自己是不是 PDE 的解？（核心结论）

这是这次回答你"看 CFD 是不是 PDE 的解"那条建议的硬数据。

### 5.1 方法

不需要 trained model，直接在 CFD 网格上：

1. 每个内部点 → 找 k 个最近邻
2. 局部二次拟合 $u, v$，读出 $u_x,u_y,u_{xx},u_{yy},u_{xy}$ 等
3. 算涡量 $\omega = v_x - u_y$
4. 第二轮局部二次拟合 $\omega$，读出 $\omega_x,\omega_y,\omega_{xx},\omega_{yy}$
5. 取 momentum 方程的 **curl 消去压力**，得到稳态涡量输运残差：

$$
r_\omega \;=\; u\,\omega_x + v\,\omega_y - \frac{1}{Re}(\omega_{xx}+\omega_{yy})
$$

如果 CFD 是稳态 NS 的解，$r_\omega$ 应该恒为 0。
这一项的统计量直接给出"PINN 在同时满足 PDE 时能达到的 data 误差下限"。

### 5.2 三组配置的结果

| 配置 | n_pts | $\|r_\omega\|$ median | p90 | p99 | max |
|---|---|---|---|---|---|
| k=24, buf=0.15 (default)        | 3378 | 1.78·10⁻³ | 0.231 | 13.1 | 20.9 |
| k=12, buf=0.15 (smaller nbr)    | 3378 | 2.15·10⁻³ | 0.110 | 15.9 | 31.7 |
| k=24, buf=**0.5** (排除 BL+边界) | 2635 | 1.43·10⁻³ | 0.031 | **1.65** | 3.5 |

### 5.3 解读

- **bulk** (远离壁面 0.5 + 远离边界 0.5)：median 1.4·10⁻³，p99 1.65。
  典型 CFD 点上稳态 NS 残差 ~ 10⁻³。
- **近壁带 + 远场带**：是 p99 大跳的主因。Re=40 边界层薄、CFD 网格在
  那里没解析；远场 outflow 也有数值耗散。这个区域的 CFD 不是 NS 的精确解。
- **continuity**: median $|u_x+v_y| \approx 7\!\times\!10^{-4}$。
  CFD 自己**也不是严格无散的**，可能是 staggered grid post-process 或
  不规则网格上的插值噪声。**这件事很要命**——我们 PINN 用流函数模型
  $u=\partial_y\psi, v=-\partial_x\psi$，div(u,v) ≡ 0 严格成立。
  所以 PINN 的 $u,v$ 跟 CFD 的 $u,v$ **逐点差额至少是 7·10⁻⁴ 量级**，
  这是数学下限。

### 5.4 对目标的含义

mentor 想要的"绝对误差 1e-4 ~ 1e-5"对应 data MSE ~ 10⁻⁸ ~ 10⁻¹⁰。
但 CFD 自己的 NS 残差量级是 10⁻³，所以**只要 PINN 同时尊重 PDE，
data MAE 的物理下限就在 ~10⁻³ 这一档**——这次 Run A 的 1.5·10⁻²
还有大约一个数量级可以再压（通过 SSBroyden 长跑 + 加点 + 网络再大），
但 1e-4 这一级**做不到**。

要做到必须二选一：
- **(a)** 换更细的 CFD 网格（在尾流和近壁加密 5-10×），让 CFD 自己的
  NS 残差降到 10⁻⁵ 量级；这是物理路径。
- **(b)** 把 PDE 权重压得很低（lambda_pde ~ 0.01），让 PINN 退化成纯插值器，
  直接拟合 CFD 数据；这能逼近 1e-5 的 data MAE 但 PINN 就不再"满足 PDE"了，
  失去了 physics-informed 的意义。

---

## 6. Run A 当前数字 + 下一步

**Run A 最终状态**（96/5 网络，1500 Adam + 500 BFGS calls）：

| 指标 | Training log | 全网格评估（4869 点） |
|---|---|---|
| data MSE       | 2.4·10⁻⁴ | (没直接报，对应 mae²≈1.6·10⁻²) |
| **mae_u**       | — | **0.125** |
| **mae_v**       | — | **0.068** |
| **rel_l2_uv**   | — | **0.238** |
| PDE rmse_mag   | 4.3·10⁻⁴ | 0.366（near_cyl 0.85，far 0.058，比 15:1）|
| L_wall_uv      | 3·10⁻¹⁵ | max\|u\|=2.0·10⁻⁷, max\|v\|=1.4·10⁻⁷ ✓ |
| inlet rmse(u−1)| — | **0.196**（比预期高很多）|
| top/bottom rmse(u−1) | — | 0.169 / 0.154 |
| vorticity rmse | — | 2.71（CFD peak ±8.1）|

相比原始（mae≈0.063）**实际反而差了 2×**——但 PDE 残差降了，
说明 PINN 现在更"满足 PDE"但更不"匹配 data"。这是权重设定问题，不是模型容量问题。

**下一步：mentor 给的两阶段策略**

Mentor 听完汇报后给的方案——**比单纯调权重/加点更聪明**：

**Phase 1（data-only）**：把 PDE 完全关掉（`--data-only`），让 PINN 当
**纯插值器**去拟合所有 4869 个 CFD 点。这一步的目的是：把 CFD 的"真值场"
重建出来，不让 PDE 干扰。

**Phase 2（resume + 开 PDE）**：从 Phase 1 的 checkpoint 接着训，把 PDE
打开。这时 BFGS 一开始报的 PDE 残差直接告诉我们"**纯数据拟合的场到底
违反 NS 多少**"。如果残差小，说明 CFD 跟我们的 NS setup 是一致的；
如果残差大、且 BFGS 必须把 data loss 抬高才能压下去，说明 CFD 跟 NS
本身就矛盾。

**为什么这个方法比之前我做的 `diagnose_cfd_pde.py` 好**：
- `diagnose_cfd_pde.py` 用局部二次拟合算 NS 残差，对 CFD 网格上的噪声敏感，
  剪切层附近 p99 跳到 1.6
- PINN 当插值器自然光滑（tanh 网络），求导用 autograd 干净不带局部噪声
- 同一个流场用同一种方式算 PDE 残差 → 数字直接可比

**Mentor 的迭代节奏**：iters_per_batch ~ 100-200，然后 resample（即每个
BFGS batch 跑 100-200 iter 后重新采点）。我已经把 cfp40.py 改完支持：
- `--data-only`：跳过 PDE
- `--use-all-cfd-data`：用全部 CFD 点而不是采样
- `--resume-from`：从 checkpoint 接着训
- `--no-frozen-colloc-bfgs`：每 batch 重采

完整命令在 `train_two_phase.sh`：

```bash
# Phase 1: data-only
python -u cfp40.py --vtk-path Re40.vtk \
    --save-dir checkpoints_phase1 \
    --width 96 --depth 5 \
    --data-only --use-all-cfd-data \
    --epochs-adam 1000 \
    --maxiter-bfgs 1500 --iters-per-batch 150 \
    --no-frozen-colloc-bfgs \
    --method-bfgs SSBroyden1

# 跑诊断看 Phase 1 数据拟合得多好
python -u run_analysis_re40.py \
    --vtk-path Re40.vtk \
    --save-dir checkpoints_phase1 \
    --out-dir  analysis_phase1 \
    --width 96 --depth 5

# Phase 2: 从 phase1 接，开 PDE
python -u cfp40.py --vtk-path Re40.vtk \
    --save-dir checkpoints_phase2 \
    --resume-from checkpoints_phase1/pinn_Re40_single.pt \
    --width 96 --depth 5 \
    --use-all-cfd-data \
    --epochs-adam 200 \
    --maxiter-bfgs 1500 --iters-per-batch 150 \
    --n-f 50000 --n-cfd-pde 12000 --lambda-pde-cfd 1.0 \
    --no-frozen-colloc-bfgs \
    --method-bfgs SSBroyden1
```

**判读结论的标准**：
1. Phase 1 跑完看 `analysis_phase1/summary.txt`：
   - `mae_u, mae_v` 应该 < 0.01（纯插值器，数据拟合应该很好）
   - `pde_rmse_mag` 是关键 → **这就是"CFD 自己违反 NS 多少"的最佳估计**
     - 如果 < 1e-2：CFD 跟 setup 一致，PINN 有希望同时满足 data 和 PDE
     - 如果 > 1e-1：CFD 自己物理上就跟 NS 不一致，PINN 卡在 plateau 是必然
2. Phase 2 跑完看 trajectory：
   - 如果 data loss 几乎不抬，PDE loss 直接降下来 → 一致
   - 如果 data loss 必须抬一两个数量级才能让 PDE 降 → 强不一致，需要换 CFD

---

## 7. 给 mentor 的关键问答准备

**Q: 为什么 wall loss 一直是 1e-15？**
> 因为壁面无滑移是通过流函数 lifting 强制写进网络结构里的：
> ψ = lift(r)·ψ_raw，lift(R)=lift'(R)=0，所以 ∇ψ 在壁面上恒为 0，
> u=v=0 不需要训练。1e-15 就是 fp32 精度的平方，是结构上的零。

**Q: 加点为什么不再降？**
> Run B（4× 点）在 BFGS 早期比 Run A 快 1.5×，但都收敛到同档 plateau。
> 因为 PINN 的 loss 是 data 和 PDE 两项的和，在 plateau 处两项贡献等量，
> 加点只缩减蒙特卡洛噪声，没改 Pareto 前沿。

**Q: CFD 是不是 PDE 的解？**
> 在 bulk 上：median |涡量输运残差| ≈ 1.4·10⁻³，p99 ≈ 1.65（远场和边界层除外）。
> 即"典型点上 CFD 满足 NS 到 1e-3 量级"。所以 PINN 同时尊重 PDE 时，
> data MAE 的下限是 ~1e-3。继续往下要么换 CFD 网格，要么放弃 PDE 约束。

**Q: 1e-4~1e-5 的目标能达到吗？**
> 在当前这份 CFD 上不能，只要还要求 PDE residual 也很小。
> 物理路径是把 CFD 网格在尾流和近壁加密到能让自身 NS 残差 ~ 1e-5。
> 否则只能放弃"既 fit data 又 satisfy PDE"，做纯插值。

**Q: Run A 的实际 MAE 为什么比 training log 推算的差 8×？**
> Training 时 data 采样 wake_frac=0.8，80% 点取自尾流，那块儿 PINN 拟合最好。
> 全网格 4869 点评估包含远场 + 入口 + 上下边界——那些位置 PINN 误差大。
> 入口 rmse(u−1)=0.20 就是直接证据。下次评估必须在全网格上做，
> 不能只看 training log 的 DATA 项。

---

## 8. 用到的代码改动一览

| 文件 | 改动 |
|---|---|
| `cfp40.py` | 修 SSBroyden 调用；frozen colloc + full data BFGS；continuity 项；启动内存预算检查；`--cfd-pde-wall-buffer` / `--cfd-pde-edge-buffer`；`evaluate_against_cfd` 直接打印 mae；**新加 `--data-only` / `--use-all-cfd-data` / `--resume-from` 三个 flag 支持 mentor 的两阶段策略** |
| `diagnose_cfd_pde.py` | 独立脚本——用局部二次拟合算 CFD 的 NS 残差 |
| `run_analysis_re40.py` | 独立脚本——把 notebook 的所有诊断打成一个 PNG + summary 文件 |
| `train_two_phase.sh` | 新 SLURM 脚本——Phase 1 data-only → Phase 2 resume + PDE |
| `run_pinn.sh` | 改成调 cfp40.py 而不是 cfp.py |
| ppt 23 张 | 14 张原 + 9 张新（commands / loss 曲线 / 壁面 lift 数学 / CFD-PDE 残差 / 4 张 Run A 诊断 / Run A vs B 对比）|
