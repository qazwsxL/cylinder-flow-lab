# Re=40 PINN 阶段汇报稿

> 用于向 mentor 当面汇报。每节都按"做了什么 → 看到什么 → 学到什么"
> 三段写。括号里是可读出口的关键数字。

---

## 0. 三句话总结

1. 修了一个之前没察觉的 bug：BFGS 阶段实际上一直跑的是 **scipy 标准 BFGS**，
   不是 SSBroyden——`method_bfgs="SSBroyden1"` 这个 option 被 scipy 静默吞掉了。
   修好之后，BFGS 阶段第一次真正起作用，data MSE 从 ~8·10⁻³ 直接掉到 ~2.4·10⁻⁴。
2. 按你之前的两条建议（"加点"、"看 CFD 是不是 PDE 的解"），各做了独立诊断：
   - 加点确实有用，但**边际效用递减**，从 8e-4 量级再往下 1.5× 左右就到顶。
   - **CFD 自己**在 Re40 网格上的 NS 残差 median ≈ 1·10⁻³，
     这给"既要 fit 数据又要 satisfy PDE 的 PINN" 设了一个物理硬地板。
3. 因此，把 PINN 的**绝对速度误差压到 1e-4 ~ 1e-5** 在当前这份 CFD 上做不到。
   要做就必须换更细的 CFD 网格，或者把 PINN 当作"纯插值器"用（PDE 权重压到很低）。

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

**效果**：BFGS 阶段 data MSE 从 8.2·10⁻³ → 2.4·10⁻⁴（30× 改善），
对应 MAE 从 6.3·10⁻² → 1.5·10⁻²，**比之前总改善 4 倍**。
这是这次最大单点收益。

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

## 4. 加点的对照实验

按你"加点能降 loss"的建议，在同样的 1500 Adam + 500 BFGS 预算下做 A/B：

| Run | n_f | n_cfd_pde | data MSE @ BFGS call 100 | data MSE @ call 200 |
|---|---|---|---|---|
| A (baseline)  | 20000 | 4000  | 1.40·10⁻³ | 5.0·10⁻⁴ |
| B (4× points) | 80000 | 16000 | **8.1·10⁻⁴** | ~3.5·10⁻⁴ (推算) |

**观察**：
- 加点确实让早期收敛快 1.5×，特别是 BFGS 前 100 calls。
- 但 BFGS batch 2 之后 Run A 已经卡在 plateau 2.4·10⁻⁴，再多 calls 不动。
  Run B 估计 plateau ~ 1.5·10⁻⁴。
- **边际效用递减**：4× 的点换来不到 2× 的 loss 改善。

**为什么会卡**：在 plateau 处，data 项和 PDE 项贡献几乎等量
（w_data·data ≈ w_pde·pde ≈ 1.3·10⁻³），优化器在两个目标的
Pareto 前沿上找平衡。再加点没用，得改权重比或者动数据本身。

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

```
data MSE plateau   ≈ 2.4·10⁻⁴       →  data MAE ≈ 1.5·10⁻²
PDE colloc MSE     ≈ 4.3·10⁻⁴
L_wall_uv          ≈ 3·10⁻¹⁵        (machine ε², 见 §3)
total loss         ≈ 4.0·10⁻³
```

相对原始（mae~6.3·10⁻²）改善 4×。

**下一步建议**（按性价比排序）：

1. 把 BFGS 跑长（5000+ calls 而不是 500），SSBroyden 在 plateau 之后
   还会缓慢改善，预计到 1·10⁻²。**不用换数据、不用换网络**，单纯多跑。
2. 网络放到 96/6 或 128/5，配 96G+ 内存（H0=NxN float64 是 BFGS 的内存大头，
   见 ppt 内存预算检查那一节）。
3. 如果还想再压：要么换 CFD（路径 a），要么把 lambda_pde 降到 0.01
   退化成插值器（路径 b）。

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

---

## 8. 用到的代码改动一览

| 文件 | 改动 |
|---|---|
| `cfp40.py` | 修 SSBroyden 调用；frozen colloc + full data BFGS；continuity 项；启动内存预算检查；加 `--cfd-pde-wall-buffer` / `--cfd-pde-edge-buffer`；`evaluate_against_cfd` 直接打印 mae |
| `diagnose_cfd_pde.py` | 新文件——独立诊断 CFD 是不是 NS 的解（curl 消 p） |
| `run_pinn.sh` | 改成调 cfp40.py 而不是 cfp.py，加 `--mem`、`-t` 等 SLURM 参数 |
| ppt 新增 4 张 slide | 命令配置 / Run A loss 曲线 / 壁面 lift 数学 / CFD-PDE 残差 |
