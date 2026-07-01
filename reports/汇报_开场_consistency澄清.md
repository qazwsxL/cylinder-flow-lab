# 汇报开场：先和 Mentor 确认 consistency 含义

> 用法：在正式陈述前花 60-90 秒先报告我的理解 + 摊开数据请 mentor 确认。
> 三个含义的"我们手上有什么"以表格形式呈现，方便 mentor 当场点选。
> 确认完之后再走 `汇报稿_v2_consistency.md` 里的主体。

---

## 开场陈述（约 60 秒）

> 老师好。在开始之前，我想先和你确认一下汇报的中心问题。这次我把
> "consistency"理解成三件不太一样的事；v2 的实验主要测了前两件，
> 第三件只有两个数据点，还没做完整的诊断。我想先把三件事和我手上的
> 数据摆出来，请你说一下这次你最想听的是哪一件，或者三件都需要
> 我都讲到——这样我可以分配陈述时间，必要的话补一个短的 follow-up
> 实验提案。

---

## 三个 candidate 含义 + 我们的数据

### **C1.  CFD 自己满不满足 NS？** （CFD ↔ NS）

> "PINN 拟合的 CFD field，本身是不是稳态 Navier–Stokes 的解？"

| 维度 | 我们已经有 |
|---|---|
| 方法 | `diagnose_cfd_pde.py`：每个内部 cell 做 k-NN + 二次拟合 → 直接算 $r_\omega = u\omega_x + v\omega_y - Re^{-1}\nabla^2\omega$ |
| 数字 | **bulk 区域**（远壁 0.5、远场 0.5 buffer）median \|$r_\omega$\| ≈ **1.4·10⁻³**，p99 ≈ 1.65 |
| 边角 | 近壁带 + 远场带 p99 跳到 ~10（OpenFOAM 没解析） |
| continuity | median \|$u_x + v_y$\| ≈ 7·10⁻⁴ —— CFD 不严格无散 |
| 物理含义 | 给所有 PINN 在这份 CFD 上的 PDE residual 设了 **~1e-3 的硬地板** |

→ **已经有定量答案。** 如果你说 C1 是中心问题，我可以 5 分钟讲完。

### **C2.  PINN 找到的极小点是不是 CFD 这个解？** （PINN ↔ CFD）

> "在容量给定的情况下，PINN 的'fit CFD'极小和'satisfy NS'极小是不是
> 参数空间里的同一个点？"

| 实验 | 网络 | mae drift (P2/P1) | PDE drop | 两极小同点？ |
|---|---|---:|---:|:-:|
| strict consistency (Phase 2 去 data) | 32/3 | 3.20× | huge (→0) | × （trivial attractor） |
| all-CFD anchor sweep prio=0.5..5 | 32/3 | 1.48× → 1.11× 单调 | 1.57× → 0.95× 单调 | × （单调 trade-off，两个 sheet） |
| scale-up (同 protocol 换 64/4) | 64/4 | **1.15×** | **5.97×** | **✓** |

→ **已经有阳性 + 阴性证据。** 32/3 上明确不 consist，64/4 上明确 consist
（PDE residual 0.014 也已经触及 C1 的 1e-3 物理下限）。
对应 slide 24-27 + 28-31。

### **C4.  capacity → ∞ 时 PINN 收敛到真解吗？** （数学 convergence）

> "随网络容量增大，PINN 的逼近误差是不是单调下降并收敛到（CFD-NS 共同
> 满足的）真解？这是数值方法标准定义下的 consistency。"

| 我们手上的数据点 | 网络 | params | mae_u | speed-norm 误差 |
|---|---|---:|---:|---:|
| 数据点 1 | 32/3 | 2.8 k | 0.106 | 17.0 % |
| 数据点 2 | 64/4 | 14.9 k | 0.019 | 2.7 % |

→ **只有两个点，趋势看起来对（5.7× 改善，比 params 比 5.3× 略强一点），
但还不是真正的 convergence 曲线。** 没有 96/5、128/6 这两档数据，
也没有验证是不是单调或者会不会饱和。

**如果你说 C4 是中心问题，我建议补这两组实验**：

- 96/5（params ≈ 44 k）：复用现有 train_v2_scaleup_prio2_smoke.sh，
  改 WIDTH/DEPTH，预计 ~3 小时。
- 128/6（params ≈ 120 k）：同上，预计 ~6 小时（取决于 BFGS 长度）。

跑完就能画出 mae_u vs params 的曲线，看是不是
> mae_u(N) ~ N⁻α

形式，能给出 PINN 在这份 CFD 上的 empirical convergence rate。这件事
原本是 publication 数字要做的，C4 这个问题会直接把它前置。

---

## 请 Mentor 选一下

我想确认这三个含义里你最想听哪个，或者三个都讲：

- **如果是 C1 + C2**：我按现在 `汇报稿_v2_consistency.md` 那份讲，8 分钟。
- **如果还包括 C4**：我先讲 C1 + C2（5 分钟），然后把 64/4 当成 C4
  曲线上的第二个数据点报告，并提议补 96/5、128/6 做完整 convergence
  scan。这次先报阶段结果，下次正式答这个问题。
- **如果你主要想问 C4**：我把 C1 + C2 压成 3 分钟铺垫，剩下时间专门
  讨论 capacity scan 的设计——比如要不要也扫 fourier_features、要不要
  跨 Re 做、convergence rate 怎么报告才是有信息量的。

> 我个人猜你这次主要是 C1 + C2 + C4 的混合，C2 和 C4 哪一个权重大我
> 没有把握，所以想先问一句。

---

## 备查：三个含义之间的逻辑关系

> 这一节给我自己看的，不用读出来。如果 mentor 问起再用。

```
            ┌─ C1 ─┐   CFD 是不是 NS 的解？  (物理基础)
            │      │
            │      ▼  bulk |r_ω| ~ 1e-3  → 物理硬地板
            │
            ├─ C2 ─┐   PINN 找到的极小 = CFD 这个解？  (这次主轴)
            │      │
            │      ▼  在 capacity-sufficient + data-anchored 时 ✓
            │         在 capacity-deficient 或去 anchor 时 ×
            │
            └─ C4 ─┐   capacity → ∞ 时 PINN → 真解？  (数学 convergence)
                   │
                   ▼  N = 2.8k → 14.9k 这一段是单调下降的；
                      96/5、128/6 没跑，convergence rate 未知
```

- C2 是 C4 在**有限容量某一点**上的快照。如果 C4 成立，那么对足够大的
  容量 C2 自动成立——所以 64/4 的 C2 通过其实也算 C4 一个支持证据。
- C1 是 C2 和 C4 都依赖的物理前提。如果 CFD 自己不是 NS 解，C4 收敛到
  的"真解"和 CFD 就是两个不同的东西，那 C2 永远不可能 strictly 成立——
  只能成立到 CFD 残差的水平（我们就是这样：PDE residual 0.014，CFD 残差
  1.4·10⁻³，差 10× —— 接近但还有空间）。
- 跑 96/5 + 128/6 之后能同时给 C2（看那一点是否仍然双通过）和 C4
  （看趋势）两个新信息。

所以无论 mentor 想要哪个，**capacity scan 都是下一步最划算的实验**。
区别只是讲稿里把它定位成"publication 准备"还是"C4 直接回答"。
