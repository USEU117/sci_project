# 广度探针 R9 结案（2026-09-09 夜，续）— DST/PLC 未过门

预注册见 `ROUND9_PLAN_CN.md`；门同全 breadth；parity k2 0.343706 / k4 0.388328 **精确复现**。

## 逐族结果（Δ 相对 C0=冻结 A1 concat top-1；宏为六类）

| 族 | lead | k2 宏 ΔAP / k2 worst | k4 宏 ΔAP / k4 worst | k2 ΔAUROC / k4 ΔAUROC | 判定 |
|---|---|---|---|---|---|
| DST 距离度规（L1 / Chebyshev） | DST_L1 | +0.0049 / −0.0064 | +0.0015 / **−0.0334** | +0.0029 / +0.0030 | FAIL（宏 < 门；k4 worst 破门） |
| PLC 位置锁定匹配（对照） | PLC | −0.2686 / −0.7037 | −0.2917 / −0.7049 | −0.0829 / −0.0517 | FAIL（灾难，预期对照） |

成员明细：DST_cheb k2 −0.099/worst −0.316、k4 −0.089/worst −0.205（L∞ 被单维最大差主导 → 深负）。

## 解读
1. **DST_L1**：把 1−cos(L2) 换成 L1 后行 unit 特征的 top-1，两 shot 增益集中在**强类**（k2 metal_plate +0.023/tubes +0.017；k4 connector +0.021/metal +0.016/tubes +0.012），
   弱类受损（k2 最差 bracket_white −0.006；k4 最差 bracket_black −0.033）。
   这是继 PA/CB（救硬/中类）之后**第三种方向相反**的良性亚门：L1 的"全维等权求和"对强类的均匀大缺陷更稳健，但对弱类（采样脆弱的 bracket_black 在 k4）有真实回归。
   宏 0.005/0.0015 与 worst 门（k4 −0.033 < −0.03）双重不过 → 结案；与 PA 不同，DST_L1 有真实的 worst 权衡，不能作"无损"注记，仅记录"度规敏感性"方法学事实。
2. **DST_cheb / PLC**：L∞ 与"无配准位置锁定"在 MPDD 均不可用；PLC 与 GV/COP/旋转对齐证据链合拢：
   **MPDD 各图间无逐像素配准先验**（括号件存在位姿/旋转差异），任何依赖坐标对应的廉价先验族在真实门都会被 A1 的全局样例 KNN 完胜。
3. 账目：**累计 30 机制族在真实 MPDD s0 k2/k4 闭合为负（R1–R9 27 族 + Web1 2 族 + FastRef 1 族）**；A1 仍唯一冻结方法；
   剩余可离线尝试的匹配/度量变体已基本穷尽（余弦、L1、L∞、L2、top-k、池化、通道、密度、度规、位置全部单测）。

## 文件
- 预注册：`ROUND9_PLAN_CN.md`；脚本：`probe_breadth8.py`（脚本名承接 R8 自增）
- 结果：`round9/RESULTS_s0_k2.json` / `RESULTS_s0_k4.json`
