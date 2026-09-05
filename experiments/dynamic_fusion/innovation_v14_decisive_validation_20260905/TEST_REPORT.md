# V14 P0 TEST_REPORT（doc28 §4.3/§4.4 确定性测试）

日期：2026-09-05。命令：`python scripts/innovation_v14_decisive_validation_20260905/test_p0.py`
结果：**18/18 通过**（exit 0）。哈希见 RUN_MANIFEST.json。

## DNC-C 选择器（修复版）
1. λ=0：DNC-C 集合 == DNC-I（D 与 C 各自集合相等）— 通过
2. 构造性翻盘：C 分支 argmax-q 通道与已选 D 通道相关 0.99，λ=8 时被次高 q（相关 0.05）替换；
   D 分支集合不变；chosen 跨分支 max 相关 0.99→0.05 — 通过
3. 每分支恰好 256 个唯一索引；重复运行完全一致（确定性）— 通过
附带：768 通道×λ=0.5 实际尺度 Jaccard(DNC-C, DNC-I) = 0.977（D），冗余惩罚确有集合效应窗口。

## 半松弛容量 OT（广义 Sinkhorn，行固定/列软 KL）
1. 长方形 Q=64×A=128 可运行；行边缘误差 1.4e-17 < 1e-5 — 通过
2. τ→0：与逐行熵 soft matching（`_row_project`）逐位一致（无虚构全局一一配对）— 通过
3. τ 增大：列质量向 ρ 靠拢（L1 0.890→0.028）；Q=A 且 ρ 均匀、τ 大 → 逼近平衡 OT
   （max|ΔP|=0.000 相对参考 Sinkhorn）— 通过
4. 同时置换 query 行与 anchor 列后逆置换还原（permutation invariant）— 通过
5. identical query/support：容量 premium ≈ 0（4.7e-25）— 通过
6. 复制某 normal anchor 内容 d∈{2,4,8}：复制行 premium 随 d 单调上升
   （0.48/0.65/0.78）；远行（非复制）premium 0.012 ≪ 复制行 0.779（spillover 受控）— 通过

## 备注
- 求解器实现为 log-domain 广义 Sinkhorn：`P_ij = a_i K_ij b_j / Z_i`，
  `b_j = (c_j/ρ_j)^(-τ/(ε+τ))`（指数 ∈(0,1)，避免空列 log(0) 陷阱；1 行情形解析解一致）。
- 大矩阵（P2 真实门 1024×(K·1024)）的逐图流式与性能待 P2 前 profile（iters/收敛容差可调）。
