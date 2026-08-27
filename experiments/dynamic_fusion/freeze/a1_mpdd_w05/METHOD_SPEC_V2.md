# METHOD_SPEC_V2.md — 投稿版 A1 方法说明（2026-08-27）

本文件是当前投稿口径的方法规格。历史冻结文件（如 `METHOD_CARD.md`）保留原样作为
历史证据，不在此处修改；二者冲突时以本 V2 为准。

## 1. 方法（固定，不再改动）

双编码器**视觉** patch 特征融合 + 正常记忆库 KNN：

1. **编码器 A（DINO 分支）**：DINOv2 ViT-B/14，输入短边 448，patch 网格约 `32×32`，
   每 patch 768 维。
2. **编码器 B（CLIP 图像分支）**：AnomalyCLIP 的 ViT-L/14@336 图像塔，输入 518，
   patch 网格 `37×37`，每 patch 768 维；测试时通过 DAPM 层抽取 patch 特征。
3. **对齐**：将 CLIP 分支网格按 bilinear resize 对齐到 DINO 网格（`resize_patches`）。
4. **分支归一化**：两个分支分别做 L2 归一化。
5. **拼接**：等权 concat（w=0.5/0.5），维度 **768+768=1536**；再整体 L2 归一化。
6. **记忆库**：用每类别 K 张正常参考图（K∈{1,2,4}，seed∈{0,1,2}）的 patch 特征建 FAISS
   IndexFlatL2；测试 patch 查最近邻（k=1），距离/2 为像素异常分数。
7. **上采样**：patch 异常图经 gaussian_filter(sigma=4) + INTER_LINEAR resize 到 448×448。
8. **指标**：448 图上按 stride=8 采样，计算 Pixel-AUROC / Pixel-AP / Pixel-AUPRO(30%)。

matched baseline：同一管线仅用 DINO 分支（第 4、6-8 步），不拼接 CLIP。

## 2. 维度与命名修正（相对旧文档）

- **concat 维度 = 1536**（768+768）。旧文档中的 **1152 是错误的**，已由 P0-2 smoke
  实测消除（`outputs/p0_2_smoke/smoke_report.json`，`concat_dim_resolved=1536`）。
- **`anomalyclip_text` 仅是历史目录名**，不代表文本特征参与 A1 推理。A1 最终推理
  `explicit_text_features_used_at_inference = false`，只用 CLIP 图像塔 patch 特征。
- 论文方法命名：**dual-encoder visual feature fusion**，不使用 multimodal /
  vision-language / dynamic router 表述。

## 3. 与历史 METHOD_CARD 的关系

历史 `METHOD_CARD.md`（freeze/a1_mpdd_w05/）保留 1152 与 `Multimodal` 字样，是
冻结时刻的证据快照，**禁止静默修改**；其数值口径已被本 V2 取代。

## 4. 泄漏契约

- 记忆库、归一化、KNN 均在正常参考图上构建；未使用测试标签、mask 或测试集统计。
- 所有数据集（MPDD/BTAD/VisA/MVTec）的 5 项泄漏 flag 均为 false。
- VisA 为 AnomalyCLIP checkpoint 的训练域内验证集，论文不得写成独立外部验证。
