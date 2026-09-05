# DATA_ROLES（doc 27 冻结；2026-09-04 深夜）

- MPDD development（seed0 × shot{1,2,4}）：唯一允许的真实异常评估集；已被历史大量开发消耗。
  今晚只产生 development 候选，**不产生外部泛化结论**。
- 拟合数据（normal-only）：只允许
  (a) 该类别 shot support 的 K 张正常参考图（JTD k2/k4 用 leave-one-image-out；k1 仅空间排除探索，不作独立样本声称）；
  (b) 合成干预/正常图的通道选择（N3 只用支持集 normal 图 + ntof 干预/合成生成器，禁止用真实缺陷标签选通道）。
- 目标类正常 **test/good** 图：不得自动进入拟合（G0）。只有 shot support 允许时，不额外用全部 train/good 暗中加参考。
- 其余类别 normal（MPDD 其他类）：O1/O2 需要用户对"源类训练"明确授权后才可用；**今晚无授权，不启用**。
- BTAD/MVTec AD：旧历史诊断用，今晚不再为新方法调参。
- VisA：既有 checkpoint 的源域；今晚不用于新训练。
- Real-IAD / MVTec AD 2 / MVTec LOCO：未接触，保持未接触；需要新授权方案才讨论（G4 今晚不做）。
- 无新下载、无新模型权重、无新大依赖。

## 每路线拟合/评估分离
- 评估只用：预注册冻结后的分数 + GT（AP/AUROC）。
- 任何数据驱动规则（联合直方图、容量、通道、memory）只按各协议允许的 normal/源数据拟合，保存拟合数据 ID 清单。
