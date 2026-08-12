# DynamicFusion V2 数据准备与冻结记录

更新时间：2026-08-10

## 1. 数据用途边界

- MPDD：V2 开发集，只用于选择校准、支持范围、路由阈值和权重上限。
- BTAD：独立保持集。V2 参数冻结之前禁止读取其总体指标或逐类别指标。
- MVTec AD、VisA：只允许在 V2 冻结后做回顾验证，禁止继续用于 V2 参数选择。

## 2. 下载来源

### MPDD

- 权威说明与许可证：https://github.com/stepanje/MPDD
- 作者提供的 SharePoint 下载链接在 2026-08-10 跳转到机构登录，无法直接自动下载。
- 当前下载镜像：https://huggingface.co/datasets/meksamiao/mpdd
- 镜像归档：`data/downloads/MPDD.zip`
- 预期大小：1,825,041,283 bytes。
- 预期 LFS SHA256：`69f8da73eea4a31451a50251e5c261e83e0c53f2d1a39a7d4dfc78b5c434ddd6`。
- 镜像身份不能只凭文件名确认；下载后还必须通过 ZIP、6 类目录、图像和掩码关系审计。

### BTAD

- 公开归档：https://avires.dimi.uniud.it/papers/btad/btad.zip
- 本地归档：`data/downloads/btad.zip`
- 服务器报告大小：1,229,193,337 bytes。
- 来源未发布 SHA256；下载完成后计算本地 SHA256，并把结果写入归档审计报告。

## 3. 自动流程

下载采用 `curl --continue-at -` 断点续传。下载完成后，`scripts/complete_v2_data_preparation.ps1` 自动执行：

1. 检查归档大小。
2. 计算 SHA256；MPDD 还要匹配预期 LFS SHA256。
3. 检查 ZIP 中央目录、逐成员 CRC 和路径穿越风险。
4. 解压到独立目录，不覆盖已有非空目录。
5. 识别唯一数据根目录。
6. MPDD 校验 6 类，BTAD 校验 3 类。
7. 检查正常训练图、测试图、异常图和掩码关系。
8. 生成 1/2/4-shot、seed 0/1/2 嵌套 manifest。
9. 为所有入选正常参考图保存 SHA256。
10. 验证 manifest 本身的 SHA256、路径存在性、数量和嵌套性。
11. 生成数据协议冻结文件；冻结文件仍明确写明 `parameters_frozen=false` 和 `holdout_metrics_allowed=false`。

## 4. 状态和日志

- 自动流程状态：`experiments/dynamic_fusion/v2/data_preparation/automation_status.json`
- 数据就绪状态：`experiments/dynamic_fusion/v2/data_preparation/readiness.json`
- 总日志：`outputs/logs/v2_data_preparation.log`
- 下载错误日志：`outputs/logs/mpdd_download.stderr.log`、`outputs/logs/btad_download.stderr.log`
- 最终数据冻结：`experiments/dynamic_fusion/v2/data_protocol_freeze/manifest.json`

只有自动状态为 `complete`、数据就绪状态为 `ready`、数据冻结复核为 `passed` 时，才允许准备 MPDD 分支缓存。BTAD 缓存可以提前生成，但其标签指标必须在参数冻结前保持不可见。
