# 数据目录

数据文件不提交 Git。

## MVTec AD

- 官方页面：<https://www.mvtec.com/research-teaching/datasets/mvtec-ad>
- 许可：CC BY-NC-SA 4.0，非商业使用。
- 目标目录：`data/mvtec_ad/`

官方页面要求填写表单后下载。下载完成后保留原压缩包校验和，并核验 15 个类别及像素掩码。

## VisA

- 官方登记页：<https://registry.opendata.aws/visa/>
- 官方项目：<https://github.com/amazon-research/spot-diff>
- 许可：CC BY 4.0。
- AWS 对象：`s3://amazon-visual-anomaly/VisA_20220922.tar`
- 原始目录：`data/visa_raw/`
- 单类协议目录：`data/visa_1cls/`

匿名下载命令：

```powershell
aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar data/downloads/VisA_20220922.tar
```

## Shot 清单

统一 1/2/4-shot 清单保存到 `data/splits/` 并提交 Git。清单只写相对路径，不复制数据。

## MPDD

- 来源: Hugging Face 镜像 (原始 SharePoint 下载需要机构登录)
- LFS SHA256: `69f8da73eea4a31451a50251e5c261e83e0c53f2d1a39a7d4dfc78b5c434ddd6`
- 原始目录: `data/mpdd_raw/MPDD/`
- 6个类别: bracket_black, bracket_brown, bracket_white, connector, metal_plate, tubes
- 测试图片: 458张, 异常图片: 282张 (含hole类异常)
- 清单: `data/splits/mpdd/manifest.json` — SHA256 `5a6a42dd12de1de9c977c2b10695f35b474d19b37f0c1492f64a7989226a9bd8`

## BTAD

- 来源: 公共服务器 (BTAD 文献引用)
- 原始目录: `data/btad_raw/`
- 3个类别 (类别03使用BMP遮罩, 需特殊处理)
- 清单: `data/splits/btad/manifest.json` — SHA256 `40696d901a78006c342dce98625dc21221b8ee9f642ebb74b7c3f3ffc5a1d215`

