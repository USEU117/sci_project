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

