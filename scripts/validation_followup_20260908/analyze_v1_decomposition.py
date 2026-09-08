"""Validation step 1: per-defect-type decomposition of the real 448 vs 896 image-level gain.

Reads the completed overnight real-resolution diagnostic (RESULTS.json + PER_IMAGE.jsonl)
and breaks the macro image-AUROC / pixel-AP differences down to (category, defect type),
cross-referencing the D6-known gap groups (bracket_brown context family, hole,
defective_painting, small/weak surface defects).  Pure read-only diagnosis.
"""
from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/dynamic_fusion/overnight_20260908_real_resolution"
OUT = ROOT / "experiments/dynamic_fusion/validation_followup_20260908/v1_decomposition"

# Groups the D6 decision (doc37) identified as the real remaining A1 gap, with the
# observed A1-fused (seed0 k2/k4, stride-8) image-AUROC as context.
D6_GAP_GROUPS = {
    "bracket_brown/parts_mismatch": "D6: A1 fused imgAUROC ~0.55/0.54 (detection near-random)",
    "bracket_brown/bend_and_parts_mismatch": "D6: A1 fused imgAUROC ~0.61/0.63",
    "bracket_black/hole": "D6: A1 fused imgAUROC ~0.42/0.66 (small-area, detection-limited)",
    "bracket_white/defective_painting": "D6: A1 fused imgAUROC ~0.79/0.79",
}


def _fin(v):
    return None if v is None else float(v)


def main() -> int:
    res = json.loads((SRC / "RESULTS.json").read_text(encoding="utf-8"))
    groups = res["summary"]["defect_type_macro"]["groups"]
    cat_macro = res["summary"]["category_macro"]["arms"]
    type_macro = res["summary"]["defect_type_macro"]["arms"]

    # within-image (anomaly only) pixel-AP macro per category, from per-image rows
    per_cat = {c: {"full44832": [], "full89664": []} for c in
               ["bracket_black", "bracket_brown", "bracket_white", "connector",
                "metal_plate", "tubes"]}
    for line in open(SRC / "PER_IMAGE.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["anomaly_type"] == "good":
            continue
        v = r["pixel_ap_full448_per_image"]
        if v is not None:
            per_cat[r["category"]][r["arm"]].append(float(v))

    rows = []
    for key, g in groups.items():
        a, b = g["full44832"], g["full89664"]
        pa = _fin(a["full_pixel"]["pixel_ap"]); pb = _fin(b["full_pixel"]["pixel_ap"])
        ia = _fin(a["image"]["image_auroc"]); ib = _fin(b["image"]["image_auroc"])
        iap = _fin(a["image"]["image_ap"]); ibp = _fin(b["image"]["image_ap"])
        rows.append({"group": key, "n_anomaly": int(g["n_anomaly"]),
                     "pixel_ap_448": pa, "pixel_ap_896": pb,
                     "image_auroc_448": ia, "image_auroc_896": ib,
                     "image_ap_448": iap, "image_ap_896": ibp,
                     "d6_gap": D6_GAP_GROUPS.get(key)})

    def mean(xs):
        return float(np.mean(xs)) if xs else None

    within = {c: {"within_ap_448": mean(per_cat[c]["full44832"]),
                  "within_ap_896": mean(per_cat[c]["full89664"])} for c in per_cat}

    def arm_macro(arms, field, metric):
        return {k: _fin(v[field][metric]) for k, v in arms.items()}

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "outputs/dynamic_fusion/overnight_20260908_real_resolution (complete, 916 rows)",
        "macro": {
            "category_macro_image_auroc": arm_macro(cat_macro, "image", "image_auroc"),
            "category_macro_full_pixel_ap": arm_macro(cat_macro, "full_pixel", "pixel_ap"),
            "defect_type_macro_image_auroc": arm_macro(type_macro, "image", "image_auroc"),
        },
        "within_anomaly_pixel_ap_per_category": within,
        "per_group": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "V1_JSON.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "# V1 逐 type 分解：真实 448 vs 896（DINO-only, seed0 k2, full-pixel mask）", "",
        f"生成：{payload['generated_utc']}（UTC）。数据源：overnight real_resolution（complete，458 图 × 2 臂，916 行）。", "",
        "## 宏对账", "",
        "| 口径 | full44832 | full89664 | Δ |", "|---|---:|---:|---:|",
    ]
    m = payload["macro"]
    for name, key in (("image-AUROC（category 宏）", "category_macro_image_auroc"),
                      ("full-pixel Pixel-AP（category 宏）", "category_macro_full_pixel_ap"),
                      ("image-AUROC（defect-type 宏）", "defect_type_macro_image_auroc")):
        a, b = m[key]["full44832"], m[key]["full89664"]
        lines.append(f"| {name} | {a:.6f} | {b:.6f} | {b - a:+.6f} |")
    lines += ["", "## 逐 (类, defect type)（按 image-AUROC 增益降序）", "",
              "| group | n | PixAP448 | PixAP896 | dPix | imgAUC448 | imgAUC896 | dimgAUC | D6 关联 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda x: -(x["image_auroc_896"] - x["image_auroc_448"])):
        d = r["image_auroc_896"] - r["image_auroc_448"] if r["image_auroc_448"] is not None and r["image_auroc_896"] is not None else None
        lines.append("| {g} | {n} | {pa:.4f} | {pb:.4f} | {dp:+.4f} | {ia:.4f} | {ib:.4f} | {d:+.4f} | {d6} |".format(
            g=r["group"], n=r["n_anomaly"], pa=r["pixel_ap_448"], pb=r["pixel_ap_896"],
            dp=(r["pixel_ap_896"] - r["pixel_ap_448"]) if r["pixel_ap_448"] is not None and r["pixel_ap_896"] is not None else 0.0,
            ia=r["image_auroc_448"], ib=r["image_auroc_896"], d=d,
            d6="是" if r["d6_gap"] else ""))
    lines += ["", "## 异常图内定位（per-image Pixel-AP，仅 anomaly，剔除正常图影响）", "",
              "| category | within448 | within896 | Δ |", "|---|---:|---:|---:|"]
    for c, v in within.items():
        a, b = v["within_ap_448"], v["within_ap_896"]
        lines.append(f"| {c} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    lines += ["", "## 读数要点", "",
              "1. **image-AUROC 宏观增益不平均**：主要来自 bracket_black/hole（0.414→0.724，+0.31）、connector/parts_mismatch（0.752→0.829）、defective_painting（0.751→0.795）、tubes（0.957→0.978）。",
              "2. **D6 核心缺口（bracket_brown 上下文族）基本未被 896 改善**：parts_mismatch 0.432→0.448、bend 0.493→0.509，仍在近随机（0.45–0.51）。高分辨率不构成上下文装配缺陷的机制。",
              "3. **回归真实存在**：bracket_white/scratches image-AUROC −0.067、Pixel-AP −0.092；connector/parts_mismatch 像素 AP −0.039；bracket_brown 异常图内定位 −0.046、connector −0.038。",
              "4. **两类定位口径分离**：pooled(category 宏) Pixel-AP 448→896 略降（0.3176→0.3105），但异常图内定位（每类 anomaly 图 per-image AP 的类均值再宏）反而升（0.4466→0.4606，bracket_black +0.113、metal_plate +0.047）；pooled 退化集中在 bracket_brown/connector/bracket_white，异常图内退化同样集中在 bracket_brown（−0.046）与 connector（−0.038）。",
              "5. 结论：**896/细节分辨率只对稀疏小面积表面缺陷（hole、划痕类）有稳定收益，对 D6 定位的真缺口（bracket_brown 上下文族）无效，并对部分类造成回归** → 不作为新机制继续；不推翻 A1。",
    ]
    (OUT / "V1_REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote V1_JSON.json + V1_REPORT_CN.md -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
