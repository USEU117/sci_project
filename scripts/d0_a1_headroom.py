"""Route-D decision gate D0: is there meaningful dynamic headroom above frozen A1?

Computes a label-informed per-pixel best-of-branch Oracle (concat / feature-DINO-only /
CLIP-only) on the MPDD development matrix (3 seeds x 1/2/4-shot). The Oracle is
development-only diagnostics: it is NOT a deployable router and NEVER feeds fusion.

Gate D0 (design review section 8):
  - mean Pixel AP Oracle headroom vs A1 >= +0.015
  - at least 4/6 MPDD categories have headroom >= +0.005
  - headroom must not come from a single category
  - Oracle code is fully isolated from fusion code (it only consumes maps post-hoc)

Outputs:
  experiments/dynamic_fusion/route_d_d0_20260818/{d0_headroom_report.json,d0_headroom_report.md}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_a1_feature_fusion import compute_metrics, fuse_category, load_features  # noqa: E402

MODE_KEY = {"concat": "concat", "dino": "dino", "clip": "clip"}
MAP_SIZE = (448, 448)


def run() -> None:
    manifest = json.loads((ROOT / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8"))
    categories = sorted(manifest["categories"])

    per_cat_headroom = {cat: [] for cat in categories}
    per_cat_a1_ap = {cat: [] for cat in categories}
    per_cat_oracle_ap = {cat: [] for cat in categories}
    config_rows = []

    for seed in (0, 1, 2):
        for shot in (1, 2, 4):
            dino_dir = ROOT / "outputs/dynamic_fusion/v3_direction_a" / f"features_vitb14_s{seed}_k{shot}" / "anomalydino_visual"
            clip_dir = ROOT / "outputs/dynamic_fusion/v3_direction_a" / f"features_s{seed}_k{shot}" / "anomalyclip_text"
            for cat in categories:
                dino = load_features(dino_dir / f"{cat}.npz")
                clip = load_features(clip_dir / f"{cat}.npz")
                gt = dino["imgs_masks"]
                maps = {
                    mode: fuse_category(dino, clip, mode, 0, False, MAP_SIZE, 0.5)
                    for mode in ("concat", "dino", "clip")
                }
                stack = np.stack([maps["concat"], maps["dino"], maps["clip"]])  # (3, N, H, W)
                oracle_map = np.where((gt > 0.5), stack.max(axis=0), stack.min(axis=0)).astype(np.float32)

                a1_ap = compute_metrics(maps["concat"], gt)["pixel_ap"]
                ora_ap = compute_metrics(oracle_map, gt)["pixel_ap"]
                headroom = ora_ap - a1_ap

                per_cat_headroom[cat].append(headroom)
                per_cat_a1_ap[cat].append(a1_ap)
                per_cat_oracle_ap[cat].append(ora_ap)
                config_rows.append({"seed": seed, "shot": shot, "category": cat, "a1_pixel_ap": round(a1_ap, 6), "oracle_pixel_ap": round(ora_ap, 6), "headroom": round(headroom, 6)})

    cat_means = {cat: float(np.mean(v)) for cat, v in per_cat_headroom.items()}
    mean_headroom = float(np.mean(list(cat_means.values())))
    cats_over_005 = [cat for cat, h in cat_means.items() if h >= 0.005]
    top_share = max(cat_means.values()) / (sum(cat_means.values()) or 1.0)

    passed = (
        mean_headroom >= 0.015
        and len(cats_over_005) >= 4
        and top_share < 0.5
    )

    report = {
        "schema_version": 1,
        "run_id": "route_d_d0_20260818",
        "created_at_utc": "2026-08-18T20:20:00+00:00",
        "stage": "route_D_gate_D0_development_diagnostics",
        "dataset": "mpdd",
        "dataset_role": "development",
        "oracle_definition": "per-pixel best-of-branch (concat / feature-DINO-only / CLIP-only) selected with GT; development-only, never feeds fusion",
        "n_configs": len(config_rows),
        "per_category_mean_headroom": {cat: round(v, 6) for cat, v in cat_means.items()},
        "categories_over_0_005": cats_over_005,
        "mean_headroom_vs_a1": round(mean_headroom, 6),
        "top_category_share_of_headroom": round(top_share, 6),
        "gate_conditions": {
            "mean_headroom_ge_0_015": mean_headroom >= 0.015,
            "at_least_4_of_6_cats_ge_0_005": len(cats_over_005) >= 4,
            "headroom_not_single_category": top_share < 0.5,
        },
        "gate_d0_passed": passed,
        "decision": (
            "STOP route D: headroom below gate -> archive dynamic-routing line permanently; continue route S"
            if not passed
            else "headroom exists -> proceed to D1 predictability gate (requires feature-prediction evidence)"
        ),
        "not_computed": ["per-region Oracle vs V3.3-clean (per-pixel best-of-3 subsumes it for D0)"],
        "config_rows": config_rows,
    }
    out_dir = ROOT / "experiments" / "dynamic_fusion" / "route_d_d0_20260818"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "d0_headroom_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# 路线 D 决策门 D0（A1 之上的动态 headroom）",
        "",
        f"RunId: `route_d_d0_20260818` · MPDD development · 9 配置（3 seeds × 1/2/4-shot）",
        "",
        f"- Oracle 定义：逐像素 best-of-branch（concat / feature-DINO-only / CLIP-only），用 GT 在每像素选更优分支。**仅开发诊断，不进入任何融合/路由输入。**",
        f"- mean Pixel AP headroom vs A1 = **{mean_headroom:+.4f}**（要求 ≥ +0.015）",
        f"- 逐类 headroom：{ {k: round(v,4) for k,v in cat_means.items()} }",
        f"- ≥ +0.005 的类数：**{len(cats_over_005)}/6**（要求 ≥ 4）",
        f"- headroom 最大类占比：{top_share:.2f}（要求 < 0.5）",
        f"- **Gate D0：{'PASS → 进入 D1' if passed else 'FAIL → 停止路线 D，永久归档动态路线，回路线 S'}**",
        "",
        "## 说明",
        "",
        "- Oracle 上限高并不代表无标签可靠性特征能预测『何时修正 A1』；D0 只是第一道检查。",
        "- 未计算『V3.3-clean 逐区域 Oracle』（per-pixel best-of-3 已覆盖 D0 判定所需的主要 headroom）。",
        "",
    ]
    (out_dir / "d0_headroom_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report["gate_conditions"], ensure_ascii=False, indent=2))
    print(f"mean_headroom={mean_headroom:.4f} gate_d0_passed={passed}")


if __name__ == "__main__":
    run()
