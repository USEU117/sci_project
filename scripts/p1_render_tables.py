"""Render P1-A/P1-B JSON evidence into paper-oriented Markdown and CSV tables.

Reads submission_repro_20260827/evidence/p1/p1_a_bootstrap_ci.json (produced by
scripts/p1_stats_bootstrap.py) and writes:
  - p1_a_bootstrap_ci.md      (per-config CI table + dataset/shot summary)
  - p1_a_shot_wise.csv        (dataset x shot mean +/- std)
  - p1_b_failure_boundaries.md
  - p1_b_failure_samples.csv  (per-image failure sample IDs)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "submission_repro_20260827" / "evidence" / "p1" / "p1_a_bootstrap_ci.json"
OUT = ROOT / "submission_repro_20260827" / "evidence" / "p1"


def fmt_ci(c) -> str:
    return f"{c['lo']:.4f}..{c['hi']:.4f}"


def main() -> int:
    payload = json.loads(SRC.read_text(encoding="utf-8"))
    configs = payload["configs"]

    md = [
        "# P1-A 统计：ΔPixel-AP 置信区间（A1 concat vs matched feature-DINO-only）",
        "",
        f"- 指标口径：每类别将测试集全部图像像素（stride=8）聚合后计算 Pixel-AP；ΔAP = concat − dino，"
        f"按类别平均（与论文主表一致）。",
        f"- Bootstrap：配对图像重采样（类别内分层、每类≥1 图）与类别重采样各 B={payload['bootstrap']['B']} 次，"
        f"{payload['bootstrap']['level']:.0%} 百分位 CI，RNG seed={payload['bootstrap']['rng_seed']}。",
        f"- 全样本 sanity：{len(configs)} 个配置相对 p0_3 参考表均在 {payload['bootstrap']['sanity_tolerance']} 容差内："
        f"`{payload['sanity_all_within_tolerance']}`。",
        "- 统计单位：图像（image bootstrap）或类别（category bootstrap）；3 seed × 3 shot 是同一测试集上的参考采样配置，不是独立数据集。",
        "",
        "## 逐配置",
        "",
        "| dataset | seed | shot | n_img | n_anom | full ΔAP | category CI | image CI (anom img) | sanity |",
        "|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for r in configs:
        cb = r["category_bootstrap"]
        ib = r["image_bootstrap"]
        md.append(f"| {r['dataset']} | {r['seed']} | {r['shot']} | {r['n_test_images']} | "
                  f"{r['n_anomalous_images']} | {r['full_sample_delta_ap']:+.4f} | {fmt_ci(cb)} | "
                  f"{fmt_ci(ib) if ib.get('lo') is not None else 'n/a'} | "
                  f"{'OK' if r['sanity_within_tolerance'] else 'FAIL'} |")

    md += ["", "## 按 dataset×shot（3 seed mean ± std）", "", "| dataset | shot | mean ΔAP | std (seeds) | 3 seeds |", "|---|---|---:|---:|---|"]
    for r in payload["shot_wise"]:
        md.append(f"| {r['dataset']} | {r['shot']} | {r['mean_delta_ap']:+.4f} | {r['std_delta_ap']:.4f} | "
                  f"{' / '.join(f'{d:+.4f}' for d in r['seeds'])} |")
    md += ["", "## 按 dataset（9 配置）", "", "| dataset | n_configs | mean ΔAP | std |", "|---|---|---:|---:|"]
    for r in payload["dataset_wise"]:
        md.append(f"| {r['dataset']} | {r['n_configs']} | {r['mean_delta_ap']:+.4f} | {r['std_delta_ap']:.4f} |")
    (OUT / "p1_a_bootstrap_ci.md").write_text("\n".join(md), encoding="utf-8")

    with (OUT / "p1_a_shot_wise.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "shot", "mean_delta_ap", "std_delta_ap", "seeds"])
        w.writeheader()
        for r in payload["shot_wise"]:
            w.writerow({**{k: r[k] for k in ("dataset", "shot", "mean_delta_ap", "std_delta_ap")},
                        "seeds": ";".join(str(d) for d in r["seeds"])})

    # ---- P1-B ----
    worst = payload["worst_categories_by_dataset"]
    neg = payload["negative_categories"]
    fails = payload["failure_samples_top5_per_config"]
    md2 = [
        "# P1-B 失败边界：worst/negative categories 与逐图失败样例",
        "",
        "## Negative-gain 类别",
        "",
        f"- 负增益类别在 36 个配置中出现 **{neg['count_across_configs']}** 次，去重后 "
        f"**{len(neg['unique_categories'])}** 个 dataset@category：`{'、'.join(neg['unique_categories']) or '无'}`。",
        "",
        "## 每 dataset 的 worst category（跨 9 配置 mean ΔAP 最低）",
        "",
        "| dataset | worst category | mean ΔAP | negative configs |",
        "|---|---|---:|---:|",
    ]
    for ds in ("mpdd", "btad", "visa", "mvtec"):
        row = next((w for w in worst if w["dataset"] == ds), None)
        if row:
            md2.append(f"| {ds} | {row['category']} | {row['mean_delta_ap']:+.4f} | {row['negative_configs']} |")
    md2 += ["", "全部类别（按 mean ΔAP 升序）：", "", "| dataset | category | n_configs | mean ΔAP | negative configs |", "|---|---|---:|---:|---:|"]
    for w in worst:
        md2.append(f"| {w['dataset']} | {w['category']} | {w['n_configs']} | {w['mean_delta_ap']:+.4f} | {w['negative_configs']} |")
    md2 += [
        "",
        "## 逐图失败样例（每配置 top-5，concat per-image AP 相对 dino 最差）",
        "",
        "| dataset | seed | shot | category | sample_id | concat AP | dino AP | ΔAP |",
        "|---|---|---|---|---|---|---:|---:|",
    ]
    for f in fails:
        md2.append(f"| {f['dataset']} | {f['seed']} | {f['shot']} | {f['category']} | `{f['sample_id']}` | "
                   f"{f['concat_pixel_ap']:.4f} | {f['dino_pixel_ap']:.4f} | {f['delta_ap']:+.4f} |")
    md2 += [
        "",
        "注：失败样例仅从本地合法数据根按 sample_id 追溯原图；包内不复制不可再分发原图。"
        "per-image Pixel-AP 只在 mask 含异常像素的测试图上定义（正常图无正像素，不参与）。",
    ]
    (OUT / "p1_b_failure_boundaries.md").write_text("\n".join(md2), encoding="utf-8")

    with (OUT / "p1_b_failure_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "seed", "shot", "category", "sample_id",
                                           "concat_pixel_ap", "dino_pixel_ap", "delta_ap"])
        w.writeheader()
        w.writerows(fails)

    print(json.dumps({"status": "passed", "outputs": [p.name for p in OUT.glob("p1_*.md")]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
