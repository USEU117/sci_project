"""P1-C: efficiency table for A1 / feature-DINO-only / CLIP-only.

Reports (all numbers traceable to project evidence):
  - trainable parameters: 0 (all pretrained backbones frozen; no training)
  - per-image feature-extraction wall time and peak VRAM: from outputs/p0_2_smoke/smoke_report.json
  - memory-bank size per (dataset, shot): counted from the rebuilt feature caches
    (ref patch count and float32 MB for DINO 768-d, CLIP 768-d, concat 1536-d)
  - compact package size: submission_repro_20260827 (bytes on disk)

Outputs: submission_repro_20260827/evidence/p1/p1_c_efficiency.json/.csv/.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "outputs" / "dynamic_fusion" / "v3_direction_a"
SMOKE_REPORT = ROOT / "outputs" / "p0_2_smoke" / "smoke_report.json"
PACKAGE_ROOT = ROOT / "submission_repro_20260827"
OUT_ROOT = PACKAGE_ROOT / "evidence" / "p1"

DATASETS = ("mpdd", "btad", "visa", "mvtec")
SEEDS = (0, 1, 2)
SHOTS = (1, 2, 4)

DINO_LAYOUT = {
    "mpdd": "features_vitb14_s{seed}_k{shot}/anomalydino_visual",
    "btad": "features_vitb14_btad_s{seed}_k{shot}/anomalydino_visual",
    "visa": "visa_features_vitb14/s{seed}_k{shot}/anomalydino_visual",
    "mvtec": "mvtec_features_vitb14/s{seed}_k{shot}/anomalydino_visual",
}
CLIP_LAYOUT = {
    "mpdd": "features_s{seed}_k{shot}/anomalyclip_text",
    "btad": "features_btad_s{seed}_k{shot}/anomalyclip_text",
    "visa": "visa_features/s{seed}_k{shot}/anomalyclip_text",
    "mvtec": "mvtec_features/s{seed}_k{shot}/anomalyclip_text",
}


def bank_stats(layout: dict[str, str], dataset: str, seed: int, shot: int) -> dict:
    rel = layout[dataset].format(seed=seed, shot=shot)
    d = CACHE_ROOT / rel
    n_patch = 0
    n_ref_img = 0
    dim = None
    for npz in sorted(d.glob("*.npz")):
        if npz.stem == "export_report":
            continue
        with np.load(npz, allow_pickle=False) as data:
            ref = np.asarray(data["ref_patch_features"])
            n_patch += int(ref.shape[0])
            n_ref_img += int(data.get("ref_ids").size) if "ref_ids" in data else 0
            dim = int(ref.shape[1])
    return {"n_ref_patches": n_patch, "n_ref_images": n_ref_img, "dim": dim}


def main() -> int:
    smoke = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    pkg_bytes = sum(p.stat().st_size for p in PACKAGE_ROOT.rglob("*") if p.is_file())

    rows = []
    for dataset in DATASETS:
        for shot in SHOTS:
            # shot-wise: aggregate over 3 seeds
            dino_patches = [bank_stats(DINO_LAYOUT, dataset, s, shot)["n_ref_patches"]
                            for s in SEEDS]
            clip_patches = [bank_stats(CLIP_LAYOUT, dataset, s, shot)["n_ref_patches"]
                            for s in SEEDS]
            concat_patches = [a + b for a, b in zip(dino_patches, clip_patches)]
            rows.append({
                "dataset": dataset, "shot": shot, "n_seeds": len(SEEDS),
                "dino_bank_patches": int(np.mean(dino_patches)),
                "clip_bank_patches": int(np.mean(clip_patches)),
                "concat_bank_patches": int(np.mean(concat_patches)),
                "dino_bank_mb_f32": round(float(np.mean(dino_patches) * 768 * 4 / 1e6), 2),
                "clip_bank_mb_f32": round(float(np.mean(clip_patches) * 768 * 4 / 1e6), 2),
                "concat_bank_mb_f32": round(float(np.mean(concat_patches) * 1536 * 4 / 1e6), 2),
            })

    summary = {
        "schema_version": 1,
        "kind": "p1_c_efficiency",
        "created_at_utc": json.loads(open(SMOKE_REPORT, encoding="utf-8").read()).get("created_at_utc", ""),
        "method": "A1 dual-encoder visual patch fusion (frozen w=0.5, KNN k=1)",
        "trainable_parameters": 0,
        "training": "none (pretrained backbones frozen; normal-reference memory bank built at inference)",
        "per_image_feature_extraction_seconds": {
            "dino": smoke["wall_seconds"]["dino"],
            "clip": smoke["wall_seconds"]["clip"],
            "note": "single-image wall clock from P0-2 smoke (includes one-time model load); not a throughput measurement",
        },
        "peak_vram_mb": smoke["peak_vram_mb"],
        "peak_ram_mb": None,
        "peak_ram_note": "not separately measured in P0-2 smoke; CPU peak RAM only from full recompute runs",
        "compact_package_size_bytes": pkg_bytes,
        "compact_package_size_mb": round(pkg_bytes / 1e6, 1),
        "memory_bank_float32": rows,
        "note": "bank = reference patch features only (no test features stored in the bank). "
                "DINO 768-d, CLIP image-tower 768-d, concat 1536-d.",
    }

    out = OUT_ROOT
    out.mkdir(parents=True, exist_ok=True)
    (out / "p1_c_efficiency.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out / "p1_c_efficiency.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [
        "# P1-C 效率表（A1 / feature-DINO-only / CLIP-only）",
        "",
        f"- 训练参数：**0**（全部预训练 backbone 冻结；推理时仅构建正常参考记忆库，无训练）",
        f"- 单图特征提取墙钟时间（P0-2 smoke 实测，含一次性模型加载，非吞吐）：DINO {smoke['wall_seconds']['dino']} s、CLIP {smoke['wall_seconds']['clip']} s",
        f"- 峰值显存（P0-2 smoke 实测）：DINO {smoke['peak_vram_mb']['dino']} MB、CLIP {smoke['peak_vram_mb']['clip']} MB；A1 全流程按两个分支顺序执行，峰值取 max ≈ {max(smoke['peak_vram_mb'].values()):.0f} MB",
        f"- 峰值 RAM：未在 smoke 单独测量",
        f"- compact 复现包大小：**{round(pkg_bytes/1e6,1)} MB**（含 324 个逐图 float16 patch maps 与全部证据）",
        "",
        "记忆库（normal memory bank）float32 规模（ref patch 特征，按 dataset×shot 对 3 seeds 取均值）：",
        "",
        "| dataset | shot | dino bank (patches) | clip bank (patches) | concat bank (patches) | dino MB | clip MB | concat MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['shot']} | {r['dino_bank_patches']} | {r['clip_bank_patches']} | "
                  f"{r['concat_bank_patches']} | {r['dino_bank_mb_f32']} | {r['clip_bank_mb_f32']} | {r['concat_bank_mb_f32']} |")
    md.append("")
    md.append("注：bank = 正常参考 patch 特征（不含测试特征）；维度 DINO 768、CLIP image-tower 768、concat 1536。")
    (out / "p1_c_efficiency.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "passed", "package_mb": round(pkg_bytes / 1e6, 1), "rows": len(rows),
                      "output": str(out / "p1_c_efficiency.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
