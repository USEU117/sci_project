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
BENCHMARK_DIR = ROOT / "outputs" / "p1_c_benchmark"
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
            if ref.ndim != 4:
                raise ValueError(f"{npz}: expected ref features [N,H,W,D], got {ref.shape}")
            n_patch += int(np.prod(ref.shape[:-1]))
            # Historical feature caches do not store ref_ids. The leading axis
            # is the number of reference images and is independently fixed by shot.
            n_ref_img += int(ref.shape[0])
            dim = int(ref.shape[-1])
    return {"n_ref_patches": n_patch, "n_ref_images": n_ref_img, "dim": dim}


def main() -> int:
    smoke = json.loads(SMOKE_REPORT.read_text(encoding="utf-8"))
    pkg_bytes = sum(p.stat().st_size for p in PACKAGE_ROOT.rglob("*") if p.is_file())

    rows = []
    for dataset in DATASETS:
        for shot in SHOTS:
            # shot-wise: aggregate over 3 seeds
            dino_stats = [bank_stats(DINO_LAYOUT, dataset, s, shot) for s in SEEDS]
            clip_stats = [bank_stats(CLIP_LAYOUT, dataset, s, shot) for s in SEEDS]
            dino_patches = [r["n_ref_patches"] for r in dino_stats]
            clip_patches = [r["n_ref_patches"] for r in clip_stats]
            # A1 resizes CLIP's native 37x37 grid to DINO's 32x32 grid before
            # concatenation. It therefore has one 1536-d vector per DINO patch,
            # not the sum of the two branch patch counts.
            concat_patches = list(dino_patches)
            rows.append({
                "dataset": dataset, "shot": shot, "n_seeds": len(SEEDS),
                "reference_images": int(np.mean([r["n_ref_images"] for r in dino_stats])),
                "dino_bank_patches": int(np.mean(dino_patches)),
                "clip_bank_patches": int(np.mean(clip_patches)),
                "concat_bank_patches": int(np.mean(concat_patches)),
                "dino_bank_mb_f32": round(float(np.mean(dino_patches) * 768 * 4 / 1e6), 2),
                "clip_bank_mb_f32": round(float(np.mean(clip_patches) * 768 * 4 / 1e6), 2),
                "concat_bank_mb_f32": round(float(np.mean(concat_patches) * 1536 * 4 / 1e6), 2),
            })

    # Steady-state end-to-end benchmark (MVTec bottle s0/k1, warm model, 30 repeats).
    bench = {}
    for name, key in (("dino", "dino"), ("clip", "clip"), ("concat", "concat_knn")):
        path = BENCHMARK_DIR / f"{name}_benchmark.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            bench[key] = {
                "latency": payload["latency"],
                "peak_vram_mb": payload.get("peak_vram_mb"),
                "peak_ram_mb": payload.get("peak_ram_mb"),
                "category": payload.get("category"),
                "seed": payload.get("seed"),
                "shot": payload.get("shot"),
            }
    total_mean = sum(bench[k]["latency"]["mean_seconds"] for k in ("dino", "clip", "concat_knn") if k in bench)
    steady_state = None
    if len(bench) == 3:
        steady_state = {
            "category": "bottle", "seed": 0, "shot": 1,
            "warmup_passes": 3, "n_repeats": bench["dino"]["latency"]["n_repeats"],
            "dino_feature_extraction": bench["dino"]["latency"],
            "clip_feature_extraction": bench["clip"]["latency"],
            "align_concat_knn": bench["concat_knn"]["latency"],
            "end_to_end_mean_seconds": round(total_mean, 4),
            "end_to_end_throughput_images_per_second": round(1.0 / total_mean, 3),
            "note": "warm model, single image (bottle s0/k1), excludes one-time model load; "
                    "dino + clip feature extraction run on GPU (cuda:0), align+concat+KNN on CPU",
        }

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
        "steady_state_end_to_end_benchmark": steady_state,
        "steady_state_benchmark_note": "steady-state numbers from scripts/p1_c_benchmark.py (MVTec bottle s0/k1, warm model, 30 repeats); "
                                       "single-image smoke times include one-time model load and are not throughput",
        "peak_vram_mb": smoke["peak_vram_mb"],
        "peak_ram_mb": round(max((bench[k]["peak_ram_mb"] for k in bench if bench[k].get("peak_ram_mb")), default=0.0), 1),
        "peak_ram_note": "max single-process working set across dino/clip/concat benchmark runs (MVTec bottle s0/k1); "
                         "A1 runs the two feature-extraction branches sequentially so end-to-end peak ≈ max of the three",
        "compact_package_size_bytes": pkg_bytes,
        "compact_package_size_mb": round(pkg_bytes / 1e6, 1),
        "memory_bank_float32": rows,
        "note": "bank = reference patch features only (no test features stored in the bank). "
                "Branch grids are read per category from the caches (BTAD/VisA include non-square grids). A1 resizes CLIP to each category's DINO grid before concatenation. "
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
        f"- 峰值 RAM（bottle s0/k1 各阶段单进程峰值，max）：{summary['peak_ram_mb']} MB",
        f"- compact 复现包大小：**{round(pkg_bytes/1e6,1)} MB**（含 324 个逐图 float16 patch maps 与全部证据）",
        "",
        "稳态端到端 benchmark（预热 3 次后重复 30 次，bottle s0/k1，排除一次性模型加载）：",
        "",
    ]
    if steady_state is not None:
        ss = steady_state
        d, c, k = ss["dino_feature_extraction"], ss["clip_feature_extraction"], ss["align_concat_knn"]
        md += [
            "| 阶段 | mean s/img | std | p50 | p95 | throughput img/s |",
            "|---|---|---:|---:|---:|---:|",
            f"| DINO 特征提取 (GPU) | {d['mean_seconds']} | {d['std_seconds']} | {d['p50_seconds']} | {d['p95_seconds']} | {d['throughput_images_per_second']} |",
            f"| CLIP 特征提取 (GPU) | {c['mean_seconds']} | {c['std_seconds']} | {c['p50_seconds']} | {c['p95_seconds']} | {c['throughput_images_per_second']} |",
            f"| 对齐+concat+KNN (CPU) | {k['mean_seconds']} | {k['std_seconds']} | {k['p50_seconds']} | {k['p95_seconds']} | {k['throughput_images_per_second']} |",
            f"| **端到端（合计）** | **{ss['end_to_end_mean_seconds']}** | — | — | — | **{ss['end_to_end_throughput_images_per_second']}** |",
            "",
            "记忆库（normal memory bank）float32 规模（ref patch 特征，按 dataset×shot 对 3 seeds 取均值）：",
            "",
            "| dataset | shot | ref images | dino bank patches | clip-only native patches | A1 concat patches (DINO grid) | dino MB | clip MB | concat MB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        md += [
            "（benchmark 文件缺失：`outputs/p1_c_benchmark/{dino,clip,concat}_benchmark.json`）",
            "",
            "记忆库（normal memory bank）float32 规模（ref patch 特征，按 dataset×shot 对 3 seeds 取均值）：",
            "",
            "| dataset | shot | ref images | dino bank patches | clip-only native patches | A1 concat patches (DINO grid) | dino MB | clip MB | concat MB |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['shot']} | {r['reference_images']} | {r['dino_bank_patches']} | {r['clip_bank_patches']} | "
                  f"{r['concat_bank_patches']} | {r['dino_bank_mb_f32']} | {r['clip_bank_mb_f32']} | {r['concat_bank_mb_f32']} |")
    md.append("")
    md.append("注：bank = 正常参考 patch 特征（不含测试特征）；网格按类别从缓存读取，BTAD/VisA 含非方形网格。A1 将 CLIP 对齐到对应 DINO 网格后，形成每 patch 1536 维 concat bank。")
    (out / "p1_c_efficiency.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": "passed", "package_mb": round(pkg_bytes / 1e6, 1), "rows": len(rows),
                      "output": str(out / "p1_c_efficiency.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
