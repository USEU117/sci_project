"""R4: A1 qualitative success/failure figures from compact maps + local data.

Gate R4 (docs/PRE_MANUSCRIPT_READINESS_AUDIT_20260827.md): fixed case selection
from P1-B failure samples + positive-gain categories; NO parameter selection.

Each case renders one panel figure: original image | GT mask | DINO-only map |
A1 concat map, with the per-image Pixel-AP of both branches.

Selection rule (fixed, recorded in the manifest):
  - failure case: P1-B top-1 failure sample for that dataset@category (from
    submission_repro_20260827/evidence/p1/p1_b_failure_samples.csv), falling
    back to the minimum per-image ΔAP computed from the compact maps.
  - success case: the sample with the maximum per-image ΔAP among anomalous
    test images of that dataset@category (computed here from compact maps;
    identical stride=8 Pixel-AP pipeline as the frozen evaluator).

Original images and GT masks are read from user data roots and are NEVER
packaged (not redistributable). The manifest (JSON + MD) is packaged and records
source IDs, selection rule, per-image APs, and the figure SHA256.

Usage (one pass, CPU):
  .venv-patchcore\\Scripts\\python.exe scripts/build_a1_qualitative_figures.py
      --data-root mpdd=data/mpdd_raw/MPDD --data-root btad=data/btad_raw
      --data-root visa=data/visa_raw --data-root mvtec=data/mvtec
      --out-dir outputs/p1_b_figures

Outputs:
  outputs/p1_b_figures/*.png   (not committed; contain original images)
  submission_repro_20260827/evidence/p1/p1_b_figures_manifest.json
  submission_repro_20260827/evidence/p1/p1_b_figures_manifest.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "submission_repro_20260827"
MAPS_ROOT = PACKAGE_ROOT / "predictions_compact" / "maps"
P1B_CSV = PACKAGE_ROOT / "evidence" / "p1" / "p1_b_failure_samples.csv"
MANIFEST_JSON = PACKAGE_ROOT / "evidence" / "p1" / "p1_b_figures_manifest.json"
MANIFEST_MD = PACKAGE_ROOT / "evidence" / "p1" / "p1_b_figures_manifest.md"

sys.path.insert(0, str(PACKAGE_ROOT))
from recompute_tables import (  # noqa: E402
    MAP_SIZE,
    STRIDE,
    build_visa_mask_map,
    compute_metrics,
    dists2map,
    load_mask_for_sample,
)

# Fixed case set: dataset, seed, shot, category, role.
# Covers stable positive-gain categories, persistent negative categories
# (mvtec leather, visa chewinggum, mpdd bracket), and an external-frozen
# validation success (btad 01).
CASES = [
    ("mpdd", 0, 1, "metal_plate", "success"),
    ("mpdd", 0, 1, "bracket_brown", "failure"),
    ("btad", 0, 1, "01", "success"),
    ("visa", 0, 1, "cashew", "success"),
    ("visa", 0, 1, "chewinggum", "failure"),
    ("mvtec", 0, 1, "toothbrush", "success"),
    ("mvtec", 0, 1, "leather", "failure"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_p1b_failure_samples() -> dict[tuple[str, str], str]:
    """(dataset, category) -> first (top-1) failure sample_id for seed 0 shot 1."""
    out: dict[tuple[str, str], str] = {}
    with P1B_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["seed"]) == 0 and int(row["shot"]) == 1:
                key = (row["dataset"], row["category"])
                out.setdefault(key, row["sample_id"])
    return out


def per_image_ap(maps: np.ndarray, masks: np.ndarray) -> float | None:
    """[H,W] maps + masks -> stride-8 pixel AP (frozen evaluator pipeline).

    Returns None when the stride-8 mask contains a single class (per-image
    pixel-AP undefined), matching the frozen evaluator's applicability rule.
    """
    m = maps[None, :, :].astype(np.float64)
    mk = masks[None, :, :].astype(np.float32)
    if int((mk[:, ::STRIDE, ::STRIDE] > 0.5).sum()) == 0:
        return None
    return float(compute_metrics(m, mk)["pixel_ap"])


def build_case_images(npz: dict, data_root: Path, visa_mask_map: dict | None,
                      dataset: str, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (concat_448, dino_448) for one sample."""
    idx = list(npz["sample_ids"]).index(sample_id)
    concat = dists2map(npz["concat_patch_map"][idx].astype(np.float32), MAP_SIZE)
    dino = dists2map(npz["dino_patch_map"][idx].astype(np.float32), MAP_SIZE)
    return concat, dino


def load_image(data_root: Path, dataset: str, sample_id: str) -> np.ndarray:
    path = data_root / sample_id
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"image missing: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", action="append", required=True,
                        help="dataset=path (repeatable; mpdd/btad/visa/mvtec)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "p1_b_figures")
    args = parser.parse_args()

    data_roots: dict[str, Path] = {}
    for item in args.data_root:
        ds, _, path = item.partition("=")
        data_roots[ds] = ROOT / path
    missing = [ds for ds in data_roots if not data_roots[ds].is_dir()]
    if missing:
        raise SystemExit(f"data root missing for: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p1b = load_p1b_failure_samples()
    visa_mask_map = build_visa_mask_map(data_roots["visa"]) if "visa" in data_roots else None

    rows: list[dict] = []
    for dataset, seed, shot, category, role in CASES:
        npz_path = MAPS_ROOT / dataset / f"s{seed}_k{shot}" / f"{category}.npz"
        if not npz_path.is_file():
            print(f"SKIP missing compact map: {npz_path}")
            continue
        data_root = data_roots.get(dataset)
        if data_root is None:
            print(f"SKIP no data root for {dataset}")
            continue
        with np.load(npz_path, allow_pickle=False) as data:
            npz = {k: data[k] for k in data.files}

        # select sample
        if role == "failure" and (dataset, category) in p1b:
            sample_id = p1b[(dataset, category)]
            selection_rule = "P1-B top-1 failure sample"
        else:
            # fallback: pick by role direction among anomalous images.
            # failure -> min per-image ΔAP; success -> max per-image ΔAP.
            best = None
            for sid in npz["sample_ids"]:
                concat, dino = build_case_images(npz, data_root, visa_mask_map, dataset, str(sid))
                mask = load_mask_for_sample(dataset, str(sid), data_root, visa_mask_map, MAP_SIZE)
                if mask.sum() == 0:
                    continue  # normal image: per-image pixel-AP undefined
                cap = per_image_ap(concat, mask)
                dap = per_image_ap(dino, mask)
                if cap is None or dap is None:
                    continue  # stride-8 mask collapses to one class
                delta = cap - dap
                if best is None or (delta > best[0] if role == "success" else delta < best[0]):
                    best = (delta, str(sid), cap, dap)
            if best is None:
                print(f"SKIP no anomalous sample with mask: {npz_path}")
                continue
            _, sample_id, _cap, _dap = best
            selection_rule = ("max per-image ΔAP among anomalous test images" if role == "success"
                              else "min per-image ΔAP among anomalous test images")

        concat, dino = build_case_images(npz, data_root, visa_mask_map, dataset, sample_id)
        mask = load_mask_for_sample(dataset, sample_id, data_root, visa_mask_map, MAP_SIZE)
        img = load_image(data_root, dataset, sample_id)
        cap = per_image_ap(concat, mask)
        dap = per_image_ap(dino, mask)
        if cap is None or dap is None:
            print(f"SKIP selected sample has single-class stride-8 mask: {sample_id}")
            continue

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, title, arr, cmap in (
            (axes[0], "original", img, None),
            (axes[1], "GT mask", mask, "gray"),
            (axes[2], f"DINO-only  P-AP {dap:.3f}", dino, "magma"),
            (axes[3], f"A1 concat  P-AP {cap:.3f}", concat, "magma"),
        ):
            ax.imshow(arr, cmap=cmap)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
        fig.suptitle(f"{dataset} · s{seed} k{shot} · {category} · {role}\n{sample_id}",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        png = args.out_dir / f"{dataset}_s{seed}_k{shot}_{category}_{role}.png"
        fig.savefig(png, dpi=130, bbox_inches="tight")
        plt.close(fig)

        rows.append({
            "dataset": dataset, "seed": seed, "shot": shot, "category": category,
            "role": role, "sample_id": sample_id,
            "concat_pixel_ap": round(cap, 4), "dino_pixel_ap": round(dap, 4),
            "delta_ap": round(cap - dap, 4),
            "figure": str(png.name), "figure_sha256": sha256(png),
            "selection_rule": selection_rule,
        })
        print(json.dumps(rows[-1], ensure_ascii=False))

    manifest = {
        "schema_version": 1,
        "kind": "p1_b_qualitative_figures",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "R4 A1 qualitative success/failure figures",
        "selection_rule": (
            "fixed cases only; failure = P1-B top-1 failure sample (seed0/shot1); "
            "success = max per-image ΔAP among anomalous test images of that category; "
            "stride-8 pixel-AP, frozen evaluator; no parameter selection"
        ),
        "original_images_and_masks_not_packaged": True,
        "figures": rows,
    }
    MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# P1-B/R4 定性图 manifest（A1 成功/失败案例）",
        "",
        "- Gate R4：从 compact concat/DINO maps + 合法本地原图/GT mask 生成固定成功与失败案例图。",
        "- 选择规则：失败 = P1-B top-1 失败样例（seed0/shot1）；成功 = 该类别异常测试图中 per-image ΔAP 最大者。",
        "- 指标口径：stride=8 Pixel-AP，冻结 evaluator；**不进行参数选择**。",
        "- 原图与 GT mask 不打包（不可再分发）；图中含原图，仅本地 `outputs/p1_b_figures/` 保留。",
        "",
        "| dataset | s/k | category | role | sample_id | concat P-AP | dino P-AP | ΔAP | figure SHA256 |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for r in rows:
        md.append(f"| {r['dataset']} | s{r['seed']}/k{r['shot']} | {r['category']} | {r['role']} | "
                  f"`{r['sample_id']}` | {r['concat_pixel_ap']} | {r['dino_pixel_ap']} | {r['delta_ap']} | "
                  f"`{r['figure_sha256'][:16]}…` |")
    MANIFEST_MD.write_text("\n".join(md), encoding="utf-8")

    print(f"\nWrote {len(rows)} figures -> {args.out_dir}")
    print(f"Wrote {MANIFEST_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
