"""Route B — MESP dino-only PROMISE probe (explicitly NOT the pre-registered gate).

Runs the MESP mechanism on the DINO-only distance map (seed0 k1, all 6 classes)
to decide whether to invest in the full fused (dino+clip) R0. Views: original
(from cache), hflip, bright0.9, bright1.1, rot-5, rot+5. References are
transformed by the same view. Candidates:
  B1 median across views (valid views per pixel)
  B2 base * (0.75 + 0.25 * stability), stability = 1 - normalized MAD
Misalignment control: B1 with the rot+5 map displaced by 14 px (should hurt).
A promise probe can FAIL the route cheaply but can NOT pass it. Memory bounded:
metrics computed at STRIDE=8 on a per-image basis.

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v10_portfolio\\run_mesp_probe_dino.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

from industrial_ad.innovation_v10_portfolio import common
from src.utils import dists2map

VIEWS = ["hflip", "bright0.9", "bright1.1", "rot-5", "rot+5"]
ROT_DEG = {"rot-5": -5.0, "rot+5": 5.0}
SIZE = 448
STRIDE = common.STRIDE
SM = SIZE // STRIDE            # 56
MISALIGN_CELLS = 2             # 14 px at STRIDE 8
MARGIN = 24                    # rotation border margin (px)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_view(img: np.ndarray, view: str) -> np.ndarray:
    if view == "hflip":
        return img[:, ::-1].copy()
    if view == "bright0.9":
        return np.clip(img.astype(np.float32) * 0.9, 0, 255).astype(np.uint8)
    if view == "bright1.1":
        return np.clip(img.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)
    if view.startswith("rot"):
        deg = ROT_DEG[view]
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), deg, 1.0)
        return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    raise ValueError(view)


def inverse_warp_map(mapp: np.ndarray, view: str) -> np.ndarray:
    """Map in view frame -> original frame (rot borders become NaN)."""
    if view == "hflip":
        return mapp[:, ::-1].copy()
    if view in ("bright0.9", "bright1.1"):
        return mapp
    if view.startswith("rot"):
        deg = ROT_DEG[view]
        m = cv2.getRotationMatrix2D((mapp.shape[1] / 2, mapp.shape[0] / 2), -deg, 1.0)
        return cv2.warpAffine(mapp, m, (mapp.shape[1], mapp.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=float("nan"))
    raise ValueError(view)


def valid_strided(view: str) -> np.ndarray:
    m = np.ones((SM, SM), dtype=bool)
    if view.startswith("rot"):
        c = int(np.ceil(MARGIN / STRIDE))
        m[:c, :] = False
        m[-c:, :] = False
        m[:, :c] = False
        m[:, -c:] = False
    return m


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mpdd_raw/MPDD")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/mpdd/manifest.json")
    parser.add_argument("--dino-cache", type=Path,
                        default=ROOT / "outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k1/anomalydino_visual")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v10_portfolio/mesp")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from src.backbones import get_model

    model = get_model("dinov2_vitb14", args.device, smaller_edge_size=SIZE)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    def extract(image_rgb: np.ndarray) -> np.ndarray:
        tensor, g = model.prepare_image(image_rgb)
        tok = model.extract_features(tensor).astype(np.float32)
        return tok.reshape(g[0], g[1], -1)

    def dmin_grid(qfeat: np.ndarray, rfeat: np.ndarray, dim: int, hh: int, ww: int) -> np.ndarray:
        import faiss

        q = qfeat.reshape(-1, dim).astype(np.float32)
        r = rfeat.reshape(-1, dim).astype(np.float32)
        faiss.normalize_L2(q)
        faiss.normalize_L2(r)
        index = faiss.IndexFlatL2(dim)
        index.add(r)
        dists, _ = index.search(q, 1)
        return (dists[:, 0] / 2.0).reshape(hh, ww)

    def map2d(gmap: np.ndarray) -> np.ndarray:
        return dists2map(gmap, (SIZE, SIZE)).astype(np.float32)

    rows = []
    t0 = time.time()
    with torch.inference_mode():
        for cat in sorted(manifest["categories"]):
            p = args.dino_cache / f"{cat}.npz"
            if not p.is_file():
                continue
            d = common.load_features(p)
            feat = d["patch_features"]
            ref_orig = d["ref_patch_features"]
            masks = d["imgs_masks"]
            n, hh, ww, dim = feat.shape
            assert (hh, ww) == (32, 32)

            # reference per view (k1 reference transformed + features)
            ref_path = manifest["categories"][cat]["0"]["1"][0]
            ref_img = cv2.cvtColor(cv2.imread(str(args.data_root / ref_path)),
                                   cv2.COLOR_BGR2RGB)
            ref_blocks = {"orig": ref_orig[0]}
            for view in VIEWS:
                ref_blocks[view] = extract(make_view(ref_img, view))

            base_maps = []
            for i in range(n):
                base_maps.append(map2d(dmin_grid(feat[i], ref_orig[0], dim, hh, ww)))
            base_maps = np.stack(base_maps).astype(np.float32)  # [N,448,448]

            valid_s = np.stack([valid_strided(v) for v in VIEWS])  # [5,56,56]

            sm_base = base_maps[:, ::STRIDE, ::STRIDE]
            sm_view = []  # per-image list of [5,56,56]
            for i in range(n):
                sid = str(d["sample_ids"][i])
                img = cv2.cvtColor(cv2.imread(str(args.data_root / sid)), cv2.COLOR_BGR2RGB)
                imaps = []
                for view in VIEWS:
                    qfeat = extract(make_view(img, view))
                    vmap = map2d(dmin_grid(qfeat, ref_blocks[view], dim, hh, ww))
                    omap = inverse_warp_map(vmap, view)
                    imaps.append(omap[::STRIDE, ::STRIDE])
                sm_view.append(np.stack(imaps).astype(np.float32))  # [5,56,56]

            b1_all = np.empty((n, SM, SM), dtype=np.float32)
            b2_all = np.empty((n, SM, SM), dtype=np.float32)
            ctrl_all = np.empty((n, SM, SM), dtype=np.float32)
            for i in range(n):
                stack = np.concatenate([sm_base[i][None], sm_view[i]], axis=0)  # [6,56,56]
                isvalid = np.ones((6, SM, SM), dtype=bool)
                isvalid[0] = True
                isvalid[1:] = valid_s
                clean = stack.copy()
                clean[~isvalid] = np.nan
                b1_all[i] = np.nanmedian(clean, axis=0)
                med = np.nanmedian(clean, axis=0)
                mad = np.nanmedian(np.abs(clean - med), axis=0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    stability = np.clip(1.0 - mad / np.maximum(med, 1e-6), 0.0, 1.0)
                stability[np.isnan(stability)] = 1.0
                b2_all[i] = sm_base[i] * (0.75 + 0.25 * stability)
                # misalignment control: roll the rot+5 view (stack index 1+4=5)
                r5_idx = 1 + VIEWS.index("rot+5")
                sh = np.roll(np.roll(clean[r5_idx], MISALIGN_CELLS, axis=0),
                             MISALIGN_CELLS, axis=1)
                ctrl_stack = clean.copy()
                ctrl_stack[r5_idx] = sh
                ctrl_all[i] = np.nanmedian(ctrl_stack, axis=0)

            def metrics(maps_sm: np.ndarray) -> dict:
                from evaluate_unified import aupro_fast
                from sklearn.metrics import average_precision_score, roc_auc_score

                masks_sm = masks[:, ::STRIDE, ::STRIDE]
                flat_maps = maps_sm.ravel()
                flat_labels = (masks_sm.ravel() > 0.5).astype(np.int32)
                return {
                    "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
                    "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
                    "pixel_aupro": float(aupro_fast(masks_sm, maps_sm)),
                }

            met_base = metrics(sm_base)
            met_b1 = metrics(b1_all)
            met_b2 = metrics(b2_all)
            met_ctrl = metrics(ctrl_all)
            row = {
                "category": cat,
                "n_images": n,
                "base_ap": round(met_base["pixel_ap"], 6),
                "b1_ap": round(met_b1["pixel_ap"], 6),
                "b2_ap": round(met_b2["pixel_ap"], 6),
                "ctrl_ap": round(met_ctrl["pixel_ap"], 6),
                "delta_b1": round(met_b1["pixel_ap"] - met_base["pixel_ap"], 6),
                "delta_b2": round(met_b2["pixel_ap"] - met_base["pixel_ap"], 6),
                "delta_ctrl": round(met_ctrl["pixel_ap"] - met_base["pixel_ap"], 6),
                "b1_auroc": round(met_b1["pixel_auroc"], 6),
            }
            rows.append(row)
            print(f"[MESP-probe {cat}] base AP={row['base_ap']:.4f} "
                  f"B1 Δ={row['delta_b1']:+.4f} B2 Δ={row['delta_b2']:+.4f} "
                  f"ctrl Δ={row['delta_ctrl']:+.4f}", flush=True)

    report = {
        "route": "B_MESP",
        "pipeline": "v10_portfolio_promise_probe_dino_only",
        "seed": 0, "shot": 1,
        "scope": "PROMISE PROBE ONLY - dino-only maps; NOT the pre-registered fused R0 gate",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(args.out_dir / "R0_PROTOCOL.json"),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "per_category": rows,
        "mean_delta_b1": round(float(np.mean([r["delta_b1"] for r in rows])), 6),
        "mean_delta_b2": round(float(np.mean([r["delta_b2"] for r in rows])), 6),
        "mean_delta_ctrl": round(float(np.mean([r["delta_ctrl"] for r in rows])), 6),
        "n_positive_b1": sum(1 for r in rows if r["delta_b1"] > 0),
        "n_positive_b2": sum(1 for r in rows if r["delta_b2"] > 0),
        "elapsed_s": round(time.time() - t0, 1),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "R0_PROMISE_PROBE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmean Δ B1 = {report['mean_delta_b1']} ({report['n_positive_b1']}/6 pos) | "
          f"B2 = {report['mean_delta_b2']} ({report['n_positive_b2']}/6 pos) | "
          f"ctrl = {report['mean_delta_ctrl']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
