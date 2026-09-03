"""Run R0 (region information value) for Route E STR (task book 19 §8).

MPDD seed0, k4 normal references. STR is calibrated on the 4 normal reference
images ONLY (median/MAD of absolute Haar subband coefficients, gray + R-G + B-Y).
Diagnostic (supervised) task: can the STR residual identify pixels of GT defect
components that the frozen A1 fused map MISSED, vs normal pixels, better than the
A1 score itself? No fusion is performed in R0.

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v10_portfolio\\run_r0_str.py
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

from industrial_ad.innovation_v10_portfolio import common, spectral
from industrial_ad.innovation_v10_portfolio.common import build_fused_blocks, load_features
from skimage import measure
from sklearn.metrics import average_precision_score

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
MISSED_AREA_FRAC = 0.20      # component is "A1-missed" if <20% of its pixels are flagged by A1
MAX_NORMAL_IMAGES = 30       # negative-image budget per class
PX_POS_CAP = 30000
PX_NEG_CAP = 90000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rgb(data_root: Path, rel: str) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(data_root / rel)), cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(rel)
    return img


def loo_dmin_threshold(ref: np.ndarray, q: float = 0.95) -> float:
    """Reference-only fused d_min distribution (leave-one-reference-out) -> theta.

    No test images, no test statistics, no GT.
    """
    import faiss

    k = ref.shape[0]
    grid_area = int(np.prod(ref.shape[1:3]))
    d = ref.shape[-1]
    ref_per_img = ref.reshape(k, grid_area, d).astype(np.float32)
    mins: list[np.ndarray] = []
    for qq in range(k):
        banks = [r for r in range(k) if r != qq]
        dists = np.empty((grid_area, len(banks)), dtype=np.float32)
        for j, r in enumerate(banks):
            index = faiss.IndexFlatL2(d)
            index.add(ref_per_img[r])
            distances, _ = index.search(ref_per_img[qq], k=1)
            dists[:, j] = distances[:, 0]
        mins.append(dists.min(axis=1) / 2.0)
    all_min = np.concatenate(mins)
    return float(np.percentile(all_min, 100.0 * q))
def _stack(pred_key: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(pred_key) if pred_key else np.empty(0, dtype=np.float32)


def run_category(data_root: Path, dino_cache: Path, clip_cache: Path, manifest: dict,
                 cat: str) -> dict:
    t0 = time.time()
    dino = load_features(dino_cache / f"{cat}.npz")
    clip = load_features(clip_cache / f"{cat}.npz")
    feat, ref, _, masks, grid = build_fused_blocks(dino, clip, dino_weight=0.5)
    n, d = feat.shape[0], feat.shape[-1]
    feat_flat = feat.reshape(-1, d).astype(np.float32)
    dr = common.per_reference_distances(feat_flat, ref)
    dmin_grid = dr.min(axis=-1).reshape(n, *grid)  # fused A1 (A0) distance map grid
    a1_map = common.maps_from_patches(dmin_grid)   # [N, 448, 448]

    theta = loo_dmin_threshold(ref)
    gt_sp = dino["gt_sp"]
    sample_ids = dino["sample_ids"]

    # ---- STR calibration on the 4 normal references ----
    ref_rels = manifest["categories"][cat]["0"]["4"]
    planes_by_ch: dict[str, list[np.ndarray]] = {"gray": [], "rg": [], "by": []}
    for rrel in ref_rels:
        chans = spectral._channels(read_rgb(data_root, rrel))
        for ch in ("gray", "rg", "by"):
            planes_by_ch[ch].append(chans[ch])
    stats = spectral.band_stats_from_references(planes_by_ch)

    # ---- diagnostic loop ----
    n_defect = int((gt_sp > 0).sum())
    rng = np.random.default_rng(20260903)
    labels: list[np.ndarray] = []
    pred: dict[str, list[np.ndarray]] = {"str": [], "a1": [], "grad": [], "str_misaligned": []}
    n_normal = 0
    n_pos = 0
    for i in range(n):
        sid = str(sample_ids[i])
        m = masks[i]
        amap = a1_map[i]
        is_defect = bool(gt_sp[i] > 0)
        if not is_defect:
            if n_normal >= MAX_NORMAL_IMAGES:
                continue
            n_normal += 1
            img = read_rgb(data_root, sid)
            str_map = spectral.residual_map(img, stats)
            grad_map = spectral.gradient_magnitude_map(img)
            yy, xx = np.mgrid[0:str_map.shape[0]:4, 0:str_map.shape[1]:4]
            yy, xx = yy.ravel(), xx.ravel()
            labels.append(np.zeros(yy.size, dtype=np.int64))
            pred["str"].append(str_map[yy, xx].astype(np.float32))
            pred["a1"].append(amap[yy, xx].astype(np.float32))
            pred["grad"].append(grad_map[yy, xx].astype(np.float32))
            shifted = np.roll(np.roll(str_map, 56, axis=0), 56, axis=1)
            pred["str_misaligned"].append(shifted[yy, xx].astype(np.float32))
        else:
            if m.max() == 0:
                continue
            lbl = measure.label((m > 0).astype(np.uint8), connectivity=2)
            flagged = amap > theta
            missed = np.zeros_like(m, dtype=bool)
            for comp in np.unique(lbl[lbl > 0]):
                cmask = lbl == comp
                frac = float((flagged & cmask).sum()) / max(1.0, float(cmask.sum()))
                if frac < MISSED_AREA_FRAC:
                    missed |= cmask
            yy, xx = np.nonzero(missed)
            if yy.size == 0:
                continue
            if yy.size > PX_POS_CAP:
                sel = rng.choice(yy.size, PX_POS_CAP, replace=False)
                yy, xx = yy[sel], xx[sel]
            img = read_rgb(data_root, sid)
            str_map = spectral.residual_map(img, stats)
            grad_map = spectral.gradient_magnitude_map(img)
            n_pos += yy.size
            labels.append(np.ones(yy.size, dtype=np.int64))
            pred["str"].append(str_map[yy, xx].astype(np.float32))
            pred["a1"].append(amap[yy, xx].astype(np.float32))
            pred["grad"].append(grad_map[yy, xx].astype(np.float32))
            shifted = np.roll(np.roll(str_map, 56, axis=0), 56, axis=1)
            pred["str_misaligned"].append(shifted[yy, xx].astype(np.float32))

    y_all = np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)
    ap = {}
    for key in pred:
        v = _stack(pred[key])
        if v.shape[0] != y_all.shape[0]:
            raise RuntimeError(f"label/pred size mismatch for {key}")
        if y_all.size == 0 or y_all.sum() == 0 or y_all.sum() == y_all.size:
            ap[key] = float("nan")
        else:
            ap[key] = float(average_precision_score(y_all, v))

    delta = ap["str"] - ap["a1"] if np.isfinite(ap["str"]) and np.isfinite(ap["a1"]) else None
    row = {
        "category": cat,
        "n_test_images": n,
        "n_defect_images": n_defect,
        "n_normal_used": n_normal,
        "n_missed_pos_px": n_pos,
        "n_px_total": int(y_all.size),
        "theta_q95_loo": round(theta, 6),
        "ap_str": round(ap["str"], 6),
        "ap_a1": round(ap["a1"], 6),
        "ap_gradient_control": round(ap["grad"], 6),
        "ap_str_misaligned": round(ap["str_misaligned"], 6),
        "delta_str_minus_a1": round(delta, 6) if delta is not None else None,
        "delta_true_vs_misaligned": round(ap["str"] - ap["str_misaligned"], 6)
        if np.isfinite(ap["str"]) and np.isfinite(ap["str_misaligned"]) else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(f"[STR {cat}] pos={n_pos} total_px={y_all.size} AP_str={ap['str']:.4f} "
          f"AP_a1={ap['a1']:.4f} Δ={row['delta_str_minus_a1']} "
          f"(θ={theta:.4f})", flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mpdd_raw/MPDD")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/mpdd/manifest.json")
    parser.add_argument("--dino-cache", type=Path,
                        default=ROOT / "outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k4/anomalydino_visual")
    parser.add_argument("--clip-cache", type=Path,
                        default=ROOT / "outputs/dynamic_fusion/v3_direction_a/features_s0_k4/anomalyclip_text")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v10_portfolio/str")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    rows = []
    for cat in args.categories:
        if not (args.dino_cache / f"{cat}.npz").is_file():
            print(f"skip {cat}: cache missing")
            continue
        rows.append(run_category(args.data_root, args.dino_cache, args.clip_cache, manifest, cat))

    finite = [r for r in rows if r["delta_str_minus_a1"] is not None]
    report = {
        "route": "E_STR",
        "pipeline": "v10_portfolio_r0_info_value",
        "seed": 0, "shot": 4, "data_role": "development_diagnostic_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {
            "common": sha256_file(ROOT / "src/industrial_ad/innovation_v10_portfolio/common.py"),
            "spectral": sha256_file(ROOT / "src/industrial_ad/innovation_v10_portfolio/spectral.py"),
            "runner": sha256_file(Path(__file__)),
        },
        "per_category": rows,
        "mean_delta_str_minus_a1": round(float(np.mean([r["delta_str_minus_a1"] for r in finite])), 6)
        if finite else None,
        "n_positive_delta": sum(1 for r in finite if r["delta_str_minus_a1"] > 0),
        "mean_true_vs_misaligned": round(float(np.mean(
            [r["delta_true_vs_misaligned"] for r in finite
             if r["delta_true_vs_misaligned"] is not None])), 6) if finite else None,
        "decision": "PENDING",
    }
    (out_dir / "R0_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmean Δ(STR-A1) = {report['mean_delta_str_minus_a1']} | "
          f"positive {report['n_positive_delta']}/6 | true-vs-misaligned "
          f"{report['mean_true_vs_misaligned']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
