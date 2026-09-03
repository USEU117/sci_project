"""Route B — MESP geometry audit (task book 19 §5): 10 normal reference images.

Measures per pre-registered view (hflip, brightness x0.9/x1.1, rot -5/+5):
1. inverse-transform image RMSE vs original over the valid interior (checks the
   chain is faithful; rotation borders excluded by a pre-registered margin mask);
2. DINO-vitb14 feature stability: mean cosine(original patch, feature of the
   inverse-transformed view) over the interior grid cells;
3. per-image added inference time (GPU) for one extra view.

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v10_portfolio\\run_mesp_geometry_audit.py
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))

VIEWS = ["hflip", "bright0.9", "bright1.1", "rot-5", "rot+5"]
ROT_DEG = {"rot-5": -5.0, "rot+5": 5.0}
RESOLUTION = 448
INTERIOR_PX = 24   # rotation margin (~19.6 px max displacement at +-5 deg + slack)


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


def inverse_view(img: np.ndarray, view: str) -> np.ndarray:
    """Transform the view back to the original frame (chain check)."""
    if view == "hflip":
        return make_view(img, "hflip")
    if view == "bright0.9":
        return np.clip(img.astype(np.float32) / 0.9, 0, 255).astype(np.uint8)
    if view == "bright1.1":
        return np.clip(img.astype(np.float32) / 1.1, 0, 255).astype(np.uint8)
    if view.startswith("rot"):
        deg = ROT_DEG[view]
        m = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), -deg, 1.0)
        return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    raise ValueError(view)


def interior_mask(h: int, w: int, margin: int = INTERIOR_PX) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[margin:h - margin, margin:w - margin] = True
    return m


def gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mpdd_raw/MPDD")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/mpdd/manifest.json")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v10_portfolio/mesp")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from src.backbones import get_model

    model = get_model("dinov2_vitb14", args.device, smaller_edge_size=RESOLUTION)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    # pre-registered deterministic image list: the 6 k1 references + 2 extra good
    # images of bracket_black and tubes (first two alphabetically)
    images: list[str] = []
    for cat in sorted(manifest["categories"]):
        images.append(manifest["categories"][cat]["0"]["1"][0])
    for cat in ("bracket_black", "tubes"):
        goods = sorted(glob.glob(str(args.data_root / cat / "train/good/*.png")))
        images.extend([goods[0], goods[1]])
    images = images[:10]
    assert len(images) == 10

    rows = []
    torch.cuda.reset_peak_memory_stats()
    t_feat = []
    with torch.inference_mode():
        for rel in images:
            img = cv2.cvtColor(cv2.imread(str(args.data_root / rel)), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA)

            def feats(image: np.ndarray):
                tensor, grid = model.prepare_image(image)
                t0 = time.time()
                tok = model.extract_features(tensor).astype(np.float32)
                t_feat.append(time.time() - t0)
                return tok.reshape(grid[0], grid[1], -1), grid

            f_orig, grid = feats(img)
            per_view = {}
            for view in VIEWS:
                v = make_view(img, view)
                iv = inverse_view(v, view)
                # chain RMSE over interior
                rm = gray(iv) - gray(img)
                mask = interior_mask(img.shape[0], img.shape[1])
                if view.startswith("rot"):
                    # rotation also displaces the valid area slightly; margin handles it
                    pass
                rmse = float(np.sqrt((rm[mask] ** 2).mean()))
                # feature stability of the inverse-transformed view vs original
                f_inv, g2 = feats(iv)
                if tuple(g2) != tuple(grid):
                    raise RuntimeError("grid mismatch across views")
                cell = RESOLUTION // grid[0]
                cell_margin = int(np.ceil(INTERIOR_PX / cell)) + 1
                cm = np.zeros(grid, dtype=bool)
                cm[cell_margin:grid[0] - cell_margin, cell_margin:grid[1] - cell_margin] = True
                f_orig_n = f_orig[cm].reshape(-1, f_orig.shape[-1])
                f_inv_n = f_inv[cm].reshape(-1, f_inv.shape[-1])
                norms = np.linalg.norm(f_inv_n, axis=-1)
                norms[norms < 1e-9] = 1.0
                cos = (f_orig_n * f_inv_n).sum(axis=-1) / (
                    np.linalg.norm(f_orig_n, axis=-1) * norms)
                per_view[view] = {
                    "inverse_rmse_px": round(rmse, 4),
                    "interior_cos": round(float(cos.mean()), 5),
                    "interior_cos_min": round(float(cos.min()), 5),
                }
                print(f"[{Path(rel).parent.parent.name}/{Path(rel).name}] {view}: "
                      f"RMSE={rmse:.3f}px cos={cos.mean():.4f}", flush=True)
            rows.append({"image": rel, "views": per_view})

    report = {
        "route": "B_MESP",
        "pipeline": "v10_portfolio_r0_geometry_audit",
        "n_images": len(images),
        "model": "dinov2_vitb14@448",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(args.out_dir / "R0_PROTOCOL.json"),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "per_image": rows,
        "summary": {
            view: {
                "mean_inverse_rmse_px": round(float(np.mean(
                    [r["views"][view]["inverse_rmse_px"] for r in rows])), 4),
                "mean_interior_cos": round(float(np.mean(
                    [r["views"][view]["interior_cos"] for r in rows])), 5),
            }
            for view in VIEWS
        },
        "mean_per_extra_view_inference_s": round(float(np.mean(t_feat)), 4)
        if t_feat else None,
        "peak_vram_gb": round(float(torch.cuda.max_memory_allocated()) / 1e9, 3),
        "decision": "PENDING",
    }
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "R0_GEOMETRY_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=1))
    print(f"per-extra-view inference {report['mean_per_extra_view_inference_s']}s "
          f"| peak VRAM {report['peak_vram_gb']}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
