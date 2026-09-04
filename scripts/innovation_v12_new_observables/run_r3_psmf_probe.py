"""Doc 22 §7 PSMF R0 probe (GPU; .venv-anomalyclip).

Phase-Stable Micro-defect Field on the micro-defect cats bracket_black /
bracket_brown / bracket_white (+ metal_plate as large-defect control).
Each phase p in {(0,0),(0,4),(4,0),(4,4)}: np.roll the native-resized image by
p (sub-patch ~0.29 token), re-extract dino@448 (L6/9/11) + clip@518 (L6/12/18/24),
score with the EXACT a1-arm recipe (per-layer L2-KNN vs the same-phase K=1 ref,
normal-only LOO z, 7-layer mean), dists2map to 448, roll the map back by -p to
image coordinates. Combine across phases:
  psmf  = per-pixel median        (primary candidate; grid-tracking alias washes out)
  ov    = per-pixel mean          (overlap-average control, same forwards)
  shuf  = median of the 4 maps after a per-image random phase-label permutation
  a1    = phase (0,0) map (identity check against the archived a1-arm per-cat AP)
Gates (R0_PROTOCOL.md): G1 identity, G2 smallest-25%-components bin AP@448,
G3 macro@56, G4 vs overlap-average, G5 vs phase-shuffle, G6 metal_plate specificity.

Self-contained scoring helpers (importing the early-fusion probes would push
src/ to the front of sys.path and shadow AnomalyCLIP's utils.get_transform).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts"),
          str(ROOT / "scripts" / "innovation_v12_new_observables"),
          str(ROOT / "methods" / "anomalydino"),
          str(ROOT / "methods" / "AnomalyCLIP-main")):
    sys.path.insert(0, p)

import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from src.utils import dists2map  # noqa: E402
from v2_mpdd_prediction_common import index_dataset  # noqa: E402

from run_r3_ef_multilayer_export import make_clip, make_dino, DATA_ROOT, MANIFEST  # noqa: E402

PHASES = [(0, 0), (0, 4), (4, 0), (4, 4)]       # offsets at the 448-map space
SHIFT_SRC = 1024                                # MPDD raw image size
IN_SHIFT = [tuple(int(round(o * SHIFT_SRC / 448.0)) for o in off) for off in PHASES]
MICRO_CATS = ["bracket_black", "bracket_brown", "bracket_white"]
CTRL_CAT = "metal_plate"
R0_CATS = MICRO_CATS + [CTRL_CAT]
D_SIZE, C_SIZE = 448, 518


# --------------------------------------------------------------------------
# small helpers (mirror the a1-arm scoring recipe)
# --------------------------------------------------------------------------

def _l2(x: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    shp = x.shape
    flat = x.reshape(-1, shp[-1])
    n = np.linalg.norm(flat, axis=1, keepdims=True)
    return (flat / np.maximum(n, 1e-8)).reshape(shp)


def _nearest_dist(q_flat, bank):
    idx = faiss.IndexFlatL2(bank.shape[1])
    idx.add(bank.astype(np.float32))
    d2, _ = idx.search(q_flat.astype(np.float32), 1)
    return np.sqrt(np.maximum(d2[:, 0], 0.0).astype(np.float64)).astype(np.float32)


def _loo_support_stats(layer, excl_radius=1):
    """Normal-only (mean,std) of L2 residual distances on the K=1 support with a
    Chebyshev spatial exclusion radius (neighbouring normal patches are not
    independent; doc 26 §4.1 convention)."""
    H = layer.shape[0]
    feats = _l2(layer).reshape(H * H, -1)
    idx = faiss.IndexFlatL2(feats.shape[1])
    idx.add(feats.astype(np.float32))
    d2, nbi = idx.search(feats.astype(np.float32), H * H)
    r = []
    for i in range(H):
        for j in range(H):
            row = i * H + j
            dmin = None
            for d, nb in zip(d2[row], nbi[row]):
                ni, nj = int(nb) // H, int(nb) % H
                if max(abs(ni - i), abs(nj - j)) > excl_radius:
                    dmin = float(d)
                    break
            r.append(dmin if dmin is not None else float(d2[row, -1]))
    rr = np.asarray(r, dtype=np.float64)
    return float(rr.mean()), float(rr.std() + 1e-8)


def _bilinear(feat, grid):
    f = np.ascontiguousarray(feat, dtype=np.float32)
    h, w, d = f.shape
    t = torch.from_numpy(f).permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=(grid, grid), mode="bilinear",
                                        align_corners=False)
    return t[0].permute(1, 2, 0).numpy()


def _pooled_ap(maps, m):
    y = (m.ravel() > 0.5).astype(np.int32)
    s = maps.ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


# --------------------------------------------------------------------------
# per-cat
# --------------------------------------------------------------------------

def _load_meta(cat: str, shot: int = 1):
    z = np.load(ROOT / "outputs/dynamic_fusion/v12_early_fusion"
                / f"ml_dino_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    masks = np.asarray(z["imgs_masks"], dtype=np.uint8)   # [N,448,448]
    ids = np.asarray(z["sample_ids"])
    del z
    samples = index_dataset("mpdd", DATA_ROOT)[cat]
    assert np.array_equal(ids, [s.sample_id for s in samples]), "order mismatch"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    refs_rel = manifest["categories"][cat]["0"][str(shot)]
    return samples, masks, refs_rel


def _phase_rgb(rgb: np.ndarray, sy: int, sx: int) -> np.ndarray:
    """Cyclic phase shift at the ORIGINAL resolution (extractors resize internally,
    so the grid alignment relative to content shifts by the same physical amount)."""
    if sy or sx:
        return np.roll(rgb, (sy, sx), axis=(0, 1))
    return rgb


def _area_threshold(masks: np.ndarray) -> float:
    from scipy.ndimage import label

    areas = []
    for i in range(masks.shape[0]):
        if masks[i].sum() == 0:
            continue
        lbl, n = label(masks[i] > 0, structure=np.ones((3, 3), dtype=int))
        areas.extend(np.bincount(lbl.ravel())[1:].tolist())
    areas = np.asarray(areas)
    return float(np.quantile(areas, 0.25)) if areas.size else float("nan")


def _smallest25_bin(masks: np.ndarray, thr_area: float) -> np.ndarray:
    from scipy.ndimage import label

    out = np.zeros_like(masks, dtype=bool)
    for i in range(masks.shape[0]):
        if masks[i].sum() == 0:
            continue
        lbl, n = label(masks[i] > 0, structure=np.ones((3, 3), dtype=int))
        areas = np.bincount(lbl.ravel())[1:]
        small = np.where(areas <= thr_area)[0] + 1
        if small.size:
            out[i] = np.isin(lbl, small)
    return out


def run_cat(cat: str, device: str) -> dict:
    samples, masks448, refs_rel = _load_meta(cat)
    n = masks448.shape[0]
    ex_d, ex_c = make_dino(device), make_clip(device)
    rgb_ref = cv2.cvtColor(cv2.imread(str(DATA_ROOT / refs_rel[0])), cv2.COLOR_BGR2RGB)

    # ---- per phase: ref banks/stats (K=1, normal-only) ----
    phase_meta = []   # per phase: {"d": [{bank,mean,std} x3], "c": [x4]}
    for pi, (dy, dx) in enumerate(PHASES):
        sy, sx = IN_SHIFT[pi]
        dl = ex_d(_phase_rgb(rgb_ref, sy, sx))      # [3,32,32,768]
        cl = ex_c(_phase_rgb(rgb_ref, sy, sx))      # [4,37,37,768]
        cl32 = [_bilinear(c, 32) for c in cl]
        meta = {"d": [], "c": []}
        for li, f in enumerate(dl):
            mean, std = _loo_support_stats(f, 1)
            meta["d"].append({"bank": _l2(f).reshape(-1, 768), "mean": mean, "std": std})
        for li, f in enumerate(cl32):
            mean, std = _loo_support_stats(f, 1)
            meta["c"].append({"bank": _l2(f).reshape(-1, 768), "mean": mean, "std": std})
        phase_meta.append(meta)
        del dl, cl, cl32

    # ---- per phase per image: 7-layer z-mean at 32 grid ----
    zph = np.zeros((len(PHASES), n, 32, 32), dtype=np.float32)
    for pi, (dy, dx) in enumerate(PHASES):
        sy, sx = IN_SHIFT[pi]
        meta = phase_meta[pi]
        for i in range(n):
            rgb = cv2.cvtColor(cv2.imread(str(samples[i].image_path)), cv2.COLOR_BGR2RGB)
            dl = ex_d(_phase_rgb(rgb, sy, sx))
            cl = ex_c(_phase_rgb(rgb, sy, sx))
            zacc = []
            for li, f in enumerate(dl):
                e = meta["d"][li]
                d = _nearest_dist(_l2(f).reshape(-1, 768), e["bank"]).reshape(32, 32)
                zacc.append((d - e["mean"]) / e["std"])
            for li, f in enumerate(cl):
                f32 = _bilinear(f, 32)
                e = meta["c"][li]
                d = _nearest_dist(_l2(f32).reshape(-1, 768), e["bank"]).reshape(32, 32)
                zacc.append((d - e["mean"]) / e["std"])
            zph[pi, i] = np.mean(zacc, axis=0)
        print(f"  [{cat}] phase {dy},{dx} done", flush=True)

    # ---- combine in image coordinates ----
    maps448 = {k: np.zeros((n, 448, 448), dtype=np.float32)
               for k in ("a1", "psmf", "ov", "shuf")}
    rng = np.random.default_rng(0)
    for i in range(n):
        # stack of maps each rolled back to image coordinates by its OWN phase offset
        own = np.stack([np.roll(dists2map(zph[pi, i], (448, 448)), (-dy, -dx),
                                axis=(0, 1)) for pi, (dy, dx) in enumerate(PHASES)])
        # phase-shuffle control: same maps, but each rolled back by a WRONG phase's
        # offset (breaks image-coordinate consistency; median is permutation-invariant
        # so a plain label shuffle would be a vacuous control).
        perm = rng.permutation(len(PHASES))
        mis = np.stack([np.roll(dists2map(zph[pi, i], (448, 448)), (-PHASES[perm[pi]][0],
                                                                    -PHASES[perm[pi]][1]),
                                axis=(0, 1)) for pi in range(len(PHASES))])
        maps448["a1"][i] = own[0]
        maps448["ov"][i] = own.mean(0)
        maps448["psmf"][i] = np.median(own, axis=0)
        maps448["shuf"][i] = np.median(mis, axis=0)
    del zph

    m56 = (masks448[:, ::8, ::8] > 0.5).astype(np.uint8)
    ap56 = {k: _pooled_ap(np.stack([maps448[k][i][::8, ::8] for i in range(n)]), m56)
            for k in ("a1", "psmf", "ov", "shuf")}
    res = {"category": cat, "n": n, "ap56": {k: round(ap56[k], 6) for k in ap56}}
    if cat in MICRO_CATS:
        thr = _area_threshold(masks448)
        binm = _smallest25_bin(masks448, thr)
        res["smallest25_area_max"] = round(thr, 1)
        res["smallest25_px"] = int(binm.sum())
        if binm.any():
            res["bin_ap448"] = {
                "a1": round(_pooled_ap(maps448["a1"], binm.astype(np.uint8)), 6),
                "psmf": round(_pooled_ap(maps448["psmf"], binm.astype(np.uint8)), 6)}
    print(f"  {cat}: " + json.dumps(res), flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else R0_CATS

    out_root = ROOT / "experiments/dynamic_fusion/innovation_v12_new_observables/psmf"
    out_root.mkdir(parents=True, exist_ok=True)
    res_path = out_root / "PSMF_R0_RESULT.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    done = {r["category"] for r in res.get("per_category", [])}
    rows = res.get("per_category", [])
    for cat in cats:
        if cat in done:
            continue
        rows.append(run_cat(cat, args.device))
    res["per_category"] = sorted(rows, key=lambda r: r["category"])
    res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, default=float)[:3000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
