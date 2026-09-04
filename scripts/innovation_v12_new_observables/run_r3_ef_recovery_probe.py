"""Doc 26 §4.3 first mechanism gate: does RECOVER-BEFORE-MATCH help?

Premise under test (doc 26 §4.3): the current A1 pipeline downsamples CLIP
37x37 -> DINO 32x32 and runs KNN on the coarse grid BEFORE any spatial recovery
("match-first"); information needed for micro-defect localisation may be lost
before matching. If we FIRST recover both branches onto a shared finer grid
(56x56) with a real pre-trained feature upsampler (AnyUp multi-backbone,
encoder-agnostic, frozen) and THEN run the same per-layer KNN, does macro
pooled Pixel-AP@56 improve?

Protocol (pre-registered; 2 micro-defect cats first, shot k1):
  cats        : bracket_black, metal_plate   (user-chosen first gate, doc 26 s4.3)
  feature set : same 7 layers as the static/A1 reference
                dino L{6,9,11} @32, clip L{6,12,18,24} @37 (ml_ caches)
  arms        :
    a1     = match-first, 32-grid, clip bilinear->32. Computed by the SAME grid
             arm at (grid=32, mode=bl): identical recipe to CL-RPF static
             "mean_std" map which matches archived A1 within 6e-4 (doc 26 s3).
    bl56   = recover-first via BILINEAR feature upsample 32/37 -> 56 -> KNN
    au56   = recover-first via AnyUp (frozen pretrained multi-backbone) -> 56 -> KNN
    au56_w= au56 with WRONG guide RGB (content-mismatch; control #7 doc 26 s4.3)
  score per arm = pooled Pixel-AP@56 (dists2map(448,sigma4)[::8,::8]) macro over cats
  gates (macro over the 2 cats):
    P1 premise : au56 - a1   >= +0.003
    P2 generic : au56 - bl56 >= +0.003   (real upsampler > cheap interpolation)
    P3 guide   : au56 - au56_w >= +0.003 (wrong-guide sanity, run only if P1 passes)
  stop rule    : P1 fail -> premise rejected, archive route negative, no coupled
                 module, no wrong-guide arm (未过即停止).

Run (GPU, .venv-anomalyclip), stages:
  --stage cpu  : a1 (identity check, all 6 cats) + bl56 (all 6 cats)   [cheap, CPU]
  --stage gpu  : au56 on the 2 pre-registered cats (needs AnyUp ckpt)
  --stage wrong: au56_w on the 2 cats (only if P1 passed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"),
          str(ROOT / "scripts" / "innovation_v12_new_observables")):
    sys.path.insert(0, p)

import faiss  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from src.utils import dists2map  # noqa: E402
from v2_mpdd_prediction_common import index_dataset  # noqa: E402

from run_r3_ef_stage0_null_audit import CATEGORIES, ML_ROOT  # noqa: E402

GRID_56 = 56
CATS_2 = ["bracket_black", "metal_plate"]
DATA_ROOT = ROOT / "data/mpdd_raw/MPDD"
MANIFEST = ROOT / "data/splits/mpdd/manifest.json"
ANYUP_CKPT = ROOT / "outputs/external_weights/anyup_multi_backbone.pth"
ANYUP_SHA256 = "B6CC407DA8986C7E5C9098E61F7531767A9ACA8FFF20A1BC6C99D488E61AAC59"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------
# small shims / helpers (mirror CL-RPF scoring helpers)
# --------------------------------------------------------------------------

def _install_rmsnorm_shim():
    """AnyUp needs torch>=2.4 nn.RMSNorm; venv is torch 2.0 -> add equivalent."""
    import torch.nn as nn

    if hasattr(nn, "RMSNorm"):
        return

    class _RMS(nn.Module):
        def __init__(self, normalized_shape, eps=None, elementwise_affine=True,
                     device=None, dtype=None):
            super().__init__()
            if isinstance(normalized_shape, int):
                normalized_shape = (normalized_shape,)
            self.normalized_shape = tuple(normalized_shape)
            self.eps = 1e-6 if eps is None else eps
            self.elementwise_affine = elementwise_affine
            if elementwise_affine:
                self.weight = nn.Parameter(
                    torch.empty(self.normalized_shape, device=device, dtype=dtype))
                nn.init.ones_(self.weight)
            else:
                self.register_parameter("weight", None)

        def forward(self, x):
            rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
            o = x * rms
            return o * self.weight if self.weight is not None else o

    nn.RMSNorm = _RMS


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


def _loo_support_stats(ref_grid, excl_radius):
    """Normal-only (mean,std) of nearest-neighbour residual distances on the
    K=1 support, excluding a Chebyshev spatial neighbourhood (radius in grid
    units scaled with the grid: 1 @32-grid, 2 @56-grid) so neighbouring normal
    patches are not treated as independent (doc 26 §4.1 convention)."""
    H = ref_grid.shape[0]
    feats = _l2(ref_grid).reshape(H * H, -1)
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


def _pooled_ap(maps56, m56):
    y = (m56.ravel() > 0.5).astype(np.int32)
    s = maps56.ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))


def _bilinear(feat, grid):
    """feat [h,w,D] -> [grid,grid,D] bilinear (torch, align_corners=False)."""
    f = np.ascontiguousarray(feat, dtype=np.float32)
    h, w, d = f.shape
    t = torch.from_numpy(f).permute(2, 0, 1)[None]  # [1,D,h,w]
    t = torch.nn.functional.interpolate(t, size=(grid, grid), mode="bilinear",
                                        align_corners=False)
    return t[0].permute(1, 2, 0).numpy()


# --------------------------------------------------------------------------
# per-cat loaders
# --------------------------------------------------------------------------

def _load_cat(cat: str, shot: int):
    zd = np.load(ML_ROOT / f"ml_dino_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    dino = {"feat": np.asarray(zd["patch_features"]),          # [N,3,32,32,768]
            "ref": np.asarray(zd["ref_patch_features"]),       # [3,K,32,32,768]
            "masks": np.asarray(zd["imgs_masks"]),
            "ids": np.asarray(zd["sample_ids"])}
    del zd
    zc = np.load(ML_ROOT / f"ml_clip_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    clip = {"feat": np.asarray(zc["patch_features"]),          # [N,4,37,37,768]
            "ref": np.asarray(zc["ref_patch_features"]),       # [4,K,37,37,768]
            "ids": np.asarray(zc["sample_ids"])}
    del zc
    samples = index_dataset("mpdd", DATA_ROOT)[cat]
    assert np.array_equal(dino["ids"], [s.sample_id for s in samples]), "dino id order"
    assert np.array_equal(clip["ids"], [s.sample_id for s in samples]), "clip id order"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    refs_rel = manifest["categories"][cat]["0"][str(shot)]
    return dino, clip, samples, refs_rel


def _guide_tensor(rgb: np.ndarray, size: int) -> torch.Tensor:
    import cv2

    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(x).permute(2, 0, 1)[None]  # [1,3,S,S]


def _load_rgb(path) -> np.ndarray:
    import cv2

    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def _make_anyup(device: str):
    _install_rmsnorm_shim()
    from anyup.model import AnyUp

    model = AnyUp().to(device)
    ck = torch.load(str(ANYUP_CKPT), map_location=device)
    missing, unexpected = model.load_state_dict(ck, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    return model


# --------------------------------------------------------------------------
# grid arm scoring (bl56 / au56 / au56_w)  -- 2-branch, 7 layers, static mean
# --------------------------------------------------------------------------

def _recover_grid_arm(cat, shot, grid, mode, device="cuda:0"):
    """Return pooled per-cat AP for the recover-first static-KNN arm.

    mode: "bl" = bilinear feature upsample (a1 uses grid=32 => identical recipe to
    the CL-RPF static "mean_std" map == archived A1 within 6e-4, doc 26 s3);
          "au" = AnyUp frozen multi-backbone recovery; "auw" = AnyUp + wrong guide.
    """
    dino, clip, samples, refs_rel = _load_cat(cat, shot)
    n = dino["feat"].shape[0]
    m56 = (dino["masks"][:, ::8, ::8] > 0.5).astype(np.uint8)
    K = dino["ref"].shape[1]
    radius = max(1, round(grid / 32.0))  # 1 @32, 2 @56 (spatially proportional)
    model = None
    wrong_rgb = None
    if mode in ("au", "auw"):
        model = _make_anyup(device)
        if mode == "auw":
            wrong_rgb = [_load_rgb(samples[(i + 1) % n].image_path) for i in range(n)]

    def recover(feat, rgb, size):
        if mode == "bl":
            return _bilinear(feat, grid)
        f = torch.from_numpy(np.ascontiguousarray(feat, dtype=np.float32))
        t = f.permute(2, 0, 1)[None].to(device)          # [1,768,h,w]
        g = _guide_tensor(rgb, size).to(device)
        with torch.inference_mode():
            out = model(g, t, output_size=(grid, grid), q_chunk_size=16)
        return out[0].permute(1, 2, 0).float().cpu().numpy()

    def rgb_for(i):
        if mode == "bl":
            return None
        if mode == "au":
            return _load_rgb(samples[i].image_path)
        return wrong_rgb[i]

    # ---- refs (support) per layer: recover to grid, bank + normal-only stats ----
    entries = {"d": [], "c": []}   # one per branch layer, order follows D_LAY/C_LAY
    for branch, arr, dsize in (("d", dino, 448), ("c", clip, 518)):
        for li in range(len(arr["ref"])):
            rf = arr["ref"][li]                              # [K,h,w,768]
            if mode in ("au", "auw"):
                rgb0 = _load_rgb(DATA_ROOT / refs_rel[0])
                refg = np.stack([recover(rf[k], rgb0, dsize) for k in range(K)])
            else:
                refg = np.stack([_bilinear(rf[k], grid) for k in range(K)])
            bank = _l2(refg[0]).reshape(-1, 768)
            mean_l, std_l = _loo_support_stats(refg[0], radius)
            entries[branch].append({"bank": bank, "mean": mean_l, "std": std_l})

    # ---- query images: per-layer recovered features -> z-map -> static mean ----
    zmean = np.zeros((n, grid, grid), dtype=np.float64)
    for i in range(n):
        rgb_i = rgb_for(i)
        zacc = []
        for branch, arr, dsize in (("d", dino, 448), ("c", clip, 518)):
            for li in range(len(entries[branch])):
                fg = recover(arr["feat"][i, li], rgb_i, dsize)   # [grid,grid,768]
                e = entries[branch][li]
                dist = _nearest_dist(_l2(fg).reshape(-1, 768), e["bank"]).reshape(grid, grid)
                zacc.append((dist - e["mean"]) / e["std"])
        zmean[i] = np.mean(zacc, axis=0)
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    maps56 = np.stack([dists2map(zmean[i], (448, 448))[::8, ::8] for i in range(n)])
    return _pooled_ap(maps56, m56), {"n": n, "grid": grid, "mode": mode, "cat": cat}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

ARMS = {"a1": (32, "bl"), "bl56": (GRID_56, "bl"), "au56": (GRID_56, "au"),
        "au56_w": (GRID_56, "auw")}


def _stage_results(arms: list[str], cats: list[str], device: str) -> list[dict]:
    rows = []
    for cat in cats:
        row = {"category": cat}
        for arm in arms:
            grid, mode = ARMS[arm]
            ap_, info = _recover_grid_arm(cat, 1, grid, mode, device)
            row[arm] = ap_
            print(f"  {cat} {arm}: {ap_:.4f} ({info})", flush=True)
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("cpu", "gpu", "wrong"), required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out_root = ROOT / "experiments/dynamic_fusion/innovation_v12_new_observables/detail_recovery"
    out_root.mkdir(parents=True, exist_ok=True)
    res_path = out_root / "RECOVERY_RESULTS.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}

    if args.stage == "cpu":
        rows = _stage_results(["a1", "bl56"], CATEGORIES, "cpu")   # 6 cats (identity + ref)
        res["cpu_all6"] = rows
    else:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable")
        arm = "au56" if args.stage == "gpu" else "au56_w"
        rows = _stage_results([arm], CATS_2, args.device)
        res[f"{arm}_rows"] = rows

    res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, default=float)[:3000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
