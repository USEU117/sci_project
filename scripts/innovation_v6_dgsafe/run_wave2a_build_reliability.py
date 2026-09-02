"""S0 Wave 2a - build normal-only reliability stats for the SUB expert (task book 16 ss.2.3-2.4).

For each MPDD (category, shot) config at seed 0 this script constructs, from the
*k-shot normal references only* (never any test GT / anomaly score):

  A : official SubspaceAD subspace (layers -12..-18, refs + 30 RandomRotation,
      exactly the export/identity-replay configuration of Wave 0).
  B : layer-group subspace G1 = {-12,-13,-14}
  C : layer-group subspace G2 = {-16,-17,-18}
  B/C are *internal stability probes*: fitted on refs + 10 RandomRotation. This
  reduced fit augmentation for the probe subspaces is frozen in
  configs/innovation_v6_dgsafe/reliability_probe.json; it does NOT affect the
  official maps / identity replay, which stay on 30 rotations.

Light-normal pool versions per reference (frozen):
  identity; brightness x1.15; contrast x1.15; shift +-2% x/y (4);
  rotate +-2 deg (2).  -> 9 versions.  Geometric versions are applied in the
  672-frame and their residual grids inverse-warped back to the identity frame.

For every subspace S in {A,B,C} and every (ref, version) we compute the raw
48x48 reconstruction-residual grid; per-pixel empirical tail probability p_b
over the config pool (pool = 9*k grids under S) and the calibrated map
z = clip(-log p_b, 0, 12).

Config-level scalars (risk direction: bigger = less reliable):
  U_aug   = mean over (ref, aug-version) of median_pixel |zA_id - zA_aug|
  U_layer = mean over refs of median_pixel |zB_id - zC_id|   (identity refs)
  B_tail  = P99 over pool pixels of zA
Then, across the 18 configs:
  risk_i = mean(percentile_rank(U_aug), percentile_rank(U_layer),
                percentile_rank(B_tail))
  q_sub  = 1 - risk_i ;  r_sub = clip((q_sub - 0.25)/0.50, 0, 1)

Outputs (no GT touched):
  reliability/reliability_raw.json   per-config scalars + formula meta
  reliability/pools/*.npz            per-config A-pool raw grids (refs x versions)
                                     + sample/version keys (reused by Wave 3)
GT-based validation is done separately by run_wave2b_diagnostic.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "methods" / "SubspaceAD")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402
from src.subspacead.core.extractor import FeatureExtractor  # noqa: E402
from src.subspacead.core.pca import PCAModel  # noqa: E402
from src.subspacead.data.datasets import get_dataset_handler  # noqa: E402
from src.subspacead.data.transforms import get_augmentation_transform  # noqa: E402
from src.subspacead.post_process.scoring import calculate_anomaly_scores  # noqa: E402

LAYERS_A = [-12, -13, -14, -15, -16, -17, -18]
LAYERS_B = [-12, -13, -14]
LAYERS_C = [-16, -17, -18]
IMAGE_RES = 672
GRID = IMAGE_RES // 14          # 48
SHOTS = (1, 2, 4)
SEED = 0
OFFICIAL_AUG = 30               # subspace A (official export policy)
PROBE_AUG = 10                  # probe subspaces B/C (frozen addendum)
Z_CAP = 12.0
PCA_EV = 0.99
POOL_VERSION_SPEC = [           # (key, kind, param)
    ("identity", "id", None),
    ("brightness1.15", "photo", 1.15),
    ("contrast1.15", "photo", 1.15),
    ("shiftx_p2", "geo", ("t", 0.02 * IMAGE_RES, 0.0)),
    ("shiftx_n2", "geo", ("t", -0.02 * IMAGE_RES, 0.0)),
    ("shifty_p2", "geo", ("t", 0.0, 0.02 * IMAGE_RES)),
    ("shifty_n2", "geo", ("t", 0.0, -0.02 * IMAGE_RES)),
    ("rot_p2", "geo", ("r", 2.0)),
    ("rot_n2", "geo", ("r", -2.0)),
]
RELIAB_OUT = maps.EXPERIMENT_ROOT / "reliability"
PROBE_SPEC = ROOT / "configs" / "innovation_v6_dgsafe" / "reliability_probe.json"


def seed_all(seed: int):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# small affine helpers (672 px frame <-> 48 grid frame)
# ---------------------------------------------------------------------------

def rotation_affine(deg: float, size: int):
    """Forward map F (content): v' = A v + t  (pixels, rotation about centre)."""
    th = math.radians(deg)
    c = np.asarray([size / 2.0, size / 2.0])
    A = np.asarray([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    t = c - A @ c
    return A, t


def translate_affine(dx: float, dy: float):
    return np.eye(2), np.asarray([dx, dy])


def affine_dest2src(A, t):
    """cv2.warpAffine M such that dst(x) = src(M x) for forward map F(v)=A v + t."""
    Ainv = np.linalg.inv(A)
    return np.hstack([Ainv, (-Ainv @ t)[:, None]]).astype(np.float64)


def inv_warp_grid(m_warp: np.ndarray, A, t) -> np.ndarray:
    """Bring a 48x48 residual grid of a geometric version back to identity frame.
    m_orig(g) = m_warp(F_g(g)) with F_g(g) = A g + t/14 (grid units)."""
    tg = t / 14.0
    xs, ys = np.meshgrid(np.arange(GRID, dtype=np.float32),
                         np.arange(GRID, dtype=np.float32))
    mx = (A[0, 0] * xs + A[0, 1] * ys + tg[0]).astype(np.float32)
    my = (A[1, 0] * xs + A[1, 1] * ys + tg[1]).astype(np.float32)
    return cv2.remap(m_warp, mx, my, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# subspace fit (mirror of the audit runner fit_pca, parameterised by layers/aug)
# ---------------------------------------------------------------------------

def fit_subspace(extractor, train_paths, layers, aug_count, image_res=IMAGE_RES,
                 batch_size=1, ev=PCA_EV):
    import torch
    all_imgs = []
    aug_tf = get_augmentation_transform(["rotate"], image_res)
    for path in train_paths:
        pil = Image.open(path).convert("RGB")
        all_imgs.append(pil)
        for _ in range(aug_count):
            all_imgs.append(aug_tf(pil))
    tmp, (h_p, w_p), _ = extractor.extract_tokens([all_imgs[0]], image_res,
                                                  layers, "mean", docrop=False)
    feat_dim = tmp.shape[-1]
    n = len(all_imgs)
    total = n * h_p * w_p
    nb = math.ceil(n / batch_size)

    def generator():
        for i in range(0, n, batch_size):
            if i > 0 and i % 20 == 0:
                torch.cuda.empty_cache()
            batch = all_imgs[i: i + batch_size]
            tok, _, _ = extractor.extract_tokens(batch, image_res, layers,
                                                 "mean", docrop=False)
            yield tok.reshape(-1, feat_dim)

    pca = PCAModel(k=None, ev=ev, whiten=False)
    pca.dtype = torch.float32
    params = pca.fit(generator, feat_dim, total, nb)
    del all_imgs
    gc.collect()
    return params


def score_grid(extractor, pil, layers, pca_params):
    tok, (h_p, w_p), _ = extractor.extract_tokens([pil], IMAGE_RES, layers,
                                                  "mean", docrop=False)
    scores = calculate_anomaly_scores(tok.reshape(-1, tok.shape[-1]),
                                      pca_params, "reconstruction", 0)
    return np.asarray(scores, dtype=np.float64).reshape(h_p, w_p)


# ---------------------------------------------------------------------------
# pool version construction
# ---------------------------------------------------------------------------

def build_pool(ref_paths: list):
    """Return list of dicts: ref, version, image (672 RGB PIL), geo(A,t|None)."""
    out = []
    for rp in ref_paths:
        base = Image.open(rp).convert("RGB").resize((IMAGE_RES, IMAGE_RES),
                                                    Image.BILINEAR)
        for key, kind, param in POOL_VERSION_SPEC:
            if kind == "id":
                pil = base.copy()
                geo = None
            elif kind == "photo":
                enh = (ImageEnhance.Brightness if key.startswith("brightness")
                       else ImageEnhance.Contrast)(base)
                pil = enh.enhance(float(param))
                geo = None
            else:
                A, t = (rotation_affine(float(param[1]), IMAGE_RES)
                        if param[0] == "r" else translate_affine(param[1], param[2]))
                M = affine_dest2src(A, t)
                warped = cv2.warpAffine(np.asarray(base), M, (IMAGE_RES, IMAGE_RES),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                pil = Image.fromarray(warped)
                geo = (A, t)
            out.append({"ref": Path(rp).name, "version": key,
                        "image": pil, "geo": geo})
    return out


def score_pool(extractor, pool, layers, pca_params):
    """Score every pool version; inverse-warp geometric grids; store 'grid'."""
    for p in pool:
        g = score_grid(extractor, p["image"], layers, pca_params)
        if p["geo"] is not None:
            A, t = p["geo"]
            g = inv_warp_grid(np.asarray(g, dtype=np.float32), A, t)
        p["grid"] = np.asarray(g, dtype=np.float64)


def z_from_pool(score_grids: list) -> list:
    """score_grids: list of 48x48 grids of the pool -> per-image z maps."""
    flat = np.stack(score_grids).reshape(len(score_grids), -1)   # (P,2304)
    P = flat.shape[0]
    sorted_pool = np.sort(flat, axis=0)          # ascending per pixel
    zs = []
    for i in range(P):
        idx = np.searchsorted(sorted_pool, flat[i], side="left")
        count_ge = P - idx
        p = (1.0 + count_ge) / (2.0 + P)
        z = np.clip(-np.log(np.maximum(p, np.finfo(np.float64).tiny)), 0.0, Z_CAP)
        zs.append(z.reshape(GRID, GRID))
    return zs


def percentile_ranks_risk(vals: list) -> list:
    """Percentile rank (0..1) of each value across configs (bigger -> riskier)."""
    vals = np.asarray(vals, dtype=np.float64)
    n = len(vals)
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    for v in np.unique(vals):
        m = float(np.mean(ranks[vals == v]))
        ranks[vals == v] = m
    return (ranks / n).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset-path",
                    default=str(ROOT / "data" / "mpdd_raw" / "MPDD"))
    ap.add_argument("--categories", nargs="+",
                    default=["bracket_black", "bracket_brown", "bracket_white",
                             "connector", "metal_plate", "tubes"])
    ap.add_argument("--shots", nargs="+", type=int, default=list(SHOTS))
    ap.add_argument("--max-configs", type=int, default=0,
                    help="debug cap (0 = all)")
    args = ap.parse_args()

    maps.assert_development_only()
    RELIAB_OUT.mkdir(parents=True, exist_ok=True)
    (RELIAB_OUT / "pools").mkdir(parents=True, exist_ok=True)
    probe_spec = {
        "frozen_addendum": "reliability probe subspaces B/C fit on refs + 10 "
                           "RandomRotation (internal stability probes; official "
                           "maps/identity replay stay on 30 rotations)",
        "official_subspace_a": {"layers": LAYERS_A, "aug_count": OFFICIAL_AUG},
        "probe_b": {"layers": LAYERS_B, "aug_count": PROBE_AUG},
        "probe_c": {"layers": LAYERS_C, "aug_count": PROBE_AUG},
        "pool_versions": [v[0] for v in POOL_VERSION_SPEC],
        "z_cap": Z_CAP,
    }
    PROBE_SPEC.parent.mkdir(parents=True, exist_ok=True)
    PROBE_SPEC.write_text(json.dumps(probe_spec, ensure_ascii=False, indent=1),
                          encoding="utf-8")

    import torch
    extractor = FeatureExtractor(args.model_dir, half=True, need_saliency=False)
    t_all = time.time()
    rows = []
    done = 0
    for cat in sorted(args.categories):
        for shot in sorted(args.shots):
            if args.max_configs and done >= args.max_configs:
                break
            seed_all(SEED)
            os.environ["V4_SMOKE_SEED"] = str(SEED)
            os.environ["V4_SMOKE_SHOT"] = str(shot)
            handler = get_dataset_handler("mpdd", args.dataset_path, cat)
            train_paths = handler.get_train_paths()
            assert len(train_paths) == shot, (cat, shot, len(train_paths))
            t0 = time.time()
            free_g, _ = torch.cuda.mem_get_info()
            print(f"[{cat}] s{SEED}/k{shot} begin  GPU free {free_g/2**20:.0f} MiB",
                  flush=True)

            pool = build_pool(train_paths)

            # ---- subspace A (official layers/policy) ----
            pa = fit_subspace(extractor, train_paths, LAYERS_A, OFFICIAL_AUG)
            score_pool(extractor, pool, LAYERS_A, pa)
            zA = z_from_pool([p["grid"] for p in pool])
            for p, z in zip(pool, zA):
                p["zA"] = z

            # ---- probe subspaces B/C ----
            pb = fit_subspace(extractor, train_paths, LAYERS_B, PROBE_AUG)
            pc = fit_subspace(extractor, train_paths, LAYERS_C, PROBE_AUG)
            score_pool(extractor, pool, LAYERS_B, pb)
            zB = z_from_pool([p["grid"] for p in pool])
            score_pool(extractor, pool, LAYERS_C, pc)
            zC = z_from_pool([p["grid"] for p in pool])
            del pa, pb, pc
            torch.cuda.empty_cache()
            gc.collect()

            # ---- config scalars ----
            n_refs = len(train_paths)
            n_ver = len(POOL_VERSION_SPEC)
            u_aug_vals = []
            for r in range(n_refs):
                zid = zA[r * n_ver]
                for v in range(1, n_ver):
                    u_aug_vals.append(float(np.median(np.abs(
                        zid - zA[r * n_ver + v]))))
            u_layer_vals = []
            for r in range(n_refs):
                u_layer_vals.append(float(np.median(np.abs(
                    zB[r * n_ver] - zC[r * n_ver]))))
            b_tail = float(np.percentile(np.stack(zA), 99))
            u_aug = float(np.mean(u_aug_vals))
            u_layer = float(np.mean(u_layer_vals))

            # ---- save A-pool raw grids (Wave 3 reuse) ----
            np.savez_compressed(
                RELIAB_OUT / "pools" / f"{cat}_s{SEED}_k{shot}.npz",
                sample_ids=np.asarray([str(p["ref"]) for p in pool]),
                version_keys=np.asarray([p["version"] for p in pool]),
                gridA=np.asarray([p["grid"] for p in pool], dtype=np.float16),
                layers=np.asarray(LAYERS_A, dtype=np.int64),
                image_res=np.asarray(IMAGE_RES, dtype=np.int64),
                aug_count_official=np.asarray(OFFICIAL_AUG, dtype=np.int64),
                probe_aug_count=np.asarray(PROBE_AUG, dtype=np.int64),
                pca_ev=np.asarray(PCA_EV, dtype=np.float64),
            )

            rows.append({"category": cat, "seed": SEED, "shot": shot,
                         "n_refs": n_refs, "n_pool": len(pool),
                         "u_aug": round(u_aug, 6),
                         "u_layer": round(u_layer, 6),
                         "b_tail": round(b_tail, 6),
                         "u_aug_raw": [round(x, 6) for x in u_aug_vals],
                         "u_layer_raw": [round(x, 6) for x in u_layer_vals],
                         "elapsed_s": round(time.time() - t0, 1)})
            print(f"[{cat}] s{SEED}/k{shot} done {time.time()-t0:.0f}s  "
                  f"U_aug={u_aug:.4f} U_layer={u_layer:.4f} "
                  f"B_tail={b_tail:.4f}", flush=True)
            del pool, zA, zB, zC
            gc.collect()
            done += 1

    # ---- reliability across configs (percentile ranks over 18 units) ----
    base_keys = ["u_aug", "u_layer", "b_tail"]
    for k in base_keys:
        pr = percentile_ranks_risk([r[k] for r in rows])
        for r, pv in zip(rows, pr):
            r[f"risk_pct_{k}"] = round(pv, 6)
    for r in rows:
        risk = float(np.mean([r[f"risk_pct_{k}"] for k in base_keys]))
        q_sub = 1.0 - risk
        r_sub = float(np.clip((q_sub - 0.25) / 0.50, 0.0, 1.0))
        r["risk_mean"] = round(risk, 6)
        r["q_sub"] = round(q_sub, 6)
        r["r_sub"] = round(r_sub, 6)

    report = {
        "program": "innovation_v6_dgsafe",
        "phase": "Wave2a_reliability_build",
        "dataset": "mpdd", "role": "development", "seed": SEED,
        "task_book_section": "16 ss.2.3-2.4 (frozen formula)",
        "formula": {
            "p_b": "(1 + #{pool >= s}) / (2 + |pool|)",
            "z": "clip(-log p_b, 0, 12)",
            "q_sub": "1 - mean(percentile_rank(U_aug), percentile_rank(U_layer), "
                     "percentile_rank(B_tail))",
            "r_sub": "clip((q_sub - 0.25) / 0.50, 0, 1)",
        },
        "probe_spec": probe_spec,
        "reliability": rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_s": round(time.time() - t_all, 1),
    }
    (RELIAB_OUT / "reliability_raw.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nWave2a reliability done in {time.time()-t_all:.0f}s -> "
          f"{RELIAB_OUT / 'reliability_raw.json'}")
    for r in rows:
        print(f"  {r['category']:16s} k{r['shot']}  U_aug={r['u_aug']:.4f} "
              f"U_layer={r['u_layer']:.4f} B_tail={r['b_tail']:.4f} "
              f"risk={r['risk_mean']:.3f} q_sub={r['q_sub']:.3f} "
              f"r_sub={r['r_sub']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
