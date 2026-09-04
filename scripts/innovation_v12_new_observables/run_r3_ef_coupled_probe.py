"""Doc 26 §4.3 coupled-recovery diagnostic (bracket_white + tubes, shot k1).

6-cat premise outcome (RECOVERY_RESULTS.json): au56 (independent AnyUp recover-
first) is NOT universal - macro-6 au56 0.2775 < a1 0.3099 < bl56 0.3186; severe
collapses on bracket_white (0.003 vs bl56 0.089) and tubes (0.519 vs a1 0.657).
Only metal_plate (+0.014) carries the 2-cat pass.

This diagnostic tests the doc 26 §4.3 core claim on the TWO COLLAPSE classes:
does CROSS-BRANCH CONDITIONED recovery (each branch's AnyUp input carries the
other branch's deepest-layer features) stabilise recovery where independent
AnyUp collapsed? If yes, coupling genuinely constrains spurious recovery and
the route has a mechanism to investigate on all 6 cats; if not, the collapse is
intrinsic to learned recovery on these features and the route archives.

Arms (all at shared 56 grid; same z-mean integration recipe as au56/a1):
  du56  : dual-conditioned recovery. For dino layer l: AnyUp(img, [dino_l ; clip_L24])
          take channel-half0 as dino_l^56; for clip layer l: AnyUp(img, [clip_l ; dino_L11])
          take half0 as clip_l^56. Split then per-branch per-layer z -> 7-layer mean.
  cu56  : SAME forward maps, but KNN on the whole recovered concat (1536-d) per pair,
          z per pair -> 7-pair mean. (control #6: concat feature, same upsampler).
  du56_m: du56 with cross-branch conditioning features taken from a DIFFERENT query
          image (content misalignment) - must drop vs du56 if conditioning informative.
Pre-registered (macro over bracket_white + tubes):
  M1c  : du56 - au56  >= +0.003   (coupling rescues collapse, vs archived au56)
  M2   : du56 - cu56  >= +0.003   (conditioning structure > mere concat availability)
  M3   : du56 - du56_m >= +0.003  (cross-branch conditioning genuinely informative)
  FP   : FP95(du56) <= 1.05 x FP95(cu56) on normal (defect-free) images
         FP95 = mean over normal images of per-image 95th-pct of the 56-map.
If M1c fails => collapse intrinsic, route archives negative (no 6-cat coupled run).
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

import torch  # noqa: E402
from src.utils import dists2map  # noqa: E402

from run_r3_ef_recovery_probe import (  # noqa: E402
    _bilinear, _guide_tensor, _l2, _load_cat, _load_rgb, _loo_support_stats,
    _make_anyup, _nearest_dist, _pooled_ap, DATA_ROOT, GRID_56)

COUPLE_CATS = ["bracket_white", "tubes"]   # the two au56-collapse classes
D_BRANCH_GRID, C_BRANCH_GRID = 32, 37
D_GUIDE, C_GUIDE = 448, 518
ND_D, ND_C = 3, 4                 # dino L{6,9,11}, clip L{6,12,18,24}


def _pair_inputs(dino_row, clip_row, dino_mis=None, clip_mis=None):
    """Concat inputs for ONE image.

    dino_row: [3,32,32,768] (dino L6/L9/L11), clip_row: [4,37,37,768] (L6/12/18/24).
    dino-conditioned pairs: concat[dino_li ; clip_L24@32]  (clip_mis if misaligned)
    clip-conditioned pairs: concat[clip_lj ; dino_L11@37]  (dino_mis if misaligned)
    Returns (dino_pairs, clip_pairs): lists of [h,w,1536].
    """
    ccond = clip_mis if clip_mis is not None else clip_row
    dcond = dino_mis if dino_mis is not None else dino_row
    dino_pairs = []
    for li in range(ND_D):
        dl = dino_row[li]                                       # [32,32,768]
        cc = _bilinear(ccond[ND_C - 1], D_BRANCH_GRID)          # clip L24 -> 32
        dino_pairs.append(np.concatenate([dl, cc], axis=-1))    # [32,32,1536]
    clip_pairs = []
    for lj in range(ND_C):
        cl = clip_row[lj]                                       # [37,37,768]
        dd = _bilinear(dcond[ND_D - 1], C_BRANCH_GRID)          # dino L11 -> 37
        clip_pairs.append(np.concatenate([cl, dd], axis=-1))    # [37,37,1536]
    return dino_pairs, clip_pairs


def run_coupled_cat(cat, shot, mode, device="cuda:0"):
    dino, clip, samples, refs_rel = _load_cat(cat, shot)
    n = dino["feat"].shape[0]
    K = dino["ref"].shape[1]
    m56 = (dino["masks"][:, ::8, ::8] > 0.5).astype(np.uint8)
    normal_rows = np.where(dino["masks"].reshape(n, -1).sum(1) == 0)[0]
    model = _make_anyup(device)

    def recover1536(feat1536, rgb, guide):
        f = torch.from_numpy(np.ascontiguousarray(feat1536, dtype=np.float32))
        t = f.permute(2, 0, 1)[None].to(device)          # [1,1536,h,w]
        g = _guide_tensor(rgb, guide).to(device)
        with torch.inference_mode():
            out = model(g, t, output_size=(GRID_56, GRID_56), q_chunk_size=16)
        return out[0].permute(1, 2, 0).float().cpu().numpy()   # [56,56,1536]

    def ref_pairs():
        # K==1 ref image (row index 0 of the ref layer arrays)
        return _pair_inputs(dino["ref"][:, 0], clip["ref"][:, 0])

    # ---- support: recover ref concat features for all 7 pairs ----
    rgb0 = _load_rgb(DATA_ROOT / refs_rel[0])
    dp0, cp0 = ref_pairs()
    ref_rec = {"d": [], "c": []}
    for li, f in enumerate(dp0):
        ref_rec["d"].append(recover1536(f, rgb0, D_GUIDE))
    for lj, f in enumerate(cp0):
        ref_rec["c"].append(recover1536(f, rgb0, C_GUIDE))

    # ---- banks & z-stats ----
    radius = 2
    banks_du, banks_cu = {"d": [], "c": []}, {"d": [], "c": []}
    # du: split halves; cu: whole concat
    for bi in ("d", "c"):
        for rec in ref_rec[bi]:
            r56 = rec                       # [56,56,1536]
            half = r56[..., :768] if bi == "d" else r56[..., :768]
            banks_du[bi].append({"bank": _l2(half).reshape(-1, 768),
                                 **dict(zip(("mean", "std"),
                                            _loo_support_stats(half, radius)))})
            banks_cu[bi].append({"bank": _l2(r56).reshape(-1, 1536),
                                 **dict(zip(("mean", "std"),
                                            _loo_support_stats(r56, radius)))})

    maps = {"du56": np.zeros((n, GRID_56, GRID_56)),
            "cu56": np.zeros((n, GRID_56, GRID_56))}
    for i in range(n):
        mis_row = (i + 1) % n
        dino_mis = dino["feat"][mis_row] if mode == "mis" else None
        clip_mis = clip["feat"][mis_row] if mode == "mis" else None
        dp, cp = _pair_inputs(dino["feat"][i], clip["feat"][i], dino_mis, clip_mis)
        rgb_i = _load_rgb(samples[i].image_path)
        z_du, z_cu = [], []
        for li, f in enumerate(dp):           # dino-conditioned pairs (guide 448)
            rec = recover1536(f, rgb_i, D_GUIDE)
            half = rec[..., :768]
            e = banks_du["d"][li]
            du_d = _nearest_dist(_l2(half).reshape(-1, 768), e["bank"]).reshape(GRID_56, GRID_56)
            z_du.append((du_d - e["mean"]) / e["std"])
            e = banks_cu["d"][li]
            cu_d = _nearest_dist(_l2(rec).reshape(-1, 1536), e["bank"]).reshape(GRID_56, GRID_56)
            z_cu.append((cu_d - e["mean"]) / e["std"])
        for lj, f in enumerate(cp):           # clip-conditioned pairs (guide 518)
            rec = recover1536(f, rgb_i, C_GUIDE)
            half = rec[..., :768]
            e = banks_du["c"][lj]
            du_d = _nearest_dist(_l2(half).reshape(-1, 768), e["bank"]).reshape(GRID_56, GRID_56)
            z_du.append((du_d - e["mean"]) / e["std"])
            e = banks_cu["c"][lj]
            cu_d = _nearest_dist(_l2(rec).reshape(-1, 1536), e["bank"]).reshape(GRID_56, GRID_56)
            z_cu.append((cu_d - e["mean"]) / e["std"])
        maps["du56"][i] = np.mean(z_du, axis=0)
        maps["cu56"][i] = np.mean(z_cu, axis=0)
    if model is not None:
        del model
        torch.cuda.empty_cache()

    out = {"category": cat, "n": n, "mode": mode}
    for arm in ("du56", "cu56"):
        m = np.stack([dists2map(maps[arm][i], (448, 448))[::8, ::8] for i in range(n)])
        out[arm] = _pooled_ap(m, m56)
        if len(normal_rows):
            p95 = [np.percentile(maps[arm][j], 95) for j in normal_rows]
            out[f"{arm}_fp95"] = float(np.mean(p95))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("self", "mis"), default="self")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cats", default=None)
    args = ap.parse_args()

    out_root = ROOT / "experiments/dynamic_fusion/innovation_v12_new_observables/detail_recovery"
    out_root.mkdir(parents=True, exist_ok=True)
    res_path = out_root / "COUPLED_RESULTS.json"
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else COUPLE_CATS
    tag = "self" if args.mode == "self" else "mis"
    rows = res.get(tag, [])
    done = {r["category"] for r in rows}
    for cat in cats:
        if cat in done:
            continue
        rows.append(run_coupled_cat(cat, 1, args.mode, args.device))
    res[tag] = sorted(rows, key=lambda r: r["category"])
    res_path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, default=float)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
