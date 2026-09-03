"""V11 Phase 1: expert-pool region-level Oracle / contribution audit (doc 21 s4.6).

Experts (all unified on 56x56 stride-8 grid of the same sample ids, MPDD seed0 k1):
  E0 A1      1-NN fused distance map (dists2map 448 -> stride-8)
  E1 TEXT    raw AnomalyCLIP text-conditioned anomaly maps (v8 text_maps), robust01
  E2 LLSE    top-8 local-linear reconstruction residual (k=8, reg=1e-3)
  E3 CSS     intra-image context self-similarity (24 neighbours r=2)

Oracle: evaluator-side offline partition = GT defect 8-components + background.
Per defect region pick the expert with max mean robust01 score (best-hit); background
picks the expert with min mean robust01 score. Oracle pixel map = chosen expert RAW
map per region. Gates per doc 21 s4.6 (g1 mean dAP>=+0.020, g2 >=4/6 classes >=+0.010,
g3 >=2 non-A1 experts >=10% of oracle-selected defect pixels, g4 LOO drop >=0.003,
g5 headroom not >50% from one class).

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v11_regret_router\\run_r1_expert_oracle_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2
from sklearn.metrics import average_precision_score, roc_auc_score

from industrial_ad.innovation_v10_portfolio import common
from industrial_ad.innovation_v10_portfolio.common import build_fused_blocks, load_features
from industrial_ad.innovation_v8_tcrr_probe.regions import robust01
from src.utils import dists2map

K_NEIGH = 8
REG = 1e-3
RADIUS = 2
CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
EXPERT_NAMES = ["E0_A1", "E1_TEXT", "E2_LLSE", "E3_CSS"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- expert grids

def a1_grid(feat_img: np.ndarray, bank: np.ndarray) -> np.ndarray:
    import faiss
    hh, ww, d = feat_img.shape
    q = feat_img.reshape(-1, d).astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(bank.astype(np.float32))
    dists, _ = index.search(q, k=1)
    return (dists[:, 0] / 2.0).reshape(hh, ww).astype(np.float32)


def _residual_from_nbr(q, nbr, k):
    nd = nbr.astype(np.float64)
    G = nd @ nd.transpose(0, 2, 1) + REG * np.eye(k, dtype=np.float64)
    rhs = np.matmul(nd, q.astype(np.float64)[..., None])[..., 0]
    coef = np.linalg.solve(G, rhs)
    resid2 = np.maximum(1.0 - (coef * rhs).sum(axis=-1), 0.0)
    return np.sqrt(resid2).astype(np.float32)


def llse_grid(feat_img: np.ndarray, bank: np.ndarray) -> np.ndarray:
    import faiss
    hh, ww, d = feat_img.shape
    q = feat_img.reshape(-1, d).astype(np.float32)
    index = faiss.IndexFlatL2(d)
    index.add(bank.astype(np.float32))
    _, idx = index.search(q, k=K_NEIGH)
    nbr = bank[idx.astype(np.int64)].astype(np.float32)  # [Q, k, D]
    resid = _residual_from_nbr(q, nbr, K_NEIGH)
    return resid.reshape(hh, ww)


def css_grid(feat_img: np.ndarray) -> np.ndarray:
    hh, ww, d = feat_img.shape
    pad = np.pad(feat_img, ((RADIUS, RADIUS), (RADIUS, RADIUS), (0, 0)), mode="edge")
    acc = np.zeros((hh, ww), dtype=np.float64)
    n_off = 0
    for dr in range(-RADIUS, RADIUS + 1):
        for dc in range(-RADIUS, RADIUS + 1):
            if dr == 0 and dc == 0:
                continue
            sl = pad[RADIUS + dr:RADIUS + dr + hh, RADIUS + dc:RADIUS + dc + ww]
            acc += 1.0 - np.einsum("hwd,hwd->hw", feat_img.astype(np.float64), sl.astype(np.float64))
            n_off += 1
    return (acc / n_off).astype(np.float32)


def to56(grid: np.ndarray) -> np.ndarray:
    return dists2map(grid, common.MAP_SIZE)[::common.STRIDE, ::common.STRIDE].astype(np.float32)


# ---------------------------------------------------------------- helpers

def align_perm(src_ids: np.ndarray, dst_ids: np.ndarray) -> np.ndarray:
    """perm such that src_ids[perm] == dst_ids (both test-image id string arrays)."""
    pos = {str(v): i for i, v in enumerate(src_ids)}
    return np.asarray([pos[str(v)] for v in dst_ids], dtype=np.int64)


def _pooled_ap_auroc(maps: np.ndarray, masks56: np.ndarray) -> tuple[float, float]:
    y = (masks56.ravel() > 0.5).astype(np.int32)
    s = maps.ravel().astype(np.float64)
    return (float(average_precision_score(y, s)), float(roc_auc_score(y, s)))


# ---------------------------------------------------------------- category audit

def run_category(dino_cache: Path, clip_cache: Path, text_dir: Path, cat: str) -> dict:
    dino = load_features(dino_cache / f"{cat}.npz")
    clip = load_features(clip_cache / f"{cat}.npz")
    feat, ref, sample_ids, masks, grid = build_fused_blocks(dino, clip, dino_weight=0.5)
    n = feat.shape[0]
    bank = ref[0].reshape(-1, feat.shape[-1]).astype(np.float32)
    masks56 = masks[:, ::common.STRIDE, ::common.STRIDE]

    # E1 text maps (aligned to sample_ids); v8 parity: resize 518->448, robust01 at 56
    with np.load(text_dir / f"{cat}.npz", allow_pickle=False) as z:
        text_ids = np.asarray(z["sample_ids"])
        text_maps = np.asarray(z["anomaly_maps"], dtype=np.float32)
    perm = align_perm(text_ids, sample_ids)
    text56 = np.stack([
        robust01(cv2.resize(m, (448, 448), interpolation=cv2.INTER_LINEAR)[::common.STRIDE, ::common.STRIDE])
        for m in text_maps[perm]
    ]).astype(np.float32)

    raw = {name: np.empty((n, 56, 56), dtype=np.float32) for name in EXPERT_NAMES}
    for i in range(n):
        d1g = a1_grid(feat[i], bank)
        raw["E0_A1"][i] = to56(d1g)
        raw["E2_LLSE"][i] = to56(llse_grid(feat[i], bank))
        raw["E3_CSS"][i] = to56(css_grid(feat[i]))
        raw["E1_TEXT"][i] = text56[i]
    norm = {name: np.stack([robust01(m) for m in raw[name]]) for name in EXPERT_NAMES}

    # identity sanity metrics
    a1_ap, a1_auroc = _pooled_ap_auroc(raw["E0_A1"], masks56)
    single_ap = {name: _pooled_ap_auroc(raw[name], masks56)[0] for name in EXPERT_NAMES}

    # per-image region oracle maps
    def build_oracle(exclude: set[str]) -> np.ndarray:
        names = [e for e in EXPERT_NAMES if e not in exclude]
        oracle = np.empty_like(raw["E0_A1"])
        contrib = {e: 0 for e in EXPERT_NAMES if e not in ("E0_A1",)}
        for i in range(n):
            lbl = (masks56[i] > 0.5).astype(np.uint8)
            num, comp = cv2.connectedComponents(lbl, connectivity=8)
            bg = lbl == 0
            # default/fallback expert = A1 for background (doc 21 selective fallback);
            # if A1 excluded (leave-one-out), background falls back to the least
            # background-active remaining expert.
            if "E0_A1" in names:
                bg_e = "E0_A1"
            else:
                bg_e = min(names, key=lambda e: float(norm[e][i][bg].mean()))
            out = raw[bg_e][i].copy()
            for cid in range(1, num):
                cm = comp == cid
                if cm.sum() < 1:
                    continue
                hit = {e: float(norm[e][i][cm].mean()) for e in names}
                best = max(names, key=lambda e: hit[e])
                out[cm] = raw[best][i][cm]
                if best != "E0_A1":
                    contrib[best] += int(cm.sum())
            oracle[i] = out
        return oracle, contrib

    full_oracle, contrib_full = build_oracle(set())
    oracle_ap, _ = _pooled_ap_auroc(full_oracle, masks56)
    loo = {}
    for e in EXPERT_NAMES:
        oap, _ = _pooled_ap_auroc(build_oracle({e})[0], masks56)
        loo[f"without_{e}"] = round(oap, 6)

    sel_total = sum(contrib_full.values())
    contrib_share = {e: (contrib_full[e] / sel_total if sel_total else 0.0) for e in contrib_full}

    return {
        "category": cat,
        "n_images": n,
        "a1_ap": round(a1_ap, 6),
        "a1_auroc": round(a1_auroc, 6),
        "single_ap": {e: round(v, 6) for e, v in single_ap.items()},
        "oracle_ap": round(oracle_ap, 6),
        "delta_oracle": round(oracle_ap - a1_ap, 6),
        "loo_oracle_ap": {k: v for k, v in loo.items()},
        "oracle_selected_defect_pixels": {e: int(contrib_full[e]) for e in contrib_full},
        "contrib_share": {e: round(v, 6) for e, v in contrib_share.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-cache", type=Path, default=None)
    parser.add_argument("--clip-cache", type=Path, default=None)
    parser.add_argument("--text-dir", type=Path,
                        default=ROOT / "outputs/dynamic_fusion/innovation_v8_tcrr_probe/text_maps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v11_regret_router/oracle_audit")
    args = parser.parse_args()

    base = ROOT / "outputs/dynamic_fusion/v3_direction_a"
    dino_cache = args.dino_cache or base / f"features_vitb14_s{args.seed}_k{args.shot}/anomalydino_visual"
    clip_cache = args.clip_cache or base / f"features_s{args.seed}_k{args.shot}/anomalyclip_text"

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    rows = []
    t0 = time.time()
    for cat in CATEGORIES:
        if not (dino_cache / f"{cat}.npz").is_file():
            print(f"[skip] {cat}: cache missing", flush=True)
            continue
        r = run_category(dino_cache, clip_cache, args.text_dir, cat)
        rows.append(r)
        print(f"[{cat}] A1 AP={r['a1_ap']:.4f} Oracle AP={r['oracle_ap']:.4f} "
              f"Δ={r['delta_oracle']:+.4f} | singles "
              + " ".join(f"{e.split('_')[0]}={v:.4f}" for e, v in r["single_ap"].items())
              + f" | contrib " + " ".join(f"{e.split('_')[0]}={v*100:.0f}%"
                                          for e, v in r["contrib_share"].items()), flush=True)

    def mean(k): return float(np.mean([r[k] for r in rows]))
    mean_delta = mean("delta_oracle")
    mean_delta_loo = {
        f"without_{e}": round(float(np.mean([r["loo_oracle_ap"][f"without_{e}"] - r["a1_ap"]
                                             for r in rows])), 6)
        for e in EXPERT_NAMES
    }
    loo_drop = {f"without_{e}": round(mean_delta - mean_delta_loo[f"without_{e}"], 6)
                for e in EXPERT_NAMES}
    classes_gt_010 = sum(1 for r in rows if r["delta_oracle"] >= 0.010)
    n_gt_020 = sum(1 for r in rows if r["delta_oracle"] >= 0.020)

    # g5 class dominance over total positive headroom
    pos_head = sum(max(r["delta_oracle"], 0.0) for r in rows)
    top_class_share = max(max(r["delta_oracle"], 0.0) / pos_head for r in rows) if pos_head > 0 else 1.0

    # g3 aggregated contribution shares over all classes
    agg_contrib = {e: sum(r["oracle_selected_defect_pixels"][e] for r in rows) for e in
                   ["E1_TEXT", "E2_LLSE", "E3_CSS"]}
    agg_sel = sum(agg_contrib.values())
    agg_share = {e: (v / agg_sel if agg_sel else 0.0) for e, v in agg_contrib.items()}
    n_experts_ge_10 = sum(1 for v in agg_share.values() if v >= 0.10)

    report = {
        "route": "V11-ORACLE",
        "pipeline": "v11_expert_oracle_audit",
        "seed": args.seed, "shot": args.shot,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "per_category": rows,
        "mean_a1_ap": round(mean("a1_ap"), 6),
        "mean_oracle_ap": round(mean("oracle_ap"), 6),
        "mean_delta_oracle": round(mean_delta, 6),
        "n_classes_headroom_ge_010": classes_gt_010,
        "n_classes_headroom_ge_020": n_gt_020,
        "agg_contrib_share": {k: round(v, 6) for k, v in agg_share.items()},
        "n_experts_contrib_ge_10pct": n_experts_ge_10,
        "loo_drop_mean_delta": loo_drop,
        "top_class_share_of_positive_headroom": round(top_class_share, 6),
        "gates": {
            "g0_identity": round(abs(mean("a1_ap") - 0.3092), 7),
            "g1_headroom_ge_020": round(mean_delta, 6) >= 0.020,
            "g2_4of6_classes_ge_010": classes_gt_010 >= 4,
            "g3_ge2_nonA1_ge_10pct": n_experts_ge_10 >= 2,
            "g4_loo_drop_ge_0003": all(v >= 0.003 for v in loo_drop.values()),
            "g5_single_class_le_50pct": top_class_share <= 0.50,
        },
        "decision": "PENDING",
    }
    (out_dir / "R0_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmean A1 AP={report['mean_a1_ap']} Oracle AP={report['mean_oracle_ap']} "
          f"Δ={report['mean_delta_oracle']:+.4f} | classes≥0.01: {classes_gt_010}/6, ≥0.02: {n_gt_020}/6 "
          f"| contrib≥10% experts: {n_experts_ge_10} {agg_share} | LOO drop {loo_drop} "
          f"| top-class share {top_class_share:.2f} | gates {report['gates']} | "
          f"elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
