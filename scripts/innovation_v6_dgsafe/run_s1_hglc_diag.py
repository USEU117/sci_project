"""S1-HGLC (task book 16 s.3.3 items 1-2) - image-level diagnostic.

Compares, per MPDD (category, seed=0, shot in {1,2,4}), the Image-AP / AUROC of:

  A1-max    : max over the frozen A1 concat 448 map (per-shot map)
  A1-top1%  : mean of the top-1% pixels of the same map
  DINO-CLS  : vit-B/14 CLS @448 cosine distance to nearest k-shot normal ref
  CLIP-glob : AnomalyCLIP ViT-L/14@336 global embedding @518 cosine distance
              to nearest k-shot normal ref (stage-1 cache)
  TEXT      : AnomalyCLIP abnormal-class text probability (stage-1 cache,
              zero-shot, config-independent)

Labels are evaluator-side (derived from "/good/" in the frozen sample id).
GT masks are never read here (only image-level metrics in this gate).

Gate (16 s.3.3 item 4): a single global score must beat A1-max by a *pooled*
mean Image-AP margin of >= +0.010 for S1-HGLC image-level part to proceed
(P5-A-lite already archived DINO CLS at -0.068 on this protocol).

Pooled = mean over categories of the per-category mean over shots (the frozen
P5-A-lite convention).  All development role (MPDD); external sets untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
METHOD_DINO = ROOT / "methods" / "anomalydino"
sys.path.insert(0, str(METHOD_DINO))

from src.backbones import get_model  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402

OUT = maps.EXPERIMENT_ROOT / "s1_hglc"
CACHE = OUT / "cache"
A1_MAPS = maps.A1_MAPS_ROOT
MPDD = maps.MPDD_DATA_ROOT
MANIFEST = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
RES = 448
TOP1_FRAC = 0.01
SIGNALS = ("a1_max", "a1_top1", "dino_cls", "clip_global", "text")


def load_cls(model, path) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    tensor, _ = model.prepare_image(img)
    with torch.inference_mode():
        x = tensor.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        out = model.model.forward_features(x)
    cls = out["x_norm_clstoken"].squeeze().float().cpu().numpy()
    return cls.astype(np.float64)


def cos_dist(test: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """1 - max cosine similarity to refs; test (N,D), refs (M,D)."""
    tn = test / (np.linalg.norm(test, axis=1, keepdims=True) + 1e-12)
    rn = refs / (np.linalg.norm(refs, axis=1, keepdims=True) + 1e-12)
    return 1.0 - (tn @ rn.T).max(axis=1)


def top1_mean(maps448: np.ndarray, frac: float = TOP1_FRAC) -> np.ndarray:
    flat = np.sort(maps448.reshape(len(maps448), -1), axis=1)
    k = max(1, int(round(frac * flat.shape[1])))
    return flat[:, -k:].mean(axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-dino-cls", action="store_true",
                    help="use archived P5-A-lite DINO CLS rows instead of recomputing")
    args = ap.parse_args()
    maps.assert_development_only()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    a1_export = json.loads((CACHE / "s1_hglc_export_report.json").read_text(
        encoding="utf-8"))
    cats = sorted(c["category"] for c in a1_export["categories"])
    rows = []

    dino_model = None if args.skip_dino_cls else get_model(
        "dinov2_vitb14", args.device, smaller_edge_size=RES)

    t_all = time.time()
    for cat in cats:
        # cache (test clip_global + text) -- order = A1 s0_k1 order
        cache = np.load(CACHE / f"{cat}.npz", allow_pickle=False)
        cache_ids = [str(s) for s in cache["sample_ids"]]
        clip_test = np.asarray(cache["clip_global_test"], dtype=np.float64)
        text_p = np.asarray(cache["text_prob_test"], dtype=np.float64)
        ref_ids = [str(s) for s in cache["ref_ids"]]
        clip_ref_map = {r: np.asarray(cache["clip_global_refs"], dtype=np.float64)[j]
                        for j, r in enumerate(ref_ids)}

        # DINO CLS test embeddings (config-independent; recompute once per image)
        dino_test = None
        if not args.skip_dino_cls:
            dino_test = np.zeros((len(cache_ids), 768), dtype=np.float64)
            for i, sid in enumerate(cache_ids):
                dino_test[i] = load_cls(dino_model, MPDD / sid)

        # reference unions per shot for DINO CLS
        ref_cls_cache = {}

        for shot in args.shots:
            npz_a1 = maps.load_a1_patch_map(cat, args.seed, shot)
            a1_ids = [str(s) for s in npz_a1["sample_ids"]]
            perm = maps.align_perm(np.asarray(cache_ids), np.asarray(a1_ids))
            n = len(a1_ids)
            a1m = maps.a1_maps448(npz_a1["patch_map"])          # (N,448,448)
            labels = np.asarray([1 if "/good/" not in s else 0 for s in a1_ids])
            refs = manifest["categories"][cat][str(args.seed)][str(shot)]
            clip_refs_shot = np.stack([clip_ref_map[r] for r in refs])
            scores = {
                "a1_max": np.asarray(a1m.reshape(n, -1).max(axis=1), dtype=np.float64),
                "a1_top1": top1_mean(a1m),
                "clip_global": cos_dist(clip_test[perm], clip_refs_shot),
                "text": text_p[perm],
            }
            if not args.skip_dino_cls:
                dino_refs = np.stack([ref_cls_cache.setdefault(
                    r, load_cls(dino_model, MPDD / r)) for r in refs])
                scores["dino_cls"] = cos_dist(dino_test[perm], dino_refs)

            rec = {"category": cat, "shot": shot, "n_test": n,
                   "n_pos": int(labels.sum())}
            for sig in SIGNALS:
                if sig not in scores:
                    continue
                y, s = labels, scores[sig]
                rec[f"ap_{sig}"] = round(float(average_precision_score(y, s)), 4)
                rec[f"auroc_{sig}"] = round(float(roc_auc_score(y, s)), 4)
                rec[f"dap_{sig}"] = round(float(average_precision_score(y, s))
                                          - rec["ap_a1_max"], 4)
            rows.append(rec)
            print(f"[{cat}] k{shot} " + " ".join(
                f"{s}={rec.get(f'ap_{s}')}" for s in ("a1_max", "a1_top1", "dino_cls",
                                                      "clip_global", "text")
                if f"ap_{s}" in rec), flush=True)

    # ---- pooled (P5-A-lite convention): mean over cats of per-cat shot mean ----
    per_cat = {}
    for cat in cats:
        rr = [r for r in rows if r["category"] == cat]
        per_cat[cat] = {}
        for sig in SIGNALS:
            vals = [r[f"ap_{sig}"] for r in rr if f"ap_{sig}" in r]
            if vals:
                per_cat[cat][f"ap_{sig}"] = round(float(np.mean(vals)), 4)
                per_cat[cat][f"auroc_{sig}"] = round(
                    float(np.mean([r[f"auroc_{sig}"] for r in rr])), 4)

    pooled = {}
    for sig in SIGNALS:
        av = [per_cat[c][f"ap_{sig}"] for c in cats if f"ap_{sig}" in per_cat[c]]
        if av:
            pooled[f"ap_{sig}"] = round(float(np.mean(av)), 4)
    for sig in SIGNALS:
        if f"ap_{sig}" in pooled:
            pooled[f"dap_{sig}_vs_a1_max"] = round(pooled[f"ap_{sig}"] - pooled["ap_a1_max"], 4)
    gate_candidates = {s: pooled[f"dap_{s}_vs_a1_max"] for s in SIGNALS
                       if f"dap_{s}_vs_a1_max" in pooled}
    best = max(gate_candidates, key=gate_candidates.get)
    gate_pass = gate_candidates[best] >= 0.010

    verdict = {
        "pooled_image_ap": pooled,
        "gate_delta_vs_a1_max": gate_candidates,
        "best_global_signal": best,
        "s1_gate_any_global_ge_a1_plus_001": bool(gate_pass),
        "note": "doc 16 s.3.3 item 4: single global score pooled Image-AP must "
                "exceed A1-max by >= +0.010 to proceed to calibration (item 3)",
    }
    report = {
        "program": "innovation_v6_dgsafe", "phase": "s1_hglc_image_level_diag",
        "dataset": "mpdd", "role": "development", "seed": args.seed,
        "task_book_section": "16 s.3.3 (items 1-2, gate item 4)",
        "dino_cls_mode": "recomputed" if not args.skip_dino_cls else "archived_p5a",
        "signals": {"a1_max": "max of frozen A1 concat 448 map",
                    "a1_top1": "mean of top-1% pixels of A1 map",
                    "dino_cls": "DINO v2-B/14 CLS @448, 1-maxcos to k refs",
                    "clip_global": "AnomalyCLIP ViT-L/14@336 global @518, "
                                   "1-maxcos to k refs",
                    "text": "AnomalyCLIP abnormal text probability (zero-shot)"},
        "identity_note": "A1 maps are the frozen artifacts (no patch re-export), so "
                         "16 s.3.3 item 1 (identity replay) does not apply; only new "
                         "image-level signals were exported (s1_hglc_export).",
        "export_report": {k: a1_export.get(k) for k in (
            "checkpoint_sha256", "text_embedding_sha256", "text_swapped",
            "image_size", "design")},
        "rows": rows,
        "per_category": per_cat,
        "verdict": verdict,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_s": round(time.time() - t_all, 1),
    }
    (OUT / "S1_HGLC_DIAG.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [
        "# S1-HGLC image-level diagnostic (doc 16 s.3.3)",
        "",
        "Pooled Image-AP (mean over categories of per-cat mean over shots):",
        f"- A1-max : {pooled.get('ap_a1_max')}",
        f"- A1-top1: {pooled.get('ap_a1_top1')}",
        f"- DINO CLS: {pooled.get('ap_dino_cls')}  "
        f"(delta {pooled.get('dap_dino_cls_vs_a1_max')})",
        f"- CLIP glob: {pooled.get('ap_clip_global')}  "
        f"(delta {pooled.get('dap_clip_global_vs_a1_max')})",
        f"- TEXT: {pooled.get('ap_text')}  "
        f"(delta {pooled.get('dap_text_vs_a1_max')})",
        "",
        f"- best global: {best} (delta {gate_candidates[best]:+.4f})",
        f"- gate (any global >= A1 + 0.010): {gate_pass}",
        "",
        "Details: S1_HGLC_DIAG.json",
    ]
    (OUT / "S1_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
