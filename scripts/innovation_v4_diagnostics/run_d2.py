"""D2 — structural-context diagnostic (task book 14 section 10 D2).

On MPDD development only (normal material, no real test masks): real normal-test
fused feature grids are synthetically perturbed (patch permutation / missing
block / duplicate block). We then measure how well three mechanisms localise the
perturbed region:
  - frozen A1 global KNN distance;
  - a non-parametric context-repair residual (predict the masked centre from its
    5x5 ring against the normal-memory ring bank — centre token never fed);
  - a node-only transport cost (set level, no edges; expected chance for
    permutation).

Decision rule (document): prefer RG-MCR when the context residual reaches
AUROC >= 0.80 on all three perturbation types and beats A1 by >= 0.10; only if
RG-MCR fails but node-only OT is clearly effective should RG-OT get a smoke.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v4_diagnostics import common, diagnostics as diag  # noqa: E402

SEED = 0
SHOT = 2          # two references -> leave-room for donor blocks
BS = 4            # perturbation block size (patches)
N_GOOD_MAX = 16   # normal query grids per category
RNG_SEED = 20260902


def category_d2(category: str) -> dict:
    aligned = common.aligned_category("mpdd", SEED, SHOT, category)
    gt = common.evaluator_gt("mpdd", SEED, SHOT, category)
    # normal query grids = test/good images (never bad, never GT)
    good_ids = [i for i, sid in enumerate(aligned.sample_ids) if "/good/" in str(sid)]
    rng_ids = good_ids[:N_GOOD_MAX]
    rng = np.random.default_rng(RNG_SEED + hash(category) % 1000)

    d_all = aligned.d_feat
    c_all = aligned.c_feat
    q = diag.fused_grid(d_all, c_all)              # [N,32,32,1536]
    r = diag.fused_grid(aligned.d_ref, aligned.c_ref)  # [S,32,32,1536]
    good_grids = [q[i] for i in rng_ids]

    # --- perturbation generation
    pert = {t: [] for t in ("permutation", "missing", "duplicate")}
    for gi, g in enumerate(good_grids):
        donor = good_grids[(gi + 1) % len(good_grids)]
        for t in pert:
            pg, m = diag.perturb_structural(g, donor, t, rng, bs=BS)
            md = diag.dilate_mask(m, iters=1)
            pert[t].append((pg, md, int(rng_ids[gi])))

    # --- A1 memory distance (label-free)
    a1_scores = diag.memory_dist1(np.stack([p for t in pert for p, _, _ in pert[t]]),
                                  r)
    off = 0
    a1_by_type = {}
    for t in ("permutation", "missing", "duplicate"):
        n = len(pert[t])
        a1_by_type[t] = a1_scores[off:off + n]
        off += n

    # --- context-repair residual
    ctx_bank = diag.ring_bank_context(r, r_in=1, r_out=2)   # [(S*H*W), 1536]
    center_bank = r.reshape(-1, r.shape[-1])
    ctx_by_type = {}
    for t in ("permutation", "missing", "duplicate"):
        grids = np.stack([p for p, _, _ in pert[t]])
        q_ctx = np.stack([diag.ring_mean(p, 1, 2) for p in grids])
        ctx_by_type[t] = diag.context_residual(grids, q_ctx, ctx_bank, center_bank)

    # --- node-only transport (set-level)
    anchors = diag.coreset_anchors(r, k=64, seed=RNG_SEED)
    good_costs = [diag.sinkhorn_ot_cost(g.reshape(-1, g.shape[-1]), anchors)
                  for g in good_grids]
    nodeot_by_type = {}
    for t in ("permutation", "missing", "duplicate"):
        qs = [p for p, _, _ in pert[t]]
        nodeot_by_type[t] = diag.node_only_ot_auroc(qs, anchors, good_costs)

    # --- patch AUROC per type
    res = {}
    for t in ("permutation", "missing", "duplicate"):
        pos_gt = np.zeros(0, dtype=bool)
        for _, m, _ in pert[t]:
            pos_gt = np.concatenate([pos_gt, m.ravel() > 0])
        neg_gt = ~pos_gt
        a1 = np.concatenate([s.ravel() for s in a1_by_type[t]])
        cx = np.concatenate([s.ravel() for s in ctx_by_type[t]])
        res[t] = {
            "a1_patch_auroc": diag.patch_auroc(a1, pos_gt, neg_gt),
            "context_patch_auroc": diag.patch_auroc(cx, pos_gt, neg_gt),
            "nodeot_image_auroc": nodeot_by_type[t],
            "n_perturbed": len(pert[t]),
        }
    return {
        "dataset": "mpdd", "role": "development", "seed": SEED, "shot": SHOT,
        "category": category, "n_good_used": len(good_grids),
        "perturbation_block_size": BS, "by_type": res,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=common.EXPERIMENT_ROOT / "D2_structure_context")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()
    common.assert_development_only()
    manifest = common.manifest_for("mpdd")
    cats = [args.category] if args.category else sorted(manifest["categories"])
    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reports = []
    for cat in cats:
        print(f"[D2] {cat}", flush=True)
        rep = category_d2(cat)
        reports.append(rep)
        (out_root / f"{cat}_s{SEED}_k{SHOT}.json").write_text(
            json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        for t, v in rep["by_type"].items():
            print(f"     {t}: a1={v['a1_patch_auroc']} "
                  f"context={v['context_patch_auroc']} nodeot={v['nodeot_image_auroc']}",
                  flush=True)
    def mean_of(field, ptype):
        vals = [r["by_type"][ptype][field] for r in reports
                if r["by_type"][ptype][field] is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    summary = {
        "schema_version": 1, "program": "innovation_v4_diagnostics",
        "diagnostic": "D2_structure_context", "dataset": "mpdd", "role": "development",
        "seed": SEED, "shot": SHOT,
        "rule": "prefer RG-MCR if context AUROC >= 0.80 on all three types and "
                "beats A1 by >= 0.10; RG-OT only if context fails but node-OT works",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
        "per_type_mean": {t: {"a1_patch_auroc": mean_of("a1_patch_auroc", t),
                              "context_patch_auroc": mean_of("context_patch_auroc", t),
                              "nodeot_image_auroc": mean_of("nodeot_image_auroc", t)}
                          for t in ("permutation", "missing", "duplicate")},
    }
    (out_root / "D2_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["per_type_mean"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
