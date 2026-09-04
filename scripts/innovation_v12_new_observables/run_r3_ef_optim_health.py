"""V12-EARLY-FUSION SCAIF short optimization health check (doc 26 §3A / §6.2).

Diagnoses why the correction-run interaction pathway decays to ~zero output
(gates ~0.0025, decoders ~1e-14) WITHOUT rerunning the full 6-fold matrix.
Runs a FIXED source-category episode and inspects steps {0, 1, 10, 50}.

Pre-registered (doc 26 §3A: fixed episode, two pre-specified source cats, no
held-out labels touched, cats not chosen by effect):
  cats  = ["connector", "bracket_white"]; K=2 support; q = <=4 def + 2 nrm rows
  steps = [0,1,10,50]; Adam lr 1e-3 wd 1e-3;  patch sampling FIXED once per run.
  run A = CONFIG loss (sparse 1.0); run B = task-only (sparse term OFF) ->
          acceptance: task loss must give finite nonzero gradients to the
          interaction path AND lower the fixed-episode task error WITHOUT the
          sparse term.

doc 26 §3A report items:
  1. per pair/direction ||g*Delta||/(||F||+eps) (pre-L2) + post-L2 row delta
  2. per-term (seg/cp/ap/sparse) grad norms by role {proj,mlp,dec,gate}
  3. Adam coupled decay wd*||theta|| vs raw task-grad magnitude (by role)
  4. fixed-episode seg loss + normal/abnormal sampled-score distributions
  5. interaction ON vs OFF: feature-row diff, score-map diff, pooled AP diff
  6. input scales (branch features, support dist features, intermediate acts)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "innovation_v12_new_observables"))

from scaif_common import ANCHORS, PAIRS, QSUBS, SEED, SCAIF, ccol, deep_rows, load_cat_features  # noqa: E402
from run_r3_ef_scaif import ML_ROOT, gt32  # noqa: E402

CATS = ["connector", "bracket_white"]   # pre-registered, not effect-selected
STEPS = [0, 1, 10, 50]
K = 2
QDEF = 4
QNRM = 2
LR = 1e-3
WD = 1e-3
EPS = 1e-12


# ----------------------------------------------------------------------
# fixed episode
# ----------------------------------------------------------------------

def make_episode(cc, device):
    rng = np.random.default_rng(SEED)
    ndef = min(QDEF, cc["def_rows"].size)
    qidx = np.concatenate([rng.choice(cc["def_rows"], size=ndef, replace=False),
                           rng.choice(cc["norm_rows"], size=QNRM, replace=False)])
    sup = rng.choice(cc["norm_rows"], size=K, replace=False)
    sup_d = [cc["d"][sup][:, di].reshape(-1, 768) for di, _ in PAIRS]
    sup_c = [cc["c"][sup][:, ccol(ci)].reshape(-1, 768) for _, ci in PAIRS]
    if sup_d[0].shape[0] > ANCHORS:           # same anchor subsample as train_step
        pick = rng.choice(sup_d[0].shape[0], size=ANCHORS, replace=False)
        sup_d = [s[pick] for s in sup_d]
        sup_c = [s[pick] for s in sup_c]
    pix = np.stack([rng.choice(1024, size=QSUBS, replace=False) for _ in range(qidx.size)])
    return qidx, sup, sup_d, sup_c, pix


# ----------------------------------------------------------------------
# roles
# ----------------------------------------------------------------------

def role_groups(model):
    groups = {"proj": [], "mlp": [], "dec": [], "gate": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".pd." in name or ".pc." in name:
            groups["proj"].append(p)
        elif ".mlp_" in name:
            groups["mlp"].append(p)
        elif ".wd." in name or ".wc." in name:
            groups["dec"].append(p)
        elif ".gate_" in name:
            groups["gate"].append(p)
    return groups


def _grad_l2(groups):
    out = {}
    for role, ps in groups.items():
        t = 0.0
        for p in ps:
            if p.grad is not None:
                t += float(p.grad.detach().pow(2).sum())
        out[role] = float(np.sqrt(t))
    return out


def _param_l2(groups):
    out = {}
    for role, ps in groups.items():
        t = sum(float(p.detach().pow(2).sum()) for p in ps)
        out[role] = float(np.sqrt(t))
    return out


# ----------------------------------------------------------------------
# episode forward (mirror of run_r3_ef_scaif.train_step, no opt step)
# ----------------------------------------------------------------------

def episode_forward(model, cc, qidx, sup, sup_d, sup_c, pix, device, with_sparse=True):
    """Returns term losses + aux stats. Caller decides backward/step."""
    dq, cq = cc["d"][qidx], cc["c"][qidx]
    fr, _, gs = model.refine(dq, cq, sup_d, sup_c)
    with torch.no_grad():                       # support bank static (registered choice)
        frs, _, _ = model.refine(cc["d"][sup], cc["c"][sup], sup_d, sup_c)
    bank = frs.reshape(-1, frs.shape[-1])

    B = dq.shape[0]
    pt = torch.from_numpy(pix).to(device)
    rows = torch.arange(B, device=device)[:, None]
    qflat = fr.reshape(B, 1024, fr.shape[-1])
    qs = qflat[rows, pt].reshape(-1, fr.shape[-1])
    d2 = torch.cdist(qs, bank).pow(2).min(dim=-1)[0]
    s = d2 / 2.0                                 # larger = more anomalous
    g = gt32(cc["masks"], qidx, device).reshape(B, 1024)[rows, pt].reshape(-1)

    with torch.no_grad():                       # A1 private-stream reference (static path)
        a1qflat = deep_rows(dq, cq).reshape(B, 1024, 1536)
        a1r = deep_rows(cc["d"][sup], cc["c"][sup]).reshape(-1, 1536)
        a1q = a1qflat[rows, pt].reshape(-1, 1536)
        sa1 = (torch.cdist(a1q, a1r).pow(2).min(dim=-1)[0]) / 2.0

    pos = g > 0.5
    neg = g <= 0.5
    wpos = max(1.0, float(neg.sum()) / max(float(pos.sum()), 1.0))
    loss_seg = torch.nn.functional.binary_cross_entropy_with_logits(
        s * 10.0, g, pos_weight=torch.tensor(wpos, device=device))
    loss_ap = (torch.nn.functional.relu(sa1[pos] - s[pos]).mean()
               if pos.sum() > 0 else torch.zeros((), device=device))
    loss_cp = (torch.nn.functional.relu(s[neg] - sa1[neg]).mean()
               if neg.sum() > 0 else torch.zeros((), device=device))
    gcat = torch.cat([gg for gd, gc in gs for gg in (gd, gc)])
    gmean = gcat.abs().mean()
    loss = loss_seg + 0.20 * loss_cp + 0.20 * loss_ap + (1.0 if with_sparse else 0.0) * gmean
    terms = {"seg": loss_seg, "cp": loss_cp, "ap": loss_ap,
             "sparse": gmean, "total": loss}
    stats = {
        "pos_s": (float(s[pos].mean()), float(s[pos].min()), float(s[pos].max())) if pos.sum() > 0 else None,
        "neg_s": (float(s[neg].mean()), float(s[neg].min()), float(s[neg].max())) if neg.sum() > 0 else None,
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
        "wpos": wpos,
    }
    return terms, stats, s, g, bank, fr


# ----------------------------------------------------------------------
# item 1: per pair/direction correction magnitude (exact via block forward)
# ----------------------------------------------------------------------

def correction_stats(model, cc, qidx, sup_d, sup_c):
    out = {}
    dq, cq = cc["d"][qidx], cc["c"][qidx]
    for p, ((di, ci), blk) in enumerate(zip(PAIRS, model.blocks)):
        d_blk = dq[:, di]
        c_blk = cq[:, ccol(ci)]
        with torch.no_grad():
            d_t, c_t, gd, gc = blk(d_blk, c_blk, sup_d[p], sup_c[p])
            del_d = (d_t - d_blk).reshape(d_blk.shape[0], -1)
            del_c = (c_t - c_blk).reshape(c_blk.shape[0], -1)
            f_d = d_blk.reshape(d_blk.shape[0], -1)
            f_c = c_blk.reshape(c_blk.shape[0], -1)
            gd_mean = float(gd.abs().mean())
            gc_mean = float(gc.abs().mean())
        out[f"pair{p}"] = {
            "d_rel_correction": float(del_d.norm() / (f_d.norm() + EPS)),
            "c_rel_correction": float(del_c.norm() / (f_c.norm() + EPS)),
            "gd_mean": gd_mean, "gc_mean": gc_mean,
        }
    return out


# ----------------------------------------------------------------------
# item 6: activation scales
# ----------------------------------------------------------------------

def activation_scales(model, cc, qidx, sup_d, sup_c):
    dq, cq = cc["d"][qidx], cc["c"][qidx]
    out = {}
    for p, ((di, ci), blk) in enumerate(zip(PAIRS, model.blocks)):
        with torch.no_grad():
            d_blk = dq[:, di]
            c_blk = cq[:, ccol(ci)]
            pd_raw = blk.pd(d_blk)
            pc_raw = blk.pc(c_blk)
            ud = F.normalize(pd_raw, dim=-1)
            uc = F.normalize(pc_raw, dim=-1)
            su = F.normalize(blk.pd(sup_d[p]), dim=-1)
            cv = F.normalize(blk.pc(sup_c[p]), dim=-1)
            d_sup = torch.cdist(ud.reshape(-1, blk.u), su.reshape(-1, blk.u)).min(dim=-1)[0]
            c_sup = torch.cdist(uc.reshape(-1, blk.u), cv.reshape(-1, blk.u)).min(dim=-1)[0]
            # cross neighbourhood + mlp input scales
            nc = blk._neigh3(uc)
            nd = blk._neigh3(ud)
            r_cd_in = torch.cat([ud, nc], dim=-1)
            r_dc_in = torch.cat([uc, nd], dim=-1)
            out[f"pair{p}"] = {
                "d_feat_row_l2": float(F.normalize(d_blk, dim=-1).norm() / np.sqrt(d_blk.shape[0])),
                "c_feat_row_l2": float(F.normalize(c_blk, dim=-1).norm() / np.sqrt(c_blk.shape[0])),
                "pd_raw_norm": float(pd_raw.norm()),
                "pc_raw_norm": float(pc_raw.norm()),
                "ud_norm": float(ud.norm()), "uc_norm": float(uc.norm()),
                "su_norm": float(su.norm()), "cv_norm": float(cv.norm()),
                "d_sup_mean": float(d_sup.mean()), "d_sup_max": float(d_sup.max()),
                "c_sup_mean": float(c_sup.mean()), "c_sup_max": float(c_sup.max()),
                "mlp_cd_in_norm": float(r_cd_in.norm()), "mlp_dc_in_norm": float(r_dc_in.norm()),
            }
    return out


# ----------------------------------------------------------------------
# item 5: ON vs OFF (feature rows, score maps, pooled AP) on the episode
# ----------------------------------------------------------------------

def onoff_stats(model, cc, qidx, sup, sup_d, sup_c, device):
    with torch.no_grad():
        dq, cq = cc["d"][qidx], cc["c"][qidx]
        fr_on, f0_on, _ = model.refine(dq, cq, sup_d, sup_c)
        fr_off, f0_off, _ = model.refine(dq, cq, sup_d, sup_c, gate_zero=True)
        bank_on, _, _ = model.refine(cc["d"][sup], cc["c"][sup], sup_d, sup_c)
        bank_off, _, _ = model.refine(cc["d"][sup], cc["c"][sup], sup_d, sup_c, gate_zero=True)
        B = dq.shape[0]
        s_on = (torch.cdist(fr_on.reshape(-1, fr_on.shape[-1]), bank_on.reshape(-1, bank_on.shape[-1]))
                .pow(2).min(dim=-1)[0] / 2.0).reshape(B, 32, 32)
        s_off = (torch.cdist(fr_off.reshape(-1, fr_off.shape[-1]),
                             bank_off.reshape(-1, bank_off.shape[-1]))
                 .pow(2).min(dim=-1)[0] / 2.0).reshape(B, 32, 32)
        g32 = gt32(cc["masks"], qidx, device).reshape(B, 32, 32)
        from sklearn.metrics import average_precision_score

        def _ap(sm):
            y = (g32 > 0.5).cpu().numpy().ravel()
            x = sm.cpu().numpy().ravel()
            return float(average_precision_score(y, x)) if y.sum() not in (0, y.size) else float("nan")

        row_diff = float((fr_on - fr_off).abs().max())
        row_rel = float((fr_on - fr_off).norm() / (fr_off.norm() + EPS))
        map_diff = float((s_on - s_off).abs().max())
        return {"row_abs_max": row_diff, "row_rel": row_rel, "map_abs_max": map_diff,
                "ap_on": _ap(s_on), "ap_off": _ap(s_off),
                "ap_on_minus_off": _ap(s_on) - _ap(s_off)}


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=max(STEPS))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    out_root = ROOT / "experiments/dynamic_fusion/innovation_v12_early_fusion/03_scaif_small_gate"
    result = {"cats": CATS, "steps": STEPS, "k": K, "lr": LR, "wd": WD,
              "device": str(device), "pre_registered_note": "doc26 s3A"}

    for cat in CATS:
        cc = load_cat_features(ML_ROOT, 1, cat, device)
        qidx, sup, sup_d, sup_c, pix = make_episode(cc, device)
        result[cat] = {"n_episode_images": int(qidx.size)}

        for run, with_sparse in (("A_config_sparse_on", True), ("B_task_only_no_sparse", False)):
            torch.manual_seed(SEED)
            model = SCAIF(variant="main").to(device)
            opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                   lr=LR, weight_decay=WD)
            groups = role_groups(model)
            row = {}
            for step in range(0, args.max_steps + 1):
                terms, stats, s, g, bank, fr = episode_forward(
                    model, cc, qidx, sup, sup_d, sup_c, pix, device, with_sparse=with_sparse)
                if step in STEPS:
                    snap = {"step": step}
                    snap["loss"] = float(terms["total"].item())
                    snap["seg"] = float(terms["seg"].item())
                    snap["ap"] = float(terms["ap"].item())
                    snap["cp"] = float(terms["cp"].item())
                    snap["sparse"] = float(terms["sparse"].item())
                    snap["score_pos"] = stats["pos_s"]
                    snap["score_neg"] = stats["neg_s"]
                    snap["n_pos_neg"] = (stats["n_pos"], stats["n_neg"])
                    snap["correction"] = correction_stats(model, cc, qidx, sup_d, sup_c)
                    snap["scales"] = activation_scales(model, cc, qidx, sup_d, sup_c)
                    snap["onoff"] = onoff_stats(model, cc, qidx, sup, sup_d, sup_c, device)
                    # item 2: per-term gradients by role (retain graph between terms)
                    term_grads = {}
                    names = ["seg", "cp", "ap", "sparse"] if with_sparse else ["seg", "cp", "ap"]
                    for tname in names:
                        opt.zero_grad(set_to_none=False)
                        terms[tname].backward(retain_graph=True)
                        term_grads[tname] = _grad_l2(groups)
                    snap["grad_by_term"] = term_grads
                    # item 3: decay vs raw seg-gradient magnitude per role
                    opt.zero_grad(set_to_none=False)
                    terms["seg"].backward(retain_graph=True)
                    raw_seg = _grad_l2(groups)
                    snap["decay_vs_seg_grad"] = {role: {
                        "raw_seg_grad_l2": raw_seg[role],
                        "decay_l2": WD * _param_l2({role: groups[role]})[role]}
                        for role in groups}
                    # ---- actual update on this step using the run's total loss ----
                    opt.zero_grad(set_to_none=True)
                    terms["total"].backward()
                    opt.step()
                    row[step] = snap
                else:
                    opt.zero_grad(set_to_none=True)
                    terms["total"].backward()
                    opt.step()
                del terms, stats, s, g, bank, fr
            result[cat][run] = row
            print(f"[{cat} {run}] step50 seg={row.get(50, {}).get('seg')} "
                  f"gate={row.get(50, {}).get('sparse')} "
                  f"step0_seg={row.get(0, {}).get('seg')}", flush=True)
            del model, opt
            torch.cuda.empty_cache()
        del cc
        torch.cuda.empty_cache()

    out_path = Path(args.out) if args.out else out_root / "RUN_OPTIM_HEALTH.json"
    out_path.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print("WROTE", out_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
