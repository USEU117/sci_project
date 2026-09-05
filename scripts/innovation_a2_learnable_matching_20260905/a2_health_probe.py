"""A2 health probe: learnable diagonal reweighting, optimization sanity (doc28 s8.2 #6).

Goal (BEFORE any real gate): on SUPPORT-only synthetic episodes, does gradient
optimisation of a bounded per-branch diagonal reweighting w=(w_D,w_C) actually
(1) improve HELD-OUT family defect ranking (defect cells above clean/nuisance in
    A1 fused min-distance) and
(2) keep the normal path stable (clean p95 / nuisance image-max scores do not
    inflate vs the identity A1 forward)?

Data: reuses the v14 support caches (outputs/dynamic_fusion/v14_p1_support_*),
which were rendered on manifest support images at 1024 and re-encoded through the
frozen DINO/CLIP extractors (no /test/ touched). No real defect is read.

Score = A1 map: z = L2([0.5*rowL2norm(w_D*d), 0.5*rowL2norm(w_C*c)]) (1536-D);
       s(cell) = 1 - max_cos(z_cell, Z_memory), memory = other (K-1) clean images.
w bounds [0, U], init 1 (identity -> exactly the A1 concat protocol on LOO memory).
Loss (L-A, margin hinge, frozen probe default):
  L = mean_{p,n} max(0, gamma + s_n - s_p) + lam_w * sum||w - 1||^2
positives: mask cells of the current image's training-family episodes;
negatives: that image's clean cells (excluding any cell positive in its training
episodes) + photometric nuisance cells.

Fold structure (per cat x shot): leave-one-family-out over the 3 synthetic kinds;
train on the other 2 kinds (K images, leave-one-image-out memory), evaluate the
held-out kind episodes. Probe gates (pre-registered defaults, NOT tuned on result):
  H1: held-family AP(trained) - AP(identity) >= +0.005 on >=2/3 evaluable folds;
  H2: normal path: macro relative rise of clean-score p95 and nuisance image-max
      p95 vs identity <= +10% (and clean p95 absolute drift <= +0.05).
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    sys.path.insert(0, p)
sys.path.insert(0, str(ROOT / "scripts" / "innovation_v14_decisive_validation_20260905"))

import cv2  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from industrial_ad.innovation_v10_portfolio.common import resize_patches  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
CACHE = ROOT / "outputs/dynamic_fusion/v14_p1_support"
OUT = ROOT / "experiments/dynamic_fusion/innovation_a2_learnable_matching_20260905"
SYN = ["cutpaste", "local_erasure", "thin_scratch"]
NCELL = 32 * 32
D = 768

# ---- frozen probe defaults (pre-registered, not tuned on results) ----
UPPER = 1.5          # weight bound w in [0, UPPER]
GAMMA = 0.05         # hinge margin on 1-cos scores
LAM_W = 1e-3         # shrink-to-identity regulariser
LR = 1e-3
STEPS = 80           # doc28 s8.2 #6: 50-100 step window
MAX_POS = 64         # positives per training episode
MAX_NEG_CLEAN = 256  # negatives from clean cells
NUI_NEG_IMGS = 2     # nuisance images sampled as negatives per step
NUI_NEG_CELLS = 64   # cells per sampled nuisance image
SEED0 = 20260905

torch.set_num_threads(max(1, torch.get_num_threads()))


def _load(cat, shot, branch):
    z = np.load(CACHE / f"v14_p1_support_{branch}_s0_k{shot}" / f"{cat}.npz", allow_pickle=False)
    return (np.asarray(z["clean_feat"]), np.asarray(z["syn_feat"]),
            np.asarray(z["syn_masks"]), np.asarray(z["nui_feat"]))


def _mask32(m1024, g=32):
    m = cv2.resize(m1024.astype(np.uint8), (g, g), interpolation=cv2.INTER_AREA)
    return (m > 127).astype(np.float32)


def _load_cell(cat, shot):
    cd, sd, md, nd = _load(cat, shot, "dino")     # [K,32,32,768] / [K,9,..] / masks / nui
    cc, sc, _mc, nc = _load(cat, shot, "clip")    # [K,37,37,768] -> 32
    cc = resize_patches(cc.reshape(-1, *cc.shape[1:]), (32, 32)).reshape(cc.shape[0], 32, 32, -1)
    sc = resize_patches(sc.reshape(-1, *sc.shape[2:]), (32, 32)).reshape(*sc.shape[:2], 32, 32, -1)
    nc = resize_patches(nc.reshape(-1, *nc.shape[2:]), (32, 32))
    K = cd.shape[0]
    nc = nc.reshape(K, 15, 32, 32, -1)
    masks = np.asarray([[_mask32(md[h, e], 32) for e in range(9)] for h in range(K)])  # [K,9,32,32]
    return cd, sd, masks, nd, cc, sc, nc, K


def _row_norm_t(x: torch.Tensor):
    return x / torch.clamp(x.norm(dim=1, keepdim=True), min=1e-12)


def _fused(feat_d: torch.Tensor, feat_c: torch.Tensor, wD: torch.Tensor, wC: torch.Tensor):
    d = _row_norm_t(wD[None, :] * feat_d)
    c = _row_norm_t(wC[None, :] * feat_c)
    z = torch.cat([0.5 * d, 0.5 * c], dim=1)
    return _row_norm_t(z)


def _scores(qd, qc, bd, bc, wD, wC):
    """Per-cell A1 score 1-max_cos against fused bank (torch, differentiable)."""
    qz = _fused(qd, qc, wD, wC)
    bz = _fused(bd, bc, wD, wC)
    cos = qz @ bz.T
    return 1.0 - cos.max(dim=1).values


def _ap(y, s):
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y.astype(int), s))


def _eval_ap_h(h, e, arrs, wD_n, wC_n, bank_idx):
    cd, sd, masks, nd, cc, sc, nc, K = arrs
    bd = torch.tensor(np.concatenate([cd[k].reshape(-1, D) for k in bank_idx]))
    bc = torch.tensor(np.concatenate([cc[k].reshape(-1, D) for k in bank_idx]))
    y = masks[h, e].ravel() > 0
    with torch.no_grad():
        s = _scores(torch.tensor(sd[h, e].reshape(-1, D)), torch.tensor(sc[h, e].reshape(-1, D)),
                    bd, bc, torch.tensor(wD_n), torch.tensor(wC_n))
    return _ap(y, s.numpy())


def _normal_stats_h(h, arrs, wD_n, wC_n, bank_idx):
    cd, sd, masks, nd, cc, sc, nc, K = arrs
    bd = torch.tensor(np.concatenate([cd[k].reshape(-1, D) for k in bank_idx]))
    bc = torch.tensor(np.concatenate([cc[k].reshape(-1, D) for k in bank_idx]))
    wD_t, wC_t = torch.tensor(wD_n), torch.tensor(wC_n)
    with torch.no_grad():
        s_clean = _scores(torch.tensor(cd[h].reshape(-1, D)), torch.tensor(cc[h].reshape(-1, D)),
                          bd, bc, wD_t, wC_t)
        clean_p95 = float(np.percentile(s_clean.numpy(), 95))
        imax = []
        for e in range(15):
            s_n = _scores(torch.tensor(nd[h, e].reshape(-1, D)),
                          torch.tensor(nc[h, e].reshape(-1, D)), bd, bc, wD_t, wC_t)
            imax.append(float(s_n.max().item()))
    return clean_p95, float(np.percentile(imax, 95))


def _train_fold(cat, shot, fi, rng, loss="lb"):
    cd, sd, masks, nd, cc, sc, nc, K = _load_cell(cat, shot)
    keep_idx = [k * 3 + s for k in range(3) if k != fi for s in range(3)]  # training eps
    held_idx = [e for e in range(9) if e // 3 == fi]

    wD = torch.ones(D, dtype=torch.float32, requires_grad=True)
    wC = torch.ones(D, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([wD, wC], lr=LR)
    loss_trace = []
    n_steps = 0

    for step in range(STEPS):
        h = step % K
        bank_idx = [k for k in range(K) if k != h]
        bd = torch.tensor(np.concatenate([cd[k].reshape(-1, D) for k in bank_idx]))
        bc = torch.tensor(np.concatenate([cc[k].reshape(-1, D) for k in bank_idx]))
        # positives: union over training episodes of h (cells inside mask)
        posD, posC, pos_set = [], [], set()
        for e in keep_idx:
            cells = np.flatnonzero(masks[h, e].ravel() > 0)
            if cells.size == 0:
                continue
            if cells.size > MAX_POS:
                cells = rng.choice(cells, MAX_POS, replace=False)
            posD.append(sd[h, e].reshape(-1, D)[cells])
            posC.append(sc[h, e].reshape(-1, D)[cells])
            pos_set.update(int(i) for i in cells)
        if not posD:
            continue
        pd, pc = torch.tensor(np.concatenate(posD)), torch.tensor(np.concatenate(posC))
        # negatives: clean cells excluding any positive cell, + nuisance cells
        clean_neg = np.asarray([i for i in range(NCELL) if i not in pos_set])
        if clean_neg.size > MAX_NEG_CLEAN:
            clean_neg = rng.choice(clean_neg, MAX_NEG_CLEAN, replace=False)
        nuiD = np.asarray(nd[h]).reshape(15, NCELL, D)
        nuiC = np.asarray(nc[h]).reshape(15, NCELL, D)
        nui_imgs = rng.choice(15, NUI_NEG_IMGS, replace=False)
        nD, nC = [], []
        for e in nui_imgs:
            cells = rng.choice(NCELL, NUI_NEG_CELLS, replace=False)
            nD.append(nuiD[e][cells])
            nC.append(nuiC[e][cells])
        qd_neg = torch.tensor(np.concatenate([cd[h].reshape(-1, D)[clean_neg], np.concatenate(nD)]))
        qc_neg = torch.tensor(np.concatenate([cc[h].reshape(-1, D)[clean_neg], np.concatenate(nC)]))
        sp = _scores(pd, pc, bd, bc, wD, wC)          # [npos]
        sn = _scores(qd_neg, qc_neg, bd, bc, wD, wC)  # [nneg]
        reg = LAM_W * ((wD - 1) ** 2).sum() + LAM_W * ((wC - 1) ** 2).sum()
        if loss == "la":
            margin = torch.clamp(GAMMA + sn[:, None] - sp[None, :], min=0.0)
            loss_v = margin.mean() + reg
        else:  # lb: AUC-logistic over pairs (never saturates on easy pairs)
            logits = sp[None, :] - sn[:, None]        # [nneg,npos] positive when defect above neg
            loss_v = torch.log1p(torch.exp(-logits)).mean() + reg
        opt.zero_grad()
        loss_v.backward()
        opt.step()
        with torch.no_grad():
            wD.clamp_(0.0, UPPER)
            wC.clamp_(0.0, UPPER)
        if step % 20 == 0:
            loss_trace.append(float(loss_v.item()))
        n_steps += 1

    wD_n, wC_n = wD.detach().numpy(), wC.detach().numpy()
    ones = np.ones(D)
    rows = {"cat": cat, "shot": shot, "held_family": SYN[fi],
            "loss": loss, "n_steps": n_steps, "loss_trace": loss_trace}
    for h in range(K):
        bank_idx = [k for k in range(K) if k != h]
        id_aps, tr_aps = [], []
        for e in held_idx:
            id_aps.append(_eval_ap_h(h, e, (cd, sd, masks, nd, cc, sc, nc, K), ones, ones, bank_idx))
            tr_aps.append(_eval_ap_h(h, e, (cd, sd, masks, nd, cc, sc, nc, K), wD_n, wC_n, bank_idx))
        rows[f"ap_id_h{h}"] = id_aps
        rows[f"ap_tr_h{h}"] = tr_aps
        cp_id, nm_id = _normal_stats_h(h, (cd, sd, masks, nd, cc, sc, nc, K), ones, ones, bank_idx)
        cp_tr, nm_tr = _normal_stats_h(h, (cd, sd, masks, nd, cc, sc, nc, K), wD_n, wC_n, bank_idx)
        rows[f"clean_p95_id_h{h}"] = cp_id
        rows[f"clean_p95_tr_h{h}"] = cp_tr
        rows[f"nui_maxp95_id_h{h}"] = nm_id
        rows[f"nui_maxp95_tr_h{h}"] = nm_tr
    rows["w_d_mean"] = float(wD_n.mean())
    rows["w_c_mean"] = float(wC_n.mean())
    rows["w_d_std"] = float(wD_n.std())
    rows["w_c_std"] = float(wC_n.std())
    return rows


def _mean_nan(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _fold_delta(row):
    """Held-family macro AP delta trained-identity across all h/episodes (eval only)."""
    d = []
    for hk in [h for h in range(10) if f"ap_id_h{h}" in row]:
        for a, b in zip(row[f"ap_id_h{hk}"], row[f"ap_tr_h{hk}"]):
            if a == a and b == b:  # not nan
                d.append(b - a)
    return float(np.mean(d)) if d else float("nan"), d


def aggregate(rows_all, out_json):
    deltas = []
    for r in rows_all:
        dv, d = _fold_delta(r)
        r["delta_held_ap"] = dv
        deltas.append(dv)
    n_eval = sum(1 for v in deltas if v == v)
    n_pass = sum(1 for v in deltas if v >= 0.005)
    h1 = {"pass": n_eval > 0 and n_pass / max(n_eval, 1) >= 2.0 / 3.0,
          "n_folds_eval": n_eval, "n_folds_pass": n_pass,
          "macro_delta": float(np.mean([v for v in deltas if v == v])) if n_eval else float("nan"),
          "deltas": deltas}
    # H2 normal path: macro relative rise of clean p95 and nuisance image-max p95
    c_rel, n_rel = [], []
    c_abs = []
    for r in rows_all:
        for hk in [h for h in range(10) if f"clean_p95_id_h{h}" in r]:
            ci, ct = r[f"clean_p95_id_h{hk}"], r[f"clean_p95_tr_h{hk}"]
            ni, nt = r[f"nui_maxp95_id_h{hk}"], r[f"nui_maxp95_tr_h{hk}"]
            c_rel.append((ct - ci) / max(ci, 1e-6))
            c_abs.append(ct - ci)
            n_rel.append((nt - ni) / max(ni, 1e-6))
    macro_c_rel = float(np.mean(c_rel))
    macro_n_rel = float(np.mean(n_rel))
    macro_c_abs = float(np.mean(c_abs))
    h2 = {"pass": macro_c_rel <= 0.10 and macro_n_rel <= 0.10 and macro_c_abs <= 0.05,
          "clean_p95_rel_rise": macro_c_rel, "nui_maxp95_rel_rise": macro_n_rel,
          "clean_p95_abs_drift": macro_c_abs}
    decision = "HEALTH_PROBE_PASS" if (h1["pass"] and h2["pass"]) else "HEALTH_PROBE_FAIL_ARCHIVE"
    out = {"h1": h1, "h2": h2, "decision": decision}
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    ap.add_argument("--shots", type=int, nargs="+", default=[2])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--loss", choices=("la", "lb"), default="lb",
                    help="lb=AUC-logistic (probe default, non-saturating); la=margin hinge")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    rows_all = []
    for shot in args.shots:
        for cat in cats:
            for fi in range(3):
                seed = (SEED0 + zlib.crc32(f"{cat}:{shot}:{fi}:{args.loss}".encode())) % 2**31
                rng = np.random.default_rng(seed)
                rows_all.append(_train_fold(cat, shot, fi, rng, loss=args.loss))
                print(f"  done {cat} k{shot} held={SYN[fi]} loss={args.loss}", flush=True)
    (OUT / "HEALTH_PROBE.json").write_text(
        json.dumps({"rows": rows_all, "config": {"loss": args.loss, "gamma": GAMMA, "lam_w": LAM_W,
                                                 "lr": LR, "steps": args.steps, "upper": UPPER,
                                                 "score": "1-max_cos A1 fused (LOO memory)"}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    gates = aggregate(rows_all, OUT / "HEALTH_PROBE_GATES.json")
    print("folds:", len(rows_all))
    print("H1:", gates["h1"]["pass"], "macro_delta=%.4f" % gates["h1"]["macro_delta"],
          "folds_pass=%d/%d" % (gates["h1"]["n_folds_pass"], gates["h1"]["n_folds_eval"]))
    print("H2:", gates["h2"]["pass"], "clean_p95_rel=%.3f nui_rel=%.3f abs=%.4f"
          % (gates["h2"]["clean_p95_rel_rise"], gates["h2"]["nui_maxp95_rel_rise"],
             gates["h2"]["clean_p95_abs_drift"]))
    print("DECISION:", gates["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
