"""V12-EARLY-FUSION Stage 1 SCAIF CLI (doc 23 s7; CONFIG.yaml frozen).

Modes:
  selfcheck     gate=0 identity (SCAIF map == static control #2, bit-exact) + param count
  references    control #1 (A1 deep) & #2 (raw 2-pair static) per cat x shot (no module)
  runfold       train one variant on one held-out category, eval at shots
  runs          runfold for all 6 held-out categories of a variant (writes runs/{variant}_all.json)
  gates         aggregate CONTROL_RESULTS.csv + MECHANISM_AUDIT.json from runs/* + references

GPU venv:
  .venv-anomalyclip\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r3_ef_scaif.py --mode runs --variant main
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))            # parent of the 'src' package (for `import src.*`)
sys.path.insert(0, str(ROOT / "src"))    # contains industrial_ad/...
sys.path.insert(0, str(ROOT / "scripts"))

from scaif_common import (ANCHORS, CATEGORIES, GATE_CAP, PAIRS, QSUBS, SEED, SHOTS,
                          TRAIN_STEPS, SCAIF, ccol, deep_rows, load_cat_features,
                          maps_to56, pair_raw_rows, pooled_ap_np)

ML_ROOT = ROOT / "outputs/dynamic_fusion/v12_early_fusion"
EXP = ROOT / "experiments/dynamic_fusion/innovation_v12_early_fusion"
RUNS = EXP / "03_scaif_small_gate" / "runs"
GRUNS = EXP / "03_scaif_small_gate"


def gt32(masks448: np.ndarray, idx: np.ndarray, device) -> torch.Tensor:
    import torch.nn.functional as F

    sub = masks448[idx].astype(np.float32)
    t = torch.from_numpy(sub).unsqueeze(1)
    pooled = F.max_pool2d(t, kernel_size=14, stride=14)
    return (pooled > 0.5).float().to(device)


# ----------------------------------------------------------------------
# reference maps (control #1, #2)
# ----------------------------------------------------------------------

def eval_static(cc: dict, kind: str, device) -> dict:
    """kind in {'a1','static2'}: maps for the whole category; returns Pixel-AP + gate stats."""
    with torch.no_grad():
        if kind == "a1":
            qf = deep_rows(cc["d"], cc["c"]).reshape(-1, 1536)
            ref = deep_rows(cc["dr"].permute(1, 0, 2, 3, 4),
                            cc["cr"].permute(1, 0, 2, 3, 4)).reshape(-1, 1536)
        else:
            qf = pair_raw_rows(cc["d"], cc["c"]).reshape(-1, 3072)
            ref = pair_raw_rows(cc["dr"].permute(1, 0, 2, 3, 4),
                                cc["cr"].permute(1, 0, 2, 3, 4)).reshape(-1, 3072)
        n = cc["d"].shape[0]
        d2 = torch.cdist(qf, ref).pow(2).min(dim=-1)[0]
        scores32 = (d2 / 2.0).reshape(n, 32, 32)   # A1 convention: larger = more anomalous
        map56 = maps_to56(scores32)
    ap = pooled_ap_np(map56, cc["masks"])
    return {"kind": kind, "pixel_ap_56": ap, "n": n}


# ----------------------------------------------------------------------
# SCAIF eval (trained or frozen; gate_zero / remove_private switches)
# ----------------------------------------------------------------------

def eval_scaif(cc: dict, model: SCAIF, device, gate_zero=False, remove_private=False) -> dict:
    n = cc["d"].shape[0]
    with torch.no_grad():
        sup_d = [cc["dr"][di].reshape(-1, 768) for di, _ in PAIRS]
        sup_c = [cc["cr"][ccol(ci)].reshape(-1, 768) for _, ci in PAIRS]
        fr, _, gs = model.refine(cc["d"], cc["c"], sup_d, sup_c,
                                 gate_zero=gate_zero, remove_private=remove_private)
        bank, _, _ = model.refine(cc["dr"].permute(1, 0, 2, 3, 4),
                                  cc["cr"].permute(1, 0, 2, 3, 4),
                                  sup_d, sup_c, gate_zero=gate_zero, remove_private=remove_private)
        qf = fr.reshape(-1, fr.shape[-1])
        bf = bank.reshape(-1, bank.shape[-1])
        d2 = torch.cdist(qf, bf).pow(2).min(dim=-1)[0]
        scores32 = (d2 / 2.0).reshape(n, 32, 32)   # A1 convention: larger = more anomalous
        map56 = maps_to56(scores32)
        gsat = torch.cat([g for gd, gc in gs for g in (gd, gc)]).abs()
        gate_frac_cap = float((gsat > 0.9 * GATE_CAP).float().mean())
    ap = pooled_ap_np(map56, cc["masks"])
    return {"kind": "scaif", "pixel_ap_56": ap, "n": n, "gate_frac_cap": gate_frac_cap}


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------

def train_step(model: SCAIF, opt, cc: dict, rng: np.random.Generator, device) -> dict:
    K = int(rng.choice([1, 2, 4]))
    ndef = min(4, cc["def_rows"].size)
    if ndef == 0:
        return None
    qdef = rng.choice(cc["def_rows"], size=ndef, replace=False)
    qnrm = rng.choice(cc["norm_rows"], size=2, replace=False)
    qidx = np.concatenate([qdef, qnrm])
    sup = rng.choice(cc["norm_rows"], size=K, replace=False)
    sup_d = [cc["d"][sup][:, di].reshape(-1, 768) for di, _ in PAIRS]
    sup_c = [cc["c"][sup][:, ccol(ci)].reshape(-1, 768) for _, ci in PAIRS]
    if sup_d[0].shape[0] > ANCHORS:
        pick = rng.choice(sup_d[0].shape[0], size=ANCHORS, replace=False)
        sup_d = [s[pick] for s in sup_d]
        sup_c = [s[pick] for s in sup_c]

    dq = cc["d"][qidx]
    cq = cc["c"][qidx]
    fr, _, gs = model.refine(dq, cq, sup_d, sup_c)
    with torch.no_grad():                       # support bank is a static memory (no grad needed)
        frs, _, _ = model.refine(cc["d"][sup], cc["c"][sup], sup_d, sup_c)
    bank = frs.reshape(-1, frs.shape[-1])

    B = dq.shape[0]
    # P0-2 fix: uniform whole-image patch sampling per image (fixed-seed rng), the SAME indices
    # applied to the SCAIF score, the GT and the A1 reference (no label misalignment).
    pix = np.stack([rng.choice(1024, size=QSUBS, replace=False) for _ in range(B)])
    pt = torch.from_numpy(pix).to(device)
    rows = torch.arange(B, device=device)[:, None]
    qflat = fr.reshape(B, 1024, fr.shape[-1])
    qs = qflat[rows, pt].reshape(-1, fr.shape[-1])
    d2 = torch.cdist(qs, bank).pow(2).min(dim=-1)[0]
    s = d2 / 2.0                     # larger = more anomalous (A1 convention)
    g = gt32(cc["masks"], qidx, device).reshape(B, 1024)[rows, pt].reshape(-1)

    with torch.no_grad():            # A1 private-stream reference is a static score path
        a1qflat = deep_rows(dq, cq).reshape(B, 1024, 1536)
        a1r = deep_rows(cc["d"][sup], cc["c"][sup]).reshape(-1, 1536)
        a1q = a1qflat[rows, pt].reshape(-1, 1536)
        sa1 = (torch.cdist(a1q, a1r).pow(2).min(dim=-1)[0]) / 2.0

    pos = g > 0.5
    neg = g <= 0.5
    wpos = max(1.0, float(neg.sum()) / max(float(pos.sum()), 1.0))
    loss_seg = torch.nn.functional.binary_cross_entropy_with_logits(
        s * 10.0, g, pos_weight=torch.tensor(wpos, device=device))
    loss_ap = torch.nn.functional.relu(sa1[pos] - s[pos]).mean() if pos.sum() > 0 else torch.zeros((), device=device)
    loss_cp = torch.nn.functional.relu(s[neg] - sa1[neg]).mean() if neg.sum() > 0 else torch.zeros((), device=device)
    # P0-1 fix: gs carries the autograd graph now -> the sparse penalty actually trains the gates.
    gd_cat = torch.cat([g for gd, gc in gs for g in (gd, gc)])
    gmean = gd_cat.abs().mean()
    loss = loss_seg + 0.20 * loss_cp + 0.20 * loss_ap + 1.0 * gmean
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return {"loss": loss.item(), "seg": loss_seg.item(), "ap": loss_ap.item(),
            "cp": loss_cp.item(), "gate": float(gmean.detach().item()),
            "pix": pix}


def train_fold(heldout: str, variant: str, device, steps=TRAIN_STEPS, tag=None) -> SCAIF:
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    model = SCAIF(variant=variant).to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=1e-3, weight_decay=1e-3)
    src_cats = [c for c in CATEGORIES if c != heldout]
    per_cat = steps // len(src_cats)
    t0 = time.time()
    step = 0
    coverage = np.zeros(1024, dtype=np.int64)
    import csv
    suf = f"_{tag}" if tag else ""
    log_path = RUNS / f"log_{variant}{suf}_{heldout}.csv"
    ckpt_path = RUNS / f"ckpt_{variant}{suf}_{heldout}.pt"
    if ckpt_path.exists():  # resumable runs: a saved fold is not retrained
        sd = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(sd["model_state"])
        print(f"  [fold-{heldout} {variant}{suf}] RESUME from {ckpt_path.name} "
              f"(coverage rows {sd.get('patch_coverage_rows')})", flush=True)
        return model
    with open(log_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "loss", "seg", "ap", "cp", "gate"])
        for scat in src_cats:
            cc = load_cat_features(ML_ROOT, 1, scat, device)
            for _ in range(per_cat):
                logs = train_step(model, opt, cc, rng, device)
                if logs is not None:
                    coverage += np.bincount(logs["pix"].ravel(), minlength=1024)
                    w.writerow([step, f"{logs['loss']:.6f}", f"{logs['seg']:.6f}",
                                f"{logs['ap']:.6f}", f"{logs['cp']:.6f}", f"{logs['gate']:.6f}"])
                    if step % 100 == 0 or step == steps - 1:
                        print(f"  [fold-{heldout} {variant}] s{step} loss={logs['loss']:.4f} "
                              f"seg={logs['seg']:.4f} ap={logs['ap']:.4f} gate={logs['gate']:.4f} "
                              f"({time.time()-t0:.0f}s)", flush=True)
                step += 1
            del cc
            torch.cuda.empty_cache()
    covered = int((coverage > 0).sum())
    torch.save({"model_state": model.state_dict(), "variant": variant, "heldout": heldout,
                "steps": steps, "patch_coverage_rows": covered, "tag": tag,
                "config": "CONFIG_v5_correction"},
               RUNS / f"ckpt_{variant}{suf}_{heldout}.pt")
    print(f"  [fold-{heldout} {variant}{suf}] patch coverage {covered}/1024 rows "
          f"min={int(coverage.min())} (log {log_path.name})", flush=True)
    return model


# ----------------------------------------------------------------------
# aggregation
# ----------------------------------------------------------------------

def macro(rows, key="pixel_ap_56"):
    vals = [r[key] for r in rows if r[key] == r[key]]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selfcheck", "references", "runfold", "runs", "gates"], required=True)
    ap.add_argument("--variant", default="main")
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--shots", type=int, nargs="+", default=SHOTS)
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--tag", default=None,
                    help="suffix for output files (e.g. 'correction'); archived results are left untouched")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    RUNS.mkdir(parents=True, exist_ok=True)

    if args.mode == "selfcheck":
        model = SCAIF(variant="main").to(device)
        print("PARAM_COUNT", model.trainable_params())
        cc = load_cat_features(ML_ROOT, 1, "bracket_black", device)
        with torch.no_grad():
            sup_d = [cc["dr"][di].reshape(-1, 768) for di, _ in PAIRS]
            sup_c = [cc["cr"][ccol(ci)].reshape(-1, 768) for _, ci in PAIRS]
            fr, f0, _ = model.refine(cc["d"], cc["c"], sup_d, sup_c, gate_zero=True)
            err = (fr - f0).abs().max().item()
            r1 = eval_scaif(cc, model, device, gate_zero=True)
            r2 = eval_static(cc, "static2", device)
            r3 = eval_static(cc, "a1", device)
        print(f"SELFCHECK gate0 rows max-abs-diff={err:.3e}")
        print(f"SELFCHECK gate0 map AP={r1['pixel_ap_56']:.6f} static2 AP={r2['pixel_ap_56']:.6f} "
              f"diff={abs(r1['pixel_ap_56']-r2['pixel_ap_56']):.3e} | a1 AP={r3['pixel_ap_56']:.6f}")
        return 0

    if args.mode == "references":
        out = []
        for shot in args.shots:
            for cat in CATEGORIES:
                cc = load_cat_features(ML_ROOT, shot, cat, device)
                for kind in ("a1", "static2"):
                    r = eval_static(cc, kind, device)
                    r.update({"category": cat, "shot": shot})
                    out.append(r)
                    print(f"  REF {kind} {cat} k{shot} AP={r['pixel_ap_56']:.6f}", flush=True)
                del cc
                torch.cuda.empty_cache()
        (GRUNS / "REFERENCES.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print("refs written", flush=True)
        return 0

    if args.mode in ("runfold", "runs"):
        cats = [args.heldout] if args.heldout else CATEGORIES
        allrows = []
        for held in cats:
            model = train_fold(held, args.variant, device, steps=args.steps, tag=args.tag)
            for shot in args.shots:
                cc = load_cat_features(ML_ROOT, shot, held, device)
                r = eval_scaif(cc, model, device)
                r.update({"category": held, "shot": shot, "variant": args.variant,
                          "tag": args.tag})
                allrows.append(r)
                print(f"  RES {args.variant}{('_'+args.tag) if args.tag else ''} {held} k{shot} "
                      f"AP={r['pixel_ap_56']:.6f} gcap={r['gate_frac_cap']:.3f}", flush=True)
                del cc
                torch.cuda.empty_cache()
        suf = f"_{args.tag}" if args.tag else ""
        fname = RUNS / f"{args.variant}{suf}_all.json"
        fname.write_text(json.dumps(allrows, indent=1), encoding="utf-8")
        print("WROTE", fname, flush=True)
        return 0

    if args.mode == "gates":
        refs = json.loads((GRUNS / "REFERENCES.json").read_text(encoding="utf-8"))
        results = {}
        for v in ["main", "dino_only", "clip_only", "no_support", "shuffled",
                  "symmetric", "no_cross"]:
            p = RUNS / f"{v}_all.json"
            if p.exists():
                results[v] = json.loads(p.read_text(encoding="utf-8"))
        # CONTROL_RESULTS.csv
        import csv

        with open(GRUNS / "CONTROL_RESULTS.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["variant", "shot", "macro_pixel_ap", "bracket_black", "bracket_brown",
                        "bracket_white", "connector", "metal_plate", "tubes"])
            for kind, rows in [("a1", [r for r in refs if r["kind"] == "a1"]),
                               ("static2", [r for r in refs if r["kind"] == "static2"])]:
                for shot in args.shots:
                    rr = [r for r in rows if r["shot"] == shot]
                    w.writerow([kind, shot, round(macro(rr) or -1, 6)] +
                               [round(next((x["pixel_ap_56"] for x in rr if x["category"] == c), float("nan")), 6)
                                for c in CATEGORIES])
            for v, rows in results.items():
                for shot in args.shots:
                    rr = [r for r in rows if r["shot"] == shot]
                    w.writerow([v, shot, round(macro(rr) or -1, 6)] +
                               [round(next((x["pixel_ap_56"] for x in rr if x["category"] == c), float("nan")), 6)
                                for c in CATEGORIES])
        print("CONTROL_RESULTS.csv written", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
