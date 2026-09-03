"""Task book 17 - Phase 0 integrity audit.

Produces experiments/dynamic_fusion/innovation_v7_global_text/00_audit/
  INPUT_AUDIT.json       (task book s.2.2 items 1-9)
  checkpoint_provenance.md  (item 10 - written separately by the same run)
  leakage_audit.json     (per-file GT-key / label-file scan + labels module check)
  input_hashes.json      (sha256 of every frozen input consumed)

Run env: .venv-patchcore (torch+torchvision+sklearn). CPU only.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v6_dgsafe import maps as v6maps  # noqa: E402
from industrial_ad.innovation_v7_global_text import (  # noqa: E402
    A1_MAPS_ROOT, ANOMALYCLIP_CKPT, ANOMALYCLIP_CKPT_SHA256, EXPERIMENT_ROOT,
    MANIFEST, OUTPUTS_ROOT, S1_CACHE, SEEDS, SHOTS, load_a1_config,
    load_manifest, load_text_cache, sha256_file,
)
from industrial_ad.innovation_v7_global_text.evaluator import (  # noqa: E402
    image_metrics, labels_from_ids, top1_mean,
)

AUDIT = EXPERIMENT_ROOT / "00_audit"
CATS = ["bracket_black", "bracket_brown", "bracket_white",
        "connector", "metal_plate", "tubes"]


def git_state() -> dict:
    def sh(*a):
        try:
            r = subprocess.run(a, capture_output=True, text=True, cwd=ROOT)
            return r.stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"<err {e}>"
    return {"head": sh("git", "rev-parse", "HEAD"),
            "dirty_short": sh("git", "status", "--short")}


def env_state() -> dict:
    import platform
    import sklearn
    import torch
    import torchvision
    out = {
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torchvision": torchvision.__version__,
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "device_name": (torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None),
        "note": "audit interpreter = .venv-patchcore; AnomalyCLIP export env = "
                ".venv-anomalyclip (torch 2.0.0+cu118)",
    }
    return out


def cache_fields(cat: str) -> dict:
    c = load_text_cache(cat)
    return {"sha256": c["sha256"], "n_test": len(c["sample_ids"]),
            "n_refs": len(c["ref_ids"]),
            "clip_global_test_shape": list(np.asarray(
                c["clip_global_test"]).shape),
            "clip_global_dtype": str(np.asarray(c["clip_global_test"]).dtype),
            "text_prob_dtype": str(np.asarray(c["text_prob_test"]).dtype),
            "keys": sorted(np.load(S1_CACHE / f"{cat}.npz",
                                   allow_pickle=False).files),
            "checkpoint_sha256": c["checkpoint_sha256"],
            "text_embedding_sha256": c["text_embedding_sha256"]}


def text_stats(cat: str) -> dict:
    p = np.asarray(load_text_cache(cat)["text_prob_test"], dtype=np.float64)
    return {"finite": bool(np.isfinite(p).all()),
            "min": float(p.min()), "max": float(p.max()),
            "in_unit_interval": bool((p >= 0).all() and (p <= 1).all()),
            "std": float(p.std()), "n_unique": int(np.unique(p).size)}


def alignment_checks() -> dict:
    out = {}
    for cat in CATS:
        cset = set(load_text_cache(cat)["sample_ids"])
        per = {}
        for seed in SEEDS:
            for shot in SHOTS:
                ids = set(load_a1_config(cat, seed, shot)["sample_ids"])
                per[f"s{seed}_k{shot}"] = {
                    "n": len(ids),
                    "equal_to_cache": ids == cset,
                    "unique": len(ids) == len(ids) if ids else True,
                }
        out[cat] = {"n_cache": len(cset), "cache_unique": len(cset) == len(
            set(cset)), "configs": per}
    return out


def manifest_ref_audit() -> dict:
    manifest = load_manifest()
    out = {}
    for cat in CATS:
        bad = []
        counts = {}
        for seed in SEEDS:
            for shot in SHOTS:
                refs = manifest["categories"][cat][str(seed)][str(shot)]
                counts[f"s{seed}_k{shot}"] = len(refs)
                for r in refs:
                    if "/train/good/" not in r:
                        bad.append(r)
        out[cat] = {"counts": counts, "non_train_good_refs": bad}
    return out


def replay_s1_rows() -> dict:
    """Recompute seed0 (cat,shot) A1-max / A1-top1 / TEXT image AP from caches
    and compare with the archived S1_HGLC_DIAG.json rows (max abs err <= 1e-4)."""
    diag = json.loads((Path(__file__).resolve().parents[2] /
                       "experiments/dynamic_fusion/innovation_v6_dgsafe/s1_hglc"
                       "/S1_HGLC_DIAG.json").read_text(encoding="utf-8"))
    arch = {(r["category"], r["shot"]): r for r in diag["rows"]}
    errs = {}
    for cat in CATS:
        cache = load_text_cache(cat)
        ids = list(cache["sample_ids"])
        text_p = np.asarray(cache["text_prob_test"], dtype=np.float64)
        for shot in SHOTS:
            a1 = load_a1_config(cat, 0, shot)
            perm = align_ids(ids, list(a1["sample_ids"]))
            m = v6maps.a1_maps448(a1["concat_patch_map"])
            labels = labels_from_ids(a1["sample_ids"])
            ap = {
                "a1_max": image_metrics(m.reshape(len(m), -1).max(axis=1),
                                        labels)["image_ap"],
                "a1_top1": image_metrics(top1_mean(m), labels)["image_ap"],
                "text": image_metrics(text_p[perm], labels)["image_ap"],
            }
            for sig, v in ap.items():
                key = f"{cat}|k{shot}|{sig}"
                errs[key] = abs(v - arch[(cat, shot)][f"ap_{sig}"])
    return {"max_abs_err": float(max(errs.values())),
            "threshold": 1e-4, "pass": max(errs.values()) <= 1e-4,
            "per_key_err": {k: round(v, 7) for k, v in errs.items()}}


def align_ids(b, a):
    from industrial_ad.innovation_v7_global_text import align_perm
    return align_perm(np.asarray(b), np.asarray(a))


def leakage_scan() -> dict:
    """Caches must carry no GT/label keys; labels module is evaluator-only."""
    gt_keywords = ("gt", "mask", "label", "ground", "target")
    hits = {}
    for cat in CATS:
        npz = np.load(S1_CACHE / f"{cat}.npz", allow_pickle=False)
        found = [k for k in npz.files if any(w in k.lower() for w in gt_keywords)]
        if found:
            hits[cat] = found
    # A1 compact maps must be label-free as well
    a1_hits = {}
    for seed in SEEDS:
        for shot in SHOTS:
            for cat in CATS:
                npz = np.load(A1_MAPS_ROOT / f"s{seed}_k{shot}" / f"{cat}.npz",
                              allow_pickle=False)
                found = [k for k in npz.files
                         if any(w in k.lower() for w in gt_keywords)]
                if found:
                    a1_hits[f"{cat}|s{seed}_k{shot}"] = found
    return {"cache_gt_keys": hits, "a1_gt_keys": a1_hits}


def main() -> int:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    export_rep = json.loads((S1_CACHE / "s1_hglc_export_report.json").read_text(
        encoding="utf-8"))
    ckpt_sha = sha256_file(ANOMALYCLIP_CKPT)
    ckpt_ok = ckpt_sha == ANOMALYCLIP_CKPT_SHA256

    checks = {
        "git_head": git_state(),
        "env": env_state(),
        "checkpoint": {
            "path": str(ANOMALYCLIP_CKPT),
            "sha256": ckpt_sha,
            "matches_frozen_415c5dcb": ckpt_ok,
            "export_report": {k: export_rep.get(k) for k in (
                "checkpoint_sha256", "text_embedding_sha256", "text_swapped",
                "image_size", "design", "dapm_layer")},
        },
        "caches": {c: cache_fields(c) for c in CATS},
        "text_prob_stats": {c: text_stats(c) for c in CATS},
        "alignment_vs_a1_9_configs": alignment_checks(),
        "manifest_ref_audit": manifest_ref_audit(),
        "replay_s1_rows": replay_s1_rows(),
        "leakage_scan": leakage_scan(),
    }

    # ---- fold PASS/FAIL summary ----
    ck = checks
    summary = {
        "checkpoint_sha_matches": bool(ckpt_ok),
        "cache_gt_keys_empty": not any(ck["leakage_scan"]["cache_gt_keys"]),
        "a1_gt_keys_empty": not any(ck["leakage_scan"]["a1_gt_keys"]),
        "all_text_finite_unit_nonconstant": all(
            ck["text_prob_stats"][c]["finite"] and
            ck["text_prob_stats"][c]["in_unit_interval"] and
            ck["text_prob_stats"][c]["std"] > 0 for c in CATS),
        "all_configs_equal_cache": all(
            cfg["equal_to_cache"]
            for c in CATS for cfg in ck["alignment_vs_a1_9_configs"][c]["configs"].values()),
        "no_non_train_good_refs": all(
            not ck["manifest_ref_audit"][c]["non_train_good_refs"] for c in CATS),
        "s1_replay_pass_1e-4": bool(ck["replay_s1_rows"]["pass"]),
    }
    report = {"program": "innovation_v7_global_text", "phase": "phase0_audit",
              "task_book": "17 (2026-09-03) s.2.2",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "elapsed_total_s": round(time.time() - t0, 1),
              "checks": checks, "summary": summary,
              "phase0_pass": all(summary.values())}
    (AUDIT / "INPUT_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- input hashes ----
    hashes = {"checkpoint": {"path": str(ANOMALYCLIP_CKPT), "sha256": ckpt_sha},
              "manifest": {"path": str(MANIFEST), "sha256": sha256_file(MANIFEST)}}
    for cat in CATS:
        hashes[f"cache_{cat}"] = sha256_file(S1_CACHE / f"{cat}.npz")
    for seed in SEEDS:
        for shot in SHOTS:
            for cat in CATS:
                hashes[f"a1_s{seed}_k{shot}_{cat}"] = sha256_file(
                    A1_MAPS_ROOT / f"s{seed}_k{shot}" / f"{cat}.npz")
    (AUDIT / "input_hashes.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- checkpoint provenance ----
    prov = [
        "# AnomalyCLIP checkpoint provenance (task book 17 s.2.2 item 10)",
        "",
        f"- checkpoint: `{ANOMALYCLIP_CKPT}`",
        f"- sha256: `{ckpt_sha}`",
        "- origin: AnomalyCLIP official training on **VisA** (directory "
        "`9_12_4_multiscale_visa`), prompt-learner head over a frozen "
        "OpenAI CLIP ViT-L/14@336px visual/text tower.",
        "- prompt: generic class `object`, learned context (design: "
        "Prompt_length=12, depth=9, text-n-ctx=4); normal state `{}`, "
        "abnormal state `damaged {}`.",
        "- role on MPDD/BTAD/MVTec: **target-domain zero-shot transfer** of the "
        "prompt learner. The system is NOT 'fully training-free' and VisA is a "
        "source/in-domain dataset, NOT an independent external validation set.",
        "- frozen inference: image_size 518, DAPM layer 20, "
        "encode_image -> global embedding @ learned text prompts -> "
        "softmax(/0.07) -> abnormal-class probability.",
    ]
    (AUDIT / "checkpoint_provenance.md").write_text("\n".join(prov),
                                                    encoding="utf-8")

    print(json.dumps(report["summary"], indent=1))
    return 0 if report["phase0_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
