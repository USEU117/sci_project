"""P1-A support-only synthetic variant export (doc28 s5.1).

Renders cutpaste / local_erasure / thin_scratch (S deterministic seeds) plus 5
photometric nuisance families x 3 strengths (15) ON the K-shot SUPPORT images at
1024, then re-encodes through the FROZEN branch extractors (dino L11 32x32 /
clip L24 37x37). Masks stored at 1024 for probe-only evaluation.

Data-role gate: only manifest support rels are touched (assert_fit_ids_are_support);
no /test/ image is read or cached.

Run (.venv-anomalyclip, single GPU proc; or --device cpu):
  .venv-anomalyclip\\Scripts\\python.exe scripts\\innovation_v14_decisive_validation_20260905\\export_p1_support_variants.py --branch dino --shot 2
  ... --branch clip --shot 2
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "innovation_v12_new_observables"),
          str(ROOT / "methods" / "anomalydino"), str(ROOT / "methods" / "AnomalyCLIP-main")):
    sys.path.insert(0, p)

import ntof_render as R  # noqa: E402

from v14_common import CATEGORIES, DATA_ROOT, load_manifest, support_paths, assert_fit_ids_are_support  # noqa: E402

OUT_ROOT = ROOT / "outputs/dynamic_fusion/v14_p1_support"
NUIS_KEYS = R.REF_KEYS            # 15 photometric variants (5 fams x 3 strengths)
SYN_KINDS = R.SYNTHETIC_KINDS      # cutpaste, local_erasure, thin_scratch


def _variant_seed(cat: str, rel: str, kind: str, idx: int) -> int:
    h = 0
    for ch in f"v14::{cat}::{rel}::{kind}::s{idx}":
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def make_extractor(branch: str, device: str):
    import torch  # noqa: F401
    if branch == "dino":
        from src.backbones import get_model
        model = get_model("dinov2_vitb14", device, smaller_edge_size=448)

        def extract(rgb: np.ndarray):
            tensor, grid = model.prepare_image(rgb)
            toks = model.extract_features(tensor).astype(np.float32)
            return toks.reshape(grid[0], grid[1], -1)
        return extract
    from types import SimpleNamespace  # clip

    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    design = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9,
              "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cpu", design_details=design)
    ck_path = ROOT / "methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth"
    pl = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    ck = torch.load(str(ck_path), map_location="cpu")
    pl.load_state_dict(ck["prompt_learner"])
    model.to(device)
    pl.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    preprocess, _ = get_transform(SimpleNamespace(image_size=518))

    def extract(rgb: np.ndarray):
        from PIL import Image
        tensor = preprocess(Image.fromarray(rgb)).reshape(1, 3, 518, 518).to(device)
        with torch.inference_mode():
            _, pf = model.encode_image(tensor, [6, 12, 18, 24], DPAM_layer=20)
        tok = pf[-1][0, 1:, :].float().cpu().numpy()
        side = int(round(tok.shape[0] ** 0.5))
        return tok.reshape(side, side, -1)
    return extract


def load_image(path: Path):
    import cv2
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def export(cat: str, shot: int, extract, manifest, syn_seeds: int, limit: int | None):
    rels = support_paths(cat, shot, "0", manifest)
    if limit:
        rels = rels[:limit]
    assert_fit_ids_are_support(rels, cat, shot, "0")
    clean, syn, syn_masks, nui = [], [], [], []
    for rel in rels:
        x0 = load_image(DATA_ROOT / rel)
        clean.append(extract(x0))
        sv, sm = [], []
        for kind in SYN_KINDS:
            for s in range(syn_seeds):
                img, m = R.render_synthetic(x0, kind, _variant_seed(cat, rel, kind, s))
                sv.append(extract(img))
                sm.append(m)
        nv = []
        for key in NUIS_KEYS:
            img, _m = R.render_by_key(x0, key, cat, rel)
            nv.append(extract(img))
        syn.append(np.stack(sv))
        syn_masks.append(np.stack(sm))
        nui.append(np.stack(nv))
        print(f"    {rel}: clean + {len(sv)} syn + {len(nv)} nui", flush=True)
    return (np.asarray(rels), np.asarray(clean, dtype=np.float32),
            np.asarray(syn, dtype=np.float32), np.asarray(syn_masks, dtype=np.uint8),
            np.asarray(nui, dtype=np.float32))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", choices=("dino", "clip"), required=True)
    ap.add_argument("--shot", type=int, default=2, choices=[2, 4])
    ap.add_argument("--category", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--syn-seeds", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable")
    manifest = load_manifest()
    extract = make_extractor(args.branch, args.device)
    import torch
    out_dir = OUT_ROOT / f"v14_p1_support_{args.branch}_s0_k{args.shot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cats = [args.category] if args.category else CATEGORIES
    rows = []
    with torch.inference_mode():
        for cat in cats:
            print(f"[export {args.branch} k{args.shot}] {cat}", flush=True)
            rels, clean, syn, smasks, nui = export(cat, args.shot, extract, manifest,
                                                   args.syn_seeds, args.limit)
            grid = list(clean.shape[1:3])
            np.savez_compressed(
                out_dir / f"{cat}.npz",
                ref_rel=np.asarray(rels),
                clean_feat=clean,
                syn_feat=syn, syn_masks=smasks,
                nui_feat=nui,
                syn_kinds=np.asarray(SYN_KINDS), nui_keys=np.asarray(NUIS_KEYS),
                syn_seeds=np.asarray(args.syn_seeds),
                grid_size=np.asarray(grid, dtype=np.int64),
                branch=np.asarray(args.branch),
            )
            rows.append({"category": cat, "refs": len(rels), "grid": grid,
                         "n_syn": int(syn.shape[1] * syn.shape[2]),
                         "n_nui": int(nui.shape[1])})
            print(f"    ok {cat} syn={rows[-1]['n_syn']} nui={rows[-1]['n_nui']}", flush=True)
    (out_dir / "export_report.json").write_text(json.dumps(
        {"created_utc": datetime.now(timezone.utc).isoformat(), "branch": args.branch,
         "shot": args.shot, "device": args.device, "syn_kinds": SYN_KINDS,
         "nui_keys": NUIS_KEYS, "categories": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
