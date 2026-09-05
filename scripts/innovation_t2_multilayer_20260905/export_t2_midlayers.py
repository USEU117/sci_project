"""Track-2 mid-layer support export (doc31 s2/s6), CPU, k2 only.

Renders the SAME support-only variants as v14 (cutpaste/local_erasure/thin_scratch
x3 seeds + 15 photometric nuisance) on manifest support images at 1024 and
re-encodes through frozen backbones, storing INTERMEDIATE layers that the v14
cache does not have:
  dino_mid = DINOv2-vitb14 block index 5 tokens (32-grid, 768-D)
  clip_6   = AnomalyCLIP DPAM forward pf[0] (resblock 6, 37-grid, 768-D)
  clip_12  = ... pf[1] (resblock 12, 37-grid, 768-D)
Final layers are already cached by v14 (v14_p1_support_*) and are NOT rewritten.

Data role: manifest support only (assert_fit_ids_are_support); no /test/ touched.
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
          str(ROOT / "scripts" / "innovation_v14_decisive_validation_20260905"),
          str(ROOT / "methods" / "anomalydino"), str(ROOT / "methods" / "AnomalyCLIP-main")):
    sys.path.insert(0, p)

import ntof_render as R  # noqa: E402

from v14_common import CATEGORIES, DATA_ROOT, support_paths, assert_fit_ids_are_support  # noqa: E402

OUT_ROOT = ROOT / "outputs/dynamic_fusion/t2_multilayer_support"
NUIS_KEYS = R.REF_KEYS
SYN_KINDS = R.SYNTHETIC_KINDS
DINO_MID_BLOCK = 5   # 0-based block index in vitb14 (12 blocks)


def _variant_seed(cat: str, rel: str, kind: str, idx: int) -> int:
    h = 0
    for ch in f"v14::{cat}::{rel}::{kind}::s{idx}":
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return h


def load_image(path: Path):
    import cv2
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def make_extractors(device: str):
    """Return (dino_mid_extract, clip_mid_extract) with their grids."""
    import torch
    from src.backbones import get_model

    dino = get_model("dinov2_vitb14", device, smaller_edge_size=448)
    m = dino.model
    blocks = list(m.blocks)
    hook_holder = {}

    def _mid_hook(_mod, _inp, out):
        hook_holder["mid"] = out

    def extract_dino_mid(rgb: np.ndarray):
        # Reuse the exact v14 forward path (extract_features -> L11 final); a
        # forward hook on block DINO_MID_BLOCK captures the intermediate tokens
        # including class token, so drop index 0 and reshape to the 2-D grid.
        t, grid = dino.prepare_image(rgb)
        h = blocks[DINO_MID_BLOCK].register_forward_hook(_mid_hook)
        try:
            with torch.inference_mode():
                dino.extract_features(t)
        finally:
            h.remove()
        mid = hook_holder.pop("mid")[0, 1:, :].float().cpu().numpy().astype(np.float32)
        return mid.reshape(grid[0], grid[1], -1), grid

    # clip (same load path as v14 export; DPAM layers list -> pf[0]=resblock6, pf[1]=resblock12)
    from types import SimpleNamespace
    import torch  # noqa: F811
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

    def extract_clip_mid(rgb: np.ndarray):
        from PIL import Image
        tensor = preprocess(Image.fromarray(rgb)).reshape(1, 3, 518, 518).to(device)
        with torch.inference_mode():
            _, pf = model.encode_image(tensor, [6, 12, 18, 24], DPAM_layer=20)
        out = []
        for pi in (0, 1):
            tok = pf[pi][0, 1:, :].float().cpu().numpy()
            side = int(round(tok.shape[0] ** 0.5))
            out.append(tok.reshape(side, side, -1).astype(np.float32))
        return out[0], out[1]

    return extract_dino_mid, extract_clip_mid


def export_cat(cat, shot, ex_d, ex_c, manifest):
    rels = support_paths(cat, shot, "0", manifest)
    assert_fit_ids_are_support(rels, cat, shot, "0")
    dm_c, dm_s, dm_n = [], [], []
    c6_c, c6_s, c6_n, c12_c, c12_s, c12_n = [], [], [], [], [], []
    for rel in rels:
        x0 = load_image(DATA_ROOT / rel)
        dm, gd = ex_d(x0)
        c6, c12 = ex_c(x0)
        dm_c.append(dm)
        c6_c.append(c6)
        c12_c.append(c12)
        dm_s_, c6_s_, c12_s_ = [], [], []
        for kind in SYN_KINDS:
            for s in range(3):
                img, _m = R.render_synthetic(x0, kind, _variant_seed(cat, rel, kind, s))
                dm_, _ = ex_d(img)
                dm_s_.append(dm_)
                c6_, c12_ = ex_c(img)
                c6_s_.append(c6_)
                c12_s_.append(c12_)
        dm_s.append(np.stack(dm_s_))
        c6_s.append(np.stack(c6_s_))
        c12_s.append(np.stack(c12_s_))
        dm_n_, c6_n_, c12_n_ = [], [], []
        for key in NUIS_KEYS:
            img, _m = R.render_by_key(x0, key, cat, rel)
            dm_, _ = ex_d(img)
            dm_n_.append(dm_)
            c6_, c12_ = ex_c(img)
            c6_n_.append(c6_)
            c12_n_.append(c12_)
        dm_n.append(np.stack(dm_n_))
        c6_n.append(np.stack(c6_n_))
        c12_n.append(np.stack(c12_n_))
        print(f"    {rel}: ok", flush=True)
    out_dir = OUT_ROOT / f"k{shot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{cat}.npz",
        ref_rel=np.asarray(rels),
        dm_clean=np.asarray(dm_c, np.float32), dm_syn=np.asarray(dm_s, np.float32),
        dm_nui=np.asarray(dm_n, np.float32),
        c6_clean=np.asarray(c6_c, np.float32), c6_syn=np.asarray(c6_s, np.float32),
        c6_nui=np.asarray(c6_n, np.float32),
        c12_clean=np.asarray(c12_c, np.float32), c12_syn=np.asarray(c12_s, np.float32),
        c12_nui=np.asarray(c12_n, np.float32),
        syn_kinds=np.asarray(SYN_KINDS), nui_keys=np.asarray(NUIS_KEYS),
        dino_mid_block=np.asarray(DINO_MID_BLOCK, np.int64),
        dino_grid=np.asarray(gd, np.int64),
        clip_grid=np.asarray([c6.shape[0], c6.shape[1]], np.int64),
    )
    return rels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default=None)
    ap.add_argument("--shot", type=int, default=2, choices=[2, 4])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable")
    manifest = __import__("json").loads((ROOT / "data/splits/mpdd/manifest.json").read_text(encoding="utf-8"))
    ex_d, ex_c = make_extractors(args.device)
    cats = [c.strip() for c in args.cats.split(",")] if args.cats else CATEGORIES
    for cat in cats:
        print(f"[export t2 mid k{args.shot}] {cat}", flush=True)
        export_cat(cat, args.shot, ex_d, ex_c, manifest)
    (OUT_ROOT / f"k{args.shot}" / "export_report.json").write_text(json.dumps(
        {"created_utc": datetime.now(timezone.utc).isoformat(), "shot": args.shot,
         "cats": cats, "dino_mid_block": DINO_MID_BLOCK,
         "clip_dpam_layers": [6, 12], "device": args.device},
        ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
