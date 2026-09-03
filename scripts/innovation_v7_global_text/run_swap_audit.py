"""Task book 17 - Phase 0 audit item 8: swap-probability complementarity.

Recomputes the frozen normal/abnormal text embeddings (same prompt learner +
checkpoint + chunk/norm code as the v6 exporter) and, from the stored
clip_global_test embeddings only (no vision forward needed), re-derives each
test image's abnormal probability under the *swapped* embedding order.

Checks (per category, per image):
  max | p_swap - (1 - p_orig_cache) |
  text_embedding_sha256(normal) == s1 cache hash e6240c...

Run env: .venv-anomalyclip (GPU for the CLIP text tower only).
Output: experiments/dynamic_fusion/innovation_v7_global_text/00_audit/swap_check.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = ROOT / "methods" / "AnomalyCLIP-main"
for p in (str(METHOD_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

import AnomalyCLIP_lib  # noqa: E402
from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402

S1_CACHE = ROOT / "experiments/dynamic_fusion/innovation_v6_dgsafe/s1_hglc/cache"
AUDIT = ROOT / "experiments/dynamic_fusion/innovation_v7_global_text/00_audit"
CKPT = (METHOD_ROOT / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth")
DESIGN = {"Prompt_length": 12,
          "learnabel_text_embedding_depth": 9,
          "learnabel_text_embedding_length": 4}
CATS = ["bracket_black", "bracket_brown", "bracket_white",
        "connector", "metal_plate", "tubes"]
EXPECTED_HASH = "e6240cf8725ad86853e89b145202e76bf8bffa1887f31a25026689ef647bda3a"


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device,
                                    design_details=DESIGN)
    model.eval()
    pl = AnomalyCLIP_PromptLearner(model.to("cpu"), DESIGN)
    pl.load_state_dict(torch.load(CKPT, map_location="cpu")["prompt_learner"])
    pl.to(device)
    model.to(device)

    prompts, toks, comp = pl(cls_id=None)
    with torch.inference_mode():
        tf = model.encode_text_learn(prompts, toks, comp).float()
    tf = torch.stack(torch.chunk(tf, dim=0, chunks=2), dim=1)
    tf = tf / tf.norm(dim=-1, keepdim=True)
    hash_n = tensor_sha256(tf)
    hash_s = tensor_sha256(tf[:, [1, 0], :])
    tf_s = tf[:, [1, 0], :].to(device)

    rows = {}
    for cat in CATS:
        d = np.load(S1_CACHE / f"{cat}.npz", allow_pickle=False)
        cg = np.asarray(d["clip_global_test"], dtype=np.float64)   # (N,768)
        p_orig = np.asarray(d["text_prob_test"], dtype=np.float64)  # fp16 stored
        cg_t = torch.from_numpy(cg).float().to(device)
        with torch.inference_mode():
            logits = cg_t @ tf_s.permute(0, 2, 1)[0]     # (N,768)x(768,2)
            p_swap = (logits / 0.07).softmax(-1)[:, 1].float().cpu().numpy()
        err = np.max(np.abs(p_swap - (1.0 - p_orig)))
        rows[cat] = {"n": len(p_orig),
                     "max_abs_swap_err": float(err),
                     "mean_orig": float(p_orig.mean()),
                     "mean_swap": float(p_swap.mean())}

    report = {"program": "innovation_v7_global_text",
              "phase": "phase0_audit_item8_swap",
              "text_embedding_sha256_normal": hash_n,
              "text_embedding_sha256_swapped": hash_s,
              "matches_s1_cache_text_hash_e6240c": hash_n == EXPECTED_HASH,
              "note": "p_orig stored fp16 in the v6 cache; tolerance ~1e-3 expected",
              "per_category": rows,
              "max_over_categories": float(max(r["max_abs_swap_err"]
                                               for r in rows.values())),
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (AUDIT / "swap_check.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "text_embedding_sha256_normal", "matches_s1_cache_text_hash_e6240c",
        "max_over_categories")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
