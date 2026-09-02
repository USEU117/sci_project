"""Route E — NCPRA: Normal-only Cross-Encoder Predictive Residual Adapter.

Task book section 9 (user-authorized, protocol change recorded in
experiments/dynamic_fusion/innovation_v2/README_STATUS.md).

Two small bottleneck adapters (backbone frozen):
    g_D2C: 768 -> r -> GELU -> 768      g_C2D: 768 -> r -> GELU -> 768
Loss on normal reference patches only:
    L = (1-cos(g_D2C(d), c)) + (1-cos(g_C2D(c), d)) + mu * L_equivariance
where L_equivariance is a fixed cycle-consistency term. The anomaly residual is
    e(q) = 0.5*[(1-cos(g_D2C(d_q), c_q)) + (1-cos(g_C2D(c_q), d_q))]
calibrated with reference-only robust statistics and combined with A1 by
lambda. Early stopping uses normal validation loss only (shot>=2:
leave-one-reference-image-out; shot=1: fixed feature-space jitter views).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion import rcec  # noqa: E402
from industrial_ad.innovation_v2.common import AlignedFeatures, InnovationError  # noqa: E402


class BottleneckAdapter(nn.Module):
    def __init__(self, r: int, in_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, r), nn.GELU(), nn.Linear(r, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(adapters: list[nn.Module]) -> int:
    return int(sum(p.numel() for a in adapters for p in a.parameters()))


def _cos_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = nn.functional.normalize(pred, dim=-1)
    target = nn.functional.normalize(target, dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def _make_views(d_ref: np.ndarray, c_ref: np.ndarray, rng: np.random.Generator,
                n_views: int, scale: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic feature-space jitter views (shot=1 normal validation)."""
    d_v = d_ref.copy()
    c_v = c_ref.copy()
    for _ in range(n_views - 1):
        d_v = np.concatenate([d_v, d_ref + rng.normal(0, scale, d_ref.shape)], axis=0)
        c_v = np.concatenate([c_v, c_ref + rng.normal(0, scale, c_ref.shape)], axis=0)
    return d_v, c_v


def train_ncpra(
    d_ref: np.ndarray, c_ref: np.ndarray,
    r: int, mu: float, shot: int, cfg: dict, device: str = "cuda:0",
) -> tuple[dict, dict]:
    """Train g_D2C / g_C2D on normal reference patches.

    Returns (checkpoints as cpu state_dicts, training report). No labels/masks.
    """
    n = d_ref.shape[0]
    if n < 16:
        raise InnovationError("NCPRA needs >= 16 normal patches")
    d_t = torch.from_numpy(d_ref.astype(np.float32)).to(device)
    c_t = torch.from_numpy(c_ref.astype(np.float32)).to(device)

    if shot >= 2:
        per_image = d_ref.shape[0] // shot
        rng_valid = np.random.default_rng(int(cfg.get("ncpra", {}).get("seed", 0)))
        # leave-one-reference-image-out validation: split by image index parity is
        # NOT used (would leak); we validate on the LAST reference image's patches.
        v_idx = np.arange(shot * per_image)
        val_mask = v_idx >= (shot - 1) * per_image
        tr_idx = v_idx[~val_mask]
        va_idx = v_idx[val_mask]
        d_tr, c_tr = d_t[tr_idx], c_t[tr_idx]
        d_va, c_va = d_t[va_idx], c_t[va_idx]
    else:
        rng_valid = np.random.default_rng(int(cfg.get("ncpra", {}).get("seed", 0)) + 1)
        d_v, c_v = _make_views(d_ref, c_ref, rng_valid, n_views=3)
        d_tr, c_tr = d_t, c_t
        d_va, c_va = (torch.from_numpy(d_v.astype(np.float32)).to(device),
                      torch.from_numpy(c_v.astype(np.float32)).to(device))

    g_d2c = BottleneckAdapter(r).to(device)
    g_c2d = BottleneckAdapter(r).to(device)
    params = list(g_d2c.parameters()) + list(g_c2d.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    max_epochs = int(cfg.get("ncpra", {}).get("max_epochs", 100))
    patience = int(cfg.get("ncpra", {}).get("patience", 10))
    best_val = float("inf")
    best = None
    best_epoch = 0
    history = []
    for epoch in range(max_epochs):
        g_d2c.train(); g_c2d.train()
        opt.zero_grad()
        p_c = g_d2c(d_tr)
        p_d = g_c2d(c_tr)
        loss = _cos_loss(p_c, c_tr) + _cos_loss(p_d, d_tr)
        loss = loss + float(mu) * 0.5 * (
            _cos_loss(g_c2d(p_c.detach()), d_tr) + _cos_loss(g_d2c(p_d.detach()), c_tr))
        loss.backward()
        opt.step()

        g_d2c.eval(); g_c2d.eval()
        with torch.no_grad():
            val = (_cos_loss(g_d2c(d_va), c_va) + _cos_loss(g_c2d(c_va), d_va)).item()
        history.append({"epoch": epoch, "train_loss": round(loss.item(), 6),
                        "val_loss": round(val, 6)})
        if val < best_val:
            best_val = val
            best = (g_d2c.state_dict(), g_c2d.state_dict())
            best_epoch = epoch
        if epoch - best_epoch >= patience:
            break

    report = {
        "r": r, "mu": mu, "shot": shot,
        "n_train_patches": int(d_tr.shape[0]),
        "n_val_patches": int(d_va.shape[0]),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val, 6),
        "epochs_run": len(history),
        "history": history,
        "parameter_count": parameter_count([g_d2c, g_c2d]),
        "early_stop_rule": "normal validation loss only",
        "seed": int(cfg.get("ncpra", {}).get("seed", 0)),
    }
    g_d2c.load_state_dict(best[0])
    g_c2d.load_state_dict(best[1])
    return {"g_d2c": g_d2c, "g_c2d": g_c2d}, report


def residual_scores(g_d2c, g_c2d, d_q: np.ndarray, c_q: np.ndarray,
                    device: str = "cuda:0", chunk: int = 8192) -> np.ndarray:
    """e(q) per patch = 0.5*[(1-cos(g_D2C(d),c)) + (1-cos(g_C2D(c),d))]."""
    g_d2c.eval(); g_c2d.eval()
    n = d_q.shape[0]
    out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for i0 in range(0, n, chunk):
            i1 = min(i0 + chunk, n)
            d = torch.from_numpy(d_q[i0:i1].astype(np.float32)).to(device)
            c = torch.from_numpy(c_q[i0:i1].astype(np.float32)).to(device)
            pc = nn.functional.normalize(g_d2c(d), dim=-1)
            pd = nn.functional.normalize(g_c2d(c), dim=-1)
            cn = nn.functional.normalize(c, dim=-1)
            dn = nn.functional.normalize(d, dim=-1)
            e_i = 0.5 * ((1 - (pc * cn).sum(-1)) + (1 - (pd * dn).sum(-1)))
            out[i0:i1] = e_i.detach().cpu().numpy()
    return out


def score_ncpra(
    aligned: AlignedFeatures,
    candidate: dict,
    cfg: dict,
    device: str = "cuda:0",
) -> tuple[np.ndarray, dict]:
    r = int(candidate["r"])
    lam = float(candidate["lambda"])
    mu = float(cfg.get("ncpra", {}).get("mu", 0.1))
    shot = int(cfg["_shot"])
    dino_weight = float(cfg.get("fixed", {}).get("dino_weight", 0.5))
    seed = int(cfg.get("ncpra", {}).get("seed", 0))

    d_ref = aligned.d_ref.reshape(-1, 768)
    c_ref = aligned.c_ref.reshape(-1, 768)
    g, train_report = train_ncpra(d_ref, c_ref, r, mu, shot, cfg, device=device)

    d_q = aligned.d_feat.reshape(-1, 768)
    c_q = aligned.c_feat.reshape(-1, 768)
    e_q = residual_scores(g["g_d2c"], g["g_c2d"], d_q, c_q, device=device)

    # Reference-only calibration: residual stats on ref patches (in-sample).
    e_ref = residual_scores(g["g_d2c"], g["g_c2d"], d_ref, c_ref, device=device)
    stats = rcec.compute_reference_stats(e_ref, epsilon=1e-6)

    # A1 dists + robust z of both signals, combine by lambda.
    mem = rcec.build_paired_reference_memory(aligned.d_ref, aligned.c_ref,
                                             aligned.n_references)
    s_a1_flat = rcec.compute_a1_dists(aligned.d_feat, aligned.c_feat, mem,
                                      dino_weight=dino_weight)
    loo = rcec.compute_reference_loo_statistics(
        aligned.d_ref, aligned.c_ref, aligned.n_references,
        direction="dino_to_clip", k=1, shot=shot)
    stats_a1 = rcec.compute_reference_stats(loo["a1_loo"], epsilon=1e-6)
    z_a1 = rcec.robust_z_from_reference(s_a1_flat, stats_a1, epsilon=1e-6,
                                        z_clip=(-5.0, 10.0))
    z_e = rcec.robust_z_from_reference(e_q, stats, epsilon=1e-6, z_clip=(-5.0, 10.0))
    s_final = (1.0 - lam) * z_a1 + lam * z_e

    n, h, w = aligned.d_feat.shape[0], *aligned.grid
    diag = {
        "r": r, "lambda": lam, "mu": mu,
        "parameter_count": train_report["parameter_count"],
        "best_epoch": train_report["best_epoch"],
        "best_val_loss": train_report["best_val_loss"],
        "epochs_run": train_report["epochs_run"],
        "residual_ref_median": round(float(stats["median"]), 6),
        "residual_ref_mad": round(float(stats["mad"]), 6),
        "protocol": "lightweight normal-only adaptation",
    }
    return s_final.reshape(n, h, w), diag
