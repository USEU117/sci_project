"""V12-EARLY-FUSION Stage 1 SCAIF shared library (cached features, doc 23 s5.2/s7).

Frozen protocol: experiments/dynamic_fusion/innovation_v12_early_fusion/
03_scaif_small_gate/CONFIG.yaml
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))           # parent of the 'src' package (for `import src.utils`)
sys.path.insert(0, str(_ROOT / "src"))   # contains industrial_ad/...

from src.utils import dists2map  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
SHOTS = [1, 2, 4]
PAIRS = [(1, 1), (2, 3)]  # (D9,C12) mid, (D11,C24) deep/A1-pair ; dino col of [6,9,11], clip ORIG col of [6,12,18,24]
CTRIM = {1: 1, 3: 2}      # clip orig col -> trimmed col in loaded [C6,C12,C24] tensors (C18 dropped)


def ccol(ci: int) -> int:
    """Map original clip layer column to the trimmed clip column used by cc['c']/cc['cr']."""
    return CTRIM[ci]
D_U = 32
GATE_CAP = 0.2
TRAIN_STEPS = 600
ANCHORS = 256
QSUBS = 512
SEED = 0


# ----------------------------------------------------------------------
# cache loading
# ----------------------------------------------------------------------

def load_cat_features(ml_root, shot: int, cat: str, device: torch.device):
    """Load one category; returns dict with cpu fp32 arrays + row split."""
    z = np.load(ml_root / f"ml_dino_s0_k{shot}/{cat}.npz", allow_pickle=False)
    d = np.asarray(z["patch_features"], dtype=np.float32)       # [N,3,32,32,768]
    dr = np.asarray(z["ref_patch_features"], dtype=np.float32)  # [3,K,32,32,768]
    masks = np.asarray(z["imgs_masks"], dtype=np.uint8)         # [N,448,448]
    del z
    zc = np.load(ml_root / f"ml_clip_s0_k{shot}/{cat}.npz", allow_pickle=False)
    c_raw = np.asarray(zc["patch_features"], dtype=np.float32)  # [N,4,37,37,768]
    cr_raw = np.asarray(zc["ref_patch_features"], dtype=np.float32)  # [4,K,37,37,768]
    del zc
    need = [0, 1, 3]  # C6, C12, C24 -> cols 0,1,2 of the trimmed set
    tt = torch.from_numpy

    def _resize5(x):
        n1 = x.shape[0] * x.shape[1]
        xt = torch.from_numpy(np.ascontiguousarray(x))
        y = F.interpolate(xt.reshape(n1, 37, 37, 768).permute(0, 3, 1, 2), size=(32, 32),
                          mode="bilinear", align_corners=False)
        return y.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[1], 32, 32, 768).numpy().astype(np.float32)

    c = _resize5(c_raw[:, need])   # patch features: layer axis = 1
    cr = _resize5(cr_raw[need])    # ref features:   layer axis = 0 ([L,K,H,W,d])
    del c_raw, cr_raw
    norm_rows = np.flatnonzero(masks.reshape(masks.shape[0], -1).sum(axis=1) == 0)
    def_rows = np.flatnonzero(masks.reshape(masks.shape[0], -1).sum(axis=1) > 0)
    return {"d": tt(d).to(device), "c": tt(c).to(device),
            "dr": tt(dr).to(device), "cr": tt(cr).to(device),
            "masks": masks, "norm_rows": norm_rows, "def_rows": def_rows}


def gt32_from_masks(masks448: np.ndarray, idx: np.ndarray, device: torch.device) -> torch.Tensor:
    """Block-max GT over 14x14 cells -> [B,32,32] float in {0,1}."""
    sub = masks448[idx]
    t = torch.from_numpy(sub.astype(np.float32)).unsqueeze(1)
    pooled = F.max_pool2d(t, kernel_size=14, stride=14)
    return (pooled > 0.5).float().to(device)


# ----------------------------------------------------------------------
# module
# ----------------------------------------------------------------------

def l2row(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


class PairBlock(nn.Module):
    def __init__(self, d_in: int = 768, u: int = D_U, variant: str = "main",
                 shuffle_seed: int = 0, active_dirs: tuple[bool, bool] = (True, True)):
        super().__init__()
        self.variant = variant
        self.u = u
        self.pd = nn.Linear(d_in, u)
        self.pc = nn.Linear(d_in, u)
        self.wd = nn.Linear(u, d_in, bias=False)
        self.wc = nn.Linear(u, d_in, bias=False)
        nn.init.zeros_(self.wd.weight)
        nn.init.zeros_(self.wc.weight)
        self.mlp_cd = nn.Sequential(nn.Linear(2 * u, 4 * u), nn.ReLU(), nn.Linear(4 * u, u))
        self.mlp_dc = nn.Sequential(nn.Linear(2 * u, 4 * u), nn.ReLU(), nn.Linear(4 * u, u))
        gate_in = 2 * u if variant == "no_support" else 2 * u + 2
        self.gate_d = nn.Sequential(nn.Linear(gate_in, u), nn.ReLU(), nn.Linear(u, 1))
        self.gate_c = nn.Sequential(nn.Linear(gate_in, u), nn.ReLU(), nn.Linear(u, 1))
        for h in (self.gate_d, self.gate_c):
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)
        # dead-direction freezing keyed on active_dirs (so param-counting with variant='main'
        # and a one-sided active_dirs is consistent with the trained control variants)
        self.dir_cd = active_dirs[0]  # C->D used
        self.dir_dc = active_dirs[1]  # D->C used
        if active_dirs == (True, False):  # dino_only
            nn.init.zeros_(self.pc.weight)
            nn.init.zeros_(self.pc.bias)
            self._freeze((self.pc, self.wc, self.mlp_dc, self.gate_c))
        elif active_dirs == (False, True):  # clip_only
            nn.init.zeros_(self.pd.weight)
            nn.init.zeros_(self.pd.bias)
            self._freeze((self.pd, self.wd, self.mlp_cd, self.gate_d))
        if variant == "shuffled":
            g = torch.Generator().manual_seed(shuffle_seed)
            self.register_buffer("sh_off",
                                 torch.randint(-3, 4, (2,), dtype=torch.long, generator=g))

    @staticmethod
    def _freeze(mods):
        for m in mods:
            for p in m.parameters():
                p.requires_grad_(False)

    def _neigh3(self, u: torch.Tensor) -> torch.Tensor:
        x = u.permute(0, 3, 1, 2)
        return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1).permute(0, 2, 3, 1)

    def _neigh3_shuffled(self, u: torch.Tensor) -> torch.Tensor:
        """Control #6: read the 3x3 neighbourhood from a spatially-shifted copy of the other
        branch (fixed per-seed integer shift) -> destroys D<->CLIP spatial correspondence."""
        dy, dx = int(self.sh_off[0].item()), int(self.sh_off[1].item())
        return self._neigh3(torch.roll(u, shifts=(dy, dx), dims=(1, 2)))

    def forward(self, d_blk, c_blk, sup_d_u, sup_c_u, gate_zero=False):
        """sup_*_u: raw support features [M,768] (same layer cols); returns (d_t,c_t,g_d,g_c).
        Query and support projections are BOTH L2-normalized before cdist so the distance is a
        proper cosine distance (identical token -> distance 0)."""
        ud = F.normalize(self.pd(d_blk), dim=-1)
        uc = F.normalize(self.pc(c_blk), dim=-1)
        su = F.normalize(self.pd(sup_d_u), dim=-1)
        cv = F.normalize(self.pc(sup_c_u), dim=-1)
        shp = ud.shape[:-1]
        d_sup = torch.cdist(ud.reshape(-1, self.u), su.reshape(-1, self.u)).min(dim=-1)[0].reshape(shp)
        c_sup = torch.cdist(uc.reshape(-1, self.u), cv.reshape(-1, self.u)).min(dim=-1)[0].reshape(shp)
        if self.variant == "shuffled":
            nc = self._neigh3_shuffled(uc)
            nd = self._neigh3_shuffled(ud)
        else:
            nc = self._neigh3(uc)
            nd = self._neigh3(ud)
        if self.variant == "no_cross":
            r_cd = self.mlp_cd(torch.cat([ud, ud], dim=-1)) if self.dir_cd else torch.zeros_like(ud)
            r_dc = self.mlp_dc(torch.cat([uc, uc], dim=-1)) if self.dir_dc else torch.zeros_like(uc)
        else:
            r_cd = self.mlp_cd(torch.cat([ud, nc], dim=-1)) if self.dir_cd else torch.zeros_like(ud)
            r_dc = self.mlp_dc(torch.cat([uc, nd], dim=-1)) if self.dir_dc else torch.zeros_like(uc)
        if gate_zero:
            gd = torch.zeros_like(ud[..., :1])
            gc = torch.zeros_like(uc[..., :1])
        else:
            gin = torch.cat([ud, uc], dim=-1)
            if self.variant != "no_support":
                gin = torch.cat([gin, d_sup[..., None], c_sup[..., None]], dim=-1)
            if self.variant == "symmetric":
                gd = gc = GATE_CAP * torch.sigmoid(self.gate_d(gin))
            else:
                gd = GATE_CAP * torch.sigmoid(self.gate_d(gin)) if self.dir_cd else torch.zeros_like(ud[..., :1])
                gc = GATE_CAP * torch.sigmoid(self.gate_c(gin)) if self.dir_dc else torch.zeros_like(uc[..., :1])
        d_t = d_blk + gd * self.wd(r_cd)
        c_t = c_blk + gc * self.wc(r_dc)
        return d_t, c_t, gd, gc


def _pair_trainable(u: int, active_dirs: tuple[bool, bool]) -> int:
    blk = PairBlock(u=u, variant="main", active_dirs=active_dirs)
    return sum(p.numel() for p in blk.parameters() if p.requires_grad)


def single_dir_matched_u(active_dirs: tuple[bool, bool]) -> int:
    """Width u for one-sided controls whose trainable count matches main (u=D_U) within 5%."""
    target = _pair_trainable(D_U, (True, True))
    lo, hi = 0.95 * target, 1.05 * target
    best = None
    for u in range(D_U + 2, 240, 2):
        n = _pair_trainable(u, active_dirs)
        if n < lo:
            continue
        best = u
        if n <= hi:
            break
        # count is monotone in u: once above hi, no later u will satisfy; keep the closest
        break
    if best is None:
        raise RuntimeError("could not match single-direction width within searched range")
    return best


class SCAIF(nn.Module):
    def __init__(self, variant="main", shuffle_seed=0, u: int | None = None):
        super().__init__()
        self.variant = variant
        dirs = {"dino_only": (True, False), "clip_only": (False, True)}.get(variant, (True, True))
        if u is None:
            u = single_dir_matched_u(dirs) if variant in ("dino_only", "clip_only") else D_U
        self.u = u
        self.blocks = nn.ModuleList(
            [PairBlock(variant=variant, u=u, shuffle_seed=shuffle_seed + p * 101, active_dirs=dirs)
             for p in range(len(PAIRS))])

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def refine(self, d, c, sup_d, sup_c, gate_zero=False, remove_private=False):
        """d/c: [B,3,32,32,768]; sup_d/sup_c: lists (per PAIR position) of raw support rows [M,768].
        Returns concat refined rows [B,32,32,3072], raw-pair rows [B,32,32,3072], gate stats.
        Gate tensors carry the autograd graph (training sparse loss needs it); callers doing
        eval-only stats detach them explicitly."""
        blocks_r, blocks_0 = [], []
        gs = []
        for p, ((di, ci), blk) in enumerate(zip(PAIRS, self.blocks)):
            dt, ct, gd, gc = blk(d[:, di], c[:, ccol(ci)], sup_d[p], sup_c[p], gate_zero=gate_zero)
            if remove_private:
                dt = dt - d[:, di]
                ct = ct - c[:, ccol(ci)]
            blocks_r += [0.5 * l2row(dt.reshape(-1, 768)), 0.5 * l2row(ct.reshape(-1, 768))]
            blocks_0 += [0.5 * l2row(d[:, di].reshape(-1, 768)),
                         0.5 * l2row(c[:, ccol(ci)].reshape(-1, 768))]
            gs.append((gd, gc))
        fr = l2row(torch.cat(blocks_r, dim=-1))
        f0 = l2row(torch.cat(blocks_0, dim=-1))
        return fr, f0, gs

    def static_deep(self, d, c):
        """A1 final-concat rows [B,32,32,1536] (deep D11 + C24, 0.5/0.5)."""
        f = torch.cat([0.5 * l2row(d[:, 2].reshape(-1, 768)), 0.5 * l2row(c[:, 2].reshape(-1, 768))], dim=-1)
        return l2row(f)


# ----------------------------------------------------------------------
# static feature rows (no module) — controls #1 (A1 deep) and #2 (raw pairs)
# ----------------------------------------------------------------------

def deep_rows(d, c):
    """A1 final-concat rows [..,32,32,1536]: 0.5*l2(D11) concat 0.5*l2(C24), then l2.
    c columns are the TRIMMED set [C6,C12,C24]; C24 is trimmed col 2."""
    f = torch.cat([0.5 * l2row(d[..., 2, :, :, :].reshape(-1, 768)),
                   0.5 * l2row(c[..., 2, :, :, :].reshape(-1, 768))], dim=-1)
    return l2row(f)


def pair_raw_rows(d, c):
    """Static 2-pair concat rows [..,32,32,3072] == SCAIF at gate=0 (control #2)."""
    blocks = []
    for di, ci in PAIRS:
        blocks += [0.5 * l2row(d[..., di, :, :, :].reshape(-1, 768)),
                   0.5 * l2row(c[..., ccol(ci), :, :, :].reshape(-1, 768))]
    return l2row(torch.cat(blocks, dim=-1))


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------

def nn_score(qf, bank_rows, grid=(32, 32)):
    """Per-patch A1 score -d/2. qf [B,32,32,F] -> [B,32,32]."""
    B = qf.shape[0]
    q = qf.reshape(B, -1, qf.shape[-1])
    d2 = torch.cdist(q, bank_rows).pow(2).min(dim=-1)[0]
    return (-d2 / 2.0).reshape(B, *grid)


def maps_to56(scores32: torch.Tensor) -> np.ndarray:
    s = scores32.cpu().numpy().astype(np.float32)
    return np.stack([dists2map(s[i], (448, 448))[::8, ::8] for i in range(s.shape[0])])


def pooled_ap_np(maps56: np.ndarray, masks: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    m56 = (masks[:, ::8, ::8] > 0.5).astype(np.int32)
    y = m56.ravel()
    s = maps56.ravel()
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    return float(average_precision_score(y, s))
