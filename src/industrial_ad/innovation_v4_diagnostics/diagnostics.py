"""D2 / D3 information-value diagnostics core (task book 14, sections 10 & 12).

All computations here happen on MPDD normal material only:
- D2 synthesises structural perturbations (patch permutation / missing block /
  duplicate block) on real normal-test feature grids, then compares how well
  frozen A1 KNN, a non-parametric context-repair residual and a node-only
  transport cost detect the perturbed regions. No real test mask is used; the
  label is the synthetic perturbation mask.
- D3 synthesises CASF-style branch-asymmetric pseudo anomalies on normal
  references and asks whether the |z_D - z_C| disagreement channel carries
  learnable supervision (asymmetric-trained head vs symmetric-trained head).

No algorithm is trained for the paper here; outputs are diagnostic numbers that
decide which route (RG-MCR / SF-NM / RG-OT / CASF) deserves a strict task book.
"""

from __future__ import annotations

import numpy as np

from industrial_ad.fusion import rcec


# ---------------------------------------------------------------------------
# A1 fused grid + memory distance helpers
# ---------------------------------------------------------------------------

def fused_grid(d_feat: np.ndarray, c_feat: np.ndarray) -> np.ndarray:
    """[..., 768] per-branch (L2 rows) -> [..., 1536] fused + L2 (A1 semantics)."""
    return rcec._concat_and_l2(d_feat, c_feat, 0.5)


def memory_dist1(feat: np.ndarray, ref: np.ndarray, chunk: int = 16384) -> np.ndarray:
    """1-NN fused distance per patch: 0.5*||z_q - z_r||^2 == 1 - cos for unit rows.

    feat [N,H,W,1536], ref [S,H,W,1536] -> [N,H,W] (higher = more anomalous).
    """
    import faiss

    q = np.ascontiguousarray(feat.reshape(-1, feat.shape[-1]), dtype=np.float32)
    r = np.ascontiguousarray(ref.reshape(-1, ref.shape[-1]), dtype=np.float32)
    faiss.normalize_L2(q)
    faiss.normalize_L2(r)
    index = faiss.IndexFlatL2(r.shape[1])
    index.add(r)
    out = np.empty(q.shape[0], dtype=np.float32)
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        d, _ = index.search(q[start:end], k=1)
        out[start:end] = d[:, 0] / 2.0
    n, h, w = feat.shape[:3]
    return out.reshape(n, h, w)


# ---------------------------------------------------------------------------
# D2: structural perturbations
# ---------------------------------------------------------------------------

def ring_mean(feat: np.ndarray, r_in: int, r_out: int) -> np.ndarray:
    """Per-patch mean feature of its ring (r_in, r_out] neighbours -> [H,W,D].

    The ring excludes the centre so a masked centre is never copied.
    """
    h, w, d = feat.shape
    out = np.zeros((h, w, d), dtype=np.float64)
    cnt = np.zeros((h, w), dtype=np.float64)
    for dr in range(-r_out, r_out + 1):
        for dc in range(-r_out, r_out + 1):
            if max(abs(dr), abs(dc)) <= r_in:
                continue
            if dr == 0 and dc == 0:
                continue
            src = feat[max(0, -dr): h - max(0, dr), max(0, -dc): w - max(0, dc)]
            dst = out[max(0, dr): h - max(0, -dr), max(0, dc): w - max(0, -dc)]
            cnt_dst = cnt[max(0, dr): h - max(0, -dr), max(0, dc): w - max(0, -dc)]
            dst += src
            cnt_dst += 1.0
    out /= np.maximum(cnt[..., None], 1.0)
    return out.astype(np.float32)


def ring_bank_context(
    feat_grid: np.ndarray,  # [S,H,W,D]
    r_in: int = 1, r_out: int = 2,
) -> np.ndarray:
    """Context bank over every memory patch -> [(S*H*W), D] row-normalised ring mean.

    5x5 ring with the 3x3 centre removed by default (16 neighbours).
    """
    s, h, w, d = feat_grid.shape
    out = np.empty((s, h, w, d), dtype=np.float32)
    for i in range(s):
        out[i] = ring_mean(feat_grid[i], r_in, r_out)
    return out.reshape(s * h * w, d)


def context_residual(
    q_feat: np.ndarray,      # [N,H,W,D] fused query (centre features)
    q_ctx: np.ndarray,       # [N,H,W,D] fused query ring means
    r_ctx_bank: np.ndarray,  # [(S*H*W),D] ring means of normal memory
    r_center_bank: np.ndarray,  # [(S*H*W),D] centre features of normal memory
    chunk: int = 2048,
) -> np.ndarray:
    """Per-query-patch 1 - cos(query centre, predicted normal centre).

    The prediction is the centre of the memory patch whose *ring* is most
    similar to the query ring — the centre token is never fed in.
    """
    import faiss

    bank = np.ascontiguousarray(r_ctx_bank, dtype=np.float32)
    faiss.normalize_L2(bank)
    index = faiss.IndexFlatL2(bank.shape[1])
    index.add(bank)

    qc = np.ascontiguousarray(q_ctx.reshape(-1, q_ctx.shape[-1]), dtype=np.float32)
    faiss.normalize_L2(qc)
    cb = np.ascontiguousarray(r_center_bank, dtype=np.float32)
    faiss.normalize_L2(cb)

    n_patch = q_feat.reshape(-1, q_feat.shape[-1])
    out = np.empty(n_patch.shape[0], dtype=np.float32)
    for start in range(0, n_patch.shape[0], chunk):
        end = min(start + chunk, n_patch.shape[0])
        d, idx = index.search(qc[start:end], k=1)
        pred = cb[idx[:, 0]]
        qn = n_patch[start:end].astype(np.float32)
        faiss.normalize_L2(qn)
        cos = (qn * pred).sum(axis=1)
        out[start:end] = np.clip(1.0 - cos, 0.0, 2.0).astype(np.float32)
    n, h, w = q_feat.shape[:3]
    return out.reshape(n, h, w)


def _block_rect(h: int, w: int, rng: np.random.Generator, bs: int) -> tuple[int, int]:
    r = int(rng.integers(0, h - bs + 1))
    c = int(rng.integers(0, w - bs + 1))
    return r, c


def perturb_structural(
    grid: np.ndarray,          # [H,W,D] one fused normal-test grid
    alt: np.ndarray,           # [H,W,D] another normal grid (donor for missing)
    ptype: str,
    rng: np.random.Generator,
    bs: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (perturbed grid, uint8 mask of changed patch region)."""
    g = grid.copy()
    h, w, _ = g.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    r1, c1 = _block_rect(h, w, rng, bs)
    if ptype == "permutation":
        r2, c2 = _block_rect(h, w, rng, bs)
        if abs(r1 - r2) < bs and abs(c1 - c2) < bs:  # force disjoint
            c2 = (c2 + bs) % max(1, w - bs)
        a = g[r1:r1 + bs, c1:c1 + bs].copy()
        b = g[r2:r2 + bs, c2:c2 + bs].copy()
        g[r1:r1 + bs, c1:c1 + bs] = b
        g[r2:r2 + bs, c2:c2 + bs] = a
        mask[r1:r1 + bs, c1:c1 + bs] = 1
        mask[r2:r2 + bs, c2:c2 + bs] = 1
    elif ptype == "missing":
        # structure wiped inside the block (filled by its own spatial mean), so
        # the centre no longer belongs to any local normal structure while the
        # surrounding ring is intact -> a genuinely "missing" structure.
        block = g[r1:r1 + bs, c1:c1 + bs]
        g[r1:r1 + bs, c1:c1 + bs] = block.mean(axis=(0, 1), keepdims=True)
        mask[r1:r1 + bs, c1:c1 + bs] = 1
    elif ptype == "duplicate":
        r2, c2 = _block_rect(h, w, rng, bs)
        if abs(r1 - r2) < bs and abs(c1 - c2) < bs:
            c2 = (c2 + bs) % max(1, w - bs)
        src = g[r2:r2 + bs, c2:c2 + bs].copy()
        g[r1:r1 + bs, c1:c1 + bs] = src
        mask[r1:r1 + bs, c1:c1 + bs] = 1
    else:
        raise ValueError(f"unknown perturbation: {ptype}")
    return g, mask


def dilate_mask(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    from scipy.ndimage import binary_dilation
    return binary_dilation(mask > 0, iterations=iters).astype(np.uint8)


def patch_auroc(scores: np.ndarray, pos: np.ndarray, neg: np.ndarray):
    from sklearn.metrics import roc_auc_score
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    y = np.concatenate([np.ones(int(pos.sum()), dtype=np.int64),
                        np.zeros(int(neg.sum()), dtype=np.int64)])
    s = np.concatenate([scores[pos > 0], scores[neg > 0]])
    if np.all(s == s[0]):
        return None
    return float(roc_auc_score(y, s))


# ---------------------------------------------------------------------------
# Node-only transport (RG-OT CTRL-OT-NODE proxy; set-level, no edges)
# ---------------------------------------------------------------------------

def sinkhorn_ot_cost(
    q: np.ndarray,   # [M,D] query node features (unit rows)
    a: np.ndarray,   # [K,D] anchor features (unit rows, from normal memory)
    eps: float = 0.05, iters: int = 20,
) -> float:
    """Entropic OT with 1-cos ground cost; q uniform -> a uniform."""
    cos = q @ a.T
    C = np.clip(1.0 - cos, 0.0, 2.0)
    K = np.exp(-C / eps)
    m, k = q.shape[0], a.shape[0]
    p = np.ones(m) / m          # uniform source measure over query nodes
    qt = np.ones(k) / k         # uniform target measure over anchors
    u = np.ones(m) / m
    v = np.ones(k) / k
    for _ in range(iters):
        u = p / (K @ v + 1e-12)
        v = qt / (K.T @ u + 1e-12)
    pi = u[:, None] * K * v[None, :]
    return float((pi * C).sum())


def node_only_ot_auroc(
    q_grids: list[np.ndarray],  # perturbed fused grids [H,W,D] (unit per-row ok)
    anchors: np.ndarray,        # [K,D] coreset of normal memory
    good_cost: list[float],
) -> float:
    """Per-image node OT cost, ranked against normal-image costs (higher = worse)."""
    costs = [sinkhorn_ot_cost(q.reshape(-1, q.shape[-1]), anchors) for q in q_grids]
    y = np.concatenate([np.ones(len(costs)), np.zeros(len(good_cost))])
    s = np.concatenate([costs, good_cost])
    from sklearn.metrics import roc_auc_score
    if np.all(s == s[0]):
        return 0.5
    return float(roc_auc_score(y, s))


def coreset_anchors(ref: np.ndarray, k: int = 64, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point coreset over normal memory patches (no GT used)."""
    flat = ref.reshape(-1, ref.shape[-1]).astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(0, flat.shape[0]))]
    dist = np.full(flat.shape[0], np.inf, dtype=np.float32)
    while len(idx) < min(k, flat.shape[0]):
        last = flat[idx[-1]]
        d = 1.0 - np.clip(flat @ last, -1.0, 1.0)
        dist = np.minimum(dist, d)
        idx.append(int(np.argmax(dist)))
    anchors = flat[idx]
    anchors = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + 1e-12)
    return anchors


# ---------------------------------------------------------------------------
# D3: branch-asymmetric pseudo anomaly supervision value
# ---------------------------------------------------------------------------

def branch_memory_z(
    d_feat: np.ndarray, c_feat: np.ndarray,
    d_ref: np.ndarray, c_ref: np.ndarray,
    n_images: int,
) -> tuple[np.ndarray, np.ndarray]:
    """z_D / z_C grids per image: 1-NN per-branch distance to normal memory,
    robustly calibrated by the reference pool median/MAD (label-free view)."""
    zd = _branch_z(d_feat, d_ref)
    zc = _branch_z(c_feat, c_ref)
    return zd.reshape(n_images, *d_ref.shape[1:3]), zc.reshape(n_images, *d_ref.shape[1:3])


def _branch_z(feat: np.ndarray, ref: np.ndarray, chunk: int = 16384) -> np.ndarray:
    import faiss
    q = np.ascontiguousarray(feat.reshape(-1, feat.shape[-1]), dtype=np.float32)
    r = np.ascontiguousarray(ref.reshape(-1, ref.shape[-1]), dtype=np.float32)
    faiss.normalize_L2(q)
    faiss.normalize_L2(r)
    index = faiss.IndexFlatL2(r.shape[1])
    index.add(r)
    d = np.empty(q.shape[0], dtype=np.float32)
    for start in range(0, q.shape[0], chunk):
        end = min(start + chunk, q.shape[0])
        dd, _ = index.search(q[start:end], k=1)
        d[start:end] = dd[:, 0] / 2.0
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-6
    return ((d - med) / mad).astype(np.float32)


def _random_mask(h: int, w: int, rng: np.random.Generator,
                 lo: float = 0.005, hi: float = 0.15) -> np.ndarray:
    """Rectangular or connected blob mask (50/50), area log-uniform in [lo,hi]."""
    n = h * w
    area = int(np.clip(n * float(np.exp(rng.uniform(np.log(lo), np.log(hi)))), 1, n))
    m = np.zeros((h, w), dtype=bool)
    if rng.random() < 0.5:  # rectangle
        bh = max(1, int(np.sqrt(area * (h / w))))
        bw = max(1, area // bh)
        r = int(rng.integers(0, h - bh + 1))
        c = int(rng.integers(0, w - bw + 1))
        m[r:r + bh, c:c + bw] = True
    else:  # connected blob by random walk from a seed
        cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
        pts = [(cy, cx)]
        m[cy, cx] = True
        while m.sum() < area:
            py, px = pts[rng.integers(0, len(pts))]
            dy, dx = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
            ny, nx = np.clip(py + dy, 0, h - 1), np.clip(px + dx, 0, w - 1)
            if not m[ny, nx]:
                m[ny, nx] = True
                pts.append((ny, nx))
    return m


def tangent_noise(x: np.ndarray, gamma: float, rng: np.random.Generator) -> np.ndarray:
    """Unit vector + in-plane orthogonal noise, renormalised (keeps L2 norm)."""
    x = x.astype(np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12
    xh = x / n
    # random orthonormal-ish perturbation in the tangent space
    eps = rng.normal(size=xh.shape)
    comp = eps - (eps * xh).sum(axis=-1, keepdims=True) * xh
    comp /= np.linalg.norm(comp, axis=-1, keepdims=True) + 1e-12
    out = xh + gamma * comp
    out /= np.linalg.norm(out, axis=-1, keepdims=True) + 1e-12
    return out.astype(np.float32)


def pseudo_anomaly_map(
    d_feat: np.ndarray, c_feat: np.ndarray,   # [H,W,768] unit-row normal ref grid
    d_alt: np.ndarray, c_alt: np.ndarray,     # donor normal ref grid (transplant)
    mode: str,  # "dino" | "clip" | "both"
    rng: np.random.Generator,
    gamma: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (d_map, c_map, mask). mode selects which branch is perturbed inside
    the mask: distant transplant (50%) or tangent noise (50%)."""
    dm = d_feat.copy()
    cm = c_feat.copy()
    m = _random_mask(*d_feat.shape[:2], rng)
    h, w = d_feat.shape[:2]
    kind = "tangent" if rng.random() < 0.5 else "transplant"
    for br in (("dino",) if mode == "dino" else
               ("clip",) if mode == "clip" else ("dino", "clip")):
        if kind == "tangent":
            src_d = tangent_noise(d_feat[m], gamma, rng) if br == "dino" else None
            src_c = tangent_noise(c_feat[m], gamma, rng) if br == "clip" else None
        else:
            src_d = d_alt[m] if br == "dino" else None
            src_c = c_alt[m] if br == "clip" else None
        if br == "dino":
            dm[m] = src_d
        else:
            cm[m] = src_c
    return dm, cm, m.astype(np.uint8)


def d3_statistics(dm: np.ndarray, cm: np.ndarray,
                  zd: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """CASF-style input statistics per patch: [H,W,4] = (z_D, z_C, |z_D-z_C|,
    signed disagreement). Logistically scaled by the head itself."""
    a = np.abs(zd - zc)
    sign = np.sign(zd - zc) * a  # retains direction of disagreement
    feats = np.stack([zd, zc, a, sign], axis=-1).reshape(-1, 4).astype(np.float32)
    return feats
