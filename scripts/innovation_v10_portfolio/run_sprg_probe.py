"""Route F — SPRG part-discovery feasibility probe (task book 19 §9.4 gate 1).

For each of the 6 MPDD classes (seed0 k4): discover k=6 part nodes per normal
reference image (spatial-constrained KMeans on DINO-vitb14 patch features) and
chain-match nodes across the 4 reference images. Report mean matching success
rate. If < 0.90 -> immediate stop (archive), no relation-graph work.

  .venv-patchcore\\Scripts\\python.exe scripts\\innovation_v10_portfolio\\run_sprg_probe.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from industrial_ad.innovation_v10_portfolio.common import load_features

K_PARTS = 6
MATCH_COS = 0.85
MATCH_POS = 0.15
SEED = 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _l2unit(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return x / norms


def discover_parts(feat: np.ndarray) -> dict:
    """feat [H, W, D] -> part nodes dict with appearance/position/area/moments.

    Spatial-constrained KMeans on [L2norm(feature), (row/16, col/16) * 0.5].
    """
    from sklearn.cluster import KMeans

    h, w, d = feat.shape
    f = feat.reshape(-1, d).astype(np.float32)
    f = _l2unit(f)
    rows, cols = np.mgrid[0:h, 0:w]
    coords = np.stack([rows.ravel(), cols.ravel()], axis=1).astype(np.float32) / 16.0 * 0.5
    x = np.concatenate([f, coords], axis=1)
    km = KMeans(n_clusters=K_PARTS, random_state=SEED, n_init=4, max_iter=100)
    labels = km.fit_predict(x)
    nodes = []
    for c in range(K_PARTS):
        member = np.flatnonzero(labels == c)
        if member.size == 0:
            continue
        y = rows.ravel()[member]
        xx = cols.ravel()[member]
        nodes.append({
            "appearance": _l2unit(f[member].mean(axis=0)).astype(np.float32),
            "pos": np.array([y.mean(), xx.mean()]) / max(h, w),       # normalized [0,1]
            "area": float(member.size),
            "spread": float(np.sqrt(((y - y.mean()) ** 2 + (xx - xx.mean()) ** 2).mean())
                            / max(h, w)),
        })
    # fill missing clusters deterministically (shouldn't happen with k < n_patches)
    while len(nodes) < K_PARTS:
        nodes.append({"appearance": nodes[0]["appearance"].copy(),
                      "pos": np.array([0.0, 0.0]), "area": 0.0, "spread": 0.0})
    return nodes


def chain_match(seq_nodes: list[list[dict]]) -> dict:
    """Hungarian matching across consecutive images; returns success statistics."""
    from scipy.optimize import linear_sum_assignment

    links = []
    for t in range(len(seq_nodes) - 1):
        a, b = seq_nodes[t], seq_nodes[t + 1]
        cost = np.zeros((K_PARTS, K_PARTS))
        for i, na in enumerate(a):
            for j, nb in enumerate(b):
                ca = 1.0 - float(na["appearance"] @ nb["appearance"])
                dp = float(np.abs(na["pos"] - nb["pos"]).sum())
                cost[i, j] = ca + 0.5 * dp
        ri, ci = linear_sum_assignment(cost)
        ok = 0
        cos_vals = []
        for i, j in zip(ri, ci):
            c = float(a[i]["appearance"] @ b[j]["appearance"])
            cos_vals.append(c)
            dp = float(np.abs(a[i]["pos"] - b[j]["pos"]).sum())
            if c >= MATCH_COS and dp <= MATCH_POS:
                ok += 1
        links.append({"ok": ok, "total": K_PARTS,
                      "mean_cos": float(np.mean(cos_vals)),
                      "rate": ok / K_PARTS})
    # chain-consistent success: node stable if it matched ok on ALL links
    chain_ok = 0
    for i in range(K_PARTS):
        stable = True
        for t in range(len(seq_nodes) - 1):
            a, b = seq_nodes[t], seq_nodes[t + 1]
            # cost-min pair for node i in link t
            j = int(np.argmin([(1.0 - float(a[i]["appearance"] @ b[kk]["appearance"]))
                               + 0.5 * float(np.abs(a[i]["pos"] - b[kk]["pos"]).sum())
                               for kk in range(K_PARTS)]))
            c = float(a[i]["appearance"] @ b[j]["appearance"])
            dp = float(np.abs(a[i]["pos"] - b[j]["pos"]).sum())
            if not (c >= MATCH_COS and dp <= MATCH_POS):
                stable = False
                break
        chain_ok += int(stable)
    return {"links": links, "chain_stable_nodes": chain_ok, "chain_total": K_PARTS,
            "chain_rate": chain_ok / K_PARTS}


def run_category(dino_cache: Path, manifest: dict, cat: str) -> dict:
    d = load_features(dino_cache / f"{cat}.npz")
    refs = d["ref_patch_features"]  # [K, H, W, D] (K=4 at shot 4)
    seq = []
    for r in range(refs.shape[0]):
        seq.append(discover_parts(refs[r]))
    res = chain_match(seq)
    return {"category": cat, "n_refs": int(refs.shape[0]),
            "chain_rate": round(res["chain_rate"], 4),
            "chain_stable_nodes": res["chain_stable_nodes"],
            "link_rates": [round(l["rate"], 3) for l in res["links"]],
            "link_mean_cos": [round(l["mean_cos"], 4) for l in res["links"]]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dino-cache", type=Path,
                        default=ROOT / "outputs/dynamic_fusion/v3_direction_a/features_vitb14_s0_k4/anomalydino_visual")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/mpdd/manifest.json")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "experiments/dynamic_fusion/innovation_v10_portfolio/sprg")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = out_dir / "R0_PROTOCOL.json"
    if not protocol.is_file():
        raise SystemExit(f"missing pre-registered protocol: {protocol}")

    rows = []
    t0 = time.time()
    for cat in sorted(manifest["categories"]):
        p = args.dino_cache / f"{cat}.npz"
        if not p.is_file():
            continue
        r = run_category(args.dino_cache, manifest, cat)
        rows.append(r)
        print(f"[SPRG {cat}] chain rate={r['chain_rate']:.2%} "
              f"links={r['link_rates']} cos={r['link_mean_cos']}", flush=True)

    mean_rate = float(np.mean([r["chain_rate"] for r in rows])) if rows else float("nan")
    report = {
        "route": "F_SPRG",
        "pipeline": "v10_portfolio_r0_feasibility_gate1",
        "seed": 0, "shot": 4, "k_parts": K_PARTS,
        "pre_registered": {"match_cos": MATCH_COS, "match_pos_norm": MATCH_POS},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(protocol),
        "code_sha256": {"runner": sha256_file(Path(__file__))},
        "per_category": rows,
        "mean_chain_rate": round(mean_rate, 4),
        "gate1_pass_ge_0.90": mean_rate >= 0.90,
        "decision": "PENDING",
    }
    (out_dir / "R0_RESULT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nmean chain rate = {report['mean_chain_rate']:.2%} | "
          f"gate1 (>=90%) = {report['gate1_pass_ge_0.90']} | elapsed {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
