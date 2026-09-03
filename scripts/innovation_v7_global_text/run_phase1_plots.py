"""Task book 17 - Phase 1 plots.

Reads 01_mpdd_full/{per_config.csv,summary.json,bootstrap.json} and writes
  bootstrap_delta_plot.png   (histogram of bootstrap macro DeltaAP + CI)
  config_delta_heatmap.png   (9 configs x 6 categories DeltaAP)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/dynamic_fusion/innovation_v7_global_text/01_mpdd_full"
CATS = ["bracket_black", "bracket_brown", "bracket_white",
        "connector", "metal_plate", "tubes"]


def main() -> int:
    rows = list(csv.DictReader(open(OUT / "per_config.csv", encoding="utf-8")))
    boot = json.loads((OUT / "bootstrap.json").read_text(encoding="utf-8"))

    # ---- heatmap ----
    cfg_labels = [f"s{s}_k{k}" for s in (0, 1, 2) for k in (1, 2, 4)]
    grid = np.zeros((len(cfg_labels), len(CATS)))
    for r in rows:
        i = cfg_labels.index(f"s{r['seed']}_k{r['shot']}")
        j = CATS.index(r["category"])
        grid[i, j] = float(r["dap_text"])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, cmap="RdBu_r", vmin=-0.20, vmax=0.25, aspect="auto")
    ax.set_xticks(range(len(CATS))); ax.set_xticklabels(CATS, rotation=30,
                                                        ha="right")
    ax.set_yticks(range(len(cfg_labels))); ax.set_yticklabels(cfg_labels)
    for i in range(len(cfg_labels)):
        for j in range(len(CATS)):
            ax.text(j, i, f"{grid[i, j]:+.3f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(grid[i, j]) > 0.12 else "black")
    ax.set_title("TEXT - A1-max: Delta Image-AP per (config x category)")
    fig.colorbar(im, label="Delta Image-AP")
    fig.tight_layout()
    fig.savefig(OUT / "config_delta_heatmap.png", dpi=150)
    plt.close(fig)

    # ---- bootstrap histogram ----
    bins = np.linspace(-0.10, 0.20, 51)
    counts = np.asarray(boot["hist_dap_bins"])
    centers = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(centers, counts / counts.sum(), width=(bins[1] - bins[0]),
           color="#9ecae1", edgecolor="white")
    pr = boot["primary"]
    for x, lab in ((pr["mean_dap"], "mean"),):
        ax.axvline(x, color="tab:red", lw=1.5)
        ax.text(x, ax.get_ylim()[1] * 0.95, f" mean={x:+.3f}",
                color="tab:red", ha="left", fontsize=9)
    lo, hi = pr["ci95_dap"]
    ax.axvspan(lo, hi, color="tab:red", alpha=0.12)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("bootstrap macro mean Delta Image-AP")
    ax.set_ylabel("density")
    ax.set_title(f"Paired stratified bootstrap (B={pr['b']}, seed={pr['seed']})\n"
                 f"95% CI [{lo:+.3f}, {hi:+.3f}]  P(d>0)={pr['p_dap_gt0']}")
    fig.tight_layout()
    fig.savefig(OUT / "bootstrap_delta_plot.png", dpi=150)
    plt.close(fig)
    print("plots written:", OUT / "config_delta_heatmap.png",
          OUT / "bootstrap_delta_plot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
