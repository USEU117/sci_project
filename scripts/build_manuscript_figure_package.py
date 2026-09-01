"""Build the manuscript-ready scientific figure package.

The script reads only frozen/project evidence and local dataset images. It
creates SVG, PDF, and 600-dpi PNG versions for quantitative diagrams, plus
600-dpi PNG qualitative panels whose source images must not be redistributed
with the compact reproducibility package.

Usage:
    .venv-patchcore/Scripts/python.exe scripts/build_manuscript_figure_package.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import colors  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / "docs" / "paper_writing_preparation_20260830"
OUT = PREP / "figures_20260830"
PNG_DIR = OUT / "png_600dpi"
SVG_DIR = OUT / "svg_editable"
PDF_DIR = OUT / "pdf_vector"
DATA_DIR = OUT / "source_data"
RESTRICTED_DIR = OUT / "qualitative_local_only"

P1 = ROOT / "submission_repro_20260827" / "evidence" / "p1"
COMPLETE_JSON = P1 / "p1_e_complete_metrics.json"
BOOTSTRAP_JSON = P1 / "p1_a_bootstrap_ci.json"
EFFICIENCY_JSON = P1 / "p1_c_efficiency.json"
QUAL_MANIFEST = P1 / "p1_b_figures_manifest.json"
CLIP_SUMMARY = (
    ROOT
    / "experiments"
    / "dynamic_fusion"
    / "v3_direction_a"
    / "clip_only_controls_20260830"
    / "summary.json"
)
MAPS_ROOT = ROOT / "submission_repro_20260827" / "predictions_compact" / "maps"

DATA_ROOTS = {
    "mpdd": ROOT / "data" / "mpdd_raw" / "MPDD",
    "btad": ROOT / "data" / "btad_raw",
    "visa": ROOT / "data" / "visa_raw",
    "mvtec": ROOT / "data" / "mvtec",
}

DATASET_LABELS = {
    "mpdd": "MPDD",
    "btad": "BTAD",
    "visa": "VisA",
    "mvtec": "MVTec AD",
}
DATASET_ORDER = ["mpdd", "btad", "visa", "mvtec"]

# Colorblind-safe palette (Okabe-Ito inspired).
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
PURPLE = "#6A51A3"
SKY = "#56B4E9"
GREY = "#6B7280"
LIGHT_GREY = "#E5E7EB"
DARK = "#111827"

FORMATS = ("png", "svg", "pdf")
QUANTITATIVE_NAMES: list[str] = []
MANIFEST_ROWS: list[dict] = []


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "patch.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def ensure_dirs() -> None:
    for path in (PNG_DIR, SVG_DIR, PDF_DIR, DATA_DIR, RESTRICTED_DIR):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def register(path: Path, figure_id: str, distribution: str) -> None:
    row = {
        "figure_id": figure_id,
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "format": path.suffix.lower().lstrip("."),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "distribution": distribution,
    }
    if path.suffix.lower() == ".png":
        with Image.open(path) as img:
            row["pixel_width"], row["pixel_height"] = img.size
            row["dpi"] = [round(float(v), 1) for v in img.info.get("dpi", (0, 0))]
    MANIFEST_ROWS.append(row)


def save_quantitative(fig: plt.Figure, name: str) -> None:
    QUANTITATIVE_NAMES.append(name)
    for fmt in FORMATS:
        if fmt == "png":
            path = PNG_DIR / f"{name}.png"
            fig.savefig(path, dpi=600)
        elif fmt == "svg":
            path = SVG_DIR / f"{name}.svg"
            fig.savefig(path)
        else:
            path = PDF_DIR / f"{name}.pdf"
            fig.savefig(path)
        register(path, name.split("_")[0], "manuscript-ready")
    plt.close(fig)


def save_qualitative(fig: plt.Figure, name: str) -> None:
    path = RESTRICTED_DIR / f"{name}.png"
    fig.savefig(path, dpi=600)
    plt.close(fig)
    register(path, name.split("_")[0], "local manuscript use only; contains dataset images")


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    path = DATA_DIR / name
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def box(ax, x, y, w, h, text, face, edge=DARK, fontsize=8.3, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face, edgecolor=edge, linewidth=0.9,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=DARK, weight=weight, linespacing=1.25)
    return patch


def arrow(ax, x1, y1, x2, y2, color=GREY, style="-|>", lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=10, linewidth=lw, color=color,
                                 shrinkA=2, shrinkB=2))


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.04, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left", color=DARK)


def fig01_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.02, 0.56, 0.14, 0.19, "Query image\n+ normal references", "#F3F4F6",
        fontsize=7.1, weight="bold")
    arrow(ax, 0.16, 0.66, 0.205, 0.78)
    arrow(ax, 0.16, 0.66, 0.205, 0.50)

    box(ax, 0.21, 0.70, 0.18, 0.17,
        "Frozen DINOv2\nViT-B/14 · 448 px\n768-D patches", "#DCEEFF", edge=BLUE,
        fontsize=7.2, weight="bold")
    box(ax, 0.21, 0.41, 0.18, 0.18,
        "Frozen AnomalyCLIP\nimage tower · 518 px\n768-D patches", "#FFF0D5", edge=ORANGE,
        fontsize=7.2, weight="bold")
    ax.text(0.30, 0.355, "No text embeddings", ha="center", va="center",
            fontsize=7.2, color=VERMILLION, weight="bold")

    arrow(ax, 0.39, 0.785, 0.425, 0.69)
    arrow(ax, 0.39, 0.50, 0.425, 0.61)
    box(ax, 0.43, 0.56, 0.15, 0.19,
        "Patch alignment\nCLIP → DINO grid\nbilinear resize", "#ECFDF5", edge=GREEN,
        fontsize=6.9)
    arrow(ax, 0.58, 0.655, 0.615, 0.655)
    box(ax, 0.62, 0.53, 0.18, 0.25,
        "Branch-wise L2\n0.5 / 0.5 concat\n768 + 768 = 1536-D\nglobal L2", "#F0EAFE", edge=PURPLE,
        fontsize=6.9)
    arrow(ax, 0.80, 0.655, 0.835, 0.655)
    box(ax, 0.84, 0.55, 0.14, 0.21,
        "Normal memory\nFAISS k = 1\ndistance / 2", "#FDE8E4", edge=VERMILLION,
        fontsize=7.1, weight="bold")

    arrow(ax, 0.91, 0.55, 0.91, 0.36)
    box(ax, 0.82, 0.15, 0.16, 0.17,
        "Anomaly map\nσ = 4 · 448 × 448\nstride-8 evaluation", "#F3F4F6",
        fontsize=7.0, weight="bold")

    box(ax, 0.05, 0.13, 0.22, 0.13, "Trainable parameters = 0", "#E8F7EE", edge=GREEN,
        fontsize=7.1, weight="bold")
    box(ax, 0.32, 0.13, 0.36, 0.13,
        "Matched control\nsame references + evaluator\nDINO branch only",
        "#EEF5FB", edge=BLUE, fontsize=6.7)
    ax.text(0.02, 0.975, "Frozen dual-encoder visual feature fusion", fontsize=9.4,
            weight="bold", color=DARK, va="top")
    save_quantitative(fig, "Fig01_method_overview")


def fig02_protocol() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.65))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    xs = [0.025, 0.27, 0.515, 0.76]
    widths = [0.20, 0.20, 0.20, 0.215]
    texts = [
        "MPDD\nDevelopment only\nmethod + weight choice",
        "Protocol freeze\nA1 fixed at 0.5 / 0.5\nno further selection",
        "Frozen validation\nBTAD · MVTec AD\nexternal",
        "Frozen validation\nVisA\nin-domain checkpoint lineage",
    ]
    colors_ = ["#DCEEFF", "#E8F7EE", "#FFF0D5", "#F0EAFE"]
    edges = [BLUE, GREEN, ORANGE, PURPLE]
    for i, (x, w, txt, fc, ec) in enumerate(zip(xs, widths, texts, colors_, edges)):
        box(ax, x, 0.50, w, 0.29, txt, fc, edge=ec, fontsize=7.0,
            weight="bold" if i == 1 else "normal")
        if i < len(xs) - 1:
            arrow(ax, x + w, 0.655, xs[i + 1], 0.655)

    ax.plot([0.03, 0.97], [0.40, 0.40], color=LIGHT_GREY, lw=1.0)
    box(ax, 0.03, 0.10, 0.27, 0.17, "3 seeds × 1/2/4 shots\nnormal references only", "#F9FAFB",
        fontsize=7.1)
    box(ax, 0.365, 0.10, 0.27, 0.17, "Same reference IDs\nsame evaluator", "#F9FAFB",
        fontsize=7.1)
    box(ax, 0.70, 0.10, 0.27, 0.17, "No test labels, masks,\nor test statistics", "#FDE8E4",
        edge=VERMILLION, fontsize=7.1, weight="bold")
    ax.text(0.03, 0.94, "Leakage-safe development and validation contract", fontsize=9.6,
            weight="bold", color=DARK, va="top")
    save_quantitative(fig, "Fig02_protocol_and_dataset_roles")


def fig03_configuration_gains(bootstrap: dict) -> None:
    rows = bootstrap["configs"]
    write_csv(
        "Fig03_configuration_gains.csv",
        ["dataset", "seed", "shot", "delta_pixel_ap", "category_ci_lo", "category_ci_hi"],
        [
            {
                "dataset": r["dataset"], "seed": r["seed"], "shot": r["shot"],
                "delta_pixel_ap": r["full_sample_delta_ap"],
                "category_ci_lo": r["category_bootstrap"]["lo"],
                "category_ci_hi": r["category_bootstrap"]["hi"],
            }
            for r in rows
        ],
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.55), sharex=True)
    seed_colors = [BLUE, ORANGE, GREEN]
    markers = ["o", "s", "^"]
    for idx, (ax, ds) in enumerate(zip(axes.flat, DATASET_ORDER)):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        for seed in (0, 1, 2):
            subset = sorted([r for r in ds_rows if r["seed"] == seed], key=lambda r: r["shot"])
            x = [r["shot"] for r in subset]
            y = [r["full_sample_delta_ap"] for r in subset]
            ax.plot(x, y, marker=markers[seed], color=seed_colors[seed], markersize=4.5,
                    label=f"Seed {seed}", zorder=3)
        ax.axhline(0, color=DARK, lw=0.8)
        ax.set_title(DATASET_LABELS[ds], weight="bold")
        ax.set_xticks([1, 2, 4], ["1", "2", "4"])
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.6, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.003, 0.068 if ds == "visa" else 0.046)
        if idx % 2 == 0:
            ax.set_ylabel("Δ Pixel AP (A1 − DINO)")
        if idx >= 2:
            ax.set_xlabel("Normal references per category (shot)")
        panel_label(ax, chr(ord("a") + idx))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.94, bottom=0.14, hspace=0.34, wspace=0.24)
    save_quantitative(fig, "Fig03_configuration_level_pixel_ap_gains")


def category_rows(bootstrap: dict) -> list[dict]:
    return sorted(bootstrap["worst_categories_by_dataset"], key=lambda r: r["mean_delta_ap"])


def draw_category_bars(rows: list[dict], name: str, height: float) -> None:
    labels = [f"{DATASET_LABELS[r['dataset']]} · {r['category']}" for r in rows]
    values = [r["mean_delta_ap"] for r in rows]
    y = np.arange(len(rows))
    bar_colors = [VERMILLION if v < 0 else BLUE for v in values]
    fig, ax = plt.subplots(figsize=(7.1, height))
    ax.barh(y, values, color=bar_colors, edgecolor="white", height=0.72)
    ax.axvline(0, color=DARK, lw=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean Δ Pixel AP across nine configurations (A1 − DINO)")
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for yi, val in zip(y, values):
        ha = "left" if val >= 0 else "right"
        offset = 0.0025 if val >= 0 else -0.0025
        ax.text(val + offset, yi, f"{val:+.3f}", va="center", ha=ha, fontsize=7.2,
                color=DARK)
    ax.set_xlim(min(values) - 0.025, max(values) + 0.027)
    fig.subplots_adjust(left=0.28, right=0.98, top=0.98, bottom=0.10)
    save_quantitative(fig, name)


def fig04_category_heterogeneity(bootstrap: dict) -> None:
    rows = category_rows(bootstrap)
    write_csv(
        "Fig04_category_heterogeneity.csv",
        ["dataset", "category", "n_configs", "mean_delta_pixel_ap", "negative_configs"],
        [
            {
                "dataset": r["dataset"], "category": r["category"],
                "n_configs": r["n_configs"], "mean_delta_pixel_ap": r["mean_delta_ap"],
                "negative_configs": r["negative_configs"],
            }
            for r in rows
        ],
    )
    main_rows = rows[:10] + rows[-10:]
    draw_category_bars(main_rows, "Fig04_category_gain_loss_extremes", 5.35)
    draw_category_bars(rows, "FigS01_all_category_gain_loss", 8.4)


def fig_s02_shot_wise(bootstrap: dict) -> None:
    rows = bootstrap["shot_wise"]
    write_csv(
        "FigS02_shot_wise_gain.csv",
        ["dataset", "shot", "mean_delta_pixel_ap", "seed_std", "n_seeds"],
        [
            {
                "dataset": r["dataset"], "shot": r["shot"],
                "mean_delta_pixel_ap": r["mean_delta_ap"], "seed_std": r["std_delta_ap"],
                "n_seeds": r["n_seeds"],
            }
            for r in rows
        ],
    )
    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    for ds, color_, marker in zip(DATASET_ORDER, [BLUE, ORANGE, PURPLE, GREEN], ["o", "s", "^", "D"]):
        subset = sorted([r for r in rows if r["dataset"] == ds], key=lambda r: r["shot"])
        ax.errorbar(
            [r["shot"] for r in subset], [r["mean_delta_ap"] for r in subset],
            yerr=[r["std_delta_ap"] for r in subset], color=color_, marker=marker,
            capsize=3, markersize=5, label=DATASET_LABELS[ds],
        )
    ax.axhline(0, color=DARK, lw=0.8)
    ax.set_xticks([1, 2, 4], ["1", "2", "4"])
    ax.set_xlabel("Normal references per category (shot)")
    ax.set_ylabel("Mean Δ Pixel AP ± seed SD")
    ax.grid(color=LIGHT_GREY, lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_ylim(0, 0.067)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.17)
    save_quantitative(fig, "FigS02_shot_wise_gain_stability")


def fig05_metric_delta_heatmap(complete: dict) -> None:
    metrics = ["image_auroc", "image_ap", "image_f1_max", "pixel_auroc", "pixel_ap", "pixel_aupro"]
    metric_labels = ["Image\nAUROC", "Image\nAP", "Image\nF1-max", "Pixel\nAUROC", "Pixel\nAP", "Pixel\nAUPRO"]
    by_ds = {r["dataset"]: r for r in complete["dataset_deltas_A1_minus_DINO"]}
    matrix = np.array([[by_ds[ds][f"delta_{m}"] for m in metrics] for ds in DATASET_ORDER])
    rows = []
    for i, ds in enumerate(DATASET_ORDER):
        for j, metric in enumerate(metrics):
            rows.append({"dataset": ds, "metric": metric, "delta_A1_minus_DINO": matrix[i, j]})
    write_csv("Fig05_complete_metric_deltas.csv", ["dataset", "metric", "delta_A1_minus_DINO"], rows)

    vmax = float(np.max(np.abs(matrix)))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = colors.LinearSegmentedColormap.from_list("div", [VERMILLION, "#FFFFFF", BLUE])
    fig, ax = plt.subplots(figsize=(7.1, 3.05))
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)), metric_labels)
    ax.set_yticks(np.arange(len(DATASET_ORDER)), [DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.tick_params(length=0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if abs(val) > 0.036 else DARK
            ax.text(j, i, f"{val:+.4f}", ha="center", va="center", fontsize=8.2,
                    color=color, weight="bold" if abs(val) > 0.02 else "normal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.025)
    cbar.set_label("A1 − DINO", rotation=270, labelpad=12)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Dataset")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.13, right=0.94, top=0.95, bottom=0.22)
    save_quantitative(fig, "Fig05_complete_metric_delta_heatmap")


def fig06_three_way(complete: dict, clip: dict) -> None:
    a1_rows = {(r["dataset"], r["method"]): r for r in complete["dataset_wise"]}
    clip_rows = {r["dataset"]: r for r in clip["dataset_rows"]}
    methods = ["CLIP image only", "DINO only", "A1 fixed concat"]
    method_colors = [ORANGE, BLUE, GREEN]
    source_rows = []
    values = {}
    for ds in ("btad", "mvtec"):
        values[ds] = [
            (clip_rows[ds]["pixel_ap_mean"], clip_rows[ds]["pixel_ap_std"]),
            (a1_rows[(ds, "feature_DINO_only")]["pixel_ap_mean"], a1_rows[(ds, "feature_DINO_only")]["pixel_ap_std"]),
            (a1_rows[(ds, "A1_concat")]["pixel_ap_mean"], a1_rows[(ds, "A1_concat")]["pixel_ap_std"]),
        ]
        for method, (mean, std) in zip(methods, values[ds]):
            source_rows.append({"dataset": ds, "method": method, "pixel_ap_mean": mean, "pixel_ap_std": std})
    write_csv("Fig06_three_way_pixel_ap.csv", ["dataset", "method", "pixel_ap_mean", "pixel_ap_std"], source_rows)

    fig, ax = plt.subplots(figsize=(7.1, 3.35))
    x = np.arange(2)
    width = 0.22
    for j, (method, color_) in enumerate(zip(methods, method_colors)):
        means = [values[ds][j][0] for ds in ("btad", "mvtec")]
        stds = [values[ds][j][1] for ds in ("btad", "mvtec")]
        bars = ax.bar(x + (j - 1) * width, means, width, yerr=stds, capsize=3,
                      color=color_, edgecolor="white", label=method)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.018, f"{mean:.4f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=0)
    ax.set_xticks(x, ["BTAD", "MVTec AD"])
    ax.set_ylabel("Pixel AP (mean ± SD)")
    ax.set_ylim(0, 0.74)
    ax.grid(axis="y", color=LIGHT_GREY, lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.15))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.16)
    save_quantitative(fig, "Fig06_three_way_clip_dino_a1_pixel_ap")


def fig07_efficiency(eff: dict) -> None:
    bench = eff["steady_state_end_to_end_benchmark"]
    stages = [
        ("DINO extraction", bench["dino_feature_extraction"]["mean_seconds"], BLUE),
        ("CLIP extraction", bench["clip_feature_extraction"]["mean_seconds"], ORANGE),
        ("Align + concat + k-NN", bench["align_concat_knn"]["mean_seconds"], GREEN),
    ]
    banks = eff["memory_bank_float32"]
    write_csv(
        "Fig07_efficiency_latency.csv", ["stage", "seconds_per_image"],
        [{"stage": n, "seconds_per_image": v} for n, v, _ in stages],
    )
    write_csv(
        "Fig07_efficiency_memory.csv", ["dataset", "shot", "concat_bank_mb_f32"],
        [{"dataset": r["dataset"], "shot": r["shot"], "concat_bank_mb_f32": r["concat_bank_mb_f32"]} for r in banks],
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25))
    ax = axes[0]
    left = 0.0
    for name, value, color_ in stages:
        ax.barh([0], [value], left=left, height=0.42, color=color_, edgecolor="white", label=name)
        if value > 0.055:
            ax.text(left + value / 2, 0, f"{value:.3f}s", ha="center", va="center",
                    fontsize=7.3, color="white", weight="bold")
        left += value
    ax.set_yticks([0], ["A1"])
    ax.set_xlabel("Steady-state latency (s/image)")
    ax.set_xlim(0, 0.45)
    ax.grid(axis="x", color=LIGHT_GREY, lw=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)
    ax.text(0.4146, 0.31, "Total 0.4146 s/image\n2.412 image/s", ha="right", va="bottom",
            fontsize=7.5, color=DARK)
    ax.set_title("Runtime breakdown", weight="bold")
    panel_label(ax, "a")

    ax = axes[1]
    for ds, color_, marker in zip(DATASET_ORDER, [BLUE, ORANGE, PURPLE, GREEN], ["o", "s", "^", "D"]):
        subset = sorted([r for r in banks if r["dataset"] == ds], key=lambda r: r["shot"])
        ax.plot([r["shot"] for r in subset], [r["concat_bank_mb_f32"] for r in subset],
                color=color_, marker=marker, label=DATASET_LABELS[ds], markersize=4.5)
    ax.set_xticks([1, 2, 4], ["1", "2", "4"])
    ax.set_xlabel("Normal references per category (shot)")
    ax.set_ylabel("A1 memory bank (MB, float32)")
    ax.grid(color=LIGHT_GREY, lw=0.6, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title("Reference-memory scaling", weight="bold")
    panel_label(ax, "b")
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.28, wspace=0.33)
    save_quantitative(fig, "Fig07_efficiency_and_memory_cost")


def load_recompute_helpers():
    package_root = ROOT / "submission_repro_20260827"
    sys.path.insert(0, str(package_root))
    from recompute_tables import (  # type: ignore
        MAP_SIZE, build_visa_mask_map, dists2map, load_mask_for_sample,
    )
    return MAP_SIZE, build_visa_mask_map, dists2map, load_mask_for_sample


def qualitative_case_arrays(case: dict, visa_mask_map, helpers):
    map_size, _, dists2map, load_mask_for_sample = helpers
    ds = case["dataset"]
    npz_path = MAPS_ROOT / ds / f"s{case['seed']}_k{case['shot']}" / f"{case['category']}.npz"
    with np.load(npz_path, allow_pickle=False) as data:
        sample_ids = [str(x) for x in data["sample_ids"]]
        idx = sample_ids.index(case["sample_id"])
        dino = dists2map(data["dino_patch_map"][idx].astype(np.float32), map_size)
        a1 = dists2map(data["concat_patch_map"][idx].astype(np.float32), map_size)
    root = DATA_ROOTS[ds]
    image_path = root / case["sample_id"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_size = tuple(map_size) if isinstance(map_size, (tuple, list)) else (map_size, map_size)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    mask = load_mask_for_sample(ds, case["sample_id"], root, visa_mask_map, map_size)
    return image, mask, dino, a1


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, vmax: float) -> np.ndarray:
    normed = np.clip(heatmap / max(vmax, 1e-12), 0, 1)
    colored = plt.get_cmap("magma")(normed)[..., :3]
    base = image.astype(np.float32) / 255.0
    return np.clip(0.52 * base + 0.48 * colored, 0, 1)


def draw_qualitative(cases: list[dict], name: str, role_label: str, helpers, visa_mask_map) -> None:
    n = len(cases)
    fig, axes = plt.subplots(
        n, 5, figsize=(7.1, 2.35 * n), squeeze=False,
        gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.045]},
    )
    for i, case in enumerate(cases):
        image, mask, dino, a1 = qualitative_case_arrays(case, visa_mask_map, helpers)
        vmax = float(max(np.max(dino), np.max(a1)))
        gt_overlay = image.astype(np.float32) / 255.0
        contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        gt_bgr = cv2.cvtColor((gt_overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.drawContours(gt_bgr, contours, -1, (255, 255, 0), 3)
        gt_overlay = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        panels = [image, gt_overlay, overlay_heatmap(image, dino, vmax), overlay_heatmap(image, a1, vmax)]
        titles = [
            "Input", "Ground truth",
            f"DINO only\nP-AP {case['dino_pixel_ap']:.3f}",
            f"A1 concat\nP-AP {case['concat_pixel_ap']:.3f}",
        ]
        for j, (arr, title) in enumerate(zip(panels, titles)):
            axes[i, j].imshow(arr)
            axes[i, j].set_title(title, fontsize=8.2, weight="bold" if j >= 2 else "normal", pad=3)
            axes[i, j].axis("off")
        label = f"{DATASET_LABELS[case['dataset']]} · {case['category']} · ΔAP {case['delta_ap']:+.3f}"
        axes[i, 0].text(0.0, -0.08, label, transform=axes[i, 0].transAxes,
                        ha="left", va="top", fontsize=7.5, color=DARK)
        # A compact per-row score scale. Both method overlays share this raw-score maximum.
        sm = plt.cm.ScalarMappable(cmap="magma", norm=colors.Normalize(vmin=0, vmax=vmax))
        cbar = fig.colorbar(sm, cax=axes[i, 4])
        cbar.ax.tick_params(labelsize=6.5, length=2)
        cbar.set_label("Anomaly score", fontsize=7, labelpad=3)
    fig.text(0.01, 0.995, role_label, ha="left", va="top", fontsize=9.5, weight="bold", color=DARK)
    fig.subplots_adjust(left=0.015, right=0.985, top=0.88, bottom=0.045, wspace=0.045, hspace=0.48)
    save_qualitative(fig, name)


def fig08_09_qualitative(qual: dict) -> None:
    helpers = load_recompute_helpers()
    _, build_visa_mask_map, _, _ = helpers
    visa_mask_map = build_visa_mask_map(DATA_ROOTS["visa"])
    figures = qual["figures"]
    success_keys = {("visa", "cashew"), ("mvtec", "toothbrush")}
    failure_keys = {("visa", "chewinggum"), ("mvtec", "leather")}
    successes = [r for r in figures if (r["dataset"], r["category"]) in success_keys]
    failures = [r for r in figures if (r["dataset"], r["category"]) in failure_keys]
    draw_qualitative(successes, "Fig08_qualitative_successes", "Representative localization improvements", helpers, visa_mask_map)
    draw_qualitative(failures, "Fig09_qualitative_failures", "Representative negative-transfer cases", helpers, visa_mask_map)

    rows = []
    for case in successes + failures:
        rows.append({
            "figure": "Fig08" if case in successes else "Fig09",
            "dataset": case["dataset"], "category": case["category"],
            "seed": case["seed"], "shot": case["shot"], "sample_id": case["sample_id"],
            "dino_pixel_ap": case["dino_pixel_ap"], "a1_pixel_ap": case["concat_pixel_ap"],
            "delta_pixel_ap": case["delta_ap"], "selection_rule": case["selection_rule"],
        })
    write_csv(
        "Fig08_Fig09_qualitative_cases.csv",
        ["figure", "dataset", "category", "seed", "shot", "sample_id", "dino_pixel_ap",
         "a1_pixel_ap", "delta_pixel_ap", "selection_rule"], rows,
    )


def make_contact_sheet() -> None:
    paths = [PNG_DIR / f"{name}.png" for name in QUANTITATIVE_NAMES]
    paths += [RESTRICTED_DIR / "Fig08_qualitative_successes.png",
              RESTRICTED_DIR / "Fig09_qualitative_failures.png"]
    thumbs = []
    for path in paths:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            rgb.thumbnail((900, 560), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (940, 620), "white")
            canvas.paste(rgb, ((940 - rgb.width) // 2, 35))
            thumbs.append((path.stem, canvas))
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * 940, rows * 660), "#D1D5DB")
    for idx, (name, img) in enumerate(thumbs):
        x = (idx % cols) * 940
        y = (idx // cols) * 660
        sheet.paste(img, (x, y))
        cv_img = np.array(sheet)
        cv2.putText(cv_img, name, (x + 18, y + 642), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (17, 24, 39), 1, cv2.LINE_AA)
        sheet = Image.fromarray(cv_img)
    path = OUT / "CONTACT_SHEET.png"
    sheet.save(path, dpi=(150, 150))
    register(path, "QA", "internal visual QA")


def write_caption_guide() -> None:
    text = """# Manuscript Figure Package

This directory contains Word-ready manuscript figures generated from frozen project evidence.

## Recommended Word use

- Prefer the SVG file in modern Microsoft Word because labels remain sharp and editable.
- If the journal system or Word version mishandles SVG, use the 600-dpi PNG.
- Insert at the intended final width; do not enlarge beyond 17.8 cm (7.0 in).
- Keep aspect ratio locked. Do not crop axes, legends, color bars, or panel labels.
- PDF files are archival/vector alternatives and are usually best for the final publisher upload.
- Quantitative files are publication-ready. `qualitative_local_only/` contains benchmark images and must not be copied into the public reproducibility package.

## Figure captions

**Figure 1. Overview of the frozen dual-encoder visual feature fusion pipeline.** DINOv2 and the AnomalyCLIP image tower independently extract 768-D patch descriptors. After spatial alignment and branch-wise normalization, the descriptors are concatenated with fixed equal scaling and stored in a category-specific normal memory. Nearest-neighbour distance produces the anomaly map. No text embedding or trainable parameter is used.

**Figure 2. Development and frozen-validation protocol.** MPDD is used for method development before the A1 configuration is frozen. BTAD and MVTec AD provide external frozen validation, whereas VisA is reported as in-domain frozen validation because of the AnomalyCLIP checkpoint lineage. All comparisons use identical normal-reference identities and evaluation code without test-label-based selection.

**Figure 3. Configuration-level Pixel-AP gains over the matched DINO-only control.** Each point is one reference-sampling configuration; lines connect the 1-, 2-, and 4-shot values for the same seed. All 36 configurations have positive ΔPixel AP. Configurations within a dataset share the same test set and are not independent datasets.

**Figure 4. Category-level heterogeneity of fusion gains.** Bars show mean category ΔPixel AP across the nine reference configurations. The panel contains the ten lowest- and ten highest-gain categories according to a fixed ranking; the full 36-category version is Figure S1. Blue and vermilion indicate positive and negative mean changes, respectively.

**Figure 5. Complete-metric change relative to the matched DINO-only control.** Cells report the difference between nine-configuration means. BTAD improves in all pixel-level metrics but loses Image AP and Image F1-max, preventing an all-metric dominance claim.

**Figure 6. Three-way control on BTAD and MVTec AD.** Bars show mean ± standard deviation across the nine reference configurations. The AnomalyCLIP image tower is weaker in isolation, whereas fixed fusion exceeds both single-encoder controls in Pixel AP.

**Figure 7. Runtime composition and reference-memory scaling.** Runtime is measured in steady state on MVTec bottle (seed 0, one shot; three warm-up passes and 30 repetitions). Memory values are float32 normal-reference patch banks and exclude test features. The hardware-specific CLIP stage dominates latency.

**Figure 8. Representative localization improvements.** Cases follow the frozen R4 manifest. Heatmaps are overlaid on the input image; DINO and A1 use the same raw anomaly-score scale within each row. Ground-truth contours are shown only for evaluation and visualization.

**Figure 9. Representative negative-transfer cases.** Cases follow the frozen R4 manifest. Shared per-row score scales prevent independent heatmap normalization from exaggerating differences. These examples illustrate that average complementarity does not eliminate category- or image-level failures.

**Figure S1. Full category-level gain/loss distribution.** All 36 dataset-category units are ordered by mean ΔPixel AP across nine reference configurations.

**Figure S2. Shot-wise gain stability.** Points show the mean ΔPixel AP across three reference seeds and error bars show the corresponding descriptive standard deviation. The plot does not imply monotonic improvement with increasing shot count.

## Integrity notes

- Figure values are read from the frozen P1 evidence and the post-freeze BTAD/MVTec CLIP-image-only control.
- Error bars represent descriptive standard deviation across reference-sampling configurations, not independent-dataset uncertainty.
- Qualitative heatmaps are never normalized independently between methods within a sample.
- Sample identities and deterministic selection rules are retained in `source_data/Fig08_Fig09_qualitative_cases.csv`.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "schema_version": 1,
        "kind": "manuscript_figure_package",
        "source_policy": "frozen project evidence; no test-driven figure selection beyond recorded manifests",
        "quantitative_formats": ["SVG editable", "PDF vector", "PNG 600 dpi"],
        "word_target_width_cm": 17.8,
        "files": MANIFEST_ROWS,
    }
    (OUT / "FIGURE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_package() -> None:
    expected_quant = 9  # Fig01-07 plus Fig04 main/S1 = nine files per format.
    pngs = list(PNG_DIR.glob("*.png"))
    svgs = list(SVG_DIR.glob("*.svg"))
    pdfs = list(PDF_DIR.glob("*.pdf"))
    qual = list(RESTRICTED_DIR.glob("*.png"))
    if not (len(pngs) == len(svgs) == len(pdfs) == expected_quant):
        raise RuntimeError(f"quantitative count mismatch: png={len(pngs)}, svg={len(svgs)}, pdf={len(pdfs)}")
    if len(qual) != 2:
        raise RuntimeError(f"qualitative count mismatch: {len(qual)}")
    for path in pngs + qual:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            dpi = img.info.get("dpi", (0, 0))
            if path.parent == PNG_DIR and min(dpi) < 590:
                raise RuntimeError(f"DPI check failed: {path} -> {dpi}")
            if img.width < 1800:
                raise RuntimeError(f"pixel width too small: {path} -> {img.width}")
    for path in svgs + pdfs:
        if path.stat().st_size < 1000:
            raise RuntimeError(f"vector output unexpectedly small: {path}")


def main() -> int:
    configure_style()
    ensure_dirs()
    # Remove only generated contents under the exact package directories.
    for directory in (PNG_DIR, SVG_DIR, PDF_DIR, DATA_DIR, RESTRICTED_DIR):
        for path in directory.iterdir():
            if path.is_file():
                path.unlink()

    complete = json.loads(COMPLETE_JSON.read_text(encoding="utf-8"))
    bootstrap = json.loads(BOOTSTRAP_JSON.read_text(encoding="utf-8"))
    efficiency = json.loads(EFFICIENCY_JSON.read_text(encoding="utf-8"))
    clip = json.loads(CLIP_SUMMARY.read_text(encoding="utf-8"))
    qual = json.loads(QUAL_MANIFEST.read_text(encoding="utf-8"))

    fig01_method_overview()
    fig02_protocol()
    fig03_configuration_gains(bootstrap)
    fig04_category_heterogeneity(bootstrap)
    fig_s02_shot_wise(bootstrap)
    fig05_metric_delta_heatmap(complete)
    fig06_three_way(complete, clip)
    fig07_efficiency(efficiency)
    fig08_09_qualitative(qual)
    make_contact_sheet()
    write_caption_guide()
    write_manifest()
    validate_package()
    print(json.dumps({
        "status": "complete",
        "output": str(OUT),
        "quantitative_figures": len(QUANTITATIVE_NAMES),
        "qualitative_figures": 2,
        "manifest_files": len(MANIFEST_ROWS),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
