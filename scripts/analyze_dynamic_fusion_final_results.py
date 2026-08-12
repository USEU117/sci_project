"""Analyze frozen dynamic-fusion results without tuning or rerunning models."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion.alignment import build_alignment_plan
from industrial_ad.fusion.calibration import load_category_calibrations


def macro_row(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8") as stream:
        row = next(row for row in csv.DictReader(stream) if row["category"] == "macro_mean")
    return {key: float(value) for key, value in row.items() if key not in ("category", "sample_count")}


def parse_run(name: str) -> tuple[str, int, int]:
    dataset = "mvtec" if "mvtec" in name else "visa"
    seed = int(re.search(r"_s(\d+)_", name).group(1))
    shot = int(re.search(r"_k(\d+)$", name).group(1))
    return dataset, seed, shot


def load_cache(
    path: Path, sidecar: Path | None = None, *, include_maps: bool = True
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        keys = ["gt_sp", "pr_sp"]
        if include_maps:
            keys.extend(["imgs_masks", "anomaly_maps"])
        result = {key: data[key] for key in keys}
        if "sample_ids" in data.files:
            result["sample_ids"] = data["sample_ids"].astype(str)
    if "sample_ids" not in result:
        if sidecar is None:
            raise ValueError(f"missing sample IDs: {path}")
        with np.load(sidecar, allow_pickle=False) as data:
            result["sample_ids"] = data["sample_ids"].astype(str)
    for key in ("imgs_masks", "anomaly_maps"):
        if key not in result:
            continue
        if result[key].ndim == 4 and result[key].shape[1] == 1:
            result[key] = result[key][:, 0]
    return result


def aligned(cache: dict[str, np.ndarray], target_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Reorder a cache using the canonical-ID rules used by fusion inference."""
    plan = build_alignment_plan(target_ids, cache["sample_ids"])
    source_count = len(cache["sample_ids"])
    result = {
        key: value[plan.candidate_order]
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == source_count
        else value
        for key, value in cache.items()
    }
    result["sample_ids"] = plan.reference_ids
    return result


def resize_map(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if value.shape == shape:
        return value.astype(np.float32)
    return np.asarray(Image.fromarray(value.astype(np.float32), mode="F").resize((shape[1], shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else float("nan")


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return float("nan")
    result = spearmanr(left, right)
    return float(getattr(result, "statistic", result.correlation))


def normalize_map(value: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(value)), float(np.max(value))
    return np.zeros_like(value, dtype=np.float32) if hi <= lo else ((value - lo) / (hi - lo)).astype(np.float32)


def contrast(value: np.ndarray, mask: np.ndarray) -> float:
    value = normalize_map(value)
    inside = value[mask > 0]
    outside = value[mask == 0]
    return float(inside.mean() - outside.mean()) if len(inside) and len(outside) else float("nan")


def original_path(dataset: str, category: str, sample_id: str) -> Path | None:
    if dataset != "mvtec":
        return None
    prefix = f"{category}-"
    rest = sample_id[len(prefix):] if sample_id.startswith(prefix) else sample_id
    defect, index = rest.rsplit("-", 1)
    path = ROOT / "data/mvtec" / category / "test" / defect / f"{index}.png"
    return path if path.exists() else None


def analyze_run(directory: Path, category_rows: list[dict], route_rows: list[dict], provenance_rows: list[dict], case_rows: list[dict]) -> None:
    dataset, seed, shot = parse_run(directory.name)
    dynamic_category = pd.read_csv(directory / "evaluation/per_category.csv").set_index("category")
    calibration_seen: set[str] = set()
    run_sample_stats: list[dict] = []

    for path in sorted(directory.glob("*.npz")):
        category = path.stem
        include_maps = dataset == "mvtec" and seed == 0 and shot == 4
        with np.load(path, allow_pickle=False) as dynamic:
            ids = dynamic["sample_ids"].astype(str)
            labels = dynamic["gt_sp"].astype(int)
            dynamic_scores = dynamic["pr_sp"].astype(float)
            visual_weights = dynamic["visual_weights"].astype(float)
            pixel_weights = dynamic["visual_pixel_weights"].astype(float)
            decisions = dynamic["route_decisions"].astype(str)
            dynamic_maps = dynamic["anomaly_maps"].astype(np.float32) if include_maps else None
            masks = dynamic["imgs_masks"] if include_maps else None
            calibration_path = Path(str(np.asarray(dynamic["calibration_path"]).item()))
            calibration_sha = str(np.asarray(dynamic["calibration_sha256"]).item())

        if dataset == "mvtec":
            visual_path = ROOT / f"outputs/anomalydino/unified_matrix/seed_{seed}_shot_{shot}/predictions/{category}.npz"
            text_path = ROOT / f"outputs/anomalyclip/mvtec_npz/{category}.npz"
            text_sidecar = ROOT / f"outputs/dynamic_fusion/sidecars/anomalyclip_mvtec_518/{category}.sample_ids.npz"
        else:
            visual_path = ROOT / f"outputs/anomalydino/unified_matrix/seed_{seed}_shot_{shot}/predictions/{category}.npz"
            text_path = ROOT / f"outputs/anomalyclip/visa_all_518_cached/{category}.npz"
            text_sidecar = ROOT / f"outputs/dynamic_fusion/sidecars/anomalyclip_visa_518_verified/{category}.sample_ids.npz"

        visual = aligned(load_cache(visual_path, include_maps=include_maps), ids)
        text = aligned(load_cache(text_path, text_sidecar, include_maps=include_maps), ids)
        if not np.array_equal(labels, visual["gt_sp"].astype(int)) or not np.array_equal(labels, text["gt_sp"].astype(int)):
            raise ValueError(f"label alignment failed: {directory.name}/{category}")

        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        visual_cal, text_cal = load_category_calibrations(payload, category)
        raw_visual = visual["pr_sp"].astype(float)
        raw_text = text["pr_sp"].astype(float)
        cal_visual = visual_cal.image.transform(raw_visual).astype(np.float32)
        cal_text = text_cal.image.transform(raw_text).astype(np.float32)
        dyn = dynamic_category.loc[category]

        pixel_sample = pixel_weights[:, ::16, ::16].reshape(-1)
        normal = labels == 0
        anomaly = labels == 1
        visual_advantage = safe_auc(labels, raw_visual) - safe_auc(labels, raw_text)
        row = {
            "dataset": dataset,
            "seed": seed,
            "shot": shot,
            "category": category,
            "sample_count": len(labels),
            "raw_visual_image_auroc": safe_auc(labels, raw_visual),
            "calibrated_visual_image_auroc": safe_auc(labels, cal_visual),
            "raw_text_image_auroc": safe_auc(labels, raw_text),
            "calibrated_text_image_auroc": safe_auc(labels, cal_text),
            "dynamic_image_auroc": float(dyn["image_auroc"]),
            "dynamic_pixel_auroc": float(dyn["pixel_auroc"]),
            "dynamic_pixel_ap": float(dyn["pixel_ap"]),
            "dynamic_aupro": float(dyn["aupro"]),
            "dynamic_minus_raw_visual_image_auroc": float(dyn["image_auroc"]) - safe_auc(labels, raw_visual),
            "calibration_loss_visual_image_auroc": safe_auc(labels, cal_visual) - safe_auc(labels, raw_visual),
            "calibration_loss_text_image_auroc": safe_auc(labels, cal_text) - safe_auc(labels, raw_text),
            "raw_visual_minus_raw_text_image_auroc": visual_advantage,
            "visual_calibration_high_saturation_fraction": float(np.mean(cal_visual >= 0.999)),
            "visual_calibration_low_saturation_fraction": float(np.mean(cal_visual <= 0.001)),
            "text_calibration_high_saturation_fraction": float(np.mean(cal_text >= 0.999)),
            "text_calibration_low_saturation_fraction": float(np.mean(cal_text <= 0.001)),
            "visual_calibrated_unique_fraction": float(len(np.unique(cal_visual)) / len(cal_visual)),
            "text_calibrated_unique_fraction": float(len(np.unique(cal_text)) / len(cal_text)),
            "raw_to_calibrated_visual_spearman": safe_spearman(raw_visual, cal_visual),
            "raw_to_calibrated_text_spearman": safe_spearman(raw_text, cal_text),
            "dynamic_to_raw_visual_spearman": safe_spearman(dynamic_scores, raw_visual),
            "mean_visual_weight": float(np.mean(visual_weights)),
            "mean_visual_weight_normal": float(np.mean(visual_weights[normal])) if np.any(normal) else float("nan"),
            "mean_visual_weight_anomaly": float(np.mean(visual_weights[anomaly])) if np.any(anomaly) else float("nan"),
            "visual_weight_label_gap": float(np.mean(visual_weights[anomaly]) - np.mean(visual_weights[normal])) if np.any(normal) and np.any(anomaly) else float("nan"),
            "visual_weight_at_upper_clip_fraction": float(np.mean(visual_weights >= 0.9499)),
            "visual_weight_at_lower_clip_fraction": float(np.mean(visual_weights <= 0.0501)),
            "mean_pixel_visual_weight": float(np.mean(pixel_weights)),
            "pixel_visual_weight_p10_sampled": float(np.quantile(pixel_sample, 0.10)),
            "pixel_visual_weight_p50_sampled": float(np.quantile(pixel_sample, 0.50)),
            "pixel_visual_weight_p90_sampled": float(np.quantile(pixel_sample, 0.90)),
            "route_visual_count": int(np.sum(decisions == "visual")),
            "route_text_count": int(np.sum(decisions == "text")),
            "route_weighted_count": int(np.sum(decisions == "weighted_fusion")),
        }
        category_rows.append(row)
        run_sample_stats.append(row)

        calibration_key = str(calibration_path)
        if calibration_key not in calibration_seen:
            calibration_seen.add(calibration_key)
            provenance_rows.append({
                "dataset": dataset,
                "seed": seed,
                "shot": shot,
                "visual_branch": payload.get("visual_branch"),
                "text_branch": payload.get("text_branch"),
                "calibration_method": payload.get("method"),
                "pixel_scale_fit_statistic": payload.get("pixel_scale_fit_statistic"),
                "test_predictions_used": payload.get("test_predictions_used"),
                "test_labels_used": payload.get("test_labels_used"),
                "calibration_path": str(calibration_path),
                "calibration_sha256_embedded": calibration_sha,
                "visual_cache_pattern": str(visual_path.parent),
                "text_cache_pattern": str(text_path.parent),
            })

        if dataset == "mvtec" and seed == 0 and shot == 4:
            visual_maps = visual["anomaly_maps"]
            text_maps = text["anomaly_maps"]
            for index, sample_id in enumerate(ids):
                if not labels[index] or not np.any(masks[index]):
                    continue
                target_shape = dynamic_maps[index].shape
                vmap = resize_map(visual_maps[index], target_shape)
                tmap = resize_map(text_maps[index], target_shape)
                visual_contrast = contrast(vmap, masks[index])
                dynamic_contrast = contrast(dynamic_maps[index], masks[index])
                case_rows.append({
                    "dataset": dataset, "seed": seed, "shot": shot, "category": category,
                    "sample_id": sample_id, "sample_index": index,
                    "original_path": str(original_path(dataset, category, sample_id) or ""),
                    "visual_contrast": visual_contrast,
                    "text_contrast": contrast(tmap, masks[index]),
                    "dynamic_contrast": dynamic_contrast,
                    "dynamic_minus_visual_contrast": dynamic_contrast - visual_contrast,
                    "visual_image_score": float(raw_visual[index]),
                    "text_image_score": float(raw_text[index]),
                    "dynamic_image_score": float(dynamic_scores[index]),
                    "visual_weight": float(visual_weights[index]),
                })

    frame = pd.DataFrame(run_sample_stats)
    route_rows.append({
        "dataset": dataset, "seed": seed, "shot": shot,
        "category_count": len(frame), "sample_count": int(frame["sample_count"].sum()),
        "mean_visual_weight": float(np.average(frame["mean_visual_weight"], weights=frame["sample_count"])),
        "mean_visual_weight_normal": float(np.average(frame["mean_visual_weight_normal"], weights=frame["sample_count"])),
        "mean_visual_weight_anomaly": float(np.average(frame["mean_visual_weight_anomaly"], weights=frame["sample_count"])),
        "mean_pixel_visual_weight": float(np.average(frame["mean_pixel_visual_weight"], weights=frame["sample_count"])),
        "visual_route_count": int(frame["route_visual_count"].sum()),
        "text_route_count": int(frame["route_text_count"].sum()),
        "weighted_route_count": int(frame["route_weighted_count"].sum()),
        "visual_upper_clip_count_approx": int(round(np.sum(frame["visual_weight_at_upper_clip_fraction"] * frame["sample_count"]))),
        "category_advantage_weight_spearman": safe_spearman(frame["raw_visual_minus_raw_text_image_auroc"].to_numpy(), frame["mean_visual_weight"].to_numpy()),
        "mean_visual_calibration_auroc_loss": float(frame["calibration_loss_visual_image_auroc"].mean()),
        "mean_dynamic_minus_raw_visual_auroc": float(frame["dynamic_minus_raw_visual_image_auroc"].mean()),
    })


def build_run_comparison(category: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "raw_visual_image_auroc": "mean",
        "calibrated_visual_image_auroc": "mean",
        "raw_text_image_auroc": "mean",
        "calibrated_text_image_auroc": "mean",
        "dynamic_image_auroc": "mean",
        "dynamic_pixel_auroc": "mean",
        "dynamic_pixel_ap": "mean",
        "dynamic_aupro": "mean",
        "dynamic_minus_raw_visual_image_auroc": "mean",
        "calibration_loss_visual_image_auroc": "mean",
        "visual_calibration_high_saturation_fraction": "mean",
        "mean_visual_weight": "mean",
        "mean_pixel_visual_weight": "mean",
    }
    return category.groupby(["dataset", "seed", "shot"], as_index=False).agg(metrics)


def build_ablation() -> pd.DataFrame:
    roots = {
        1: ROOT / "outputs/dynamic_fusion/development_matrix/20260731_visa_s0_k1_calibrated_development_matrix",
        2: ROOT / "outputs/dynamic_fusion/development_matrix/20260804_visa_s0_k2_calibrated_development_matrix_v3",
        4: ROOT / "outputs/dynamic_fusion/development_matrix/20260804_visa_s0_k4_calibrated_development_matrix_v1",
    }
    mode_names = {
        "visual": "calibrated_visual_only", "text": "calibrated_text_only",
        "fixed_w0": "fixed_visual_0.00", "fixed_w025": "fixed_visual_0.25",
        "fixed_w05": "fixed_visual_0.50", "fixed_w075": "fixed_visual_0.75",
        "fixed_w1": "fixed_visual_1.00", "dynamic": "single_temperature_0.20",
    }
    rows = []
    for shot, root in roots.items():
        for folder, label in mode_names.items():
            summary = root / folder / "evaluation/summary.csv"
            if summary.exists():
                rows.append({"dataset": "visa", "seed": 0, "shot": shot, "variant": label, **macro_row(summary)})
    split_root = ROOT / "outputs/dynamic_fusion/split_temperature/20260805_visa_s0_split_temperature_k1_check"
    for shot in (1, 2, 4):
        summary = split_root / f"k{shot}_image_t050_pixel_t020/evaluation/summary.csv"
        rows.append({"dataset": "visa", "seed": 0, "shot": shot, "variant": "split_image_0.50_pixel_0.20", **macro_row(summary)})
    for shot in (2, 4):
        summary = ROOT / f"outputs/dynamic_fusion/selected_candidate_20260805/k{shot}/evaluation/summary.csv"
        rows.append({"dataset": "visa", "seed": 0, "shot": shot, "variant": "single_temperature_0.50", **macro_row(summary)})
    return pd.DataFrame(rows)


def save_figures(category: pd.DataFrame, run: pd.DataFrame, route: pd.DataFrame, ablation: pd.DataFrame, cases: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    colors = {"Raw AnomalyDINO": "#2F6B9A", "Dynamic fusion": "#D97941"}

    mvtec = run[run.dataset == "mvtec"].groupby("shot").agg({"raw_visual_image_auroc": ["mean", "std"], "dynamic_image_auroc": ["mean", "std"]})
    shots = mvtec.index.to_numpy(); x = np.arange(len(shots)); width = 0.34
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (column, label) in enumerate((("raw_visual_image_auroc", "Raw AnomalyDINO"), ("dynamic_image_auroc", "Dynamic fusion"))):
        ax.bar(x + (i - .5) * width, mvtec[(column, "mean")] * 100, width, yerr=mvtec[(column, "std")] * 100, capsize=4, label=label, color=colors[label])
    ax.set_xticks(x, [f"{v}-shot" for v in shots]); ax.set_ylabel("Image AUROC (%)"); ax.set_title("MVTec: fusion remains below the raw visual branch"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(output / "mvtec_visual_vs_dynamic_by_shot.png", dpi=220); plt.close(fig)

    delta = category[category.dataset == "mvtec"].groupby(["shot", "category"])["dynamic_minus_raw_visual_image_auroc"].mean().unstack()
    fig, ax = plt.subplots(figsize=(15, 3.5)); image = ax.imshow(delta.to_numpy() * 100, cmap="RdBu", vmin=-50, vmax=50, aspect="auto")
    ax.set_yticks(range(len(delta.index)), [f"{v}-shot" for v in delta.index]); ax.set_xticks(range(len(delta.columns)), delta.columns, rotation=45, ha="right")
    ax.set_title("Dynamic fusion minus raw AnomalyDINO Image AUROC (percentage points)"); fig.colorbar(image, ax=ax, label="Δ AUROC (pp)")
    fig.tight_layout(); fig.savefig(output / "mvtec_category_image_auroc_delta_heatmap.png", dpi=220); plt.close(fig)

    route_mean = route.groupby(["dataset", "shot"], as_index=False).agg({"mean_visual_weight": "mean", "mean_pixel_visual_weight": "mean", "mean_visual_weight_normal": "mean", "mean_visual_weight_anomaly": "mean"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, dataset in zip(axes, ("visa", "mvtec")):
        part = route_mean[route_mean.dataset == dataset]; x = np.arange(len(part));
        ax.plot(x, part.mean_visual_weight, "o-", label="Image visual weight", color="#2F6B9A")
        ax.plot(x, part.mean_pixel_visual_weight, "o-", label="Pixel visual weight", color="#D97941")
        ax.plot(x, part.mean_visual_weight_normal, "--", label="Image weight: normal", color="#6A9F58")
        ax.plot(x, part.mean_visual_weight_anomaly, ":", label="Image weight: anomaly", color="#A44A4A")
        ax.set_xticks(x, [f"{v}-shot" for v in part.shot]); ax.set_title(dataset.upper()); ax.set_ylim(0, 1); ax.grid(alpha=.25)
    axes[0].set_ylabel("Mean visual weight"); axes[1].legend(frameon=False, fontsize=8); fig.suptitle("Image and pixel routing use materially different evidence weights")
    fig.tight_layout(); fig.savefig(output / "route_weight_summary.png", dpi=220); plt.close(fig)

    sat = category.groupby(["dataset", "shot"], as_index=False).agg({"visual_calibration_high_saturation_fraction": "mean", "calibration_loss_visual_image_auroc": "mean"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for dataset, marker in (("visa", "o"), ("mvtec", "s")):
        part = sat[sat.dataset == dataset]
        axes[0].plot(part.shot, part.visual_calibration_high_saturation_fraction * 100, marker=marker, label=dataset.upper())
        axes[1].plot(part.shot, part.calibration_loss_visual_image_auroc * 100, marker=marker, label=dataset.upper())
    axes[0].set_title("Visual scores saturated at ≥0.999"); axes[0].set_ylabel("Category-average fraction (%)"); axes[0].set_xticks([1,2,4])
    axes[1].set_title("AUROC change after visual calibration"); axes[1].set_ylabel("Δ Image AUROC (pp)"); axes[1].set_xticks([1,2,4])
    for ax in axes: ax.grid(alpha=.25); ax.axhline(0, color="#666", lw=.8); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(output / "calibration_saturation_diagnostic.png", dpi=220); plt.close(fig)

    selected = ablation[ablation.variant.isin(["fixed_visual_0.50", "fixed_visual_0.75", "single_temperature_0.20", "single_temperature_0.50", "split_image_0.50_pixel_0.20"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, title in ((axes[0], "image_auroc", "Image AUROC"), (axes[1], "aupro", "AUPRO")):
        pivot = selected.pivot(index="shot", columns="variant", values=metric)
        for variant in pivot.columns:
            ax.plot(pivot.index, pivot[variant] * 100, marker="o", label=variant.replace("_", " "))
        ax.set_xticks([1,2,4]); ax.set_xlabel("Shot"); ax.set_ylabel("Score (%)"); ax.set_title(title); ax.grid(alpha=.25)
    axes[1].legend(frameon=False, fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("VisA seed-0 ablation: split temperature restores localization while improving image ranking")
    fig.tight_layout(); fig.savefig(output / "visa_ablation_split_temperature.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    chosen = pd.concat([cases.nlargest(3, "dynamic_minus_visual_contrast").assign(case_type="success"), cases.nsmallest(3, "dynamic_minus_visual_contrast").assign(case_type="failure")], ignore_index=True)
    chosen.to_csv(output / "selected_cases.csv", index=False)
    run_dir = ROOT / "outputs/dynamic_fusion/final_validation/20260805_mvtec_final_validation_s0_k4"
    fig, axes = plt.subplots(len(chosen), 6, figsize=(17, 3 * len(chosen)))
    titles = ["Original", "Ground truth", "AnomalyDINO", "AnomalyCLIP", "Dynamic fusion", "Visual pixel weight"]
    for row_index, record in chosen.iterrows():
        category = record.category; sample_index = int(record.sample_index)
        with np.load(run_dir / f"{category}.npz", allow_pickle=False) as d:
            ids=d["sample_ids"].astype(str); target=np.flatnonzero(ids == record.sample_id)[0]
            mask=d["imgs_masks"][target]; dyn=d["anomaly_maps"][target]; weight=d["visual_pixel_weights"][target]
        visual=load_cache(ROOT / f"outputs/anomalydino/unified_matrix/seed_0_shot_4/predictions/{category}.npz")
        text=aligned(load_cache(ROOT / f"outputs/anomalyclip/mvtec_npz/{category}.npz", ROOT / f"outputs/dynamic_fusion/sidecars/anomalyclip_mvtec_518/{category}.sample_ids.npz"), ids)
        visual=aligned(visual, ids)
        vmap=resize_map(visual["anomaly_maps"][target], dyn.shape); tmap=resize_map(text["anomaly_maps"][target], dyn.shape)
        original=np.asarray(Image.open(record.original_path).convert("RGB")) if record.original_path else np.zeros((*dyn.shape,3),dtype=np.uint8)
        panels=[original,mask,vmap,tmap,dyn,weight]; cmaps=[None,"gray","inferno","inferno","inferno","viridis"]
        for col,(panel,cmap) in enumerate(zip(panels,cmaps)):
            axes[row_index,col].imshow(panel,cmap=cmap); axes[row_index,col].axis("off")
            if row_index == 0: axes[row_index,col].set_title(titles[col])
        axes[row_index,0].set_ylabel(f"{record.case_type}: {category}\n{record.sample_id}\nΔcontrast={record.dynamic_minus_visual_contrast:+.3f}", fontsize=8)
    fig.suptitle("MVTec K=4 qualitative cases selected by localization-contrast change", y=1.002)
    fig.tight_layout(); fig.savefig(output / "mvtec_success_failure_cases.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260809")
    args = parser.parse_args()
    analysis_root = ROOT / f"experiments/summaries/dynamic_fusion_scientific_analysis_{args.date}"
    figure_root = ROOT / f"outputs/dynamic_fusion/figures/{args.date}_scientific_analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    category_rows: list[dict] = []; route_rows: list[dict] = []; provenance_rows: list[dict] = []; case_rows: list[dict] = []
    final_root = ROOT / "outputs/dynamic_fusion/final_validation"
    directories = [path for path in sorted(final_root.iterdir()) if path.is_dir() and ("mvtec_final" in path.name or "visa_final" in path.name)]
    for index, directory in enumerate(directories, 1):
        print(f"[{index}/{len(directories)}] {directory.name}", flush=True)
        analyze_run(directory, category_rows, route_rows, provenance_rows, case_rows)

    category = pd.DataFrame(category_rows); route = pd.DataFrame(route_rows); provenance = pd.DataFrame(provenance_rows); cases = pd.DataFrame(case_rows)
    run = build_run_comparison(category); ablation = build_ablation()
    category.to_csv(analysis_root / "category_diagnostics.csv", index=False)
    run.to_csv(analysis_root / "run_comparison.csv", index=False)
    route.to_csv(analysis_root / "route_statistics.csv", index=False)
    provenance.to_csv(analysis_root / "provenance.csv", index=False)
    ablation.to_csv(analysis_root / "ablation_summary.csv", index=False)
    cases.to_csv(analysis_root / "case_candidates.csv", index=False)

    mvtec = run[run.dataset == "mvtec"]
    visa = run[run.dataset == "visa"]
    summary = {
        "schema_version": 1, "status": "passed", "frozen_v1_no_retuning": True,
        "run_count": len(run), "category_run_count": len(category),
        "provenance_run_count": len(provenance),
        "mvtec": {
            "mean_dynamic_minus_raw_visual_image_auroc": float(mvtec.dynamic_minus_raw_visual_image_auroc.mean()),
            "mean_visual_calibration_auroc_loss": float(mvtec.calibration_loss_visual_image_auroc.mean()),
            "mean_visual_high_saturation_fraction": float(mvtec.visual_calibration_high_saturation_fraction.mean()),
            "mean_dynamic_to_raw_visual_spearman": float(category[category.dataset == "mvtec"].dynamic_to_raw_visual_spearman.mean()),
        },
        "visa": {
            "mean_dynamic_minus_raw_visual_image_auroc": float(visa.dynamic_minus_raw_visual_image_auroc.mean()),
            "mean_visual_calibration_auroc_loss": float(visa.calibration_loss_visual_image_auroc.mean()),
            "mean_visual_high_saturation_fraction": float(visa.visual_calibration_high_saturation_fraction.mean()),
        },
        "interpretation": [
            "Normal-reference median/MAD sigmoid calibration saturates many test scores and introduces ties, destroying strong raw-visual rankings.",
            "Entropy reliability is low at both probability extremes; saturated probabilities therefore look over-confident rather than unreliable.",
            "Sample-dependent routing weights can further alter image ranking, while the weaker text branch often becomes the only remaining ranking signal when visual scores saturate.",
            "Split image/pixel temperatures fix the seed-0 fixed-weight comparison but do not repair calibration saturation against the raw AnomalyDINO baseline.",
        ],
    }
    (analysis_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_figures(category, run, route, ablation, cases, figure_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
