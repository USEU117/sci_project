"""Cross-method MVTec macro mean comparison from unified summary.csv files.

Reads all MVTec summary.csv files from outputs/unified/ for PatchCore, WinCLIP+,
AnomalyDINO, PromptAD, and AnomalyCLIP, computes mean/std across seed/shot combos,
and writes two summary CSV files plus a stdout table.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UNIFIED_DIR = ROOT / "outputs" / "unified"
OUTPUT_DIR = ROOT / "outputs" / "logs"

METHOD_SPECS: list[dict] = [
    {
        "name": "PatchCore",
        "prefix": "patchcore_mvtec_seed",
        "seeds": [0, 1, 2],
        "shots": [1, 2, 4],
        "fmt": "patchcore_mvtec_seed_{seed}_shot_{shot}",
    },
    {
        "name": "WinCLIP+",
        "prefix": "winclip_mvtec_seed",
        "seeds": [0, 1, 2],
        "shots": [1, 2, 4],
        "fmt": "winclip_mvtec_seed_{seed}_shot_{shot}",
    },
    {
        "name": "AnomalyDINO",
        "prefix": "anomalydino_mvtec_full_s",
        "seeds": [0, 1, 2],
        "shots": [1, 2, 4],
        "fmt": "anomalydino_mvtec_full_s{seed}_k{shot}",
    },
    {
        "name": "PromptAD",
        "prefix": "promptad_mvtec_seed",
        "seeds": [0, 1, 2],
        "shots": [1, 2, 4],
        "fmt": "promptad_mvtec_seed_{seed}_shot_{shot}",
    },
    {
        "name": "AnomalyCLIP",
        "prefix": "anomalyclip_mvtec_official",
        "seeds": [],
        "shots": [],
        "fmt": "anomalyclip_mvtec_official",
    },
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # ── collect macro_mean per combo ──────────────────────────────────────
    macro_rows: list[dict] = []  # for macro mean CSV
    per_cat_rows: list[dict] = []  # for per-category CSV
    method_macros: dict[str, list[dict]] = {}  # method -> list of {combo_id, image_auroc, ...}

    for spec in METHOD_SPECS:
        name = spec["name"]
        combos: list[dict] = []

        if name == "AnomalyCLIP":
            # Zero-shot only — single fixed directory
            d = UNIFIED_DIR / spec["fmt"]
            summary_path = d / "summary.csv"
            per_cat_path = d / "per_category.csv"
            if summary_path.is_file():
                with summary_path.open(newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row["category"] == "macro_mean":
                            combos.append({
                                "combo_id": "zero_shot",
                                "image_auroc": float(row["image_auroc"]),
                                "pixel_auroc": float(row["pixel_auroc"]),
                                "aupro": float(row["aupro"]),
                            })
                            break
                if per_cat_path.is_file():
                    with per_cat_path.open(newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            cat = row["category"]
                            if cat == "macro_mean":
                                continue
                            per_cat_rows.append({
                                "method": name,
                                "category": cat,
                                "image_auroc": float(row["image_auroc"]),
                                "pixel_auroc": float(row["pixel_auroc"]),
                                "aupro": float(row["aupro"]),
                                "combo_id": "zero_shot",
                            })
        else:
            for seed in spec["seeds"]:
                for shot in spec["shots"]:
                    dir_name = spec["fmt"].format(seed=seed, shot=shot)
                    d = UNIFIED_DIR / dir_name
                    summary_path = d / "summary.csv"
                    per_cat_path = d / "per_category.csv"
                    combo_id = f"s{seed}_k{shot}"

                    if not summary_path.is_file():
                        print(f"  [SKIP] {name} {combo_id}: summary.csv not found")
                        continue

                    with summary_path.open(newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row["category"] == "macro_mean":
                                combos.append({
                                    "combo_id": combo_id,
                                    "image_auroc": float(row["image_auroc"]),
                                    "pixel_auroc": float(row["pixel_auroc"]),
                                    "aupro": float(row["aupro"]),
                                })
                                break

                    if per_cat_path.is_file():
                        with per_cat_path.open(newline="", encoding="utf-8") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                cat = row["category"]
                                if cat == "macro_mean":
                                    continue
                                per_cat_rows.append({
                                    "method": name,
                                    "category": cat,
                                    "image_auroc": float(row["image_auroc"]),
                                    "pixel_auroc": float(row["pixel_auroc"]),
                                    "aupro": float(row["aupro"]),
                                    "combo_id": combo_id,
                                })

        method_macros[name] = combos

    # ── compute mean / std across combos ──────────────────────────────────
    results: list[dict] = []
    for spec in METHOD_SPECS:
        name = spec["name"]
        combos = method_macros[name]
        n = len(combos)

        if n == 0:
            results.append({
                "method": name,
                "image_auroc_mean": "",
                "image_auroc_std": "",
                "pixel_auroc_mean": "",
                "pixel_auroc_std": "",
                "aupro_mean": "",
                "aupro_std": "",
                "n_combos": 0,
            })
            continue

        image_vals = [c["image_auroc"] for c in combos]
        pixel_vals = [c["pixel_auroc"] for c in combos]
        aupro_vals = [c["aupro"] for c in combos]

        if n >= 2:
            results.append({
                "method": name,
                "image_auroc_mean": round(statistics.mean(image_vals), 4),
                "image_auroc_std": round(statistics.stdev(image_vals), 4),
                "pixel_auroc_mean": round(statistics.mean(pixel_vals), 4),
                "pixel_auroc_std": round(statistics.stdev(pixel_vals), 4),
                "aupro_mean": round(statistics.mean(aupro_vals), 4),
                "aupro_std": round(statistics.stdev(aupro_vals), 4),
                "n_combos": n,
            })
        else:
            results.append({
                "method": name,
                "image_auroc_mean": round(image_vals[0], 4),
                "image_auroc_std": "",
                "pixel_auroc_mean": round(pixel_vals[0], 4),
                "pixel_auroc_std": "",
                "aupro_mean": round(aupro_vals[0], 4),
                "aupro_std": "",
                "n_combos": n,
            })

    # ── write CSVs ────────────────────────────────────────────────────────
    write_csv(OUTPUT_DIR / "mvtec_cross_method_macro_mean.csv", results)

    per_cat_fieldnames = ["method", "category", "image_auroc", "pixel_auroc", "aupro", "combo_id"]
    write_csv(OUTPUT_DIR / "mvtec_cross_method_per_category.csv",
              [dict((k, row[k]) for k in per_cat_fieldnames) for row in per_cat_rows])

    # ── print table ───────────────────────────────────────────────────────
    header = f"{'Method':<14} {'I-AUROC':>10} {'I-σ':>8} {'P-AUROC':>10} {'P-σ':>8} {'AUPRO':>10} {'A-σ':>8} {'N':>4}"
    sep = "-" * len(header)
    print("\nMVTec Cross-Method Macro Mean Comparison\n")
    print(header)
    print(sep)
    for r in results:
        ia = f"{r['image_auroc_mean']}" if r["image_auroc_mean"] != "" else "-"
        i_std = f"{r['image_auroc_std']}" if r["image_auroc_std"] != "" else "-"
        pa = f"{r['pixel_auroc_mean']}" if r["pixel_auroc_mean"] != "" else "-"
        p_std = f"{r['pixel_auroc_std']}" if r["pixel_auroc_std"] != "" else "-"
        aa = f"{r['aupro_mean']}" if r["aupro_mean"] != "" else "-"
        a_std = f"{r['aupro_std']}" if r["aupro_std"] != "" else "-"
        print(f"{r['method']:<14} {ia:>10} {i_std:>8} {pa:>10} {p_std:>8} {aa:>10} {a_std:>8} {r['n_combos']:>4}")

    # ── per-method detail ─────────────────────────────────────────────────
    print("\nPer-combo details:")
    for spec in METHOD_SPECS:
        name = spec["name"]
        combos = method_macros[name]
        cids = [c["combo_id"] for c in combos]
        print(f"  {name} ({len(combos)} combos): {', '.join(cids)}")

    print(f"\nWrote {OUTPUT_DIR / 'mvtec_cross_method_macro_mean.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'mvtec_cross_method_per_category.csv'}")


if __name__ == "__main__":
    main()
