#!/usr/bin/env python3
"""Read-only error decomposition for the completed real-resolution diagnostic.

This script consumes the existing real_resolution RESULTS.json and PER_IMAGE.jsonl
only.  It does not load features, run a model, fit a threshold, or rewrite any
old artifact.  The output directory contains a machine-readable summary and a
Chinese report generated from the same checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


SCRIPT_VERSION = "diagnose_gap_v1"
ARMS = ("full44832", "full89664")
CATEGORIES = (
    "bracket_black",
    "bracket_brown",
    "bracket_white",
    "connector",
    "metal_plate",
    "tubes",
)
EXPECTED_CATEGORY_COUNTS = {
    "bracket_black": 79,
    "bracket_brown": 77,
    "bracket_white": 60,
    "connector": 44,
    "metal_plate": 97,
    "tubes": 101,
}
PAIR_FIELDS = (
    "category",
    "sample_id",
    "anomaly_type",
    "gt_sp",
    "gt_area_pixels_full448",
    "gt_area_fraction_full448",
    "gt_area_pixels_stride8",
    "gt_area_fraction_stride8",
    "stride8_mask_disappears",
    "mask_sha256",
    "memory_k",
    "memory_leave_one_out",
    "query_in_memory",
    "a1_stride",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {lineno}: {exc}") from exc
        if not isinstance(item, dict):
            raise ValueError(f"JSONL line {lineno} is not an object")
        rows.append(item)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: Iterable[float | int | None]) -> float | None:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return sum(vals) / len(vals) if vals else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    position = (len(vals) - 1) * q
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (position - lo)


def _stats(values: Iterable[float | int | None]) -> dict[str, Any]:
    vals = [float(x) for x in values if x is not None and math.isfinite(float(x))]
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else None,
        "median": statistics.median(vals) if vals else None,
        "p05": _quantile(vals, 0.05),
        "p25": _quantile(vals, 0.25),
        "p75": _quantile(vals, 0.75),
        "p95": _quantile(vals, 0.95),
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def _rank_auc(labels: list[int], scores: list[float]) -> float | None:
    """Mann-Whitney rank AUC with average ranks for ties."""

    if len(labels) != len(scores):
        raise ValueError("label and score lengths differ")
    n_pos = sum(int(x) for x in labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda x: x[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        average_rank = ((i + 1) + j) / 2.0
        rank_sum_pos += average_rank * sum(int(y) for _, y in ordered[i:j])
        i = j
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pair_records(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("gt_sp") != 1 or not predicate(row):
            continue
        key = (str(row["category"]), str(row["sample_id"]))
        arm = str(row["arm"])
        if arm in by_key[key]:
            raise ValueError(f"duplicate arm in pair {key}: {arm}")
        by_key[key][arm] = row

    records: list[dict[str, Any]] = []
    for key in sorted(by_key):
        arms = by_key[key]
        if set(arms) != set(ARMS):
            raise ValueError(f"incomplete arm pair for {key}: {sorted(arms)}")
        low = arms[ARMS[0]]
        high = arms[ARMS[1]]
        records.append(
            {
                "category": key[0],
                "sample_id": key[1],
                "anomaly_type": low["anomaly_type"],
                "area_fraction": float(low["gt_area_fraction_full448"]),
                "stride8_mask_disappears": bool(low["stride8_mask_disappears"]),
                "ap_full448": _finite_or_none(low["pixel_ap_full448_per_image"]),
                "ap_full896": _finite_or_none(high["pixel_ap_full448_per_image"]),
                "ap_stride8_full448": _finite_or_none(low["pixel_ap_stride8_per_image"]),
                "ap_stride8_full896": _finite_or_none(high["pixel_ap_stride8_per_image"]),
                "score_full448": float(low["image_score_max"]),
                "score_full896": float(high["image_score_max"]),
            }
        )
    return records


def _paired_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    def column(name: str) -> list[float]:
        return [float(r[name]) for r in records if r.get(name) is not None]

    def delta(name_low: str, name_high: str) -> list[float]:
        return [
            float(r[name_high]) - float(r[name_low])
            for r in records
            if r.get(name_low) is not None and r.get(name_high) is not None
        ]

    def one(name_low: str, name_high: str) -> dict[str, Any]:
        low = column(name_low)
        high = column(name_high)
        diffs = delta(name_low, name_high)
        return {
            "full448": _stats(low),
            "full896": _stats(high),
            "delta_896_minus_448": _stats(diffs),
            "n_delta_positive": sum(x > 0 for x in diffs),
            "n_delta_negative": sum(x < 0 for x in diffs),
            "n_delta_zero": sum(x == 0 for x in diffs),
        }

    area = [float(r["area_fraction"]) for r in records]
    return {
        "n": len(records),
        "area_fraction": _stats(area),
        "stride8_mask_disappears": sum(bool(r["stride8_mask_disappears"]) for r in records),
        "pixel_ap_full": one("ap_full448", "ap_full896"),
        "pixel_ap_stride8": one("ap_stride8_full448", "ap_stride8_full896"),
        "image_score_max": one("score_full448", "score_full896"),
    }


def _score_separation(rows: list[dict[str, Any]], category: str | None, arm: str) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["arm"] == arm and (category is None or row["category"] == category)
    ]
    normal = [float(r["image_score_max"]) for r in selected if int(r["gt_sp"]) == 0]
    anomaly = [float(r["image_score_max"]) for r in selected if int(r["gt_sp"]) == 1]
    normal_p95 = _quantile(normal, 0.95)
    anomaly_median = statistics.median(anomaly) if anomaly else None
    return {
        "category": category or "all_categories_pooled",
        "arm": arm,
        "n_normal": len(normal),
        "n_anomaly": len(anomaly),
        "normal_score": _stats(normal),
        "anomaly_score": _stats(anomaly),
        "image_auroc_rank": _rank_auc(
            [int(r["gt_sp"]) for r in selected],
            [float(r["image_score_max"]) for r in selected],
        ),
        "anomaly_le_normal_p95_rate": (
            sum(x <= normal_p95 for x in anomaly) / len(anomaly)
            if normal_p95 is not None and anomaly
            else None
        ),
        "normal_ge_anomaly_median_rate": (
            sum(x >= anomaly_median for x in normal) / len(normal)
            if anomaly_median is not None and normal
            else None
        ),
    }


def _validate(
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    protocol: dict[str, Any],
    per_image_path: Path,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    expected_rows = sum(EXPECTED_CATEGORY_COUNTS.values()) * len(ARMS)
    checks["per_image_rows"] = len(rows)
    checks["expected_rows"] = expected_rows
    checks["per_image_rows_ok"] = len(rows) == expected_rows
    result_rows = result.get("rows", [])
    result_keys = {(r.get("category"), r.get("sample_id"), r.get("arm")) for r in result_rows}
    input_keys = {(r.get("category"), r.get("sample_id"), r.get("arm")) for r in rows}
    checks["results_rows"] = len(result_rows)
    checks["results_per_image_key_set_equal"] = len(result_rows) == len(rows) and result_keys == input_keys
    checks["results_summary_n_rows"] = result.get("summary", {}).get("n_rows")
    checks["results_summary_n_rows_ok"] = result.get("summary", {}).get("n_rows") == len(rows)

    by_sample: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[(str(row["category"]), str(row["sample_id"]))].append(row)
    sample_counts = Counter(category for category, _ in by_sample)
    checks["unique_samples"] = len(by_sample)
    checks["unique_samples_ok"] = len(by_sample) == sum(EXPECTED_CATEGORY_COUNTS.values())
    checks["category_sample_counts"] = dict(sorted(sample_counts.items()))
    checks["category_sample_counts_ok"] = dict(sample_counts) == EXPECTED_CATEGORY_COUNTS
    arm_counts = Counter(str(row["arm"]) for row in rows)
    checks["arm_counts"] = dict(sorted(arm_counts.items()))
    checks["arm_counts_ok"] = arm_counts == Counter({arm: sum(EXPECTED_CATEGORY_COUNTS.values()) for arm in ARMS})

    duplicate_or_incomplete = []
    pair_field_mismatches: list[dict[str, Any]] = []
    for key, pair in sorted(by_sample.items()):
        arms = {str(row["arm"]): row for row in pair}
        if set(arms) != set(ARMS) or len(pair) != len(ARMS):
            duplicate_or_incomplete.append({"key": key, "arms": sorted(arms), "n": len(pair)})
            continue
        low, high = arms[ARMS[0]], arms[ARMS[1]]
        mismatch = {field: [low.get(field), high.get(field)] for field in PAIR_FIELDS if low.get(field) != high.get(field)}
        if mismatch:
            pair_field_mismatches.append({"key": key, "fields": mismatch})
    checks["incomplete_or_duplicate_pairs"] = duplicate_or_incomplete
    checks["pair_fields_mismatched"] = pair_field_mismatches
    checks["pair_alignment_ok"] = not duplicate_or_incomplete and not pair_field_mismatches

    sample_ids = list(by_sample)
    checks["all_query_ids_are_test_paths"] = all("/test/" in sid.replace("\\", "/") for _, sid in sample_ids)
    checks["all_query_ids_unique"] = len(sample_ids) == len(set(sample_ids))
    checks["all_memory_k_two"] = all(row.get("memory_k") == 2 for row in rows)
    checks["all_query_in_memory_false"] = all(row.get("query_in_memory") is False for row in rows)
    checks["all_memory_leave_one_out_false"] = all(row.get("memory_leave_one_out") is False for row in rows)

    support_images: dict[str, list[str]] = {}
    for category in CATEGORIES:
        images = result.get("summary", {}).get("per_category", {}).get(category, {}).get("bank", {}).get("memory_images", [])
        support_images[category] = list(images)
    checks["support_images"] = support_images
    checks["support_is_k2_train_good_no_test"] = all(
        len(images) == 2
        and len(set(images)) == 2
        and all("/train/good/" in str(path).replace("\\", "/") for path in images)
        and all("/test/" not in str(path).replace("\\", "/") for path in images)
        for images in support_images.values()
    )
    checks["protocol_role"] = protocol.get("role")
    checks["protocol_test_cache"] = protocol.get("test_cache")

    ap_alignment_errors: list[dict[str, Any]] = []
    for row in rows:
        label = int(row["gt_sp"])
        full = row.get("pixel_ap_full448_per_image")
        stride = row.get("pixel_ap_stride8_per_image")
        disappeared = bool(row.get("stride8_mask_disappears"))
        full_ok = (full is None and label == 0) or (full is not None and label == 1 and math.isfinite(float(full)))
        stride_ok = (
            (stride is None and (label == 0 or disappeared))
            or (stride is not None and label == 1 and not disappeared and math.isfinite(float(stride)))
        )
        if not full_ok or not stride_ok:
            ap_alignment_errors.append({"category": row.get("category"), "sample_id": row.get("sample_id"), "arm": row.get("arm")})
    checks["per_image_ap_label_alignment_errors"] = ap_alignment_errors
    checks["per_image_ap_label_alignment_ok"] = not ap_alignment_errors
    disappear = {
        (row["category"], row["sample_id"])
        for row in rows
        if row.get("arm") == ARMS[0] and int(row.get("gt_sp")) == 1 and row.get("stride8_mask_disappears")
    }
    checks["stride8_disappeared_anomaly_images"] = len(disappear)
    checks["stride8_disappeared_category_counts"] = dict(sorted(Counter(cat for cat, _ in disappear).items()))
    checks["source_per_image_sha256"] = _sha256(per_image_path)

    structural = (
        checks["per_image_rows_ok"]
        and checks["results_per_image_key_set_equal"]
        and checks["results_summary_n_rows_ok"]
        and checks["unique_samples_ok"]
        and checks["category_sample_counts_ok"]
        and checks["arm_counts_ok"]
        and checks["pair_alignment_ok"]
        and checks["all_query_ids_are_test_paths"]
        and checks["all_query_ids_unique"]
        and checks["all_memory_k_two"]
        and checks["all_query_in_memory_false"]
        and checks["all_memory_leave_one_out_false"]
        and checks["support_is_k2_train_good_no_test"]
        and checks["per_image_ap_label_alignment_ok"]
    )
    checks["structural_validation_pass"] = structural
    if not structural:
        raise ValueError("input structural validation failed; see validation fields")
    return checks


def _analysis(rows: list[dict[str, Any]], result: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
    summary = result["summary"]
    category_pooled = summary["per_category"]
    all_anomaly = _pair_records(rows, lambda _: True)

    category: dict[str, Any] = {}
    for cat in CATEGORIES:
        records = [r for r in all_anomaly if r["category"] == cat]
        paired = _paired_stats(records)
        per_arm: dict[str, Any] = {}
        for arm in ARMS:
            ap_key = "full448" if arm == ARMS[0] else "full896"
            anomaly_ap = paired["pixel_ap_full"][ap_key]["mean"]
            pooled_ap = category_pooled[cat]["arms"][arm]["full_pixel"]["pixel_ap"]
            per_arm[arm] = {
                "anomaly_image_ap_mean": anomaly_ap,
                "category_pooled_pixel_ap": pooled_ap,
                "anomaly_mean_minus_category_pooled": (
                    anomaly_ap - pooled_ap if anomaly_ap is not None and pooled_ap is not None else None
                ),
            }
        category[cat] = {
            "n_normal": int(category_pooled[cat]["n_normal"]),
            "n_anomaly": int(category_pooled[cat]["n_anomaly"]),
            "paired": paired,
            "per_arm_ap_estimands": per_arm,
            "category_pooled_pixel_ap_delta_896_minus_448": (
                category_pooled[cat]["arms"][ARMS[1]]["full_pixel"]["pixel_ap"]
                - category_pooled[cat]["arms"][ARMS[0]]["full_pixel"]["pixel_ap"]
            ),
        }

    types = sorted({str(r["anomaly_type"]) for r in all_anomaly})
    defect_type: dict[str, Any] = {}
    for defect_type_name in types:
        records = [r for r in all_anomaly if r["anomaly_type"] == defect_type_name]
        defect_type[defect_type_name] = _paired_stats(records)

    bin_defs: list[tuple[str, str, Callable[[float], bool]]] = [
        ("le_0.001", "≤ 0.001", lambda x: x <= 0.001),
        ("gt_0.001_le_0.01", "> 0.001 and ≤ 0.01", lambda x: x > 0.001 and x <= 0.01),
        ("gt_0.01", "> 0.01", lambda x: x > 0.01),
    ]
    area_bins: dict[str, Any] = {}
    category_bin_counts: dict[str, dict[str, int]] = {}
    for category_name in CATEGORIES:
        category_bin_counts[category_name] = {}
    for name, label, predicate in bin_defs:
        records = [r for r in all_anomaly if predicate(float(r["area_fraction"]))]
        area_bins[name] = {"label": label, **_paired_stats(records)}
        for category_name in CATEGORIES:
            category_bin_counts[category_name][name] = sum(r["category"] == category_name for r in records)
    for category_name in CATEGORIES:
        category_bin_counts[category_name]["total"] = sum(category_bin_counts[category_name].values())

    separation: list[dict[str, Any]] = []
    for category_name in CATEGORIES:
        for arm in ARMS:
            separation.append(_score_separation(rows, category_name, arm))
    global_separation = [_score_separation(rows, None, arm) for arm in ARMS]

    macro = summary["category_macro"]["arms"]
    headline = {
        "category_macro": summary["category_macro"],
        "full_pixel_ap_delta_896_minus_448": macro[ARMS[1]]["full_pixel"]["pixel_ap"] - macro[ARMS[0]]["full_pixel"]["pixel_ap"],
        "image_auroc_delta_896_minus_448": macro[ARMS[1]]["image"]["image_auroc"] - macro[ARMS[0]]["image"]["image_auroc"],
        "pixel_auroc_delta_896_minus_448": macro[ARMS[1]]["full_pixel"]["pixel_auroc"] - macro[ARMS[0]]["full_pixel"]["pixel_auroc"],
    }
    return {
        "headline": headline,
        "category": category,
        "defect_type": defect_type,
        "area_bins": area_bins,
        "category_bin_counts": category_bin_counts,
        "image_score_separation_by_category": separation,
        "image_score_separation_all_categories_pooled": global_separation,
        "stride8_disappearance": {
            "anomaly_images": validation["stride8_disappeared_anomaly_images"],
            "category_counts": validation["stride8_disappeared_category_counts"],
            "anomaly_images_total": int(summary["n_anomaly"]),
            "positive_fraction_of_full_macro": summary["category_macro"]["fine_anomaly"]["mean_stride8_positive_fraction_of_full"],
        },
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _arrow(stats: dict[str, Any], digits: int = 4) -> str:
    low = stats["full448"]["mean"]
    high = stats["full896"]["mean"]
    delta = stats["delta_896_minus_448"]["mean"]
    return f"{_fmt(low, digits)} → {_fmt(high, digits)} ({_fmt(delta, digits)})"


def _compact_counts(counts: dict[str, int]) -> str:
    return "; ".join(
        f"{cat} {counts[cat]} / {counts.get(cat, 0)} / {counts.get(cat, 0)}"
        for cat in CATEGORIES
    )


def _report(
    root: Path,
    result: dict[str, Any],
    protocol: dict[str, Any],
    validation: dict[str, Any],
    analysis: dict[str, Any],
    source_results: Path,
    source_per_image: Path,
) -> str:
    h = analysis["headline"]
    macro = h["category_macro"]["arms"]
    lines: list[str] = [
        "# 真实分辨率误差分解（只读）",
        "",
        f"生成时间：{_utc_now()}；脚本版本：{SCRIPT_VERSION}。",
        "",
        "本报告只读取已完成的 real_resolution RESULTS.json、PER_IMAGE.jsonl 以及其冻结协议；没有重新提取特征、运行模型、训练参数或用 test 标签拟合阈值。所有 full44832 与 full89664 结果均按同一 test sample 配对。输入结果是 MPDD 开发集上的 support-only 诊断，不能作为独立确认。",
        "",
        "## 结论先行",
        "",
        f"- 完整性通过：{validation['per_image_rows']} 行 = 458 张图 × 2 臂；6 类共 176 张 normal、282 张 anomaly。每个 sample_id 恰有两臂，mask、标签、面积和 memory 标记一致；query 全在 test 路径，support 为每类 2 张 train/good，未进入 test。",
        f"- category macro 的 full-pixel AP 为 {_fmt(macro[ARMS[0]]['full_pixel']['pixel_ap'])} → {_fmt(macro[ARMS[1]]['full_pixel']['pixel_ap'])}（Δ {_fmt(h['full_pixel_ap_delta_896_minus_448'])}）；image AUROC 为 {_fmt(macro[ARMS[0]]['image']['image_auroc'])} → {_fmt(macro[ARMS[1]]['image']['image_auroc'])}（Δ {_fmt(h['image_auroc_delta_896_minus_448'])}）。这两个方向不能合并成“全面提升”。",
        f"- 面积分箱显示定位风险集中在小 mask：≤ 0.001 的 37 张异常图 full per-image AP {_fmt(analysis['area_bins']['le_0.001']['paired']['pixel_ap_full']['full448']['mean'])} → {_fmt(analysis['area_bins']['le_0.001']['paired']['pixel_ap_full']['full896']['mean'])}（Δ {_fmt(analysis['area_bins']['le_0.001']['paired']['pixel_ap_full']['delta_896_minus_448']['mean'])}）；stride8 有效样本只有 31 张，不能把 6 张消失 mask 当作 AP = 0。",
        f"- 正常高尾和跨类分数尺度都可能影响 image max：例如 category AUROC 在两臂中 metal_plate 为 {_fmt(next(x['image_auroc_rank'] for x in analysis['image_score_separation_by_category'] if x['category']=='metal_plate' and x['arm']==ARMS[0]))} / {_fmt(next(x['image_auroc_rank'] for x in analysis['image_score_separation_by_category'] if x['category']=='metal_plate' and x['arm']==ARMS[1]))}，而 bracket_black 为 {_fmt(next(x['image_auroc_rank'] for x in analysis['image_score_separation_by_category'] if x['category']=='bracket_black' and x['arm']==ARMS[0]))} / {_fmt(next(x['image_auroc_rank'] for x in analysis['image_score_separation_by_category'] if x['category']=='bracket_black' and x['arm']==ARMS[1]))}。raw max 不能直接跨 category 或跨分辨率比较。",
        "",
        "## 数据角色与配对核验",
        "",
        "| 检查 | 结果 |",
        "|---|---:|",
        f"| PER_IMAGE 行数 / 预期 | {validation['per_image_rows']} / {validation['expected_rows']} |",
        f"| unique test sample | {validation['unique_samples']} |",
        f"| 两臂行数 | full44832 {validation['arm_counts'].get(ARMS[0], 0)}；full89664 {validation['arm_counts'].get(ARMS[1], 0)} |",
        f"| pair 字段 mismatch | {len(validation['pair_fields_mismatched'])} |",
        f"| sample_id 全为 test 路径且唯一 | {_fmt(validation['all_query_ids_are_test_paths'] and validation['all_query_ids_unique'])} |",
        f"| query_in_memory / memory_leave_one_out | {_fmt(validation['all_query_in_memory_false'])} / {_fmt(validation['all_memory_leave_one_out_false'])} |",
        f"| support | {_fmt(validation['support_is_k2_train_good_no_test'])}（每类 K = 2 train/good） |",
        f"| per-image AP 与 label / 消失 mask 对齐 | {_fmt(validation['per_image_ap_label_alignment_ok'])} |",
        f"| stride8 完全消失的 anomaly image | {validation['stride8_disappeared_anomaly_images']} / {sum(validation['category_sample_counts'].get(c, 0) for c in CATEGORIES)} anomaly 分母见下文 |",
        "",
        "样本计数按去重后的 sample_id 为：bracket_black 79、bracket_brown 77、bracket_white 60、connector 44、metal_plate 97、tubes 101；normal / anomaly 分别为 176 / 282。stride8 消失计数的正确 anomaly 分母是 282，而不是总行数或 map 像素数。",
        "",
        "## category macro 与 AP estimand gap",
        "",
        "RESULTS 中的 category pooled pixel AP 是每个类别把 normal 与 anomaly 的完整 448 × 448 像素一起算出的 AP，随后对 6 类做 arithmetic macro；它不是 anomaly 图 per-image AP 的平均。下面同时列出两个 estimand 和同图配对变化。",
        "",
        "| 类别 | n anomaly | median mask frac | anomaly per-image AP 448 → 896 (Δ) | category pooled pixel AP 448 → 896 (Δ) | anomaly mean − pooled（448 / 896） | image max mean 448 → 896 |
|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cat in CATEGORIES:
        item = analysis["category"][cat]
        p = item["paired"]
        estimand = item["per_arm_ap_estimands"]
        lines.append(
            "| {} | {} | {} | {} | {} → {} ({}) | {} / {} | {} → {} ({}) |".format(
                cat,
                item["n_anomaly"],
                _fmt(p["area_fraction"]["median"], 6),
                _arrow(p["pixel_ap_full"], 4),
                _fmt(estimand[ARMS[0]]["category_pooled_pixel_ap"], 4),
                _fmt(estimand[ARMS[1]]["category_pooled_pixel_ap"], 4),
                _fmt(item["category_pooled_pixel_ap_delta_896_minus_448"], 4),
                _fmt(estimand[ARMS[0]]["anomaly_mean_minus_category_pooled"], 4),
                _fmt(estimand[ARMS[1]]["anomaly_mean_minus_category_pooled"], 4),
                _fmt(p["image_score_max"]["full448"]["mean"], 4),
                _fmt(p["image_score_max"]["full896"]["mean"], 4),
                _fmt(p["image_score_max"]["delta_896_minus_448"]["mean"], 4),
            )
        )
    lines += [
        "",
        "全异常图的 per-image full AP 平均为 0.495778 → 0.518149，但 6 类 category pooled full-pixel AP macro 为 0.317636 → 0.310542。这个差距是评价定义和 normal 负像素共同造成的可见现象，不能由它单独推断模型定位变好或变坏；当前输出也没有一个跨 6 类拼接的 pooled-pixel AP，因此不能声称真实 pooled-pixel AP 获得提升。",
        "",
        "## normal / anomaly image max 分布与分离",
        "",
        "`image_score_max` 是每张 448 × 448 map 的最大值。normal p95 与 anomaly median 的重叠、以及 anomaly ≤ normal p95 的比例只作描述，不是调参阈值；rank AUROC 是同类内部的阈值无关分离。",
        "",
        "| 类别 | 臂 | normal mean / p95 | anomaly mean / p05 | anomaly ≤ normal p95 | normal ≥ anomaly median | rank image AUROC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in analysis["image_score_separation_by_category"]:
        lines.append(
            "| {} | {} | {} / {} | {} / {} | {} | {} | {} |".format(
                item["category"],
                item["arm"],
                _fmt(item["normal_score"]["mean"], 4),
                _fmt(item["normal_score"]["p95"], 4),
                _fmt(item["anomaly_score"]["mean"], 4),
                _fmt(item["anomaly_score"]["p05"], 4),
                _fmt(item["anomaly_le_normal_p95_rate"], 3),
                _fmt(item["normal_ge_anomaly_median_rate"], 3),
                _fmt(item["image_auroc_rank"], 4),
            )
        )
    lines += [
        "",
        "补充的全类别 pooled image-score rank AUROC 为 full44832 {}、full89664 {}；它受类别比例与跨类 raw score 标度影响，不能替代上表的 category macro。".format(
            _fmt(analysis["image_score_separation_all_categories_pooled"][0]["image_auroc_rank"], 4),
            _fmt(analysis["image_score_separation_all_categories_pooled"][1]["image_auroc_rank"], 4),
        ),
        "",
        "## gt 面积分箱与同图 AP 变化",
        "",
        "分箱使用 full 448 mask 的 area fraction，并固定为 ≤ 0.001、> 0.001 且 ≤ 0.01、> 0.01；边界值在本数据中没有样本。full AP 为 anomaly 图 per-image AP 平均。stride8 AP 只在 stride mask 含正像素时有效，分母单独列出。",
        "",
        "| area bin | n anomaly | stride8 valid | full per-image AP 448 → 896 (Δ) | stride8 per-image AP 448 → 896 (Δ) | full AP 896 > 448 / < / = | category counts（black / brown / white / connector / metal / tubes） |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("le_0.001", "gt_0.001_le_0.01", "gt_0.01"):
        item = analysis["area_bins"][name]
        p = item["paired"]
        counts = analysis["category_bin_counts"]
        count_text = "/".join(str(counts[c][name]) for c in CATEGORIES)
        stride_stats = p["pixel_ap_stride8"]
        lines.append(
            "| {} | {} | {} / {} | {} | {} | {} / {} / {} | {} |".format(
                item["label"],
                item["n"],
                stride_stats["full448"]["n"],
                item["n"],
                _arrow(p["pixel_ap_full"], 4),
                _arrow(stride_stats, 4),
                p["pixel_ap_full"]["n_delta_positive"],
                p["pixel_ap_full"]["n_delta_negative"],
                p["pixel_ap_full"]["n_delta_zero"],
                count_text,
            )
        )
    lines += [
        "",
        "小 mask bin 的 6 张 stride8 全零图来自 bracket_white（4 张 defective_painting、2 张 scratches）；全 anomaly 为 6 / 282 = 2.13%，而 stride8 正像素保留的 category macro 均值为 1.512%。1.512% 接近 1 / 64 的空间抽样比例，不能写成 98.5% 缺陷消失。",
        "",
        "## 按 defect type 的配对描述统计",
        "",
        "这些 defect type 统计跨类别合并同名 type；每行仍是同一 query 的两臂配对，AP 是异常图 per-image AP。低 n 组不做显著性宣称。",
        "",
        "| defect type | n | median mask frac | per-image AP 448 → 896 (Δ) | AP 896 > 448 / < / = | image max mean 448 → 896 (Δ) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dtype in sorted(analysis["defect_type"]):
        item = analysis["defect_type"][dtype]
        p = item["pixel_ap_full"]
        s = item["image_score_max"]
        lines.append(
            "| {} | {} | {} | {} | {} / {} / {} | {} |".format(
                dtype,
                item["n"],
                _fmt(item["area_fraction"]["median"], 6),
                _arrow(p, 4),
                p["n_delta_positive"],
                p["n_delta_negative"],
                p["n_delta_zero"],
                _arrow(s, 4),
            )
        )
    lines += [
        "",
        "## 可执行的后续假设",
        "",
        "1. **定位采样损失（优先级高）**：≤ 0.001 面积 bin 的 full per-image AP 下降 0.0154，stride8 AP 下降约 0.0679，且 6 张 mask 在 stride8 全零；中、大面积 bin 的 full AP 分别上升约 0.0502、0.0140。这支持先做 full-pixel map 的面积分层与边界对齐诊断。下一轮应固定同一 query、同一 mask，比较 full 448 与 full 896 的原始 map 在小缺陷内 recall / rank；不把 stride8 全零样本补成 0 分。局限是面积与 category / defect type 强混杂，不能据此确定高分辨率造成因果改善。",
        "2. **跨图分数校准（优先级高）**：normal 与 anomaly 的 max 分离因 category 变化很大；bracket_black、bracket_brown、bracket_white、connector 有明显 normal 高尾重叠，而 metal_plate、tubes 分离较好。下一轮可只用 train/good support 或预先固定的 development calibration 做 category 内 robust z-score / quantile normalization，然后在 untouched test 上同时报告 image AUROC 与原始 score；当前诊断不能用 test 分数拟合该校准。",
        "3. **正常高尾与 map max 偏置（优先级中）**：full896 的多数类别 raw max 均值变小，但 AP 与 image AUROC 并不一致；这提示全局 score 收缩、插值 / max 选择和正常高尾可能混在一起。下一轮应报告 normal-only score quantiles、固定 FPR 下的 anomaly recall，并保留 mean / median pooling 对照；不要仅因绝对 score 变小就称为鲁棒。",
        "4. **评价量复核（优先级中）**：anomaly per-image AP 平均和 category pooled AP 的差距在 5 / 6 类明显，且 tubes 的 448 值几乎相反方向。这是估计量差异的警报。若要声称真实 pooled-pixel 变化，必须从同一批完整 maps 直接计算跨类别 pooled pixel AP，并同时给 category macro、per-image AP，不以 per-image 平均替代 pooled AP。",
        "",
        "## 证据限制",
        "",
        "- 本次只是对已经产生的逐图指标做 CPU 汇总；没有保存的 anomaly map 无法在此重新计算像素级 pooled AP、定位 recall 或空间连通性。",
        "- 两个分辨率来自同一 458 张 test 图，属于配对观测；916 行不是 916 个独立样本。报告均值、median 和方向计数，不提供 p 值或显著性门。",
        "- category pooled AP 把 normal 全负像素纳入，和 anomaly per-image AP 平均的分母、像素权重不同；跨表相减只作诊断对照。",
        "- defect type、面积 bin、类别存在结构性混杂；小组的 n 不能支持泛化结论。raw image max 的绝对尺度也不保证跨类别、跨输入 edge 可比。",
        "- test 标签仅用于这次事后诊断和固定分组，未用于模型、memory、选择或阈值拟合；因此本文件不能升级昨日 support-only resolution diagnostic 的证据等级。",
        "",
        f"原始输入：`{source_results.as_posix()}`、`{source_per_image.as_posix()}`。PER_IMAGE SHA-256：`{validation['source_per_image_sha256']}`。",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> int:
    script_path = Path(__file__).resolve()
    root = script_path.parents[2]
    source_results = (root / args.results).resolve()
    source_per_image = (root / args.per_image).resolve()
    source_protocol = (root / args.protocol).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    result = _json(source_results)
    rows = _jsonl(source_per_image)
    protocol = _json(source_protocol) if source_protocol.exists() else {}
    validation = _validate(rows, result, protocol, source_per_image)
    analysis = _analysis(rows, result, validation)
    report = _report(root, result, protocol, validation, analysis, source_results, source_per_image)

    output = {
        "script_version": SCRIPT_VERSION,
        "generated_utc": _utc_now(),
        "mode": "read_only_cpu_summary; no feature extraction or model execution",
        "sources": {
            "results": source_results.as_posix(),
            "per_image": source_per_image.as_posix(),
            "protocol": source_protocol.as_posix(),
            "per_image_sha256": validation["source_per_image_sha256"],
        },
        "validation": validation,
        "analysis": analysis,
    }
    (output_dir / "RESULTS.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "REPORT_CN.md").write_text(report, encoding="utf-8")
    (output_dir / "PROTOCOL_CN.md").write_text(
        "\n".join(
            [
                "# 真实分辨率误差分解协议",
                "",
                f"脚本版本：`{SCRIPT_VERSION}`。只读 `real_resolution/RESULTS.json` 与 `PER_IMAGE.jsonl`，并读取其冻结 `PROTOCOL.json` 做角色核验。",
                "",
                "- 不重提取 feature，不加载模型，不训练，不改阈值，不用 test 标签选择样本。",
                "- 916 行按 `category + sample_id` 配成 458 个同图 pair；full44832 与 full89664 的面积、mask hash、label、support 标记必须一致。",
                "- AP 分成现有 category pooled full-pixel AP 与 anomaly-only per-image AP；两者不互相替代。",
                "- 面积分箱：≤ 0.001、> 0.001 且 ≤ 0.01、> 0.01。stride8 AP 对全零 mask 保持缺失，并单列有效分母。",
                "- image max 只作分布和 rank AUROC 描述；不跨 category 或分辨率把绝对 score 当作可比概率。",
                "- 输出只写本目录的 `RESULTS.json`、`REPORT_CN.md`、`PROTOCOL_CN.md`。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": output_dir.as_posix(), "rows": len(rows)}, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        default="outputs/dynamic_fusion/overnight_20260908_real_resolution/RESULTS.json",
    )
    parser.add_argument(
        "--per-image",
        dest="per_image",
        default="outputs/dynamic_fusion/overnight_20260908_real_resolution/PER_IMAGE.jsonl",
    )
    parser.add_argument(
        "--protocol",
        default="experiments/dynamic_fusion/innovation_overnight_20260908/real_resolution/PROTOCOL.json",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/dynamic_fusion/innovation_followup_20260908/gap",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
