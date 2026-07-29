"""Merge PromptAD classification scores with segmentation anomaly maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--segmentation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.classification, allow_pickle=False) as cls, np.load(
        args.segmentation, allow_pickle=False
    ) as seg:
        if not np.array_equal(cls["sample_ids"], seg["sample_ids"]):
            raise ValueError("Classification and segmentation sample order differs")
        if not np.array_equal(cls["gt_sp"], seg["gt_sp"]):
            raise ValueError("Classification and segmentation labels differ")
        if len(cls["pr_sp"]) != len(seg["anomaly_maps"]):
            raise ValueError("Prediction counts differ")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            gt_sp=cls["gt_sp"],
            pr_sp=cls["pr_sp"],
            imgs_masks=seg["imgs_masks"],
            anomaly_maps=seg["anomaly_maps"],
            sample_ids=cls["sample_ids"],
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
