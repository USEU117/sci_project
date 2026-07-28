# Unified prediction schema

Version: 1

The unified evaluator reads one compressed NumPy file per dataset category.
This keeps large pixel maps out of CSV files while allowing every method to use
the same metric implementation.

Required arrays in `<category>.npz`:

| Key | Shape | Meaning |
|---|---|---|
| `gt_sp` | `[N]` | Image labels: `0` normal, `1` anomalous |
| `pr_sp` | `[N]` | Image anomaly scores; larger means more anomalous |
| `imgs_masks` | `[N,H,W]` or `[N,1,H,W]` | Binary ground-truth masks |
| `anomaly_maps` | `[N,H,W]` or `[N,1,H,W]` | Pixel anomaly scores |

Optional arrays:

| Key | Shape | Meaning |
|---|---|---|
| `sample_ids` | `[N]` | Stable image identifiers or relative paths |

Rules:

- All four required arrays must contain the same number of samples.
- Pixel maps and ground-truth masks must have exactly the same shape after an
  optional singleton channel is removed. The evaluator does not silently
  resize predictions.
- Ground-truth masks are binarized with `mask > 0`.
- Image and pixel scores must follow `higher_is_more_anomalous`.
- Each category is evaluated separately and the final value is the unweighted
  macro mean over categories.
- AUPRO uses 200 thresholds and maximum false-positive rate 0.30.

Outputs:

- `per_image.csv`: sample-level labels and image scores.
- `per_category.csv`: all unified metrics for each category.
- `summary.csv`: macro mean over categories.
- `evaluation_report.json`: schema, input categories and validation details.

Method adapters may copy or convert native predictions into this schema. They
must not change scores, resize maps, or recompute model inference during metric
evaluation.
