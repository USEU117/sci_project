# Dynamic Fusion Scientific Analysis 2026-08-09

Status: `passed`  
Frozen V1 retuning: `forbidden`  
GPU used: `false`

Reproduce:

```powershell
.\.venv-anomalyclip\Scripts\python.exe scripts\analyze_dynamic_fusion_final_results.py
```

Outputs:

- `category_diagnostics.csv`: 231 category-run diagnostics.
- `run_comparison.csv`: 17 frozen-run macro comparisons.
- `route_statistics.csv`: image/pixel weights and route counts.
- `provenance.csv`: branch and calibration provenance.
- `input_provenance_sha256.csv`: 693 run/category/role rows covering 285
  unique frozen input files with size, mtime and SHA256.
- `ablation_summary.csv`: VisA seed-0 frozen ablations.
- `case_candidates.csv`: auditable MVTec K=4 qualitative candidates.
- `summary.json`: machine-readable scientific conclusion.

The first analysis attempt failed because raw baseline caches and fused caches
used different sample-ID representations. The analysis now uses the same
canonical alignment routine as fusion inference. No result file was changed.
