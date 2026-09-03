# Phase 0 reproduce commands

```powershell
# 1. integrity audit (CPU, .venv-patchcore)
.\.venv-patchcore\Scripts\python.exe scripts/innovation_v7_global_text/run_phase0_audit.py
# 2. swap complementarity (GPU text tower, .venv-anomalyclip)
.\.venv-anomalyclip\Scripts\python.exe scripts/innovation_v7_global_text/run_swap_audit.py
# 3. unit tests
.\.venv-patchcore\Scripts\python.exe -m pytest tests/innovation_v7_global_text -q
# 4. this finalize
python scripts/innovation_v7_global_text/run_phase0_finalize.py
```