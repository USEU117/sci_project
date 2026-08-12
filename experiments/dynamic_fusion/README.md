# Dynamic fusion run registry

This directory stores small, reviewable metadata for second-stage runs.
Large NPZ files and figures remain under `outputs/dynamic_fusion/`.

Every completed or failed run must have its own immutable directory containing
the command, configuration snapshot, log, machine-readable report and a short
decision note.  See `docs/dynamic_fusion_experiment_protocol.md`.
