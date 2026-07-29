# Superseded job manifests

Kept as a record of what was run, not for reuse.

| Manifest | Why superseded |
|---|---|
| `job-bioclip-sweep.yaml` | Ran 2026-07-29, killed by `DeadlineExceeded` at 12h after 3 of 6 runs. Its completed runs are on the PVC and are skipped by `--skip_existing`. |
| `job-backbone-baselines.yaml` | Folded into `job-backbone-comparison.yaml` — one queue wait instead of two, which matters under GPU contention. |

Current entrypoint: `k8s/job-backbone-comparison.yaml`.
