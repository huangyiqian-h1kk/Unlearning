# Script entry points

| Script area | Status | Purpose |
| --- | --- | --- |
| `scripts/results/` | Validated, CPU-only | Extract archived metrics and rebuild paper tables. |
| `scripts/configs/validate_historical_experiments.py` | Validated, CPU-only | Check immutable historical records against the paper manifest. |
| `scripts/configs/validate_reproduction_configs.py` | Validated, CPU-only | Check portable future-run candidates and component references. |
| `scripts/repository/` | Validated, CPU-only | Audit source/release inventories and support cleanup recovery. |
| `scripts/train_conrep.py` | Stable dispatch only | List or dispatch to preserved ConRep implementations. |
| `scripts/evaluate_clinicia.py` | Stable dispatch only | List or dispatch to preserved ClinicIA evaluators. |

Listing stable model-facing targets is dependency-free. Actual training and
evaluation are not part of repository validation and require separately
reviewed environments, data, checkpoints, and compute.
