# Configuration layers

| Directory | Meaning |
| --- | --- |
| `paper/historical/` | Resolved, evidence-backed descriptions of the 25 paper cells. These are records, not portable launch configs. |
| `components/` | Semantic model, method, dataset, and protocol registries. |
| `reproduction/` | Portable future-run candidates with explicit readiness gates. |
| `sweeps/` | Compact review matrices; expanded generated jobs are not tracked. |
| `historical/` | Schema and notes retained from the earlier provenance phase. |

For an actual selected ConRep run, start from
[`experiments/paper_runs/`](../experiments/paper_runs/README.md). Its historical
files remain byte-identical, while the launcher creates a temporary
machine-local view.
