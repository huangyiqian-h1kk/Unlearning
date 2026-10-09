# Llama M-S completion, 2026-10-09

For the combined Llama completion + Gemma retain grid, use the
[mixed-grid campaign](conrep-mixed-grid.md). It includes these same 14 Llama
runs in one shared three-worker pool; do not also start this standalone campaign.

This campaign adds only Llama-3.1-8B M-S, each at seeds 42 and 43: 14 runs,
125 optimizer steps per run, and 13 checkpoint validations per run (182 total).
The previous 32-run night campaign and 27-run follow-up remain unchanged.

The `llama-ms` profile uses the exact same M-S factor definitions as the Gemma
experiments in [conrep-followup.md](conrep-followup.md). It derives each task
from the original Llama A/42 configuration and the same injected SFT checkpoint;
it does not continue training an already-unlearned J/K checkpoint.

| ID | Parent configuration | Change |
|---|---|---|
| M | K | rank256/alpha512 |
| N | K | learning rate 5e-6 |
| O | J | learning rate 2e-5 |
| P | K | forget CL weight 1 |
| Q | J | one complete fact-preserving retain paraphrase positive |
| R | J | detach the corrupted branch in forget CL |
| S | J | detach retain/general negative representations in forget CL |

J is rank64/alpha128 and K is rank128/alpha256, both with forget CL weight 5,
learning rate 1e-5, four corruption views, corruption probability .7, and
LoRA dropout .05. All alpha/rank ratios remain 2. Other loss weights remain 1.
Q audits all Llama retain facts before publishing a plan. R/S retain the
original gradient-path semantics; detachment does not freeze the encoder.

Three workers share the task pool, initially M/42, N/42 and O/42. Remaining
seed-42 tasks take priority over seed-43 replications. A free worker immediately
claims another task inside the same allocation. There is no model selection
or test evaluation. All existing step-zero references, answer log probabilities,
representation observations and sampled-gradient diagnostics are enabled.

Submission keeps the account-wide cap of three active jobs/nodes, including
queued/held jobs and jobs outside this campaign. Each job uses one eight-GPU
HF node in R9920261000, project gcg51557, a 0390_ job name and 12-hour walltime.
Administrator/queued termination is retried after 20 minutes without a retry
count ceiling, while the campaign budget remains open. Manual qdel is still
intentional cancellation; deterministic errors retain their bounded retry policy.

## Start on an ABCI login node

The bootstrap installs into a separate frozen campaign snapshot. It preserves
the original server evaluator/model/data helpers and the current checkout.
Default `--profile original` continues to mean the previous 27-run matrix;
include `--profile llama-ms` for these 14 new tasks.

```bash
(
set -euo pipefail
cd /groups/gcg51557/experiments/0390_rlsd/unlearning/Unlearning
source local.env
source "$CONREP_ENV/bin/activate"

git fetch origin codex/0390-conrep-night-20261008
llama_ms_ref="$(git rev-parse FETCH_HEAD)"
llama_ms_dir="$(mktemp -d /tmp/0390_conrep_llama_ms.XXXXXX)"
git show "${llama_ms_ref}:scripts/abci/0390_prepare_conrep_followup.py" > "$llama_ms_dir/prepare.py"

campaign="$PWD/results/validated_v2/0390/conrep-llama-ms-20261009"
python -I "$llama_ms_dir/prepare.py" \
  --root "$PWD" --ref "$llama_ms_ref" \
  --source-campaign "$PWD/results/validated_v2/0390/conrep-night-20261008" \
  --campaign "$campaign" --profile llama-ms --hours 10

entry="$campaign/code/scripts/abci/0390_conrep_night.py"
python -I "$entry" start --campaign "$campaign"
python -I "$entry" status --campaign "$campaign"
qstat -u "$USER"
)
```

Preparation should report `profile=llama-ms`, `tasks=14`, `llama_runs=14`,
zero Gemma runs, three workers, and `pbs_walltime=12:00:00`. Preparation itself
does not submit jobs; `start` starts the existing supervisor. Its log contains
the actual submitted PBS job IDs. Repeating preparation with the same revision
and profile preserves the task state. A different revision/profile cannot
silently repurpose an existing campaign.

The ten-hour campaign budget includes queue wait and is distinct from each
job's twelve-hour allocation. To extend a paused/expired campaign explicitly:

```bash
campaign="$PWD/results/validated_v2/0390/conrep-llama-ms-20261009"
python -I "$campaign/code/scripts/abci/0390_conrep_night.py" \
  resume --campaign "$campaign" --hours 10
```

## Collect all results

The updated collector includes this campaign automatically when its directory
exists, alongside the two original campaigns. Explicit repeated `--campaign`
arguments still override the defaults. Extract the updated collector from the
branch without replacing scripts in the server checkout:

```bash
git fetch origin codex/0390-conrep-night-20261008
collect_dir="$(mktemp -d /tmp/0390_conrep_collect.XXXXXX)"
git show FETCH_HEAD:scripts/abci/0390_collect_conrep_results.py > "$collect_dir/collect.py"
python -I "$collect_dir/collect.py" --root "$PWD"
```

After every task completes, the combined inventory should contain 73 complete
experiments and 949 validated checkpoints (59/767 previous + 14/182 new).
Collection is read-only and can run before completion as well.
