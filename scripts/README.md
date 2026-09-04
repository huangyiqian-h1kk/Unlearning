# Public commands

| Command | Purpose | Output | Loads a model? |
| --- | --- | --- | --- |
| `python scripts/reproduce.py rebuild-tables` | Rebuild archived Tables 1-6 from evidence | `results/paper/raw_metrics/`, `results/paper/reconstructed/` | No |
| `python scripts/reproduce.py data-status` | Distinguish materialized ClinicIA files from LFS pointers | Status only | No |
| `python scripts/reproduce.py sft-pmc --output-dir PATH` | Train the current Regime B PMC SFT candidate | `PATH` | Yes unless `--dry-run` |
| `python scripts/reproduce.py baseline-unlearn ID ... --output-dir PATH` | Run a paper GradDiff/NPO/RMU config | `PATH` | Yes unless `--dry-run` |
| `python scripts/train_conrep.py list` | List five selected ConRep paper runs | IDs on stdout | No |
| `python scripts/train_conrep.py show RUN` | Show historical and portable config locations | JSON on stdout | No |
| `python scripts/train_conrep.py config RUN --output-root PATH` | Print the complete normalized training config | JSON on stdout | No |
| `python scripts/train_conrep.py run RUN --output-root PATH` | Launch a selected trainer after environment preparation | `PATH/RUN/model/` | Yes |
| `python scripts/evaluate_clinicia.py list` | List selected ClinicIA evaluations | IDs/roles on stdout | No |
| `python scripts/evaluate_clinicia.py show RUN` | Show evaluation configs/probes | JSON on stdout | No |
| `python scripts/evaluate_clinicia.py run RUN --role ROLE --output-root PATH` | Launch a preserved selected-run evaluator | `PATH/RUN/<role>/` | Yes |
| `python scripts/evaluate_clinicia.py paper-config ID --model-path MODEL --output-root PATH` | Print the ClinicIA config for any paper cell | JSON on stdout | No |
| `python scripts/evaluate_clinicia.py run-model ID --model-path MODEL --output-root PATH` | Evaluate any paper-cell model with ClinicIA | `PATH/ID/clinicia/` | Yes |

`scripts/configs/`, `scripts/results/`, and `scripts/repository/` contain
offline validators and evidence builders. Exploratory job generators are under
`legacy/`; upstream framework scripts are under `third_party/`.
