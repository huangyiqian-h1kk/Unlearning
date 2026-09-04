# Public commands

| Command | Purpose | Loads a model? |
| --- | --- | --- |
| `python scripts/reproduce.py table N` | Verify evidence and rebuild/show paper Tables 1–6 | No |
| `python scripts/train_conrep.py list` | List five selected ConRep paper runs | No |
| `python scripts/train_conrep.py show RUN` | Show historical and portable config locations | No |
| `python scripts/train_conrep.py run RUN --output-root PATH` | Launch a selected trainer after environment preparation | Yes |
| `python scripts/evaluate_clinicia.py list` | List selected ClinicIA evaluations | No |
| `python scripts/evaluate_clinicia.py show RUN` | Show evaluation configs/probes | No |
| `python scripts/evaluate_clinicia.py run RUN --role ROLE --output-root PATH` | Launch a preserved evaluator | Yes |

`scripts/configs/`, `scripts/results/`, and `scripts/repository/` contain
offline validators and evidence builders. Exploratory job generators are under
`legacy/`; upstream framework scripts are under `third_party/`.
