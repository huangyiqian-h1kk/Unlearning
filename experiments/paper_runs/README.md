# Selected paper runs

This directory contains the five ConRep runs for which the repository can
identify the exact paper-selected configuration and supporting job command.

| Regime | Target | Model | Capsule |
| --- | --- | --- | --- |
| A | diagnosis | Llama-2 7B Chat | `a-diagnosis-llama2-conrep/` |
| A | diagnosis | Mistral 7B Instruct v0.2 | `a-diagnosis-mistral-conrep/` |
| A | deaths | Llama-2 7B Chat | `a-deaths-llama2-conrep/` |
| A | deaths | Mistral 7B Instruct v0.2 | `a-deaths-mistral-conrep/` |
| B | injected PMC | Mistral 7B Instruct v0.2 | `b-pmc-mistral-conrep/` |

Each `historical/` directory is byte-identical to the selected Git blob. The
absolute cluster paths are historical evidence, not portable defaults. Use:

```bash
python scripts/train_conrep.py list
python scripts/train_conrep.py show a-diagnosis-mistral-conrep
python scripts/evaluate_clinicia.py show a-diagnosis-mistral-conrep
```

The launchers build transient configs with repository-local dataset paths and
a caller-selected output root. They never rewrite these historical files.

The other 20 paper cells are still indexed in `index.json` and their archived
metrics remain reconstructable. They do not pretend to have an exact runnable
job capsule when the evidence does not establish one.
