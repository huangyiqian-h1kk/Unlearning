# ConRep + ClinicIA

**ConRep** removes identifier–attribute knowledge in representation space.
**ClinicIA** tests whether that knowledge is still recoverable through multiple
linguistic expressions—not only the prompt form seen during unlearning.

This repository is the artifact for *Towards Unlearning Beyond Textual Expressions for LLMs*
(ICLR 2026 review copy). It now separates the paper's
code, selected runs, data, evidence, upstream dependencies, and exploratory
history so a researcher can follow one result end to end.

## The paper in one minute

An unlearning method can appear successful on one textual question while the
same identifier–attribute relation remains extractable through a paraphrase,
cloze prompt, background question, or reversed multiple-choice query.

- **ConRep** targets the internal representation of the relation. The selected
  experiments use the symmetric objective described in Eqs. 11–13.
- **ClinicIA** evaluates six views: generated **QA**, **Cloze**, and
  **Background (BG)** answers, plus **ATT**, **IDeq**, and **ID** MCQs.
- **Regime A** forgets pre-existing celebrity diagnosis or death attributes in
  Llama-2 and Mistral.
- **Regime B** injects PMC clinical relations into Mistral, then forgets a
  designated subset while measuring a retain subset.

See [the paper-to-code map](docs/paper-to-code.md) for the equation, module,
dataset, and table correspondence.

## Start here

Rebuild the archived paper evidence without a GPU, model, or network access:

```bash
python scripts/reproduce.py tables
python scripts/reproduce.py table 1
```

Inspect the five exact ConRep run capsules selected for the paper:

```bash
python scripts/train_conrep.py list
python scripts/train_conrep.py show a-diagnosis-mistral-conrep
python scripts/evaluate_clinicia.py show a-diagnosis-mistral-conrep
```

After preparing the model environment and materializing required data, create a
portable transient config and launch a selected run:

```bash
python scripts/train_conrep.py run a-diagnosis-mistral-conrep \
  --output-root /path/to/new/artifacts
```

The original capsule is never edited. Old cluster paths remain visible as
historical evidence; the launcher maps recognized datasets to `data/clinicia/`
and writes new outputs under the path you provide.

## Paper matrix

| Paper tables | Regime and target | Models | Compared methods |
| --- | --- | --- | --- |
| 1 and 4 | A: celebrity diagnosis | Llama-2 7B Chat, Mistral 7B Instruct | Baseline, GradDiff, NPO, RMU, ConRep |
| 2 and 6 | B: injected PMC | Mistral 7B Instruct | Baseline, GradDiff, NPO, RMU, ConRep |
| 3 and 5 | A: celebrity deaths | Llama-2 7B Chat, Mistral 7B Instruct | Baseline, GradDiff, NPO, RMU, ConRep |

All 25 cells are indexed in
[`experiments/paper_runs/index.json`](experiments/paper_runs/index.json).
Exact train/evaluation/job files are materialized only for the five selected
ConRep cells for which the repository establishes that identity. The remaining
20 cells retain evidence records and table reconstruction without pretending
that an arbitrary historical job script was the selected run.

## Repository map

| Path | What belongs there |
| --- | --- |
| [`src/conrep/`](src/conrep/) | Project-owned method concepts, stable dispatch, and preserved trainers. |
| [`src/clinicia/`](src/clinicia/) | Probe definitions, metrics, dataset integrity, and evaluators. |
| [`experiments/paper_runs/`](experiments/paper_runs/) | All paper cells plus five exact ConRep run capsules. |
| [`data/clinicia/`](data/clinicia/) | Semantic Regime A/B dataset layout and historical-path catalog. |
| [`configs/paper/historical/`](configs/paper/historical/) | Evidence-backed records for all 25 paper cells. |
| [`results/paper/`](results/paper/) | Immutable evidence manifest, extracted metrics, and reconstructed tables. |
| [`third_party/`](third_party/) | Upstream-derived LLM2Vec and OpenUnlearning snapshots with documented local deltas. |
| [`legacy/`](legacy/) | Exploratory/superseded jobs and code not claimed as selected paper runs. |
| [`docs/`](docs/) | Reproduction guide, paper map, provenance, and historical reorganization records. |

The previous all-purpose top-level `llm2vec/` directory no longer exists:
upstream code, project code, paper jobs, data, and discarded explorations now
have different owners and different locations.

## What is reproducible today?

| Level | Status |
| --- | --- |
| Rebuild archived metrics and tables | **Runnable now**, standard library only |
| Trace each table cell to evidence | **Available now** for all 25 cells |
| Inspect exact selected ConRep configs/jobs | **Available now** for five cells |
| Rerun a selected ConRep job | Requires compatible model environment, checkpoints, and materialized datasets |
| Claim numerical end-to-end reproduction | Not yet; unresolved gates remain explicit |

Read [the reproduction guide](docs/reproduction.md) before model execution.
New measurements must use a separate `results/validated_v2/` namespace and
must not overwrite archived paper evidence.

## Important limitations

- Several datasets are Git LFS pointers; a pointer proves identity, not local
  availability or redistribution permission.
- Historical run files contain machine-specific paths and scheduler commands.
  They are preserved verbatim; use the public launchers for a portable view.
- LLM2Vec and OpenUnlearning require incompatible Transformers versions and
  therefore remain separate environments.
- The repository does not bundle model weights or selected checkpoints.
- The review PDF does not provide public citation metadata, so no DOI, arXiv
  identifier, author list, or `CITATION.cff` is fabricated.

## Validation

```bash
python scripts/configs/validate_historical_experiments.py \
  --index configs/paper/historical/index.json \
  --manifest results/paper/manifest.json
python scripts/configs/validate_reproduction_configs.py \
  --index configs/reproduction/index.json
python scripts/repository/validate_release_inventory.py \
  --manifest data/lfs_manifest.json \
  --dependencies docs/dependency_matrix.json
python -m unittest discover -s tests -p 'test_*.py'
```

These checks are offline and do not download models or LFS data, initialize a
model, submit a scheduler job, train, or evaluate.
