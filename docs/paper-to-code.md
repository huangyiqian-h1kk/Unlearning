# Paper-to-code map

The paper *Towards Unlearning Beyond Textual Expressions for LLMs* contributes
two connected pieces: **ConRep**, a representation-space unlearning method, and
**ClinicIA**, a diversified identifier–attribute evaluation.

| Paper concept | Where to read | Executable or evidence path |
| --- | --- | --- |
| General representation objective (Eqs. 5–6) | `src/conrep/objectives.py` | preserved trainers in `src/conrep/legacy/` |
| Token-swap corruption (Eq. 7) | `src/conrep/corruption.py` | selected trainer variant |
| Combined objective (Eqs. 8–10) | `src/conrep/objectives.py` | `adaptive-random-token-lmloss-margin` |
| Symmetric specialization (Eqs. 11–13) | `src/conrep/objectives.py` | all five selected ConRep capsules |
| ClinicIA generation probes | `src/clinicia/probes.py` | `data/clinicia/**/probes/generation/` |
| ClinicIA MCQ probes | `src/clinicia/probes.py` | `data/clinicia/**/probes/mcq/` |
| Paper normalization formulas | `src/clinicia/metrics.py` | `scripts/results/build_tables.py` |
| Tables 1–6 | `src/clinicia/runs.py` | `python scripts/reproduce.py table N` |

## Experimental regimes

**Regime A** removes pre-existing celebrity diagnosis or death attributes from
Llama-2 and Mistral models. A complementary celebrity target acts as retention
knowledge.

**Regime B** first injects clinical identifier–attribute pairs from PMC into
Mistral, then separates forget and retain subsets. It therefore tests
unlearning of deliberately introduced knowledge.

## Reading a result end to end

For a selected ConRep cell:

1. Find the semantic experiment in `experiments/paper_runs/index.json`.
2. Inspect the byte-identical files in its `historical/` capsule.
3. Read the resolved evidence record in `configs/paper/historical/`.
4. Inspect the extracted metrics in `results/paper/raw_metrics/`.
5. Rebuild the normalized table with `scripts/reproduce.py`.

This chain separates what was historically observed from what is portable
today. It does not silently turn an unresolved baseline job into a claimed
paper reproduction.
