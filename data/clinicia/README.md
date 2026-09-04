# ClinicIA data

ClinicIA evaluates identifier–attribute knowledge through multiple linguistic
expressions rather than one fixed prompt.

- `regime_a/` contains celebrity diagnosis/death training sets and probes.
- `regime_b/pmc/` contains the injected clinical-knowledge setting, split
  into retain and forget probes.
- `catalog.json` gives stable semantic names and maps every current path to
  its historical `llm2vec/` path.

Generation probes cover direct QA, cloze, and background formulations. MCQ
probes cover attribute selection (ATT), exact-identifier recovery (IDeq), and
related-identifier discrimination (ID).

Many files are Git LFS pointers. A pointer records identity but is not the
dataset content; materialize only after reviewing the relevant data terms.
The protected paper inventory remains under `results/paper/` and deliberately
retains historical paths as provenance.
