# Archived paper-result layer

`manifest.json` identifies immutable archive members. `raw_metrics/` contains hash-verified compact snapshots. `reconstructed/` contains deterministic CSV, Markdown, and LaTeX tables. Table reconstruction does not train or evaluate models.

Status flags and footnotes are documented in the manifest and provenance documents. `†` denotes an MCQ baseline not significantly above chance when exact counts support the test; `‡` denotes a non-comparable starting checkpoint; `N/A` denotes missing or unresolved evidence; `–` denotes significance masking. In CSV and Markdown, a masked baseline is `–†`; in LaTeX it is `\textemdash{}\textsuperscript{\dagger}`.

Exact MCQ dataset identities and counts are recorded in `mcq_dataset_inventory.json`. Regime B averages include unmasked R% and F% values but exclude Delta, MMLU, missing, and masked cells. Compact snapshots include parsed tracked configurations, selected terminal trainer state, and selected MMLU log-block evidence; they are not claims of full end-to-end reproduction.
