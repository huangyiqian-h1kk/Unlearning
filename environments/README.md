# Environments

Use separate model environments:

1. **ConRep/LLM2Vec:** install the pinned package in
   `third_party/llm2vec/` and the project package from the repository root.
2. **OpenUnlearning baselines:** install
   `third_party/open-unlearning/` in a separate environment.

Their Transformers constraints conflict. The standard-library evidence tools
(`scripts/reproduce.py` and validators) need neither model environment.

Historical jobs name the original cluster modules, conda path, caches, and
scheduler directives. Treat those as provenance. The public launchers replace
recognized dataset/output paths but do not install CUDA, download weights,
fetch LFS data, or claim that the historical checkpoint is available.
