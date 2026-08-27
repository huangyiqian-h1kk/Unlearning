| Experiment | Model | Method | Status | R-QA | R-Cloze | R-BG | F-QA | F-Cloze | F-BG | R-ATT | R-IDeq | R-ID | F-ATT | F-IDeq | F-ID | MMLU | Average |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a-diagnosis-llama2-baseline | meta-llama/Llama-2-7b-chat-hf | Baseline | verified | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | –† | 100.00 | 0.00 | N/A | –† | 0.46 | 55.56 |
| a-diagnosis-llama2-graddiff | meta-llama/Llama-2-7b-chat-hf | GradDiff | verified | 92.50 | 100.00 | 100.00 | 33.33 | 60.00 | 0.00 | 87.88 | – | 90.48 | 30.19 | N/A | – | N/A | 66.04 |
| a-diagnosis-llama2-npo | meta-llama/Llama-2-7b-chat-hf | NPO | verified;manuscript_transcription_error | 97.50 | 99.15 | 100.00 | 25.00 | 0.00 | 28.57 | 100.00 | – | 90.48 | 0.00 | N/A | – | N/A | 60.08 |
| a-diagnosis-llama2-rmu | meta-llama/Llama-2-7b-chat-hf | RMU | verified;manuscript_transcription_error | 100.00 | 100.00 | 100.00 | 16.67 | 40.00 | 14.29 | 93.94 | – | 100.00 | 83.02 | N/A | – | N/A | 71.99 |
| a-diagnosis-mistral-baseline | mistralai/Mistral-7B-Instruct-v0.2 | Baseline | verified | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.00 | 100.00 | –† | –† | 0.00 | N/A | –† | 0.59 | 50.00 |
| a-diagnosis-mistral-graddiff | mistralai/Mistral-7B-Instruct-v0.2 | GradDiff | verified;manuscript_transcription_error | 0.00 | 0.00 | 1.50 | 100.00 | 100.00 | 100.00 | 51.28 | – | – | 98.36 | N/A | – | N/A | 56.39 |
| a-diagnosis-mistral-npo | mistralai/Mistral-7B-Instruct-v0.2 | NPO | verified | 15.65 | 48.77 | 8.27 | 100.00 | 100.00 | 95.45 | 74.36 | – | – | 100.00 | N/A | – | N/A | 67.81 |
| a-diagnosis-mistral-rmu | mistralai/Mistral-7B-Instruct-v0.2 | RMU | verified;manuscript_transcription_error | 91.84 | 91.36 | 57.14 | 100.00 | 100.00 | 54.55 | 73.08 | – | – | 100.00 | N/A | – | N/A | 83.49 |
| a-diagnosis-llama2-conrep | meta-llama/Llama-2-7b-chat-hf | ConRep | verified;label_corrected | 95.83 | 90.68 | 75.86 | 91.67 | 20.00 | 42.86 | 87.88 | – | 100.00 | 67.92 | N/A | – | N/A | 74.74 |
| a-diagnosis-mistral-conrep | mistralai/Mistral-7B-Instruct-v0.2 | ConRep | verified | 91.84 | 83.95 | 64.66 | 100.00 | 81.82 | 86.36 | 97.44 | – | – | 100.00 | N/A | – | N/A | 88.26 |
