# Result reconciliation

This document records conflicts without altering archived measurements.

| Finding | Status | Release treatment |
|---|---|---|
| Deaths/Llama-2 ConRep archive orders MCQs as ATT/IDeq/ID ≈ `.37/.32/.37`; an appendix appears to swap the first two. | `manuscript_transcription_error` | Preserve archive order. |
| PMC ConRep retain ATT is `.51`; manuscript presentation is approximately `.50`. | `manuscript_transcription_error` | Preserve `.51` in raw snapshot. |
| A Mistral GradDiff appendix Diagnosis row may contain Deaths values. | `unresolved`, `manuscript_transcription_error` | Use archived Diagnosis identity only when configuration and metric source agree. |
| Llama-2 NPO Diagnosis ATT is approximately `.49` in archived/main evidence versus `.51` in one appendix location. | `manuscript_transcription_error` | Preserve archive value. |
| RMU retain-Diagnosis MCQs conflict with an appendix. | `unresolved`, `manuscript_transcription_error` | Do not substitute appendix values. |
| `rmu_celebrity_death_llama2` resolves to Mistral and Diagnosis despite its label; MMLU ≈ `.4603`. | `unresolved` | Excluded from intended Deaths cell. |
| `rmu_celebrity_death_Mixtral` resolves to Llama-2 and Deaths despite its label; MMLU ≈ `.4608`. | `unresolved` | Not automatically reassigned. |
| PMC GradDiff begins at the CSV checkpoint rather than universal continuation. | `non_comparable` | Display with `‡ different starting checkpoint`. |
| PMC ConRep MMLU `.2659` belongs to `lm0.01`, four epochs, not selected `lm0`, five epochs. | `unresolved` for selected cell | Selected ConRep MMLU remains `N/A`. |

No item above is a corrected experiment; each is a provenance or transcription finding.
