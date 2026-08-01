# Spoken Stroop Trial-Level Speech Phenotyping

This repository contains the public code, anonymized feature data, and result
tables for the manuscript:

**Automated Phenotyping Reveals How Task Load and Age Modulate Speech Motor Control**

The repository follows the order of the Results section. It includes the full
anonymized feature tables for the healthy reference, device-repeatability,
Wilson's disease (WD), and substance-use cohorts. Original recordings are not
public because voice data are personally identifiable. Two anonymized healthy
examples are provided to illustrate the speech-processing workflow.

The four cohorts are independent and contain no overlapping participants.

## Repository Layout

```text
data/
  example/
    audio/                         Two anonymized wav examples
    asr_json/                      Example Qwen3-ASR outputs
    features/                      Two-subject trial-level feature example
    manual_textgrid/               Example Praat annotations
    participant_info.xlsx          Information for the two examples
  public/
    features/
      Automated_feature.csv        Healthy automated features (N = 158)
      Praat_manual_feature.csv      Healthy manually segmented features
    participant_info_HC_158.xlsx    Healthy reference participant information
    device_repeatability/
      Stroop_features_device_repeatability.csv
      participant_info_device_repeatability_51.csv
    wd/
      features/
        Stroop_features_WD_CI.csv
        Stroop_features_WD_CN.csv
      WD_participant_info.xlsx
    substance_use/
      features/
        Stroop_features_Drug.csv
        Stroop_features_HC.csv
      participant_info_substance_use_58.csv

results/
  00_behavioral_load_gradient/
  01_trial_level_parsing_validation/
  02_age_related_acoustic_responses/
  03_repeatability_cross_device/
  04_wd_external_validation/
  05_substance_use_external_validation/

scripts/
  common_paths.py
  01_speech_processing/
  02_behavioral_load_gradient/
  03_trial_level_parsing_validation/
  04_age_related_acoustic_analysis/
  05_repeatability_cross_device/
  06_external_validation/
  07_visualization/

outputs/                            Generated locally; ignored by git
```

## Anonymous IDs

The same anonymous ID is used consistently within each cohort and across its
public feature, participant-information, and result tables.

| Cohort | N | Anonymous ID range |
|---|---:|---|
| Healthy reference | 158 | `xxxxxxxx001`-`xxxxxxxx158` |
| Wilson's disease | 43 | `xxxxxxxx159`-`xxxxxxxx201` |
| Device repeatability | 51 | `xxxxxxxx202`-`xxxxxxxx252` |
| Substance use | 58 | `xxxxxxxx253`-`xxxxxxxx310` |

For the device-repeatability feature table, `Participant_ID` uses the range
above and `Subject_ID` appends the recording device, for example
`xxxxxxxx202_D1`. Names, original IDs, source folders, and local audio paths
were removed. The substance-use tables use `Drug` and `HC` group labels and
English categorical values.

## Results Order

### 1. Behavioral Load Gradient

`results/00_behavioral_load_gradient/task_level_behavioral_metrics.csv`
contains overall accuracy and total hesitation duration for Hanzi, Dots, and
STT. The same directory also contains separate descriptive-statistics tables
for the two behavioral measures.

Scripts:

- `scripts/02_behavioral_load_gradient/01_extract_behavioral_metrics_from_textgrid.py`
- `scripts/02_behavioral_load_gradient/02_plot_behavioral_task_metrics.py`

### 2. Validation of Trial-Level Speech Parsing

`results/01_trial_level_parsing_validation/` contains the Table 1 boundary
localization results, the Table S2 subject-task-feature agreement summary, and
the manual-annotation LME benchmark used for comparison with the automated
analysis.

Scripts:

- `scripts/03_trial_level_parsing_validation/01_validate_token_timestamps.py`
- `scripts/03_trial_level_parsing_validation/02_subject_task_feature_spearman.py`

### 3. Age-Related Acoustic Responses

`results/02_age_related_acoustic_responses/01_trial_vs_utterance_ablation/`
contains the utterance- versus trial-level comparison reported in Table 2.
`results/02_age_related_acoustic_responses/02_lme_task_contrasts/` contains the
three automated Age x Task contrast tables underlying Table 3 and the reported
feature-level results.

Scripts:

- `scripts/04_age_related_acoustic_analysis/01_lme_task_load_models.py`
- `scripts/04_age_related_acoustic_analysis/02_trial_level_ablation_models.py`

### 4. Repeatability and Cross-Device Reproducibility

The independent device cohort contains 51 participants, two sessions, and
three devices. Public trial-level features are in
`data/public/device_repeatability/`. The manuscript-ready tables are:

- `results/03_repeatability_cross_device/table4_within_device_retest_summary.csv`
- `results/03_repeatability_cross_device/table_s7_cross_device_summary.csv`
- `results/03_repeatability_cross_device/table_s8_cross_device_features_icc_ge_0_50.csv`

Feature-level ICC, pairwise agreement, and device-distribution tables used to
derive these summaries are provided in the same result directory.

Scripts:

- `scripts/05_repeatability_cross_device/01_device_feature_pipeline.py`
- `scripts/05_repeatability_cross_device/02_within_device_retest_icc.py`
- `scripts/05_repeatability_cross_device/03_cross_device_reproducibility.py`
- `scripts/05_repeatability_cross_device/04_build_manuscript_tables.py`

The first script requires private recordings and metadata. The reliability and
summary scripts run directly from the public feature and result tables.

### 5. External Validation in Wilson's Disease

The WD feature tables and English participant-information table are under
`data/public/wd/`. WD-CI and WD-CN deviation-score results corresponding to
Table 5 are under `results/04_wd_external_validation/`.

Scripts:

- `scripts/06_external_validation/01_wd_deviation_scores.py`
- `scripts/06_external_validation/02_wd_feature_recovery_pipeline.py`

The recovery pipeline documents processing of private WD audio; the deviation
analysis uses the public feature and participant-information tables.

### 6. External Validation in the Substance-Use Cohort

The independent cohort contains 28 Drug participants and 30 healthy controls.
Public participant information and features are under
`data/public/substance_use/`. Public outputs are:

- `results/05_substance_use_external_validation/subject_level_deviation_indices.csv`
- `results/05_substance_use_external_validation/table6_moca_correlations.csv`
- `results/05_substance_use_external_validation/table_s9_drug_vs_hc.csv`
- `results/05_substance_use_external_validation/table_s10_moca_groups.csv`

For categorical analyses, one point is added when education is 12 years or
less, capped at 30; corrected MoCA scores below 26 define MoCA-CI. Spearman
correlations use the original MoCA score. FDR-BH correction is applied across
the four deviation indices separately within each analysis set.

Scripts:

- `scripts/06_external_validation/03_substance_use_feature_pipeline.py`
- `scripts/06_external_validation/04_substance_use_deviation_scores.py`
- `scripts/06_external_validation/05_substance_use_statistical_validation.py`

The feature pipeline requires private audio. The deviation and statistical
validation scripts use the public feature and participant-information tables.

## Speech-Processing Workflow

The primary automated pipeline is organized as follows:

1. `scripts/01_speech_processing/01_qwen3_asr_alignment_pipeline.py`
   performs Qwen3 ASR, forced alignment, and adaptive boundary refinement.
2. `scripts/01_speech_processing/02_extract_trial_acoustic_features.py`
   extracts 88 openSMILE eGeMAPS functionals plus word duration and pre-word
   interval for each trial.
3. `scripts/01_speech_processing/03_impute_missing_acoustic_features.py`
   performs hierarchical within-sequence and task/global median imputation.

The Chinese color words (`red`, `yellow`, `blue`, and `green` in the task
materials) remain encoded as their original Mandarin characters in the scripts
because they are experimental stimuli, while code, comments, paths, and output
labels are in English.

## Installation

Create a Python environment and install the listed dependencies:

```bash
pip install -r requirements.txt
```

The ASR and audio-recovery scripts additionally require Qwen3-ASR-0.6B and
Qwen3-ForcedAligner-0.6B. Official model pages:

- [Qwen3-ASR-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ASR-0.6B on ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ForcedAligner-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [Qwen3-ForcedAligner-0.6B on ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B)

Set model locations with environment variables when using local downloads:

```powershell
$env:QWEN3_ASR_MODEL = "path\to\Qwen3-ASR-0.6B"
$env:QWEN3_ALIGNER_MODEL = "path\to\Qwen3-ForcedAligner-0.6B"
```

All public scripts resolve inputs and outputs relative to the repository root
through `scripts/common_paths.py`. Generated files are written under
`outputs/`. Private-audio pipelines accept `DEVICE_AUDIO_ROOT`,
`DEVICE_METADATA_CSV`, `WD_AUDIO_ROOT`, and `SUBSTANCE_USE_AUDIO_ROOT` when the
corresponding restricted data are available.

## Privacy and Data Availability

- Original voice recordings are not public because of privacy and ethical
  restrictions.
- Full anonymized extracted features and statistical result tables are public.
- Two anonymized healthy audio/TextGrid examples are included solely to
  demonstrate the processing workflow.
- Direct identifiers, original subject IDs, local source folders, and private
  audio paths are excluded from the public device and substance-use data.
