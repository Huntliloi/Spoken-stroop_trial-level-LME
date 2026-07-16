# Spoken Stroop Trial-Level Speech Phenotyping

This repository contains public code, full anonymized healthy-subject feature
tables, and two anonymized audio/TextGrid examples for the manuscript:

**Automated Phenotyping Reveals How Task Load and Age Modulate Speech Motor Control**

Only key experimental and analysis code is included. Healthy-subject
trial-level feature tables and participant demographics are public after
subject-ID anonymization. Raw audio/TextGrid examples are limited to two
subjects. WD clinical validation feature tables and the filtered WD information
table are public after anonymization; original WD audio and segment files are
not included for privacy reasons.

## Repository Layout

```text
data/example/
  audio/                 Anonymized example wav files
  asr_json/              Example Qwen3-ASR JSON outputs
  features/              Precomputed two-subject acoustic feature table
  manual_textgrid/       Example manual TextGrid annotations
  participant_info.xlsx  English participant table for the two examples

data/public/
  features/
    Automated_feature.csv       Full anonymized automatic feature table
    Praat_manual_feature.csv    Full anonymized Praat/manual feature table
  participant_info_HC_158.xlsx  Full anonymized HC participant information table
  wd/
    features/
      Stroop_features_WD_CI.csv Full anonymized WD cognitive-impairment feature table
      Stroop_features_WD_CN.csv Full anonymized WD cognitively normal feature table
    WD_participant_info.xlsx    English WD information table for the 43 public WD samples

results/
  00_trial_level_parsing_validation/
    boundary_localization_table1.csv
    subject_task_feature_correlation_summary.csv
  Stroop_TaskLevel_NewFeatures.csv  Full anonymized task-level result table
  01_lme_task_contrasts/            Automated LME task-contrast result tables
    manual_annotations/             Manual-annotation LME benchmark tables
  02_ablation_analysis/             Utterance- versus trial-level ablation tables
  03_behavior_speech_coupling/      Behavior-speech coupling result tables
  04_wd_external_validation/        WD_CI and WD_CN external-validation result tables

scripts/
  common_paths.py        Shared repository-relative paths
  01_speech_processing/  Methods 2.3: ASR, alignment, acoustic features
  02_feature_validation/ Results 3.1: parsing validation
  03_behavioral_analysis/ Results 3.4: behavioral metrics and correlations
  04_mixed_effects_modeling/ Methods 2.4 and Results 3.2-3.3
  05_external_validation/ Results 3.5: WD external clinical validation
  06_visualization/      Waveform and boundary plotting utilities

outputs/                 Created by scripts, ignored by git
```

The full public feature tables are:

```text
data/public/features/Automated_feature.csv
data/public/features/Praat_manual_feature.csv
data/public/participant_info_HC_158.xlsx
```

All public subject IDs are anonymized and sorted as `xxxxxxxx001` through
`xxxxxxxx158`. The same anonymous ID is used consistently across the automatic
feature table, Praat/manual feature table, and participant information table.

The public WD validation files are:

```text
data/public/wd/features/Stroop_features_WD_CI.csv
data/public/wd/features/Stroop_features_WD_CN.csv
data/public/wd/WD_participant_info.xlsx
```

WD subject IDs continue the same anonymous sequence as `xxxxxxxx159` through
`xxxxxxxx201`. The WD information table was filtered to the 43 WD samples used
by the public feature tables, and its column names and categorical values were
translated to English. Original WD audio and segment files are not shared; path
columns in the WD feature tables contain anonymized placeholder paths.

The two-subject example feature table is:

```text
data/example/features/Stroop_features_imputed_example.csv
```

It contains a small anonymized example version of the imputed trial-level
acoustic features. Downstream scripts automatically use
`outputs/Stroop_features_imputed_add.csv` if you generated it locally; otherwise
they fall back to `data/public/features/Automated_feature.csv`, and finally to
the two-subject example file if the full public table is unavailable.

Public raw result tables are provided under `results/`. The
`00_trial_level_parsing_validation/` directory contains the
boundary-localization summary table for Table 1 and the subject-task-feature
correlation summary table for Table S2. The `manual_annotations/` directory
under `01_lme_task_contrasts/` contains the three manual-annotation benchmark
tables corresponding to the automated LME task-contrast result tables. Subject
identifiers in public result tables use the same anonymous ID sequence as the
public feature and participant-information files. The clinical validation
labels use `WD_CI` for WD participants with cognitive impairment and `WD_CN` for
cognitively normal WD participants.

## Script Order

Run scripts from the repository root.

1. `scripts/01_speech_processing/01_qwen3_asr_alignment_pipeline.py`
   Runs ASR, forced alignment, and adaptive boundary refinement. This requires
   Qwen3 ASR and forced-aligner model files or compatible model IDs.

2. `scripts/01_speech_processing/02_extract_trial_acoustic_features.py`
   Uses `outputs/asr_qwen3/token_results.csv` and example wav files to extract
   trial-level OpenSMILE features.

3. `scripts/01_speech_processing/03_impute_missing_acoustic_features.py`
   Imputes missing OpenSMILE feature values.

   If you only want to inspect or run the downstream modeling code with the
   two-subject example, you can skip steps 1-3 and use the provided file in
   `data/example/features/`.

4. `scripts/02_feature_validation/01_validate_token_timestamps.py`
   Compares automatic token timestamps with manual TextGrid annotations.

5. `scripts/02_feature_validation/02_subject_task_feature_spearman.py`
   Evaluates whether automated timestamps preserve within-subject variation at
   the subject-task-feature level using
   `data/public/features/Automated_feature.csv` and
   `data/public/features/Praat_manual_feature.csv`.

6. `scripts/03_behavioral_analysis/01_extract_behavioral_metrics_from_textgrid.py`
   Computes task-level accuracy and hesitation duration from TextGrid files.

7. `scripts/03_behavioral_analysis/02_plot_behavioral_task_metrics.py` and
   `scripts/03_behavioral_analysis/03_behavior_speech_shift_correlation.py`
   Plot behavioral summaries and compute behavior-speech shift correlations.

8. `scripts/04_mixed_effects_modeling/01_lme_task_load_models.py`
   Fits trial-level linear mixed-effects models.

9. `scripts/04_mixed_effects_modeling/02_trial_level_ablation_models.py`
   Compares utterance-level and trial-level modeling.

10. `scripts/05_external_validation/01_wd_deviation_scores_unshared_data.py`
    Computes WD deviation scores using the public WD_CI and WD_CN feature tables
    and the English WD information table.

11. `scripts/05_external_validation/02_recovered_wd_stroop_pipeline.py`
    Documents the recovered WD ASR and OpenSMILE feature-extraction pipeline for
    the additional WD validation samples. The raw WD audio is not public; if
    private audio is available, set `WD_AUDIO_ROOT`, `QWEN3_ASR_MODEL`, and
    `QWEN3_ALIGNER_MODEL`, then write regenerated tables under
    `outputs/wd_recovered_pipeline/`.

12. `scripts/06_visualization/01_plot_waveforms_with_token_boundaries.py`
    Generates waveform plots with token boundaries.

## Paths

All scripts use repository-relative paths from `scripts/common_paths.py`.
No script should require the original local project path. Standard outputs are
written under `outputs/`.

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

The ASR step additionally requires the Qwen3 ASR and forced-aligner models.
Download pages:

- Qwen3-ASR-0.6B: [Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) or [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B)
- Qwen3-ForcedAligner-0.6B: [Hugging Face](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) or [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B)

After downloading the models, point the script to the local model folders with:

```bash
set QWEN3_ASR_MODEL=path\to\Qwen3-ASR-0.6B
set QWEN3_ALIGNER_MODEL=path\to\Qwen3-ForcedAligner-0.6B
```

On Linux or macOS, use `export` instead of `set`.

## Privacy Notes

- Subject identifiers and folder names are anonymized.
- Full healthy-subject feature tables and participant information are included
  with IDs anonymized as `xxxxxxxx001` to `xxxxxxxx158`.
- Raw audio/TextGrid examples are limited to two anonymized subjects.
- Public WD feature tables and the filtered WD information table are included
  with IDs anonymized as `xxxxxxxx159` to `xxxxxxxx201`.
- Original WD audio, segment files, and original project paths are not included.
