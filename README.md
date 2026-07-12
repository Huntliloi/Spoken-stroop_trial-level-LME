# Spoken Stroop Trial-Level Speech Phenotyping

This repository contains the public example code and two anonymized example
subjects for the manuscript:

**Automated Phenotyping Reveals How Task Load and Age Modulate Speech Motor Control**

Only key experimental and analysis code is included. The public data are limited
to two anonymized example subjects so reviewers can inspect the file structure
and run lightweight checks. Full cohort data and WD clinical validation data are
not included for privacy reasons.

## Repository Layout

```text
data/example/
  audio/                 Anonymized example wav files
  asr_json/              Example Qwen3-ASR JSON outputs
  features/              Precomputed two-subject acoustic feature table
  manual_textgrid/       Example manual TextGrid annotations
  participant_info.xlsx  English participant table for the two examples
  participant_info.csv   CSV copy of the same table

scripts/
  common_paths.py        Shared repository-relative paths
  01_speech_processing/  Methods 2.3: ASR, alignment, acoustic features
  02_feature_validation/ Results 3.1: timestamp and feature validation
  03_behavioral_analysis/ Results 3.4: behavioral metrics and correlations
  04_mixed_effects_modeling/ Methods 2.4 and Results 3.2-3.3
  05_external_validation/ Results 3.5: WD validation template, data not shared
  06_visualization/      Waveform and boundary plotting utilities

outputs/                 Created by scripts, ignored by git
```

The precomputed feature table is:

```text
data/example/features/Stroop_features_imputed_example.csv
```

It contains the anonymized two-subject example version of the imputed
trial-level acoustic features used by the modeling scripts. Downstream scripts
automatically use `outputs/Stroop_features_imputed_add.csv` if you generated it
locally; otherwise they fall back to this public example feature table.

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

5. `scripts/02_feature_validation/02_feature_level_correlation.py` and
   `scripts/02_feature_validation/03_subject_task_feature_spearman.py`
   Validate automatic versus manual feature tables. The original full manual
   feature table is not included; provide it at the path shown in each script if
   you want to reproduce this analysis fully.

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
    Documents the WD deviation-score analysis. WD participant data are not
    public, so this script is included as a transparent analysis template only.

11. `scripts/06_visualization/01_plot_waveforms_with_token_boundaries.py`
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
- The participant information table contains only two anonymous example rows.
- Full cohort data, clinical WD data, and original project paths are not
  included.
- Scripts that require unavailable private data are retained as templates and
  clearly marked by placeholder paths under `outputs/`.
