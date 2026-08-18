# Beyond Speed and Accuracy: Expanding the Measurement Space of the Stroop Test With Automated Item-Level Acoustic Process Analysis

This repository provides the task materials, English-language analysis code,
anonymized feature data, and statistical results accompanying the article.

The folders follow the order of the manuscript Results section. The four
cohorts are independent and contain no overlapping participants. Original
voice recordings are not public because voice data are personally
identifiable; full anonymized extracted features and statistical results are
provided.

## Repository Layout

```text
task_materials/                    Mandarin instructions and stimulus screens

data/
  example/                         Two anonymized workflow examples
    audio/
    asr_json/
    features/
    manual_textgrid/
    participant_info.xlsx
  public/
    features/                      Healthy automated and manual features (N=158)
    participant_info_HC_158.xlsx
    device_repeatability/          Independent repeatability cohort (N=51)
    wd/                            Independent WD cohort (N=43)
    substance_use/                 Independent substance-use cohort (N=58)

results/
  00_behavioral_load_gradient/
  01_item_level_parsing_validation/
  02_age_related_acoustic_responses/
  03_repeatability_cross_device/
  04_wd_external_validation/
  05_substance_use_external_validation/
  06_behavioral_acoustic_comparison/

scripts/
  01_speech_processing/
  02_behavioral_load_gradient/
  03_item_level_parsing_validation/
  04_age_related_acoustic_analysis/
  05_repeatability_cross_device/
  06_external_validation/
  07_visualization/

outputs/                            Locally generated files; ignored by git
```

## Anonymous IDs

Anonymous IDs are consistent across feature, participant-information, and
result tables within each cohort.

| Cohort | N | Anonymous ID range |
|---|---:|---|
| Healthy reference | 158 | `xxxxxxxx001`-`xxxxxxxx158` |
| Wilson's disease | 43 | `xxxxxxxx159`-`xxxxxxxx201` |
| Device repeatability | 51 | `xxxxxxxx202`-`xxxxxxxx252` |
| Substance use | 58 | `xxxxxxxx253`-`xxxxxxxx310` |

For the device cohort, `Participant_ID` uses the range above and `Subject_ID`
also identifies the recording device, for example `xxxxxxxx202_D1`. Direct
identifiers, original IDs, names, source folders, and private audio paths were
removed.

## Results Order

### 1. Behavioral Load Gradient

`results/00_behavioral_load_gradient/` contains item-derived task-level
accuracy and total hesitation duration for Hanzi, Dots, and STT, together with
their descriptive summaries. The corresponding scripts are in
`scripts/02_behavioral_load_gradient/`.

### 2. Validation of Item-Level Speech Parsing

`results/01_item_level_parsing_validation/` contains the Table 1 boundary
localization errors, Table S2 subject-task-feature agreement summary, and the
manual-annotation LME benchmark underlying Tables S3-S6. Reproduction scripts
are in `scripts/03_item_level_parsing_validation/`.

### 3. Age-Related Acoustic Responses

`results/02_age_related_acoustic_responses/lme_task_contrasts/` contains the
three automated Age x Task contrast tables underlying Table 2 and Tables
S4-S6. The primary LME script is
`scripts/04_age_related_acoustic_analysis/01_lme_task_load_models.py`.

The targeted education-adjusted sensitivity analysis of the 25 prespecified
clinical features is provided as:

- `scripts/04_age_related_acoustic_analysis/02_education_adjusted_sensitivity.py`
- `results/02_age_related_acoustic_responses/education_adjusted_sensitivity/`

All 25 adjusted Age x Task coefficients retained their original direction;
15 remained significant at `q <= .05`, 6 at `q <= .01`, and no Education x
Task term survived FDR correction. These results correspond to Table S7.

### 4. Repeatability and Cross-Device Reproducibility

The independent cohort includes 51 participants recorded in two sessions on
three devices. Public features are under `data/public/device_repeatability/`.
Manuscript-ready summaries are:

- `results/03_repeatability_cross_device/table3_within_device_retest_summary.csv`
- `results/03_repeatability_cross_device/table_s8_cross_device_summary.csv`
- `results/03_repeatability_cross_device/table_s9_cross_device_features_icc_ge_0_50.csv`

Scripts are in `scripts/05_repeatability_cross_device/`. The feature-recovery
pipeline requires restricted recordings; the ICC and table-building scripts
operate on the public feature table.

### 5. External Evaluation in Wilson's Disease

`data/public/wd/` contains 12 WD-CI and 31 WD-CN participants. The fixed
healthy-reference deviation results corresponding to Table 4 are under
`results/04_wd_external_validation/`. Scripts 01-02 in
`scripts/06_external_validation/` document deviation scoring and feature
recovery. Audio recovery requires restricted recordings; deviation scoring
uses the public feature and participant-information tables.

### 6. External Evaluation in the Substance-Use Cohort

`data/public/substance_use/` contains 28 Drug participants and 30 healthy
controls. Public statistical outputs are:

- `results/05_substance_use_external_validation/table5_moca_correlations.csv`
- `results/05_substance_use_external_validation/table_s10_drug_vs_hc.csv`
- `results/05_substance_use_external_validation/table_s11_moca_groups.csv`

For categorical analyses, one point is added to MoCA when education is 12
years or less, capped at 30; corrected scores below 26 define MoCA-CI.
Spearman correlations use uncorrected MoCA scores. Scripts 03-05 in
`scripts/06_external_validation/` reproduce this workflow from the available
features, except for the restricted-audio feature extraction step.

### 7. Comparison with Traditional Stroop Scoring

The final clinical comparison uses six age-referenced traditional Stroop
measures: Hanzi and STT accuracy, Hanzi and STT card completion time, and the
STT-versus-Hanzi interference costs for accuracy and completion time. These
measures are compared with four acoustic deviation indices and a combined
model. Predictions are generated with leave-one-subject-out cross-validated
L2 logistic regression; median imputation and standardization are fitted
within each training fold, and AUC confidence intervals use DeLong's method.

- `scripts/06_external_validation/06_wd_behavioral_acoustic_auc.py`
- `scripts/06_external_validation/07_substance_use_behavioral_acoustic_auc.py`
- `results/06_behavioral_acoustic_comparison/table6_behavioral_acoustic_auc.csv`

The cohort-specific directories contain the healthy behavioral reference
models, anonymous subject-level deviations, univariate results, and complete
cross-validated AUC outputs.

## Speech-Processing Workflow

1. `scripts/01_speech_processing/01_qwen3_asr_alignment_pipeline.py` performs
   ASR, forced alignment, and adaptive boundary refinement.
2. `scripts/01_speech_processing/02_extract_item_acoustic_features.py`
   extracts 88 openSMILE eGeMAPS functionals plus item duration and the
   preceding hesitation interval.
3. `scripts/01_speech_processing/03_impute_missing_acoustic_features.py`
   performs hierarchical within-sequence and task/global median imputation.

The two anonymized examples under `data/example/` illustrate this workflow.
Mandarin stimulus labels are retained where needed; code, comments, paths, and
output labels are in English.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The ASR and alignment scripts additionally require the Qwen models:

- [Qwen3-ASR-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ASR-0.6B on ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ASR-0.6B)
- [Qwen3-ForcedAligner-0.6B on Hugging Face](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [Qwen3-ForcedAligner-0.6B on ModelScope](https://modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B)

For local model downloads, set:

```powershell
$env:QWEN3_ASR_MODEL = "path\to\Qwen3-ASR-0.6B"
$env:QWEN3_ALIGNER_MODEL = "path\to\Qwen3-ForcedAligner-0.6B"
```

All public scripts use repository-relative defaults. Reanalysis outputs are
written to `outputs/` unless an explicit `--output-dir` is supplied. For
example:

```bash
python scripts/04_age_related_acoustic_analysis/02_education_adjusted_sensitivity.py
python scripts/06_external_validation/06_wd_behavioral_acoustic_auc.py
python scripts/06_external_validation/07_substance_use_behavioral_acoustic_auc.py
```

## Data Availability

Original audio recordings are not publicly available because of privacy and
ethical restrictions. Complete anonymized feature-extraction results,
statistical results, and documented analysis code are provided in this
repository. The corresponding author may be contacted regarding restricted
source data at `lzyang@cmpt.ac.cn`.
