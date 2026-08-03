"""Shared paths for the public spoken Stroop example repository.

All scripts resolve paths relative to the repository root so reviewers can run
the example after cloning the GitHub repository without editing local absolute
paths.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
EXAMPLE_DATA_DIR = DATA_DIR / "example"
EXAMPLE_FEATURE_DIR = EXAMPLE_DATA_DIR / "features"
PUBLIC_DATA_DIR = DATA_DIR / "public"
PUBLIC_FEATURE_DIR = PUBLIC_DATA_DIR / "features"
PUBLIC_DEVICE_DIR = PUBLIC_DATA_DIR / "device_repeatability"
PUBLIC_WD_DIR = PUBLIC_DATA_DIR / "wd"
PUBLIC_WD_FEATURE_DIR = PUBLIC_WD_DIR / "features"
PUBLIC_SUBSTANCE_DIR = PUBLIC_DATA_DIR / "substance_use"
PUBLIC_SUBSTANCE_FEATURE_DIR = PUBLIC_SUBSTANCE_DIR / "features"
OUTPUT_DIR = REPO_ROOT / "outputs"
RESULTS_DIR = REPO_ROOT / "results"

AUDIO_ROOT = EXAMPLE_DATA_DIR / "audio"
ASR_JSON_ROOT = EXAMPLE_DATA_DIR / "asr_json"
MANUAL_TEXTGRID_ROOT = EXAMPLE_DATA_DIR / "manual_textgrid"
EXAMPLE_PARTICIPANT_INFO_XLSX = EXAMPLE_DATA_DIR / "participant_info.xlsx"
PARTICIPANT_INFO_XLSX = PUBLIC_DATA_DIR / "participant_info_HC_158.xlsx"

TOKEN_RESULTS_CSV = OUTPUT_DIR / "asr_qwen3" / "token_results.csv"
ACOUSTIC_FEATURES_CSV = OUTPUT_DIR / "Stroop_features_add.csv"
IMPUTED_ACOUSTIC_FEATURES_CSV = OUTPUT_DIR / "Stroop_features_imputed_add.csv"
EXAMPLE_IMPUTED_ACOUSTIC_FEATURES_CSV = EXAMPLE_FEATURE_DIR / "Stroop_features_imputed_example.csv"
PUBLIC_AUTOMATED_FEATURE_CSV = PUBLIC_FEATURE_DIR / "Automated_feature.csv"
PUBLIC_PRAAT_MANUAL_FEATURE_CSV = PUBLIC_FEATURE_DIR / "Praat_manual_feature.csv"
PUBLIC_DEVICE_FEATURE_CSV = PUBLIC_DEVICE_DIR / "Stroop_features_device_repeatability.csv"
PUBLIC_DEVICE_INFO_CSV = PUBLIC_DEVICE_DIR / "participant_info_device_repeatability_51.csv"
PUBLIC_WD_CI_FEATURE_CSV = PUBLIC_WD_FEATURE_DIR / "Stroop_features_WD_CI.csv"
PUBLIC_WD_CN_FEATURE_CSV = PUBLIC_WD_FEATURE_DIR / "Stroop_features_WD_CN.csv"
PUBLIC_WD_INFO_XLSX = PUBLIC_WD_DIR / "WD_participant_info.xlsx"
PUBLIC_SUBSTANCE_DRUG_FEATURE_CSV = PUBLIC_SUBSTANCE_FEATURE_DIR / "Stroop_features_Drug.csv"
PUBLIC_SUBSTANCE_HC_FEATURE_CSV = PUBLIC_SUBSTANCE_FEATURE_DIR / "Stroop_features_HC.csv"
PUBLIC_SUBSTANCE_INFO_CSV = PUBLIC_SUBSTANCE_DIR / "participant_info_substance_use_58.csv"
PUBLIC_BEHAVIORAL_METRICS_CSV = RESULTS_DIR / "00_behavioral_load_gradient" / "task_level_behavioral_metrics.csv"
BEHAVIORAL_METRICS_CSV = OUTPUT_DIR / "behavior" / "task_level_behavioral_metrics.csv"

FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"
VALIDATION_OUTPUT_DIR = OUTPUT_DIR / "validation"
LME_OUTPUT_DIR = OUTPUT_DIR / "lme"
BEHAVIOR_OUTPUT_DIR = OUTPUT_DIR / "behavior"
WD_RECOVERED_OUTPUT_DIR = OUTPUT_DIR / "wd_recovered_pipeline"
DEVICE_REPEATABILITY_OUTPUT_DIR = OUTPUT_DIR / "device_repeatability"
SUBSTANCE_USE_OUTPUT_DIR = OUTPUT_DIR / "substance_use"


def get_imputed_acoustic_features_csv() -> Path:
    """Return generated imputed features when present; otherwise use public data."""
    if IMPUTED_ACOUSTIC_FEATURES_CSV.exists():
        return IMPUTED_ACOUSTIC_FEATURES_CSV
    if PUBLIC_AUTOMATED_FEATURE_CSV.exists():
        return PUBLIC_AUTOMATED_FEATURE_CSV
    return EXAMPLE_IMPUTED_ACOUSTIC_FEATURES_CSV


def get_behavioral_metrics_csv() -> Path:
    """Return generated behavioral metrics, otherwise the public result table."""
    if BEHAVIORAL_METRICS_CSV.exists():
        return BEHAVIORAL_METRICS_CSV
    return PUBLIC_BEHAVIORAL_METRICS_CSV


def get_device_features_csv() -> Path:
    """Return generated device features, otherwise the public feature table."""
    generated = DEVICE_REPEATABILITY_OUTPUT_DIR / "Stroop_features_device_repeatability.csv"
    if generated.exists():
        return generated
    return PUBLIC_DEVICE_FEATURE_CSV


def get_substance_subject_scores_csv() -> Path:
    """Return generated substance-use scores, otherwise the public result table."""
    generated = SUBSTANCE_USE_OUTPUT_DIR / "subject_level_deviation_indices.csv"
    if generated.exists():
        return generated
    return RESULTS_DIR / "05_substance_use_external_validation" / "subject_level_deviation_indices.csv"


def ensure_output_dirs() -> None:
    """Create the standard output folders used by the analysis scripts."""
    for path in [
        OUTPUT_DIR,
        FIGURE_OUTPUT_DIR,
        VALIDATION_OUTPUT_DIR,
        LME_OUTPUT_DIR,
        BEHAVIOR_OUTPUT_DIR,
        WD_RECOVERED_OUTPUT_DIR,
        DEVICE_REPEATABILITY_OUTPUT_DIR,
        SUBSTANCE_USE_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
