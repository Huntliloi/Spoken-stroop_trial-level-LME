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
PUBLIC_WD_DIR = PUBLIC_DATA_DIR / "wd"
PUBLIC_WD_FEATURE_DIR = PUBLIC_WD_DIR / "features"
OUTPUT_DIR = REPO_ROOT / "outputs"

AUDIO_ROOT = EXAMPLE_DATA_DIR / "audio"
ASR_JSON_ROOT = EXAMPLE_DATA_DIR / "asr_json"
MANUAL_TEXTGRID_ROOT = EXAMPLE_DATA_DIR / "manual_textgrid"
EXAMPLE_PARTICIPANT_INFO_XLSX = EXAMPLE_DATA_DIR / "participant_info.xlsx"
PARTICIPANT_INFO_XLSX = PUBLIC_DATA_DIR / "participant_info.xlsx"

TOKEN_RESULTS_CSV = OUTPUT_DIR / "asr_qwen3" / "token_results.csv"
ACOUSTIC_FEATURES_CSV = OUTPUT_DIR / "Stroop_features_add.csv"
IMPUTED_ACOUSTIC_FEATURES_CSV = OUTPUT_DIR / "Stroop_features_imputed_add.csv"
EXAMPLE_IMPUTED_ACOUSTIC_FEATURES_CSV = EXAMPLE_FEATURE_DIR / "Stroop_features_imputed_example.csv"
PUBLIC_AUTOMATED_FEATURE_CSV = PUBLIC_FEATURE_DIR / "Automated_feature.csv"
PUBLIC_PRAAT_MANUAL_FEATURE_CSV = PUBLIC_FEATURE_DIR / "Praat_manual_feature.csv"
PUBLIC_WD_CI_FEATURE_CSV = PUBLIC_WD_FEATURE_DIR / "Stroop_features_WD_CI.csv"
PUBLIC_WD_CN_FEATURE_CSV = PUBLIC_WD_FEATURE_DIR / "Stroop_features_WD_CN.csv"
PUBLIC_WD_INFO_XLSX = PUBLIC_WD_DIR / "WD_participant_info.xlsx"
BEHAVIORAL_METRICS_CSV = OUTPUT_DIR / "Stroop_TaskLevel_NewFeatures.csv"

FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"
VALIDATION_OUTPUT_DIR = OUTPUT_DIR / "validation"
LME_OUTPUT_DIR = OUTPUT_DIR / "lme"
BEHAVIOR_OUTPUT_DIR = OUTPUT_DIR / "behavior"


def get_imputed_acoustic_features_csv() -> Path:
    """Return generated imputed features when present; otherwise use public data."""
    if IMPUTED_ACOUSTIC_FEATURES_CSV.exists():
        return IMPUTED_ACOUSTIC_FEATURES_CSV
    if PUBLIC_AUTOMATED_FEATURE_CSV.exists():
        return PUBLIC_AUTOMATED_FEATURE_CSV
    return EXAMPLE_IMPUTED_ACOUSTIC_FEATURES_CSV


def ensure_output_dirs() -> None:
    """Create the standard output folders used by the analysis scripts."""
    for path in [
        OUTPUT_DIR,
        FIGURE_OUTPUT_DIR,
        VALIDATION_OUTPUT_DIR,
        LME_OUTPUT_DIR,
        BEHAVIOR_OUTPUT_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
