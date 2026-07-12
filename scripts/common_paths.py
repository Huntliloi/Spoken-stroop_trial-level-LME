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
OUTPUT_DIR = REPO_ROOT / "outputs"

AUDIO_ROOT = EXAMPLE_DATA_DIR / "audio"
ASR_JSON_ROOT = EXAMPLE_DATA_DIR / "asr_json"
MANUAL_TEXTGRID_ROOT = EXAMPLE_DATA_DIR / "manual_textgrid"
PARTICIPANT_INFO_XLSX = EXAMPLE_DATA_DIR / "participant_info.xlsx"

TOKEN_RESULTS_CSV = OUTPUT_DIR / "asr_qwen3" / "token_results.csv"
ACOUSTIC_FEATURES_CSV = OUTPUT_DIR / "Stroop_features_add.csv"
IMPUTED_ACOUSTIC_FEATURES_CSV = OUTPUT_DIR / "Stroop_features_imputed_add.csv"
EXAMPLE_IMPUTED_ACOUSTIC_FEATURES_CSV = EXAMPLE_FEATURE_DIR / "Stroop_features_imputed_example.csv"
BEHAVIORAL_METRICS_CSV = OUTPUT_DIR / "Stroop_TaskLevel_NewFeatures.csv"

FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"
VALIDATION_OUTPUT_DIR = OUTPUT_DIR / "validation"
LME_OUTPUT_DIR = OUTPUT_DIR / "lme"
BEHAVIOR_OUTPUT_DIR = OUTPUT_DIR / "behavior"


def get_imputed_acoustic_features_csv() -> Path:
    """Return generated imputed features when present; otherwise use the public example file."""
    if IMPUTED_ACOUSTIC_FEATURES_CSV.exists():
        return IMPUTED_ACOUSTIC_FEATURES_CSV
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
