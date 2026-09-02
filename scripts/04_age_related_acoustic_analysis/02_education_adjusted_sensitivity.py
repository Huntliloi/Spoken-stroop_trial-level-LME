"""Targeted education-adjusted sensitivity analysis for the 25 core features.

The analysis uses only the 158-person healthy reference cohort and the
CW-versus-W contrast. It compares the original model

    Feature_Z ~ Age_Z * Task_Code + (1 + Task_Code | Subject_ID)

with the education-adjusted model

    Feature_Z ~ Age_Z * Task_Code + Education_Z * Task_Code
                + (1 + Task_Code | Subject_ID)

The 25 features are fixed in advance from the clinical validation analysis.
They are not reselected using the education-adjusted results.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = REPO_ROOT / "data" / "public" / "features" / "Automated_feature.csv"
DEFAULT_PARTICIPANT_INFO_PATH = REPO_ROOT / "data" / "public" / "participant_info_HC_158.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "education_adjusted_lme_sensitivity"
DEFAULT_ORIGINAL_SUMMARY_PATH = (
    REPO_ROOT
    / "results"
    / "04_wd_external_validation"
    / "LME_deviation_scores_CW_vs_W_WD_CI_FDR01"
    / "healthy_LME_model_summary_CW_vs_W_WD_CI_FDR01.csv"
)

TASKS = ("W", "CW")
TASK_CODE = {"W": 0, "CW": 1}
AGE_TASK_TERM = "Age_Z:Task_Code"
EDUCATION_TASK_TERM = "Education_Z:Task_Code"
FDR_ALPHA_STRICT = 0.01
FDR_ALPHA_CONVENTIONAL = 0.05
MIN_SUBJECTS = 20

CORE_FEATURES = [
    "Gap_Before",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
    "loudness_sma3_stddevNorm",
    "loudness_sma3_percentile20.0",
    "loudness_sma3_pctlrange0-2",
    "spectralFlux_sma3_stddevNorm",
    "mfcc1_sma3_amean",
    "mfcc1_sma3_stddevNorm",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "HNRdBACF_sma3nz_amean",
    "F1amplitudeLogRelF0_sma3nz_amean",
    "F3amplitudeLogRelF0_sma3nz_amean",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm",
    "hammarbergIndexV_sma3nz_stddevNorm",
    "spectralFluxV_sma3nz_stddevNorm",
    "mfcc1V_sma3nz_amean",
    "mfcc1V_sma3nz_stddevNorm",
    "alphaRatioUV_sma3nz_amean",
    "hammarbergIndexUV_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
    "spectralFluxUV_sma3nz_amean",
    "StddevVoicedSegmentLengthSec",
]

ADJUSTED_FORMULA = "Feature_Z ~ Age_Z * Task_Code + Education_Z * Task_Code"


def normalize_subject_id(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return str(int(value)) if float(value).is_integer() else str(value).strip()
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def resolve_participant_info_path(explicit_path: Path | None) -> Path:
    path = explicit_path or DEFAULT_PARTICIPANT_INFO_PATH
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_participant_info(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    subject_col = find_column(frame, ["Subject_ID", "Subject", "ID"])
    age_col = find_column(frame, ["Age"])
    education_col = find_column(frame, ["Education_Years", "Education"])

    if subject_col is None or age_col is None or education_col is None:
        raise ValueError(
            "Participant information must contain Subject_ID, Age, and Education_Years."
        )
    selected = frame[[subject_col, age_col, education_col]].copy()
    selected.columns = ["Subject_ID", "Age", "Education_Years"]

    selected["Subject_ID"] = selected["Subject_ID"].map(normalize_subject_id)
    selected["Age"] = pd.to_numeric(selected["Age"], errors="coerce")
    selected["Education_Years"] = pd.to_numeric(selected["Education_Years"], errors="coerce")
    selected = selected.dropna(subset=["Subject_ID", "Age", "Education_Years"])
    selected = selected.drop_duplicates("Subject_ID")
    return selected


def zscore_subject_covariate(values: pd.Series) -> tuple[pd.Series, float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Covariate SD is zero or missing.")
    return (numeric - mean) / sd, mean, sd


def prepare_data(features_path: Path, participant_info_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    features = pd.read_csv(features_path, encoding="utf-8-sig", dtype={"Subject_ID": str})
    required = {"Subject_ID", "Task", *CORE_FEATURES}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    features["Subject_ID"] = features["Subject_ID"].map(normalize_subject_id)
    features["Task"] = features["Task"].astype(str).str.strip()
    features = features[features["Task"].isin(TASKS)].copy()

    info = load_participant_info(participant_info_path)
    data = features.merge(info, on="Subject_ID", how="inner", validate="many_to_one")
    if data["Subject_ID"].nunique() != 158:
        raise ValueError(
            f"Expected 158 healthy participants after merging, found {data['Subject_ID'].nunique()}."
        )
    if data[["Age", "Education_Years"]].isna().any().any():
        raise ValueError("Age or education is missing after merging.")

    subject_info = data[["Subject_ID", "Age", "Education_Years"]].drop_duplicates("Subject_ID")
    age_z, age_mean, age_sd = zscore_subject_covariate(subject_info["Age"])
    education_z, education_mean, education_sd = zscore_subject_covariate(subject_info["Education_Years"])
    subject_info["Age_Z"] = age_z
    subject_info["Education_Z"] = education_z

    data = data.drop(columns=["Age", "Education_Years"]).merge(
        subject_info, on="Subject_ID", how="left", validate="many_to_one"
    )
    data["Task_Code"] = data["Task"].map(TASK_CODE).astype(int)

    correlation = float(subject_info[["Age", "Education_Years"]].corr().iloc[0, 1])
    vif = float(1.0 / (1.0 - correlation**2))
    metadata = {
        "Reference_N": int(len(subject_info)),
        "Age_Mean": age_mean,
        "Age_SD_DDof0": age_sd,
        "Education_Mean": education_mean,
        "Education_SD_DDof0": education_sd,
        "Age_Education_Pearson_R": correlation,
        "Age_Education_VIF": vif,
    }
    return data, metadata


def fit_mixed_model(data: pd.DataFrame, feature: str, formula: str) -> tuple[object | None, dict[str, object]]:
    columns = ["Subject_ID", "Task_Code", "Age_Z", "Education_Z", feature]
    model_data = data[columns].copy()
    model_data[feature] = pd.to_numeric(model_data[feature], errors="coerce")
    model_data = model_data.dropna()

    n_subjects = int(model_data["Subject_ID"].nunique())
    if n_subjects < MIN_SUBJECTS or model_data["Task_Code"].nunique() != 2:
        return None, {
            "Status": "insufficient_data",
            "N_Subjects": n_subjects,
            "N_Observations": len(model_data),
        }

    feature_mean = float(model_data[feature].mean())
    feature_sd = float(model_data[feature].std(ddof=0))
    if not np.isfinite(feature_sd) or feature_sd <= 0:
        return None, {
            "Status": "zero_feature_sd",
            "N_Subjects": n_subjects,
            "N_Observations": len(model_data),
        }
    model_data["Feature_Z"] = (model_data[feature] - feature_mean) / feature_sd

    messages: list[str] = []
    fallback_result = None
    fallback_optimizer = None
    for optimizer in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    formula,
                    model_data,
                    groups=model_data["Subject_ID"],
                    re_formula="~Task_Code",
                )
                result = model.fit(
                    method=optimizer,
                    reml=False,
                    maxiter=2000,
                    disp=False,
                )
            messages.extend(str(item.message) for item in caught)
            fallback_result = result
            fallback_optimizer = optimizer
            if bool(getattr(result, "converged", False)):
                return result, {
                    "Status": "ok",
                    "Optimizer": optimizer,
                    "Converged": True,
                    "N_Subjects": n_subjects,
                    "N_Observations": len(model_data),
                    "Feature_Mean": feature_mean,
                    "Feature_SD_DDof0": feature_sd,
                    "Warnings": " | ".join(dict.fromkeys(messages)),
                }
        except Exception as error:
            messages.append(f"{optimizer}: {error}")

    if fallback_result is not None:
        return fallback_result, {
            "Status": "not_converged",
            "Optimizer": fallback_optimizer,
            "Converged": False,
            "N_Subjects": n_subjects,
            "N_Observations": len(model_data),
            "Feature_Mean": feature_mean,
            "Feature_SD_DDof0": feature_sd,
            "Warnings": " | ".join(dict.fromkeys(messages)),
        }
    return None, {
        "Status": "fit_failed",
        "Converged": False,
        "N_Subjects": n_subjects,
        "N_Observations": len(model_data),
        "Warnings": " | ".join(dict.fromkeys(messages)),
    }


def extract_term(result: object | None, term: str, prefix: str) -> dict[str, float]:
    if result is None or term not in result.params.index:
        return {
            f"{prefix}_Beta": np.nan,
            f"{prefix}_SE": np.nan,
            f"{prefix}_CI_Low": np.nan,
            f"{prefix}_CI_High": np.nan,
            f"{prefix}_Z": np.nan,
            f"{prefix}_P_Raw": np.nan,
        }
    beta = float(result.params[term])
    se = float(result.bse[term])
    return {
        f"{prefix}_Beta": beta,
        f"{prefix}_SE": se,
        f"{prefix}_CI_Low": beta - 1.96 * se,
        f"{prefix}_CI_High": beta + 1.96 * se,
        f"{prefix}_Z": float(result.tvalues[term]),
        f"{prefix}_P_Raw": float(result.pvalues[term]),
    }


def add_fdr(frame: pd.DataFrame, raw_column: str, output_column: str) -> None:
    frame[output_column] = np.nan
    valid = frame[raw_column].notna()
    if valid.any():
        frame.loc[valid, output_column] = multipletests(
            frame.loc[valid, raw_column].to_numpy(float), method="fdr_bh"
        )[1]


def load_original_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame({"Feature": CORE_FEATURES})
    original = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "Feature",
        "Coef_Age_Z_x_Task_Code",
        "Interaction_PValue",
        "Interaction_FDR_PValue",
    }
    if not required.issubset(original.columns):
        raise ValueError(f"Original model summary is missing: {sorted(required.difference(original.columns))}")
    original = original[original["Feature"].isin(CORE_FEATURES)].copy()
    if set(original["Feature"]) != set(CORE_FEATURES):
        missing = sorted(set(CORE_FEATURES).difference(original["Feature"]))
        raise ValueError(f"Original model summary is missing core features: {missing}")
    return original[
        ["Feature", "Coef_Age_Z_x_Task_Code", "Interaction_PValue", "Interaction_FDR_PValue"]
    ].rename(
        columns={
            "Coef_Age_Z_x_Task_Code": "Original_90Test_Age_Task_Beta",
            "Interaction_PValue": "Original_90Test_Age_Task_P_Raw",
            "Interaction_FDR_PValue": "Original_90Test_Age_Task_Q_FDR_BH",
        }
    )


def run_analysis(data: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, feature in enumerate(CORE_FEATURES, start=1):
        print(f"[{index:02d}/{len(CORE_FEATURES)}] {feature}")
        adjusted, adjusted_info = fit_mixed_model(data, feature, ADJUSTED_FORMULA)
        row: dict[str, object] = {
            "Feature_Order": index,
            "Feature": feature,
            "Adjusted_Status": adjusted_info.get("Status"),
            "Adjusted_Optimizer": adjusted_info.get("Optimizer"),
            "Adjusted_Converged": adjusted_info.get("Converged", False),
            "N_Subjects": adjusted_info.get("N_Subjects"),
            "N_Observations": adjusted_info.get("N_Observations"),
            "Adjusted_Warnings": adjusted_info.get("Warnings", ""),
        }
        row.update(extract_term(adjusted, AGE_TASK_TERM, "Adjusted_Age_Task"))
        row.update(extract_term(adjusted, "Education_Z", "Adjusted_Education_Main"))
        row.update(extract_term(adjusted, EDUCATION_TASK_TERM, "Adjusted_Education_Task"))
        rows.append(row)

    results = pd.DataFrame(rows)
    add_fdr(results, "Adjusted_Age_Task_P_Raw", "Adjusted_Age_Task_Q_FDR_BH_25")
    add_fdr(results, "Adjusted_Education_Task_P_Raw", "Adjusted_Education_Task_Q_FDR_BH_25")
    results = results.merge(original, on="Feature", how="left", validate="one_to_one")

    reference_beta = results["Original_90Test_Age_Task_Beta"]
    adjusted_beta = results["Adjusted_Age_Task_Beta"]
    results["Age_Task_Same_Direction"] = np.sign(reference_beta) == np.sign(adjusted_beta)
    results["Age_Task_Absolute_Percent_Change"] = (
        (adjusted_beta - reference_beta).abs() / reference_beta.abs() * 100.0
    )
    results["Age_Task_Retained_Q_LE_0_01"] = results["Adjusted_Age_Task_Q_FDR_BH_25"].le(
        FDR_ALPHA_STRICT
    )
    results["Age_Task_Retained_Q_LE_0_05"] = results["Adjusted_Age_Task_Q_FDR_BH_25"].le(
        FDR_ALPHA_CONVENTIONAL
    )
    return results.sort_values("Feature_Order").reset_index(drop=True)


def make_summary(results: pd.DataFrame, metadata: dict[str, float]) -> pd.DataFrame:
    valid = results.dropna(subset=["Original_90Test_Age_Task_Beta", "Adjusted_Age_Task_Beta"])
    if len(valid) >= 3:
        pearson_r, pearson_p = stats.pearsonr(
            valid["Original_90Test_Age_Task_Beta"], valid["Adjusted_Age_Task_Beta"]
        )
        spearman_r, spearman_p = stats.spearmanr(
            valid["Original_90Test_Age_Task_Beta"], valid["Adjusted_Age_Task_Beta"]
        )
    else:
        pearson_r = pearson_p = spearman_r = spearman_p = np.nan

    row = {
        **metadata,
        "Core_Features_Planned": len(CORE_FEATURES),
        "Adjusted_Models_Fitted": int(results["Adjusted_Age_Task_Beta"].notna().sum()),
        "Adjusted_Models_Converged": int(results["Adjusted_Converged"].fillna(False).sum()),
        "Age_Task_Same_Direction_N": int(results["Age_Task_Same_Direction"].fillna(False).sum()),
        "Age_Task_Same_Direction_Percent": float(results["Age_Task_Same_Direction"].mean() * 100.0),
        "Original_90Test_Age_Task_Q_LE_0_01_N": int(
            results["Original_90Test_Age_Task_Q_FDR_BH"].le(0.01).sum()
        ),
        "Adjusted_Age_Task_Raw_P_LE_0_01_N": int(results["Adjusted_Age_Task_P_Raw"].le(0.01).sum()),
        "Adjusted_Age_Task_Raw_P_LE_0_05_N": int(results["Adjusted_Age_Task_P_Raw"].le(0.05).sum()),
        "Age_Task_Retained_Q_LE_0_01_N": int(results["Age_Task_Retained_Q_LE_0_01"].sum()),
        "Age_Task_Retained_Q_LE_0_05_N": int(results["Age_Task_Retained_Q_LE_0_05"].sum()),
        "Age_Task_Beta_Pearson_R": pearson_r,
        "Age_Task_Beta_Pearson_P": pearson_p,
        "Age_Task_Beta_Spearman_Rho": spearman_r,
        "Age_Task_Beta_Spearman_P": spearman_p,
        "Median_Absolute_Percent_Beta_Change": float(
            results["Age_Task_Absolute_Percent_Change"].median()
        ),
        "Education_Task_Q_LE_0_05_N": int(
            results["Adjusted_Education_Task_Q_FDR_BH_25"].le(0.05).sum()
        ),
    }
    return pd.DataFrame([row])


def write_notes(path: Path, features_path: Path, info_path: Path, summary_path: Path) -> None:
    def display_path(input_path: Path) -> str:
        try:
            return input_path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return input_path.name

    text = f"""Education-adjusted LME sensitivity analysis

Purpose
Assess whether the 25 prespecified CW-versus-W Age x Task effects remain stable after controlling for education in the 158-person healthy cohort.

Models
Original model results are read from the existing 90-feature analysis.
Adjusted model: {ADJUSTED_FORMULA} + (1 + Task_Code | Subject_ID)

Definitions
1. W is coded 0 and CW is coded 1.
2. Age and education are Z-standardized across the 158 unique healthy participants using population SD (ddof=0).
3. Each acoustic feature is Z-standardized across the included CW and W item-level observations.
4. The adjusted models use maximum-likelihood estimation and the same random intercept plus random Task_Code slope as the original models. Optimizers may change, but the random-effects structure is never simplified.
5. The 25 features were fixed before this sensitivity analysis from the original 90-feature clinical-model set; no feature reselection is performed.
6. BH-FDR correction is applied across the 25 targeted Age x Task tests in the adjusted model. Education x Task tests form a separate 25-test family.
7. The original 90-test FDR values are retained from the source model summary for context.
8. This is a robustness analysis of the Age x Task estimates, not evidence that education has no cognitive or clinical importance.

Inputs
Features: {display_path(features_path)}
Participant information: {display_path(info_path)}
Original model summary: {display_path(summary_path)}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--participant-info", type=Path)
    parser.add_argument("--original-summary", type=Path, default=DEFAULT_ORIGINAL_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    features_path = args.features.resolve()
    participant_info_path = resolve_participant_info_path(args.participant_info)
    original_summary_path = args.original_summary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Features: {features_path}")
    print(f"Participant information: {participant_info_path}")
    data, metadata = prepare_data(features_path, participant_info_path)
    original = load_original_results(original_summary_path)
    results = run_analysis(data, original)
    summary = make_summary(results, metadata)

    feature_output = output_dir / "education_adjusted_feature_results.csv"
    summary_output = output_dir / "education_adjusted_summary.csv"
    results.to_csv(feature_output, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_output, index=False, encoding="utf-8-sig")
    write_notes(output_dir / "analysis_notes.txt", features_path, participant_info_path, original_summary_path)

    print(f"Saved: {feature_output}")
    print(f"Saved: {summary_output}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
