"""Test 90 acoustic Age x Task effects after controlling Stroop performance.

For each feature, the original CW-versus-W model is extended only by the
two conventional task-level Stroop measures:

    Feature_Z ~ Age_Z * Task_Code + Accuracy_Z + Completion_Time_Z
                + (1 + Task_Code | Subject_ID)

Accuracy and card completion time follow the definitions used for the
traditional Stroop benchmark. All 158 healthy participants remain eligible
for the mixed-effects analysis.
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
DEFAULT_FEATURES_PATH = (
    REPO_ROOT / "data" / "public" / "features" / "Automated_feature.csv"
)
DEFAULT_PARTICIPANT_INFO_PATH = (
    REPO_ROOT / "data" / "public" / "participant_info_HC_158.xlsx"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "traditional_score_adjusted_lme"
DEFAULT_ORIGINAL_SUMMARY_PATH = (
    REPO_ROOT
    / "results"
    / "04_wd_external_validation"
    / "LME_deviation_scores_CW_vs_W_WD_CI_FDR01"
    / "healthy_LME_model_summary_CW_vs_W_WD_CI_FDR01.csv"
)

TASKS = ("W", "CW")
TASK_CODE = {"W": 0, "CW": 1}
EXPECTED_ITEMS = 12
AGE_TASK_TERM = "Age_Z:Task_Code"
ADJUSTED_FORMULA = (
    "Feature_Z ~ Age_Z * Task_Code + Accuracy_Z + Completion_Time_Z"
)
FDR_ALPHA_STRICT = 0.01
FDR_ALPHA_CONVENTIONAL = 0.05
MIN_SUBJECTS = 20

TASK_TARGETS = {
    "W": [
        "黄", "红", "绿", "蓝", "绿", "黄",
        "蓝", "红", "蓝", "绿", "红", "黄",
    ],
    "CW": [
        "黄", "红", "绿", "蓝", "绿", "蓝",
        "黄", "红", "红", "蓝", "绿", "黄",
    ],
}
COLOR_TO_CHINESE = {
    "yellow": "黄",
    "red": "红",
    "green": "绿",
    "blue": "蓝",
}


def normalize_subject_id(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return str(int(value)) if float(value).is_integer() else str(value).strip()
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_color(value: object) -> str:
    text = str(value).strip().lower()
    return COLOR_TO_CHINESE.get(text, text)


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

    if subject_col is None or age_col is None:
        if frame.shape[1] < 5:
            raise ValueError("Participant information workbook lacks ID and age columns.")
        selected = frame.iloc[:, [1, 4]].copy()
        selected.columns = ["Subject_ID", "Age"]
    else:
        selected = frame[[subject_col, age_col]].copy()
        selected.columns = ["Subject_ID", "Age"]

    selected["Subject_ID"] = selected["Subject_ID"].map(normalize_subject_id)
    selected["Age"] = pd.to_numeric(selected["Age"], errors="coerce")
    selected = selected.dropna(subset=["Subject_ID", "Age"]).drop_duplicates("Subject_ID")
    return selected


def zscore(values: pd.Series) -> tuple[pd.Series, float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    mean = float(numeric.mean())
    sd = float(numeric.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Cannot standardize a variable with zero or missing SD.")
    return (numeric - mean) / sd, mean, sd


def derive_task_metrics(features: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the accuracy and completion-time definitions used in T.3."""
    required = {"Subject_ID", "Task", "Item_Index", "Auto_Word", "Absolute_Word_End"}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Feature table is missing behavioral columns: {missing}")

    rows: list[dict[str, object]] = []
    for (subject_id, task), subset in features.groupby(["Subject_ID", "Task"], sort=True):
        if task not in TASKS:
            continue
        subset = subset.sort_values("Item_Index").head(EXPECTED_ITEMS).copy()
        words = [normalize_color(word) for word in subset["Auto_Word"].fillna("").astype(str)]
        correct = sum(
            word == TASK_TARGETS[task][index]
            for index, word in enumerate(words)
        )
        ends = pd.to_numeric(subset["Absolute_Word_End"], errors="coerce")
        completion_time = (
            float(ends.iloc[EXPECTED_ITEMS - 1])
            if len(subset) == EXPECTED_ITEMS
            else np.nan
        )
        rows.append(
            {
                "Subject_ID": subject_id,
                "Task": task,
                "Valid_Item_Count": len(subset),
                "Correct_Item_Count": correct,
                "Overall_Accuracy": correct / EXPECTED_ITEMS,
                "Card_Completion_Time": completion_time,
            }
        )
    return pd.DataFrame(rows)


def prepare_data(
    features_path: Path,
    participant_info_path: Path,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    features = pd.read_csv(features_path, encoding="utf-8-sig", dtype={"Subject_ID": str})
    required = {"Subject_ID", "Task", *feature_names}
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Feature table is missing required columns: {missing}")

    features["Subject_ID"] = features["Subject_ID"].map(normalize_subject_id)
    features["Task"] = features["Task"].astype(str).str.strip()
    features = features[features["Task"].isin(TASKS)].copy()

    info = load_participant_info(participant_info_path)
    if info["Subject_ID"].nunique() < 158:
        raise ValueError(f"Expected at least 158 participant records, found {len(info)}.")
    subject_info = info[info["Subject_ID"].isin(features["Subject_ID"].unique())].copy()
    if len(subject_info) != 158:
        raise ValueError(f"Expected 158 matched healthy participants, found {len(subject_info)}.")
    subject_info["Age_Z"], age_mean, age_sd = zscore(subject_info["Age"])

    task_metrics = derive_task_metrics(features)
    if task_metrics["Subject_ID"].nunique() != 158 or len(task_metrics) != 316:
        raise ValueError(
            "Expected one W and one CW task-level record for each of 158 participants."
        )

    task_metrics["Completion_Time_Was_Missing"] = task_metrics[
        "Card_Completion_Time"
    ].isna()
    task_medians = task_metrics.groupby("Task")["Card_Completion_Time"].transform("median")
    task_metrics["Card_Completion_Time_Imputed"] = task_metrics[
        "Card_Completion_Time"
    ].fillna(task_medians)
    if task_metrics["Card_Completion_Time_Imputed"].isna().any():
        raise ValueError("Task-specific completion-time median imputation failed.")

    task_metrics["Accuracy_Z"] = task_metrics.groupby("Task")["Overall_Accuracy"].transform(
        lambda values: (values - values.mean()) / values.std(ddof=0)
    )
    task_metrics["Completion_Time_Z"] = task_metrics.groupby("Task")[
        "Card_Completion_Time_Imputed"
    ].transform(lambda values: (values - values.mean()) / values.std(ddof=0))

    data = features.merge(
        subject_info[["Subject_ID", "Age_Z"]],
        on="Subject_ID",
        how="inner",
        validate="many_to_one",
    )
    data = data.merge(
        task_metrics[
            [
                "Subject_ID",
                "Task",
                "Accuracy_Z",
                "Completion_Time_Z",
            ]
        ],
        on=["Subject_ID", "Task"],
        how="inner",
        validate="many_to_one",
    )
    data["Task_Code"] = data["Task"].map(TASK_CODE).astype(int)
    if data["Subject_ID"].nunique() != 158:
        raise ValueError("Not all 158 participants were retained in the item-level model data.")

    metadata: dict[str, object] = {
        "Healthy_Subjects": int(data["Subject_ID"].nunique()),
        "Item_Level_Observations": int(len(data)),
        "Age_Mean": age_mean,
        "Age_SD_DDof0": age_sd,
        "Completion_Time_Imputed_Subject_Task_Records": int(
            task_metrics["Completion_Time_Was_Missing"].sum()
        ),
    }
    for task in TASKS:
        subset = task_metrics[task_metrics["Task"].eq(task)]
        metadata[f"{task}_Accuracy_Mean"] = float(subset["Overall_Accuracy"].mean())
        metadata[f"{task}_Completion_Time_Mean_Observed"] = float(
            subset["Card_Completion_Time"].mean()
        )
    return data, task_metrics, metadata


def load_original_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    original = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "Feature",
        "Coef_Age_Z_x_Task_Code",
        "Interaction_PValue",
        "Interaction_FDR_PValue",
    }
    if not required.issubset(original.columns):
        raise ValueError(f"Original summary is missing: {sorted(required.difference(original.columns))}")
    if original["Feature"].duplicated().any():
        raise ValueError("Original summary contains duplicate feature names.")
    if len(original) != 90:
        raise ValueError(f"Expected 90 original feature results, found {len(original)}.")
    return original[
        ["Feature", "Coef_Age_Z_x_Task_Code", "Interaction_PValue", "Interaction_FDR_PValue"]
    ].rename(
        columns={
            "Coef_Age_Z_x_Task_Code": "Original_Age_Task_Beta",
            "Interaction_PValue": "Original_Age_Task_P_Raw",
            "Interaction_FDR_PValue": "Original_Age_Task_Q_FDR_BH_90",
        }
    )


def fit_adjusted_model(
    data: pd.DataFrame,
    feature: str,
) -> tuple[object | None, dict[str, object]]:
    columns = [
        "Subject_ID",
        "Task_Code",
        "Age_Z",
        "Accuracy_Z",
        "Completion_Time_Z",
        feature,
    ]
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
                    ADJUSTED_FORMULA,
                    model_data,
                    groups=model_data["Subject_ID"],
                    re_formula="~Task_Code",
                )
                result = model.fit(method=optimizer, reml=False, maxiter=2000, disp=False)
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


def run_analysis(data: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    feature_names = original["Feature"].tolist()
    for index, feature in enumerate(feature_names, start=1):
        print(f"[{index:02d}/{len(feature_names)}] {feature}")
        adjusted, fit_info = fit_adjusted_model(data, feature)
        row: dict[str, object] = {
            "Feature_Order": index,
            "Feature": feature,
            "Adjusted_Status": fit_info.get("Status"),
            "Adjusted_Optimizer": fit_info.get("Optimizer"),
            "Adjusted_Converged": fit_info.get("Converged", False),
            "N_Subjects": fit_info.get("N_Subjects"),
            "N_Observations": fit_info.get("N_Observations"),
            "Feature_Mean": fit_info.get("Feature_Mean"),
            "Feature_SD_DDof0": fit_info.get("Feature_SD_DDof0"),
            "Adjusted_Warnings": fit_info.get("Warnings", ""),
        }
        row.update(extract_term(adjusted, AGE_TASK_TERM, "Adjusted_Age_Task"))
        row.update(extract_term(adjusted, "Accuracy_Z", "Adjusted_Accuracy"))
        row.update(extract_term(adjusted, "Completion_Time_Z", "Adjusted_Completion_Time"))
        rows.append(row)

    results = pd.DataFrame(rows).merge(
        original,
        on="Feature",
        how="left",
        validate="one_to_one",
    )
    add_fdr(results, "Adjusted_Age_Task_P_Raw", "Adjusted_Age_Task_Q_FDR_BH_90")

    original_beta = results["Original_Age_Task_Beta"]
    adjusted_beta = results["Adjusted_Age_Task_Beta"]
    results["Age_Task_Same_Direction"] = np.sign(original_beta) == np.sign(adjusted_beta)
    results["Age_Task_Retained_Magnitude_Percent"] = (
        adjusted_beta.abs() / original_beta.abs() * 100.0
    )
    results["Age_Task_Signed_Attenuation_Percent"] = (
        1.0 - adjusted_beta.abs() / original_beta.abs()
    ) * 100.0
    results["Age_Task_Absolute_Percent_Change"] = (
        (adjusted_beta - original_beta).abs() / original_beta.abs() * 100.0
    )
    results["Incremental_Age_Task_Effect_Q_LE_0_01"] = results[
        "Adjusted_Age_Task_Q_FDR_BH_90"
    ].le(FDR_ALPHA_STRICT)
    results["Incremental_Age_Task_Effect_Q_LE_0_05"] = results[
        "Adjusted_Age_Task_Q_FDR_BH_90"
    ].le(FDR_ALPHA_CONVENTIONAL)
    return results.sort_values("Feature_Order").reset_index(drop=True)


def make_summary(results: pd.DataFrame, metadata: dict[str, object]) -> pd.DataFrame:
    valid = results.dropna(subset=["Original_Age_Task_Beta", "Adjusted_Age_Task_Beta"])
    pearson_r, pearson_p = stats.pearsonr(
        valid["Original_Age_Task_Beta"], valid["Adjusted_Age_Task_Beta"]
    )
    spearman_r, spearman_p = stats.spearmanr(
        valid["Original_Age_Task_Beta"], valid["Adjusted_Age_Task_Beta"]
    )
    row = {
        **metadata,
        "Features_Planned": len(results),
        "Adjusted_Models_Fitted": int(results["Adjusted_Age_Task_Beta"].notna().sum()),
        "Adjusted_Models_Converged": int(results["Adjusted_Converged"].fillna(False).sum()),
        "Age_Task_Same_Direction_N": int(results["Age_Task_Same_Direction"].fillna(False).sum()),
        "Age_Task_Same_Direction_Percent": float(results["Age_Task_Same_Direction"].mean() * 100),
        "Original_Age_Task_Q_LE_0_01_N": int(
            results["Original_Age_Task_Q_FDR_BH_90"].le(FDR_ALPHA_STRICT).sum()
        ),
        "Original_Age_Task_Q_LE_0_05_N": int(
            results["Original_Age_Task_Q_FDR_BH_90"].le(FDR_ALPHA_CONVENTIONAL).sum()
        ),
        "Adjusted_Age_Task_Q_LE_0_01_N": int(
            results["Incremental_Age_Task_Effect_Q_LE_0_01"].sum()
        ),
        "Adjusted_Age_Task_Q_LE_0_05_N": int(
            results["Incremental_Age_Task_Effect_Q_LE_0_05"].sum()
        ),
        "Age_Task_Beta_Pearson_R": pearson_r,
        "Age_Task_Beta_Pearson_P": pearson_p,
        "Age_Task_Beta_Spearman_Rho": spearman_r,
        "Age_Task_Beta_Spearman_P": spearman_p,
        "Median_Retained_Beta_Magnitude_Percent": float(
            results["Age_Task_Retained_Magnitude_Percent"].median()
        ),
        "Median_Signed_Attenuation_Percent": float(
            results["Age_Task_Signed_Attenuation_Percent"].median()
        ),
        "Median_Absolute_Percent_Beta_Change": float(
            results["Age_Task_Absolute_Percent_Change"].median()
        ),
    }
    return pd.DataFrame([row])


def write_notes(
    path: Path,
    features_path: Path,
    participant_info_path: Path,
    original_summary_path: Path,
) -> None:
    def public_path(value: Path) -> str:
        try:
            return value.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(value)

    text = f"""Traditional-score-adjusted 90-feature LME analysis

Purpose
Test whether the 90 item-level acoustic features retain Age x Task effects after controlling for conventional Stroop accuracy and card completion time.

Model
{ADJUSTED_FORMULA} + (1 + Task_Code | Subject_ID)

Definitions
1. W is coded 0 and CW is coded 1.
2. Accuracy is the number of correctly paired recognized color words divided by 12. Missing recognized items therefore count as incorrect.
3. Card completion time is the end time of the 12th paired color word relative to audio onset. It is missing when fewer than 12 paired items are available.
4. To retain all 158 participants, missing completion times are median-imputed separately within W and CW, matching the traditional-score benchmark.
5. Accuracy and imputed completion time are Z-standardized separately within W and CW before entry as covariates.
6. Age is Z-standardized across the 158 healthy participants. Each acoustic feature is Z-standardized across its included W and CW item-level observations.
7. Models use maximum-likelihood estimation with the original random intercept and random Task_Code slope. The random-effects structure is not simplified.
8. BH-FDR correction is applied across all 90 adjusted Age x Task tests.
9. An incremental acoustic effect is defined as an adjusted Age x Task effect that remains significant after the 90-test BH-FDR correction.

Inputs
Features: {public_path(features_path)}
Participant information: {public_path(participant_info_path)}
Original 90-feature summary: {public_path(original_summary_path)}
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
    original = load_original_results(original_summary_path)
    data, task_metrics, metadata = prepare_data(
        features_path,
        participant_info_path,
        original["Feature"].tolist(),
    )
    results = run_analysis(data, original)
    summary = make_summary(results, metadata)

    outputs = {
        "traditional_score_adjusted_90_feature_results.csv": results,
        "traditional_score_adjusted_90_feature_summary.csv": summary,
        "task_level_traditional_scores_qc.csv": task_metrics,
    }
    for filename, frame in outputs.items():
        destination = output_dir / filename
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        print(f"Saved: {destination}")

    notes_path = output_dir / "analysis_notes.txt"
    write_notes(notes_path, features_path, participant_info_path, original_summary_path)
    print(f"Saved: {notes_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
