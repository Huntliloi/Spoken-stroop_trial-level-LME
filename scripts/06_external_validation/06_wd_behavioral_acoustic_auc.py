"""Compare age-referenced behavioral and acoustic deviations in the WD cohort.

Accuracy and total hesitation are first converted to deviation Z scores using
age-only reference regressions fitted in the independent 158-person healthy
cohort. Clinical labels are never used to fit or scale the reference models.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_WD_DIR = REPO_ROOT / "data" / "public" / "wd"
PUBLIC_WD_FEATURE_DIR = PUBLIC_WD_DIR / "features"
WD_RESULT_DIR = REPO_ROOT / "results" / "04_wd_external_validation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "behavioral_acoustic_auc" / "wd"
DEFAULT_HEALTHY_BEHAVIOR_CSV = (
    REPO_ROOT / "results" / "00_behavioral_load_gradient" / "task_level_behavioral_metrics.csv"
)
DEFAULT_HEALTHY_INFO_XLSX = REPO_ROOT / "data" / "public" / "participant_info_HC_158.xlsx"

EXPECTED_ITEMS = 12
TASK_TARGETS = {
    "Hanzi": ["yellow", "red", "green", "blue", "green", "yellow", "blue", "red", "blue", "green", "red", "yellow"],
    "Dots": ["red", "yellow", "blue", "green", "green", "blue", "yellow", "red", "yellow", "red", "green", "blue"],
    "STT": ["yellow", "red", "green", "blue", "green", "blue", "yellow", "red", "red", "blue", "green", "yellow"],
}
COLOR_LABELS = {"\u9ec4": "yellow", "\u7ea2": "red", "\u7eff": "green", "\u84dd": "blue"}

ACOUSTIC_COLUMNS = [
    "Mean_Abs_Deviation_Z",
    "RMS_Deviation_Z",
    "N_Features_AbsZ_GE_1_96",
    "N_Features_AbsZ_GE_2_58",
]
TASK_LEVEL_CONVENTIONAL_COLUMNS = [
    "Overall_Accuracy_Hanzi",
    "Overall_Accuracy_Dots",
    "Overall_Accuracy_STT",
    "Total_Hesitation_Duration_Hanzi",
    "Total_Hesitation_Duration_Dots",
    "Total_Hesitation_Duration_STT",
]
BEHAVIOR_TASK_COMPOSITE_COLUMNS = [
    "Behavior_Task_Mean_Abs_Deviation_Z",
    "Behavior_Task_RMS_Deviation_Z",
    "Behavior_Task_N_Metrics_AbsZ_GE_1_96",
    "Behavior_Task_N_Metrics_AbsZ_GE_2_58",
]
BEHAVIOR_TASK_Z_COLUMNS = [f"Behavior_ZDev__{column}" for column in TASK_LEVEL_CONVENTIONAL_COLUMNS]
MEASURES = [
    ("Hanzi accuracy Z_dev", BEHAVIOR_TASK_Z_COLUMNS[0], "Behavioral task deviation", "Lower is worse"),
    ("Dots accuracy Z_dev", BEHAVIOR_TASK_Z_COLUMNS[1], "Behavioral task deviation", "Lower is worse"),
    ("STT accuracy Z_dev", BEHAVIOR_TASK_Z_COLUMNS[2], "Behavioral task deviation", "Lower is worse"),
    ("Hanzi hesitation Z_dev", BEHAVIOR_TASK_Z_COLUMNS[3], "Behavioral task deviation", "Higher is worse"),
    ("Dots hesitation Z_dev", BEHAVIOR_TASK_Z_COLUMNS[4], "Behavioral task deviation", "Higher is worse"),
    ("STT hesitation Z_dev", BEHAVIOR_TASK_Z_COLUMNS[5], "Behavioral task deviation", "Higher is worse"),
    ("Mean |Z_dev|", ACOUSTIC_COLUMNS[0], "Acoustic", "Higher is worse"),
    ("RMS Z_dev", ACOUSTIC_COLUMNS[1], "Acoustic", "Higher is worse"),
    ("N_abn,1.96", ACOUSTIC_COLUMNS[2], "Acoustic", "Higher is worse"),
    ("N_abn,2.58", ACOUSTIC_COLUMNS[3], "Acoustic", "Higher is worse"),
]
MODEL_SETS = {
    "Behavioral_Task_Z_Deviations": BEHAVIOR_TASK_Z_COLUMNS,
    "Acoustic_Deviation": ACOUSTIC_COLUMNS,
    "Combined_Task_Behavior_Acoustic": BEHAVIOR_TASK_Z_COLUMNS + ACOUSTIC_COLUMNS,
}


def normalize_subject_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"Subject_ID": str})
    if "Subject_ID" in frame:
        frame["Subject_ID"] = frame["Subject_ID"].map(normalize_subject_id)
    return frame


def derive_task_metrics(items: pd.DataFrame) -> pd.DataFrame:
    required = {"Subject_ID", "Task", "Item_Index", "Word", "Gap_Before", "Absolute_Word_End"}
    missing = required.difference(items.columns)
    if missing:
        raise ValueError(f"Item table is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for (subject_id, task), sub in items.groupby(["Subject_ID", "Task"], sort=True):
        if task not in TASK_TARGETS:
            continue
        sub = sub.sort_values("Item_Index").head(EXPECTED_ITEMS).copy()
        words = [COLOR_LABELS.get(word, word.lower()) for word in sub["Word"].fillna("").astype(str)]
        correct = sum(word == TASK_TARGETS[task][index] for index, word in enumerate(words))
        gaps = pd.to_numeric(sub["Gap_Before"], errors="coerce").fillna(0.0).clip(lower=0.0)
        response_end = pd.to_numeric(sub["Absolute_Word_End"], errors="coerce").max()
        rows.append(
            {
                "Subject_ID": subject_id,
                "Task": task,
                "Valid_Item_Count": len(sub),
                "Correct_Item_Count": correct,
                "Overall_Accuracy": correct / EXPECTED_ITEMS,
                "Total_Hesitation_Duration": float(gaps.sum()),
                "Overall_Response_Duration": response_end,
            }
        )
    return pd.DataFrame(rows)


def build_subject_behavior(task_metrics: pd.DataFrame) -> pd.DataFrame:
    values = [
        "Valid_Item_Count",
        "Correct_Item_Count",
        "Overall_Accuracy",
        "Total_Hesitation_Duration",
        "Overall_Response_Duration",
    ]
    wide = task_metrics.pivot(index="Subject_ID", columns="Task", values=values)
    wide.columns = [f"{measure}_{task}" for measure, task in wide.columns]
    wide = wide.reset_index()
    required = TASK_LEVEL_CONVENTIONAL_COLUMNS
    missing = [column for column in required if column not in wide]
    if missing:
        raise ValueError(f"Cannot construct task-level behavioral measures; missing columns: {missing}")
    for task in ["Hanzi", "Dots", "STT"]:
        wide[f"Complete_12_Items_{task}"] = wide[f"Valid_Item_Count_{task}"].eq(EXPECTED_ITEMS)
    wide["QC_All_Tasks_Complete_12"] = wide[
        ["Complete_12_Items_Hanzi", "Complete_12_Items_Dots", "Complete_12_Items_STT"]
    ].all(axis=1)
    return wide


def load_healthy_reference(behavior_csv: Path, info_xlsx: Path) -> pd.DataFrame:
    task_metrics = read_csv(behavior_csv)
    task_metrics["Overall_Response_Duration"] = np.nan
    behavior = build_subject_behavior(task_metrics)
    info = pd.read_excel(info_xlsx, dtype={"Subject_ID": str})
    info["Subject_ID"] = info["Subject_ID"].map(normalize_subject_id)
    healthy = behavior.merge(info[["Subject_ID", "Age"]], on="Subject_ID", how="inner", validate="one_to_one")
    if len(healthy) != 158:
        raise ValueError(f"Expected 158 healthy reference participants, found {len(healthy)}.")
    return healthy


def fit_behavior_reference(healthy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in TASK_LEVEL_CONVENTIONAL_COLUMNS:
        values = pd.to_numeric(healthy[column], errors="coerce")
        ages = pd.to_numeric(healthy["Age"], errors="coerce")
        valid = values.notna() & ages.notna()
        y = values[valid].to_numpy(float)
        age = ages[valid].to_numpy(float)
        age_mean = float(age.mean())
        age_sd = float(age.std(ddof=1))
        age_z = (age - age_mean) / age_sd
        design = np.column_stack([np.ones(len(age_z)), age_z])
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        fitted = design @ beta
        residuals = y - fitted
        residual_sd = float(np.sqrt(np.sum(residuals**2) / (len(y) - design.shape[1])))
        if not np.isfinite(residual_sd) or residual_sd <= 0:
            raise ValueError(f"Invalid healthy residual SD for {column}: {residual_sd}")
        total_ss = float(np.sum((y - y.mean()) ** 2))
        rows.append(
            {
                "Metric": column,
                "Reference_N": len(y),
                "Reference_Age_Mean": age_mean,
                "Reference_Age_SD": age_sd,
                "Intercept": beta[0],
                "Age_Z_Coefficient": beta[1],
                "Residual_SD_Sqrt_SSE_Over_N_Minus_2": residual_sd,
                "R_Squared": 1.0 - float(np.sum(residuals**2)) / total_ss if total_ss > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def add_behavior_deviations(subject_data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    result = subject_data.copy()
    reference_by_metric = reference.set_index("Metric")
    for column in TASK_LEVEL_CONVENTIONAL_COLUMNS:
        model = reference_by_metric.loc[column]
        age_z = (pd.to_numeric(result["Age"], errors="coerce") - model["Reference_Age_Mean"]) / model[
            "Reference_Age_SD"
        ]
        expected = model["Intercept"] + model["Age_Z_Coefficient"] * age_z
        signed_z = (pd.to_numeric(result[column], errors="coerce") - expected) / model[
            "Residual_SD_Sqrt_SSE_Over_N_Minus_2"
        ]
        result[f"Behavior_Age_Expected__{column}"] = expected
        result[f"Behavior_ZDev__{column}"] = signed_z
        result[f"Behavior_AbsZDev__{column}"] = signed_z.abs()

    z_values = result[BEHAVIOR_TASK_Z_COLUMNS]
    result["Behavior_Task_Mean_Abs_Deviation_Z"] = z_values.abs().mean(axis=1)
    result["Behavior_Task_RMS_Deviation_Z"] = np.sqrt((z_values**2).mean(axis=1))
    result["Behavior_Task_N_Metrics_AbsZ_GE_1_96"] = z_values.abs().ge(1.96).sum(axis=1)
    result["Behavior_Task_N_Metrics_AbsZ_GE_2_58"] = z_values.abs().ge(2.58).sum(axis=1)
    return result


def load_wd_data(
    healthy_behavior_csv: Path, healthy_info_xlsx: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_frames = []
    score_frames = []
    specifications = [
        (
            "WD-CI",
            PUBLIC_WD_FEATURE_DIR / "Stroop_features_WD_CI.csv",
            WD_RESULT_DIR / "LME_deviation_scores_STT_vs_Hanzi_WD_CI_FDR01"
            / "WD_subject_level_deviation_index_STT_vs_Hanzi_WD_CI_FDR01.csv",
        ),
        (
            "WD-CN",
            PUBLIC_WD_FEATURE_DIR / "Stroop_features_WD_CN.csv",
            WD_RESULT_DIR / "LME_deviation_scores_STT_vs_Hanzi_WD_CN_FDR01"
            / "WD_subject_level_deviation_index_STT_vs_Hanzi_WD_CN_FDR01.csv",
        ),
    ]
    for group, item_path, score_path in specifications:
        item_frame = read_csv(item_path)
        item_frame["Clinical_Group"] = group
        item_frames.append(item_frame)
        score_frame = read_csv(score_path)
        score_frame["Clinical_Group"] = group
        score_frames.append(score_frame)

    items = pd.concat(item_frames, ignore_index=True)
    scores = pd.concat(score_frames, ignore_index=True)
    task_metrics = derive_task_metrics(items)
    behavior = build_subject_behavior(task_metrics)
    subject_data = behavior.merge(scores, on="Subject_ID", how="inner", validate="one_to_one")
    if len(subject_data) != 43 or subject_data["Clinical_Group"].value_counts().to_dict() != {"WD-CN": 31, "WD-CI": 12}:
        raise ValueError("Expected 43 WD participants (12 WD-CI and 31 WD-CN) after merging.")
    healthy = load_healthy_reference(healthy_behavior_csv, healthy_info_xlsx)
    reference = fit_behavior_reference(healthy)
    subject_data = add_behavior_deviations(subject_data, reference)
    return task_metrics, subject_data, reference


def auc_components(labels: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) < 2 or len(negative) < 2:
        return np.nan, np.array([]), np.array([])
    comparisons = (positive[:, None] > negative[None, :]).astype(float)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    return float(comparisons.mean()), comparisons.mean(axis=1), comparisons.mean(axis=0)


def delong_auc_ci(labels: np.ndarray, scores: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    auc, v10, v01 = auc_components(labels, scores)
    if not np.isfinite(auc):
        return np.nan, np.nan, np.nan
    variance = np.var(v10, ddof=1) / len(v10) + np.var(v01, ddof=1) / len(v01)
    standard_error = np.sqrt(max(float(variance), 0.0))
    z_value = stats.norm.ppf(0.5 + confidence / 2.0)
    return auc, max(0.0, auc - z_value * standard_error), min(1.0, auc + z_value * standard_error)


def adjust_fdr_by_family(frame: pd.DataFrame, p_column: str = "P_Raw") -> pd.DataFrame:
    frame = frame.copy()
    frame["P_FDR_BH"] = np.nan
    for _, index in frame.groupby(["Analysis_Set", "FDR_Family"]).groups.items():
        valid = frame.loc[index, p_column].notna()
        valid_index = frame.loc[index].index[valid]
        if len(valid_index):
            frame.loc[valid_index, "P_FDR_BH"] = multipletests(
                frame.loc[valid_index, p_column].to_numpy(float), method="fdr_bh"
            )[1]
    return frame


def univariate_group_comparison(subject_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis_set, subset in [
        ("Full sample", subject_data),
        ("Complete 12 items in all tasks", subject_data[subject_data["QC_All_Tasks_Complete_12"]].copy()),
    ]:
        labels_all = subset["Clinical_Group"].eq("WD-CI").astype(int)
        for measure, column, family, direction in MEASURES:
            values = pd.to_numeric(subset[column], errors="coerce")
            valid = values.notna() & labels_all.notna()
            labels = labels_all[valid].to_numpy(int)
            scores = values[valid].to_numpy(float)
            positive = scores[labels == 1]
            negative = scores[labels == 0]
            test = stats.mannwhitneyu(positive, negative, alternative="two-sided", method="asymptotic")
            risk_scores = -scores if direction == "Lower is worse" else scores
            auc, ci_low, ci_high = delong_auc_ci(labels, risk_scores)
            rows.append(
                {
                    "Analysis_Set": analysis_set,
                    "Measure": measure,
                    "Column": column,
                    "FDR_Family": family,
                    "Direction": direction,
                    "WD_CI_N": len(positive),
                    "WD_CI_Mean": np.mean(positive),
                    "WD_CI_SD": np.std(positive, ddof=1),
                    "WD_CN_N": len(negative),
                    "WD_CN_Mean": np.mean(negative),
                    "WD_CN_SD": np.std(negative, ddof=1),
                    "Mann_Whitney_U": test.statistic,
                    "P_Raw": test.pvalue,
                    "AUC": auc,
                    "AUC_CI_Low_DeLong": ci_low,
                    "AUC_CI_High_DeLong": ci_high,
                }
            )
    return adjust_fdr_by_family(pd.DataFrame(rows))


def loocv_classification(subject_data: pd.DataFrame) -> pd.DataFrame:
    model_rows = []
    for analysis_set, subset in [
        ("Full sample", subject_data),
        ("Complete 12 items in all tasks", subject_data[subject_data["QC_All_Tasks_Complete_12"]].copy()),
    ]:
        labels = subset["Clinical_Group"].eq("WD-CI").astype(int).to_numpy()
        for model_name, columns in MODEL_SETS.items():
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=2000, random_state=2026),
            )
            predicted = cross_val_predict(model, subset[columns], labels, cv=LeaveOneOut(), method="predict_proba")[:, 1]
            auc, ci_low, ci_high = delong_auc_ci(labels, predicted)
            model_rows.append(
                {
                    "Outcome": f"WD-CI vs WD-CN: {analysis_set}",
                    "Model": model_name,
                    "Predictors": "; ".join(columns),
                    "N": len(labels),
                    "N_Positive": int(labels.sum()),
                    "N_Negative": int((labels == 0).sum()),
                    "Validation": "Leave-one-subject-out cross-validation",
                    "AUC": auc,
                    "AUC_CI_Low_DeLong": ci_low,
                    "AUC_CI_High_DeLong": ci_high,
                }
            )
    return pd.DataFrame(model_rows)


def write_notes(path: Path) -> None:
    text = """Analysis definitions

1. Overall accuracy is correct recognized color words divided by 12; missing recognized items therefore count as incorrect. Accuracy and hesitation are calculated separately for Hanzi, Dots, and STT.
2. Total hesitation duration is the sum of nonnegative Gap_Before values, including the interval before item 1.
3. Every behavioral metric is age-referenced before clinical analysis. For each metric, an age-only OLS model is fitted in the independent 158-person healthy cohort. Z_dev = (observed - healthy age-expected value) / sqrt(SSE/(n-2)). Clinical labels are not used to fit or scale these reference models.
4. Age is the only reference covariate because the article's acoustic healthy-reference model is also age-adjusted.
5. The behavioral model uses six signed, age-referenced Z deviations: accuracy and total hesitation for Hanzi, Dots, and STT. The acoustic model uses four prespecified deviation indices. The combined model includes both sets.
6. Four absolute behavioral-deviation summaries are retained in the subject-level output for descriptive use, but they do not replace the directionally informative behavioral Z deviations in the model.
7. BH correction is applied separately within each analysis set across the six behavioral tests and the four acoustic tests.
8. Univariate WD-CI versus WD-CN tests use two-sided asymptotic Mann-Whitney U tests. AUC confidence intervals use DeLong's method; no formal tests compare AUCs between models.
9. Model AUCs use leave-one-subject-out predictions from L2-penalized logistic regression. The combined model evaluates whether acoustic deviations add useful information to conventional behavioral deviations; it is not an independently validated diagnostic model.
10. Full-sample analyses count missing recognized items as errors. Complete-case sensitivity analyses include only participants with 12 recognized items in all three tasks; ASR substitutions can still be indistinguishable from true response errors without manual annotation.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--healthy-behavior-csv", type=Path, default=DEFAULT_HEALTHY_BEHAVIOR_CSV)
    parser.add_argument("--healthy-info-xlsx", type=Path, default=DEFAULT_HEALTHY_INFO_XLSX)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, subject_data, reference = load_wd_data(
            args.healthy_behavior_csv.resolve(), args.healthy_info_xlsx.resolve()
        )
        univariate = univariate_group_comparison(subject_data)
        models = loocv_classification(subject_data)

    outputs = {
        "healthy_behavior_age_reference_models.csv": reference,
        "subject_level_behavioral_and_acoustic_deviations.csv": subject_data,
        "univariate_WD_CI_vs_WD_CN.csv": univariate,
        "loocv_model_auc.csv": models,
    }
    for filename, frame in outputs.items():
        destination = output_dir / filename
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        print(f"Saved: {destination}")
    write_notes(output_dir / "analysis_notes.txt")
    print(f"Saved: {output_dir / 'analysis_notes.txt'}")


if __name__ == "__main__":
    main()
