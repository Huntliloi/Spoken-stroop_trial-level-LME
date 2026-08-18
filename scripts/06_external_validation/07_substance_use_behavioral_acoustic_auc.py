"""Compare traditional Stroop and acoustic deviations in substance use.

The traditional benchmark uses Hanzi and STT accuracy, card completion time,
and STT-vs-Hanzi interference costs. Each measure is converted to an
age-referenced deviation Z score using the independent 158-person healthy
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
PUBLIC_SUBSTANCE_DIR = REPO_ROOT / "data" / "public" / "substance_use"
PUBLIC_SUBSTANCE_FEATURE_DIR = PUBLIC_SUBSTANCE_DIR / "features"
SUBSTANCE_RESULT_DIR = REPO_ROOT / "results" / "05_substance_use_external_validation"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "behavioral_acoustic_auc" / "substance_use"
DEFAULT_HEALTHY_BEHAVIOR_CSV = (
    REPO_ROOT / "results" / "00_behavioral_load_gradient" / "task_level_behavioral_metrics.csv"
)
DEFAULT_HEALTHY_FEATURE_CSV = REPO_ROOT / "data" / "public" / "features" / "Automated_feature.csv"
DEFAULT_HEALTHY_INFO_XLSX = REPO_ROOT / "data" / "public" / "participant_info_HC_158.xlsx"

EXPECTED_ITEMS = 12
TASK_TARGETS = {
    "Hanzi": ["yellow", "red", "green", "blue", "green", "yellow", "blue", "red", "blue", "green", "red", "yellow"],
    "STT": ["yellow", "red", "green", "blue", "green", "blue", "yellow", "red", "red", "blue", "green", "yellow"],
}
COLOR_LABELS = {"\u9ec4": "yellow", "\u7ea2": "red", "\u7eff": "green", "\u84dd": "blue"}

ACOUSTIC_COLUMNS = [
    "Mean_Abs_Deviation_Z",
    "RMS_Deviation_Z",
    "N_Features_AbsZ_GE_1_96",
    "N_Features_AbsZ_GE_2_58",
]
TASK_LEVEL_TRADITIONAL_COLUMNS = [
    "Overall_Accuracy_Hanzi",
    "Card_Completion_Time_Hanzi",
    "Overall_Accuracy_STT",
    "Card_Completion_Time_STT",
]
INTERFERENCE_COLUMNS = [
    "Accuracy_Cost_STT_vs_Hanzi",
    "Completion_Time_Cost_STT_vs_Hanzi",
]
TRADITIONAL_COLUMNS = TASK_LEVEL_TRADITIONAL_COLUMNS + INTERFERENCE_COLUMNS
TRADITIONAL_Z_COLUMNS = [f"Behavior_ZDev__{column}" for column in TRADITIONAL_COLUMNS]
MEASURES = [
    ("Hanzi accuracy Z_dev", TRADITIONAL_Z_COLUMNS[0], "Traditional Stroop", "Lower is worse"),
    ("Hanzi completion-time Z_dev", TRADITIONAL_Z_COLUMNS[1], "Traditional Stroop", "Higher is worse"),
    ("STT accuracy Z_dev", TRADITIONAL_Z_COLUMNS[2], "Traditional Stroop", "Lower is worse"),
    ("STT completion-time Z_dev", TRADITIONAL_Z_COLUMNS[3], "Traditional Stroop", "Higher is worse"),
    ("Accuracy-interference Z_dev: STT vs Hanzi", TRADITIONAL_Z_COLUMNS[4], "Traditional Stroop", "Higher is worse"),
    ("Completion-time-interference Z_dev: STT vs Hanzi", TRADITIONAL_Z_COLUMNS[5], "Traditional Stroop", "Higher is worse"),
    ("Mean |Z_dev|", ACOUSTIC_COLUMNS[0], "Acoustic", "Higher is worse"),
    ("RMS Z_dev", ACOUSTIC_COLUMNS[1], "Acoustic", "Higher is worse"),
    ("N_abn,1.96", ACOUSTIC_COLUMNS[2], "Acoustic", "Higher is worse"),
    ("N_abn,2.58", ACOUSTIC_COLUMNS[3], "Acoustic", "Higher is worse"),
]
MODEL_SETS = {
    "Traditional_Stroop": TRADITIONAL_Z_COLUMNS,
    "Acoustic_Deviation": ACOUSTIC_COLUMNS,
    "Combined_Traditional_Acoustic": TRADITIONAL_Z_COLUMNS + ACOUSTIC_COLUMNS,
}


def normalize_subject_id(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"Subject_ID": str, "subject_id": str})
    if "Subject_ID" in frame:
        frame["Subject_ID"] = frame["Subject_ID"].map(normalize_subject_id)
    return frame


def derive_task_metrics(items: pd.DataFrame) -> pd.DataFrame:
    required = {"Subject_ID", "Task", "Item_Index", "Word", "Absolute_Word_End"}
    missing = required.difference(items.columns)
    if missing:
        raise ValueError(f"Item table is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for (subject_id, task), sub in items.groupby(["Subject_ID", "Task"], sort=True):
        if task not in {"Hanzi", "STT"}:
            continue
        sub = sub.sort_values("Item_Index").head(EXPECTED_ITEMS).copy()
        words = [COLOR_LABELS.get(word, word.lower()) for word in sub["Word"].fillna("").astype(str)]
        correct = sum(word == TASK_TARGETS[task][index] for index, word in enumerate(words))
        ends = pd.to_numeric(sub["Absolute_Word_End"], errors="coerce")
        completion_time = float(ends.iloc[EXPECTED_ITEMS - 1]) if len(sub) == EXPECTED_ITEMS else np.nan
        rows.append(
            {
                "Subject_ID": subject_id,
                "Task": task,
                "Valid_Item_Count": len(sub),
                "Correct_Item_Count": correct,
                "Overall_Accuracy": correct / EXPECTED_ITEMS,
                "Card_Completion_Time": completion_time,
            }
        )
    return pd.DataFrame(rows)


def derive_completion_times(items: pd.DataFrame) -> pd.DataFrame:
    required = {"Subject_ID", "Task", "Item_Index", "Absolute_Word_End"}
    missing = required.difference(items.columns)
    if missing:
        raise ValueError(f"Healthy item table is missing columns: {sorted(missing)}")

    rows = []
    for (subject_id, task), sub in items.groupby(["Subject_ID", "Task"], sort=True):
        if task not in {"Hanzi", "STT"}:
            continue
        sub = sub.sort_values("Item_Index").head(EXPECTED_ITEMS)
        ends = pd.to_numeric(sub["Absolute_Word_End"], errors="coerce")
        completion_time = float(ends.iloc[EXPECTED_ITEMS - 1]) if len(sub) == EXPECTED_ITEMS else np.nan
        rows.append(
            {
                "Subject_ID": subject_id,
                "Task": task,
                "Card_Completion_Time": completion_time,
            }
        )
    return pd.DataFrame(rows)


def build_subject_behavior(task_metrics: pd.DataFrame) -> pd.DataFrame:
    values = [
        "Valid_Item_Count",
        "Correct_Item_Count",
        "Overall_Accuracy",
        "Card_Completion_Time",
    ]
    wide = task_metrics.pivot(index="Subject_ID", columns="Task", values=values)
    wide.columns = [f"{measure}_{task}" for measure, task in wide.columns]
    wide = wide.reset_index()
    required = TASK_LEVEL_TRADITIONAL_COLUMNS
    missing = [column for column in required if column not in wide]
    if missing:
        raise ValueError(f"Cannot construct STT-Hanzi contrasts; missing columns: {missing}")
    wide["Accuracy_Cost_STT_vs_Hanzi"] = wide["Overall_Accuracy_Hanzi"] - wide["Overall_Accuracy_STT"]
    wide["Completion_Time_Cost_STT_vs_Hanzi"] = (
        wide["Card_Completion_Time_STT"] - wide["Card_Completion_Time_Hanzi"]
    )
    for task in ["Hanzi", "STT"]:
        wide[f"Complete_12_Items_{task}"] = wide[f"Valid_Item_Count_{task}"].eq(EXPECTED_ITEMS)
    wide["QC_Hanzi_STT_Complete_12"] = wide[
        ["Complete_12_Items_Hanzi", "Complete_12_Items_STT"]
    ].all(axis=1)
    return wide


def load_healthy_reference(behavior_csv: Path, feature_csv: Path, info_xlsx: Path) -> pd.DataFrame:
    task_metrics = read_csv(behavior_csv).rename(
        columns={"Valid_Trial_Count": "Valid_Item_Count", "Correct_Trial_Count": "Correct_Item_Count"}
    )
    task_metrics = task_metrics[task_metrics["Task"].isin(["Hanzi", "STT"])].copy()
    completion = derive_completion_times(read_csv(feature_csv))
    task_metrics = task_metrics.drop(columns=["Card_Completion_Time"], errors="ignore").merge(
        completion,
        on=["Subject_ID", "Task"],
        how="left",
        validate="one_to_one",
    )
    behavior = build_subject_behavior(task_metrics)
    info = pd.read_excel(info_xlsx, dtype={"Subject_ID": str})
    info["Subject_ID"] = info["Subject_ID"].map(normalize_subject_id)
    healthy = behavior.merge(info[["Subject_ID", "Age"]], on="Subject_ID", how="inner", validate="one_to_one")
    if len(healthy) != 158:
        raise ValueError(f"Expected 158 healthy reference participants, found {len(healthy)}.")
    return healthy


def fit_behavior_reference(healthy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in TRADITIONAL_COLUMNS:
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
    for column in TRADITIONAL_COLUMNS:
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

    z_values = result[TRADITIONAL_Z_COLUMNS]
    result["Traditional_Mean_Abs_Deviation_Z"] = z_values.abs().mean(axis=1)
    result["Traditional_RMS_Deviation_Z"] = np.sqrt((z_values**2).mean(axis=1))
    result["Traditional_N_Metrics_AbsZ_GE_1_96"] = z_values.abs().ge(1.96).sum(axis=1)
    result["Traditional_N_Metrics_AbsZ_GE_2_58"] = z_values.abs().ge(2.58).sum(axis=1)
    return result


def load_subject_data(
    healthy_behavior_csv: Path, healthy_feature_csv: Path, healthy_info_xlsx: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    item_frames = []
    for group, filename in [("Drug", "Stroop_features_Drug.csv"), ("HC", "Stroop_features_HC.csv")]:
        frame = read_csv(PUBLIC_SUBSTANCE_FEATURE_DIR / filename)
        frame["Group_From_Feature_File"] = group
        item_frames.append(frame)
    items = pd.concat(item_frames, ignore_index=True)
    task_metrics = derive_task_metrics(items)
    behavior = build_subject_behavior(task_metrics)

    score_path = SUBSTANCE_RESULT_DIR / "subject_level_deviation_indices.csv"
    scores = read_csv(score_path)
    if "Cognitive_Group" in scores and "Group" not in scores:
        scores = scores.rename(columns={"Cognitive_Group": "Group"})
    subject_data = behavior.merge(scores, on="Subject_ID", how="inner", validate="one_to_one")
    if len(subject_data) != 58 or subject_data["Group"].value_counts().to_dict() != {"HC": 30, "Drug": 28}:
        raise ValueError("Expected 58 participants (28 Drug and 30 HC) after merging.")
    education_bonus = pd.to_numeric(subject_data["Education"], errors="coerce").le(12).astype(int)
    subject_data["MoCA_Corrected"] = np.minimum(
        pd.to_numeric(subject_data["MoCA"], errors="coerce") + education_bonus,
        30,
    )
    subject_data["MoCA_Group"] = np.where(subject_data["MoCA_Corrected"].lt(26), "MoCA-CI", "MoCA-CN")
    healthy = load_healthy_reference(healthy_behavior_csv, healthy_feature_csv, healthy_info_xlsx)
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


def analysis_sets(subject_data: pd.DataFrame) -> list[tuple[str, pd.DataFrame, str, str, str]]:
    sets = []
    for sample_label, sample in [
        ("Full sample", subject_data),
        ("Complete 12 items in Hanzi and STT", subject_data[subject_data["QC_Hanzi_STT_Complete_12"]].copy()),
    ]:
        sets.extend(
            [
                (f"Drug vs HC: {sample_label}", sample, "Group", "Drug", "HC"),
                (f"MoCA-CI vs MoCA-CN: Overall: {sample_label}", sample, "MoCA_Group", "MoCA-CI", "MoCA-CN"),
                (
                    f"MoCA-CI vs MoCA-CN: Drug: {sample_label}",
                    sample[sample["Group"].eq("Drug")].copy(),
                    "MoCA_Group",
                    "MoCA-CI",
                    "MoCA-CN",
                ),
                (
                    f"MoCA-CI vs MoCA-CN: HC: {sample_label}",
                    sample[sample["Group"].eq("HC")].copy(),
                    "MoCA_Group",
                    "MoCA-CI",
                    "MoCA-CN",
                ),
            ]
        )
    return sets


def adjust_fdr(frame: pd.DataFrame, group_columns: list[str], p_column: str = "P_Raw") -> pd.DataFrame:
    frame = frame.copy()
    frame["P_FDR_BH"] = np.nan
    for _, index in frame.groupby(group_columns).groups.items():
        valid_index = frame.loc[index].index[frame.loc[index, p_column].notna()]
        if len(valid_index):
            frame.loc[valid_index, "P_FDR_BH"] = multipletests(
                frame.loc[valid_index, p_column].to_numpy(float), method="fdr_bh"
            )[1]
    return frame


def univariate_group_comparisons(subject_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis, subset, group_column, positive_label, negative_label in analysis_sets(subject_data):
        labels_all = subset[group_column].eq(positive_label).astype(int)
        for measure, column, family, direction in MEASURES:
            values = pd.to_numeric(subset[column], errors="coerce")
            valid = values.notna()
            labels = labels_all[valid].to_numpy(int)
            scores = values[valid].to_numpy(float)
            positive = scores[labels == 1]
            negative = scores[labels == 0]
            test = stats.mannwhitneyu(positive, negative, alternative="two-sided", method="asymptotic")
            risk_scores = -scores if direction == "Lower is worse" else scores
            auc, ci_low, ci_high = delong_auc_ci(labels, risk_scores)
            rows.append(
                {
                    "Analysis": analysis,
                    "Measure": measure,
                    "Column": column,
                    "FDR_Family": family,
                    "Direction": direction,
                    "Positive_Group": positive_label,
                    "Negative_Group": negative_label,
                    "Positive_N": len(positive),
                    "Positive_Mean": np.mean(positive),
                    "Positive_SD": np.std(positive, ddof=1),
                    "Negative_N": len(negative),
                    "Negative_Mean": np.mean(negative),
                    "Negative_SD": np.std(negative, ddof=1),
                    "Mann_Whitney_U": test.statistic,
                    "P_Raw": test.pvalue,
                    "AUC": auc,
                    "AUC_CI_Low_DeLong": ci_low,
                    "AUC_CI_High_DeLong": ci_high,
                }
            )
    return adjust_fdr(pd.DataFrame(rows), ["Analysis", "FDR_Family"])


def loocv_classification(subject_data: pd.DataFrame) -> pd.DataFrame:
    model_rows = []
    for analysis, subset, group_column, positive_label, negative_label in analysis_sets(subject_data):
        labels = subset[group_column].eq(positive_label).astype(int).to_numpy()
        for model_name, columns in MODEL_SETS.items():
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=2000, random_state=2026),
            )
            predicted = cross_val_predict(
                model,
                subset[columns],
                labels,
                cv=LeaveOneOut(),
                method="predict_proba",
            )[:, 1]
            auc, ci_low, ci_high = delong_auc_ci(labels, predicted)
            model_rows.append(
                {
                    "Outcome": analysis,
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

1. Overall accuracy is the number of correctly paired recognized color words divided by 12. Missing recognized items therefore count as incorrect. Accuracy is calculated separately for Hanzi and STT.
2. Card completion time is the end time of the 12th paired color word relative to audio onset. It is missing when fewer than 12 paired items are available.
3. STT-vs-Hanzi accuracy interference is Hanzi accuracy minus STT accuracy. STT-vs-Hanzi completion-time interference is STT completion time minus Hanzi completion time. Higher values indicate greater interference.
4. Each traditional measure is age-referenced using an age-only OLS model fitted in the independent 158-person healthy cohort. Z_dev = (observed - healthy age-expected value) / sqrt(SSE/(n-2)). Clinical labels are not used to fit or scale the reference models.
5. The traditional Stroop model uses six signed age-referenced Z deviations: Hanzi accuracy, Hanzi completion time, STT accuracy, STT completion time, and the two STT-vs-Hanzi interference costs.
6. Four absolute-deviation summaries of the six traditional measures are retained for descriptive use but are not model predictors.
7. BH correction is applied separately within each analysis set across the six traditional Stroop tests and the four acoustic tests.
8. Group tests use two-sided asymptotic Mann-Whitney U tests. MoCA-CI uses education-corrected MoCA: add one point for education <=12 years, cap at 30, and classify corrected scores <26 as MoCA-CI. Correlations use raw MoCA.
9. AUC confidence intervals use DeLong's method; no formal tests compare AUCs between models.
10. Model AUCs use leave-one-subject-out predictions from L2-penalized logistic regression. Median imputation and standardization are fitted within each training fold.
11. The combined model evaluates the complementary information in traditional Stroop and acoustic deviations; it is not an independently validated diagnostic model.
12. Full-sample analyses count missing recognized items as errors and impute missing completion-time predictors within each training fold. Complete-case analyses include only participants with 12 paired items in both Hanzi and STT.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--healthy-behavior-csv", type=Path, default=DEFAULT_HEALTHY_BEHAVIOR_CSV)
    parser.add_argument("--healthy-feature-csv", type=Path, default=DEFAULT_HEALTHY_FEATURE_CSV)
    parser.add_argument("--healthy-info-xlsx", type=Path, default=DEFAULT_HEALTHY_INFO_XLSX)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, subject_data, reference = load_subject_data(
            args.healthy_behavior_csv.resolve(),
            args.healthy_feature_csv.resolve(),
            args.healthy_info_xlsx.resolve(),
        )
        group_results = univariate_group_comparisons(subject_data)
        models = loocv_classification(subject_data)

    outputs = {
        "healthy_behavior_age_reference_models.csv": reference,
        "subject_level_behavioral_and_acoustic_deviations.csv": subject_data,
        "univariate_group_comparisons.csv": group_results,
        "loocv_classification_auc.csv": models,
    }
    for filename, frame in outputs.items():
        destination = output_dir / filename
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        print(f"Saved: {destination}")
    write_notes(output_dir / "analysis_notes.txt")
    print(f"Saved: {output_dir / 'analysis_notes.txt'}")


if __name__ == "__main__":
    main()
