"""
Compute test-retest reliability of six age-referenced W/CW/CW-minus-W acoustic
deviation indices in the independent device-repeatability cohort.

The deviation-index calculation follows the external clinical evaluation:
    Feature_Z ~ Age_Z * Task_Code + (1 + Task_Code | Subject_ID)

Task coding:
    W  = Word Reading, Task_Code = 0
    CW = Color-Word Interference, Task_Code = 1

For each of the 25 prespecified acoustic features, the healthy reference model
is fitted in the original 158-participant healthy cohort. For each device-cohort
participant/session/device target, W is represented by the empirical-Bayes
random intercept, CW by random intercept plus random task slope, and CW-minus-W
by the random task slope. Each effect is standardized against the corresponding
healthy reference BLUP distribution. The six subject-level indices are:

    W Mean |Z|
    W RMS Z
    CW Mean |Z|
    CW RMS Z
    CW-W Mean |Z|
    CW-W RMS Z

Reliability is computed separately for each recording device using ICC(3,1),
two-way mixed-effects consistency, single measurement, comparing session_1 and
session_2 within the same participant.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEVICE_FEATURE_CSV = (
    REPO_ROOT
    / "data"
    / "public"
    / "device_repeatability"
    / "Stroop_features_device_repeatability.csv"
)
DEFAULT_HEALTHY_FEATURES = REPO_ROOT / "data" / "public" / "features" / "Automated_feature.csv"
DEFAULT_HEALTHY_INFO = REPO_ROOT / "data" / "public" / "participant_info_HC_158.xlsx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "deviation_index_retest_reliability"

TASK_ALIASES = {"W": "W", "CW": "CW"}
TASK_CODE = {"W": 0, "CW": 1}
CONDITIONS = ("W", "CW", "CW-W")
EXPECTED_HEALTHY_N = 158

DEVICE_ORDER = ("1", "2", "3")
DEVICE_NAMES = {
    "1": "ASUS_FX63VD_laptop",
    "2": "Apple_iPad_2021",
    "3": "Huawei_P30_smartphone",
}
SESSIONS = ("session_1", "session_2")

SELECTED_FEATURES = [
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

METRIC_LABELS = {
    ("W", "Mean_Abs_Deviation_Z"): "W Mean |Z|",
    ("W", "RMS_Deviation_Z"): "W RMS Z",
    ("CW", "Mean_Abs_Deviation_Z"): "CW Mean |Z|",
    ("CW", "RMS_Deviation_Z"): "CW RMS Z",
    ("CW-W", "Mean_Abs_Deviation_Z"): "CW-W Mean |Z|",
    ("CW-W", "RMS_Deviation_Z"): "CW-W RMS Z",
}


def normalize_subject_id(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def load_age_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path)
    subject_column = find_column(frame, ["Subject_ID", "Subject", "ID"])
    age_column = find_column(frame, ["Age"])
    if subject_column is None or age_column is None:
        raise ValueError(f"{label}: subject ID or age column is missing.")
    result = frame[[subject_column, age_column]].copy()
    result.columns = ["Subject_ID", "Age"]
    result["Subject_ID"] = result["Subject_ID"].map(normalize_subject_id)
    result["Age"] = pd.to_numeric(result["Age"], errors="coerce")
    return result.dropna().drop_duplicates("Subject_ID")


def normalize_healthy_items(feature_csv: Path, info_xlsx: Path) -> pd.DataFrame:
    frame = read_csv(feature_csv)
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    subject_column = find_column(frame, ["Subject_ID", "Subject", "ID"])
    task_column = find_column(frame, ["Task", "Task_Name", "Task_Type"])
    if subject_column is None or task_column is None:
        raise ValueError("Healthy feature table lacks subject ID or task column.")
    frame["Subject_ID"] = frame[subject_column].map(normalize_subject_id)
    frame["Task"] = frame[task_column].astype(str).str.strip().map(TASK_ALIASES)
    frame = frame[frame["Task"].isin(TASK_CODE)].copy()
    ages = load_age_table(info_xlsx, "healthy information")
    frame = frame.merge(ages, on="Subject_ID", how="inner", validate="many_to_one")
    if frame["Subject_ID"].nunique() != EXPECTED_HEALTHY_N:
        raise ValueError(
            f"Expected {EXPECTED_HEALTHY_N} healthy reference participants, "
            f"found {frame['Subject_ID'].nunique()}."
        )
    missing = sorted(set(SELECTED_FEATURES).difference(frame.columns))
    if missing:
        raise ValueError(f"Healthy feature table missing selected features: {missing}")
    return frame


def normalize_device_items(feature_csv: Path) -> pd.DataFrame:
    frame = read_csv(feature_csv)
    required = {"Participant_ID", "Device_ID", "Session", "Task", "Age"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Device feature table missing columns: {missing}")
    absent = sorted(set(SELECTED_FEATURES).difference(frame.columns))
    if absent:
        raise ValueError(f"Device feature table missing selected features: {absent}")

    frame = frame.copy()
    frame["Participant_ID"] = frame["Participant_ID"].astype(str)
    frame["Device_ID"] = frame["Device_ID"].astype(str)
    frame["Session"] = frame["Session"].astype(str)
    frame["Task"] = frame["Task"].astype(str).str.strip().map(TASK_ALIASES)
    frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")
    frame = frame[
        frame["Device_ID"].isin(DEVICE_ORDER)
        & frame["Session"].isin(SESSIONS)
        & frame["Task"].isin(TASK_CODE)
        & frame["Age"].notna()
    ].copy()
    frame["Device_Name"] = frame["Device_ID"].map(DEVICE_NAMES)
    frame["Scoring_Target_ID"] = (
        frame["Participant_ID"] + "__D" + frame["Device_ID"] + "__" + frame["Session"]
    )
    frame["Subject_ID"] = frame["Scoring_Target_ID"]
    return frame


def random_effect_design(frame: pd.DataFrame, names: list[object]) -> np.ndarray:
    columns = []
    for name in names:
        key = str(name)
        if key in {"Group", "Intercept", "const", "1"}:
            columns.append(np.ones(len(frame)))
        elif key in frame:
            columns.append(pd.to_numeric(frame[key], errors="coerce").to_numpy(float))
        else:
            raise ValueError(f"Cannot construct random-effect column: {key}")
    return np.column_stack(columns)


def estimate_subject_effects(
    model_result,
    frame: pd.DataFrame,
    feature: str,
    feature_mean: float,
    feature_sd: float,
) -> pd.DataFrame:
    data = frame[["Subject_ID", "Task", "Task_Code", "Age_Z", feature]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna()
    data["Feature_Z"] = (data[feature] - feature_mean) / feature_sd
    data["Fixed_Residual_Z"] = data["Feature_Z"] - model_result.predict(data)

    covariance = model_result.cov_re
    names = list(covariance.index)
    name_strings = [str(name) for name in names]
    intercept_index = next(
        (index for index, name in enumerate(name_strings) if name in {"Group", "Intercept", "const", "1"}),
        None,
    )
    if intercept_index is None or "Task_Code" not in name_strings:
        raise RuntimeError("The fitted model lacks a random intercept or random task slope.")
    task_index = name_strings.index("Task_Code")
    g_matrix = np.asarray(covariance, dtype=float)
    sigma_squared = float(model_result.scale)
    if not np.isfinite(sigma_squared) or sigma_squared <= 0:
        raise RuntimeError("The fitted model has an invalid residual variance.")

    rows = []
    for subject_id, subset in data.groupby("Subject_ID", sort=False):
        if set(subset["Task"]) != {"W", "CW"}:
            continue
        design = random_effect_design(subset, names)
        residual = subset["Fixed_Residual_Z"].to_numpy(float)
        variance = design @ g_matrix @ design.T + sigma_squared * np.eye(len(subset))
        try:
            effects = g_matrix @ design.T @ np.linalg.solve(variance, residual)
        except np.linalg.LinAlgError:
            effects = g_matrix @ design.T @ np.linalg.pinv(variance) @ residual
        random_intercept = float(effects[intercept_index])
        random_slope = float(effects[task_index])
        rows.append(
            {
                "Subject_ID": subject_id,
                "W": random_intercept,
                "CW": random_intercept + random_slope,
                "CW-W": random_slope,
                "N_W_Items": int((subset["Task"] == "W").sum()),
                "N_CW_Items": int((subset["Task"] == "CW").sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_feature_model(feature: str, healthy: pd.DataFrame):
    data = healthy[["Subject_ID", "Task", "Task_Code", "Age_Z", feature]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna()
    feature_mean = float(data[feature].mean())
    feature_sd = float(data[feature].std(ddof=0))
    if not np.isfinite(feature_sd) or feature_sd <= 0:
        raise RuntimeError(f"Invalid healthy feature SD for {feature}.")
    data["Feature_Z"] = (data[feature] - feature_mean) / feature_sd

    last_result = None
    messages = []
    for optimizer in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    "Feature_Z ~ Age_Z * Task_Code",
                    data,
                    groups=data["Subject_ID"],
                    re_formula="~Task_Code",
                )
                result = model.fit(method=optimizer, reml=False, maxiter=2000, disp=False)
            messages.extend(str(item.message) for item in caught)
            last_result = (result, optimizer)
            if bool(getattr(result, "converged", False)):
                break
        except Exception as error:
            messages.append(f"{optimizer}: {error}")
    if last_result is None:
        raise RuntimeError(f"Could not fit the healthy reference model for {feature}.")
    return last_result[0], last_result[1], feature_mean, feature_sd, " | ".join(dict.fromkeys(messages))


def compute_deviation_indices(
    healthy: pd.DataFrame,
    scoring: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    healthy = healthy.copy()
    scoring = scoring.copy()
    healthy_ages = healthy[["Subject_ID", "Age"]].drop_duplicates("Subject_ID")
    age_mean = float(healthy_ages["Age"].mean())
    age_sd = float(healthy_ages["Age"].std(ddof=0))
    if not np.isfinite(age_sd) or age_sd <= 0:
        raise RuntimeError("Healthy age SD is invalid.")
    for frame in (healthy, scoring):
        frame["Age_Z"] = (frame["Age"] - age_mean) / age_sd
        frame["Task_Code"] = frame["Task"].map(TASK_CODE).astype(int)

    feature_rows = []
    model_rows = []
    for index, feature in enumerate(SELECTED_FEATURES, start=1):
        print(f"[{index:02d}/{len(SELECTED_FEATURES)}] {feature}")
        model, optimizer, feature_mean, feature_sd, model_warnings = fit_feature_model(feature, healthy)
        healthy_effects = estimate_subject_effects(model, healthy, feature, feature_mean, feature_sd)
        scoring_effects = estimate_subject_effects(model, scoring, feature, feature_mean, feature_sd)
        if healthy_effects["Subject_ID"].nunique() != EXPECTED_HEALTHY_N:
            raise RuntimeError(f"Incomplete healthy random effects for {feature}.")

        model_row = {
            "Feature": feature,
            "Optimizer": optimizer,
            "Converged": bool(getattr(model, "converged", False)),
            "N_Healthy_Subjects": healthy_effects["Subject_ID"].nunique(),
            "N_Scoring_Targets": scoring_effects["Subject_ID"].nunique(),
            "Feature_Mean": feature_mean,
            "Feature_SD_DDof0": feature_sd,
            "Age_Mean": age_mean,
            "Age_SD_DDof0": age_sd,
            "Warnings": model_warnings,
        }
        for condition in CONDITIONS:
            reference_mean = float(healthy_effects[condition].mean())
            reference_sd = float(healthy_effects[condition].std(ddof=1))
            if not np.isfinite(reference_sd) or reference_sd <= 0:
                raise RuntimeError(f"Invalid healthy {condition} random-effect SD for {feature}.")
            scores = scoring_effects[["Subject_ID", "N_W_Items", "N_CW_Items", condition]].copy()
            scores["Feature"] = feature
            scores["Condition"] = condition
            scores["Deviation_Z"] = (scores[condition] - reference_mean) / reference_sd
            scores["Abs_Deviation_Z"] = scores["Deviation_Z"].abs()
            scores["Healthy_Random_Effect_Mean"] = reference_mean
            scores["Healthy_Random_Effect_SD"] = reference_sd
            feature_rows.append(scores.drop(columns=[condition]))
            model_row[f"{condition}_Reference_Mean"] = reference_mean
            model_row[f"{condition}_Reference_SD_DDof1"] = reference_sd
        model_rows.append(model_row)

    feature_scores = pd.concat(feature_rows, ignore_index=True)
    keys = ["Participant_ID", "Device_ID", "Device_Name", "Session", "Scoring_Target_ID", "Age"]
    target_map = scoring[keys].drop_duplicates("Scoring_Target_ID")
    feature_scores = feature_scores.merge(
        target_map,
        left_on="Subject_ID",
        right_on="Scoring_Target_ID",
        how="left",
        validate="many_to_one",
    )

    subject_scores = (
        feature_scores.groupby(
            ["Participant_ID", "Device_ID", "Device_Name", "Session", "Scoring_Target_ID", "Age", "Condition"],
            as_index=False,
        )
        .agg(
            N_Features=("Feature", "nunique"),
            Mean_Abs_Deviation_Z=("Abs_Deviation_Z", "mean"),
            RMS_Deviation_Z=("Deviation_Z", lambda values: float(np.sqrt(np.mean(np.square(values))))),
            Median_N_W_Items=("N_W_Items", "median"),
            Median_N_CW_Items=("N_CW_Items", "median"),
        )
    )
    if not subject_scores["N_Features"].eq(len(SELECTED_FEATURES)).all():
        raise RuntimeError("Not every target-condition summary contains all 25 features.")

    wide_parts = []
    index_cols = ["Participant_ID", "Device_ID", "Device_Name", "Session", "Scoring_Target_ID", "Age"]
    for condition in CONDITIONS:
        sub = subject_scores[subject_scores["Condition"] == condition].copy()
        metric = sub.pivot_table(
            index=index_cols,
            values=["Mean_Abs_Deviation_Z", "RMS_Deviation_Z"],
            aggfunc="first",
        ).reset_index()
        metric = metric.rename(
            columns={
                "Mean_Abs_Deviation_Z": METRIC_LABELS[(condition, "Mean_Abs_Deviation_Z")],
                "RMS_Deviation_Z": METRIC_LABELS[(condition, "RMS_Deviation_Z")],
            }
        )
        wide_parts.append(metric)
    target_scores = wide_parts[0]
    for part in wide_parts[1:]:
        target_scores = target_scores.merge(part, on=index_cols, how="inner", validate="one_to_one")

    return pd.DataFrame(model_rows), feature_scores, target_scores


def icc_3_1(x: np.ndarray, y: np.ndarray) -> float:
    data = np.column_stack([x, y]).astype(float)
    n, k = data.shape
    if n < 2 or k != 2 or not np.isfinite(data).all() or np.std(data) == 0:
        return np.nan
    target_means = data.mean(axis=1, keepdims=True)
    session_means = data.mean(axis=0, keepdims=True)
    grand_mean = data.mean()
    ss_target = k * np.sum((target_means - grand_mean) ** 2)
    ss_error = np.sum((data - target_means - session_means + grand_mean) ** 2)
    ms_target = ss_target / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_target + (k - 1) * ms_error
    return np.nan if denominator == 0 else float((ms_target - ms_error) / denominator)


def icc_category(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value >= 0.90:
        return "excellent"
    if value >= 0.75:
        return "good"
    if value >= 0.50:
        return "moderate"
    return "poor"


def safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> Tuple[float, float]:
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan, np.nan
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        raise ValueError(method)
    return float(r), float(p)


def paired_t(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) < 2 or np.nanstd(y - x) == 0:
        return np.nan, np.nan
    t_value, p_value = stats.ttest_rel(y, x, nan_policy="omit")
    return float(t_value), float(p_value)


def reliability_table(target_scores: pd.DataFrame) -> pd.DataFrame:
    metric_cols = list(METRIC_LABELS.values())
    rows: List[Dict[str, object]] = []
    for device_id in DEVICE_ORDER:
        device_data = target_scores[target_scores["Device_ID"] == device_id]
        for metric in metric_cols:
            wide = device_data.pivot_table(
                index="Participant_ID",
                columns="Session",
                values=metric,
                aggfunc="mean",
            )
            row: Dict[str, object] = {
                "Device_ID": device_id,
                "Device_Name": DEVICE_NAMES[device_id],
                "Measure": metric,
            }
            if set(SESSIONS).issubset(wide.columns):
                pair = wide[list(SESSIONS)].dropna()
            else:
                pair = pd.DataFrame(columns=list(SESSIONS))
            row["N_Complete_Pairs"] = len(pair)
            if len(pair) >= 2:
                x = pair["session_1"].to_numpy(dtype=float)
                y = pair["session_2"].to_numpy(dtype=float)
                diff = y - x
                icc = icc_3_1(x, y)
                pearson_r, pearson_p = safe_corr(x, y, "pearson")
                spearman_rho, spearman_p = safe_corr(x, y, "spearman")
                t_value, t_p = paired_t(x, y)
                row.update(
                    {
                        "ICC_3_1_consistency": icc,
                        "ICC_Category": icc_category(icc),
                        "Pearson_r": pearson_r,
                        "Pearson_p": pearson_p,
                        "Spearman_rho": spearman_rho,
                        "Spearman_p": spearman_p,
                        "Session1_mean": float(np.mean(x)),
                        "Session1_sd": float(np.std(x, ddof=1)),
                        "Session2_mean": float(np.mean(y)),
                        "Session2_sd": float(np.std(y, ddof=1)),
                        "Mean_difference_S2_minus_S1": float(np.mean(diff)),
                        "SD_difference": float(np.std(diff, ddof=1)),
                        "Paired_t": t_value,
                        "Paired_t_p": t_p,
                    }
                )
            else:
                row["ICC_Category"] = "NA"
            rows.append(row)
    return pd.DataFrame(rows)


def build_table_s12(reliability: pd.DataFrame) -> pd.DataFrame:
    metric_columns = list(METRIC_LABELS.values())
    device_labels = {
        "1": "ASUS FX63VD (PC)",
        "2": "Apple iPad 2021 (iPad)",
        "3": "HUAWEI P30 (smartphone)",
    }
    table = reliability.pivot(
        index="Device_ID",
        columns="Measure",
        values="ICC_3_1_consistency",
    ).reindex(index=DEVICE_ORDER, columns=metric_columns)
    if table.isna().any().any():
        raise RuntimeError(
            "Table S12 is missing one or more device-by-index ICC values."
        )
    table.insert(0, "Device", [device_labels[device_id] for device_id in table.index])
    table = table.reset_index(drop=True)
    return table.rename(
        columns={metric: f"{metric} ICC(3,1)" for metric in metric_columns}
    )


def target_task_qc(scoring: pd.DataFrame, target_scores: pd.DataFrame) -> pd.DataFrame:
    task_counts = (
        scoring.groupby(
            ["Participant_ID", "Device_ID", "Device_Name", "Session", "Task"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "N_Items"})
    )
    wide = task_counts.pivot_table(
        index=["Participant_ID", "Device_ID", "Device_Name", "Session"],
        columns="Task",
        values="N_Items",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for task in ("W", "CW"):
        if task not in wide.columns:
            wide[task] = 0
    scored = target_scores[
        ["Participant_ID", "Device_ID", "Device_Name", "Session"]
    ].drop_duplicates()
    scored["Deviation_Indices_Computed"] = True
    qc = wide.merge(
        scored,
        on=["Participant_ID", "Device_ID", "Device_Name", "Session"],
        how="left",
    )
    qc["Deviation_Indices_Computed"] = qc["Deviation_Indices_Computed"].fillna(False)
    qc["Has_W"] = qc["W"].gt(0)
    qc["Has_CW"] = qc["CW"].gt(0)
    qc["Reason"] = np.where(
        qc["Deviation_Indices_Computed"],
        "computed",
        np.where(~qc["Has_W"], "missing W items", "missing CW items"),
    )
    return qc.sort_values(["Device_ID", "Participant_ID", "Session"]).reset_index(drop=True)


def write_notes(
    path: Path,
    device_feature_csv: Path,
    healthy_features: Path,
    healthy_info: Path,
    target_scores: pd.DataFrame,
) -> None:
    def display_path(input_path: Path) -> str:
        try:
            return input_path.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return input_path.name

    counts = (
        target_scores.groupby(["Device_ID", "Device_Name", "Session"])["Participant_ID"]
        .nunique()
        .reset_index(name="N_participants")
        .to_string(index=False)
    )
    text = f"""Device-cohort W/CW/CW-minus-W deviation-index test-retest reliability

Device feature table: {display_path(device_feature_csv)}
Healthy reference features: {display_path(healthy_features)}
Healthy reference information: {display_path(healthy_info)}
Selected acoustic features: {len(SELECTED_FEATURES)}

Deviation-index definitions
1. W is Word Reading and CW is Color-Word Interference.
2. For every selected feature, the healthy reference model is Feature_Z ~ Age_Z * Task_Code + (1 + Task_Code | Subject_ID), with W coded 0 and CW coded 1.
3. W is the subject random intercept, CW is random intercept plus random task slope, and CW-W is the random task slope.
4. Each feature-level effect is standardized against the matching empirical healthy BLUP distribution using its mean and sample SD.
5. The six target-level indices are W Mean |Z|, W RMS Z, CW Mean |Z|, CW RMS Z, CW-W Mean |Z|, and CW-W RMS Z.

Reliability model
ICC(3,1), two-way mixed-effects consistency, single measurement, comparing session_1 and session_2 within the same device.

Participant counts by device/session:
{counts}
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-feature-csv", type=Path, default=DEFAULT_DEVICE_FEATURE_CSV)
    parser.add_argument("--healthy-features", type=Path, default=DEFAULT_HEALTHY_FEATURES)
    parser.add_argument("--healthy-info", type=Path, default=DEFAULT_HEALTHY_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in ("device_feature_csv", "healthy_features", "healthy_info", "output_dir"):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    healthy = normalize_healthy_items(args.healthy_features, args.healthy_info)
    scoring = normalize_device_items(args.device_feature_csv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_summary, feature_scores, target_scores = compute_deviation_indices(healthy, scoring)

    reliability = reliability_table(target_scores)
    table_s12 = build_table_s12(reliability)
    qc = target_task_qc(scoring, target_scores)

    outputs = {
        "healthy_reference_lme_models.csv": model_summary,
        "feature_level_deviation_scores.csv": feature_scores,
        "subject_session_deviation_indices.csv": target_scores,
        "deviation_index_retest_icc.csv": reliability,
        "table_s12_deviation_index_retest_reliability.csv": table_s12,
        "target_task_qc.csv": qc,
    }
    for filename, frame in outputs.items():
        destination = args.output_dir / filename
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        print(f"Saved: {destination}")
    write_notes(
        args.output_dir / "analysis_notes.txt",
        args.device_feature_csv,
        args.healthy_features,
        args.healthy_info,
        target_scores,
    )
    print(f"Saved: {args.output_dir / 'analysis_notes.txt'}")
    print(f"Targets scored: {len(target_scores)}")
    print(f"Features used: {len(SELECTED_FEATURES)}")


if __name__ == "__main__":
    main()
