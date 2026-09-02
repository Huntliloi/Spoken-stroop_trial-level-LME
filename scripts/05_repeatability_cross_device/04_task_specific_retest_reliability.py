"""
Compare test-retest reliability of traditional Stroop scores and the 32
manuscript-significant acoustic features in the HC device cohort.

All analyses are conducted separately for each recording device. Data are
never pooled across devices.

Traditional measures:
    W accuracy
    CW accuracy
    W - CW accuracy interference
    W card completion time
    CW card completion time
    CW - W completion-time interference

Acoustic measures (32 prespecified features only):
    Overall feature level across all three conditions and items
    W task mean
    CW task mean
    Model-based subject-specific CW-versus-W task slope

The task slope is not a raw difference between two task means. For every
device, session, and feature, an item-level mixed model is fitted:

    Feature_Z ~ Age_Z * Task_Code + (1 + Task_Code | Participant_ID)

Task_Code is -0.5 for W and +0.5 for CW. The model-implied individual
task slope is the fixed task slope, plus the age-dependent fixed component,
plus the participant's empirical-Bayes random task slope. Its ICC across the
two sessions quantifies test-retest consistency of task-evoked modulation.

Reliability is ICC(3,1): two-way mixed-effects consistency, single
measurement, comparing session_1 with session_2 within the same device.
Participant bootstrap confidence intervals preserve the session pairing.
"""

from __future__ import annotations

import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURE_CSV = (
    REPO_ROOT
    / "data"
    / "public"
    / "device_repeatability"
    / "Stroop_features_device_repeatability.csv"
)
DEFAULT_PRIMARY_RESULTS_CSV = Path(
    REPO_ROOT
    / "results"
    / "04_wd_external_validation"
    / "LME_deviation_scores_CW_vs_W_WD_CI_FDR01"
    / "healthy_LME_model_summary_CW_vs_W_WD_CI_FDR01.csv"
)
DEFAULT_RESULT_DIR = REPO_ROOT / "outputs" / "task_specific_reliability"

EXPECTED_ITEMS = 12
SESSIONS = ("session_1", "session_2")
DEVICE_ORDER = ("1", "2", "3")
DEVICE_NAMES = {
    "1": "ASUS_FX63VD_laptop",
    "2": "Apple_iPad_2021",
    "3": "Huawei_P30_smartphone",
}
TASK_TARGETS = {
    "W": ["黄", "红", "绿", "蓝", "绿", "黄", "蓝", "红", "蓝", "绿", "红", "黄"],
    "C": ["红", "黄", "蓝", "绿", "绿", "蓝", "黄", "红", "黄", "红", "绿", "蓝"],
    "CW": ["黄", "红", "绿", "蓝", "绿", "蓝", "黄", "红", "红", "蓝", "绿", "黄"],
}
ID_COLS = {
    "Subject_ID",
    "Participant_ID",
    "Device_ID",
    "Device_Name",
    "Original_Subject_ID",
    "Name",
    "Age",
    "Sex",
    "Follow_Up",
    "Session",
    "Folder_Session",
    "Subject_Folder",
    "Task",
    "Item_Index",
    "Word",
    "Absolute_Word_Start",
    "Absolute_Word_End",
    "Audio_Path",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument(
        "--primary-results-csv",
        type=Path,
        default=DEFAULT_PRIMARY_RESULTS_CSV,
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    return parser.parse_args()


def clean_input(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Participant_ID",
        "Device_ID",
        "Session",
        "Task",
        "Item_Index",
        "Word",
        "Absolute_Word_End",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")

    work = frame.copy()
    work["Participant_ID"] = work["Participant_ID"].astype(str)
    work["Device_ID"] = work["Device_ID"].astype(str)
    work["Session"] = work["Session"].astype(str)
    work["Task"] = work["Task"].astype(str)
    work = work[
        work["Device_ID"].isin(DEVICE_ORDER)
        & work["Session"].isin(SESSIONS)
        & work["Task"].isin(["C", "W", "CW"])
    ].copy()
    work["Device_Name"] = work["Device_ID"].map(DEVICE_NAMES)
    return work


def load_selected_features(path: Path, feature_table: pd.DataFrame) -> List[str]:
    results = pd.read_csv(path, encoding="utf-8-sig")
    required = {"Feature", "Interaction_FDR_PValue"}
    missing = sorted(required.difference(results.columns))
    if missing:
        raise ValueError(f"Primary result table missing columns: {missing}")

    results["Interaction_FDR_PValue"] = pd.to_numeric(
        results["Interaction_FDR_PValue"], errors="coerce"
    )
    selected = results.loc[
        results["Interaction_FDR_PValue"].le(0.05), "Feature"
    ].astype(str).tolist()
    if len(selected) != 32:
        raise RuntimeError(f"Expected 32 FDR-significant features, found {len(selected)}")
    absent = sorted(set(selected).difference(feature_table.columns))
    if absent:
        raise RuntimeError(f"Selected features missing from device data: {absent}")
    return selected


def derive_traditional_metrics(items: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    keys = ["Participant_ID", "Device_ID", "Device_Name", "Session", "Task"]
    for key, sub in items.groupby(keys, sort=True):
        participant_id, device_id, device_name, session, task = key
        if task not in TASK_TARGETS:
            continue
        ordered = sub.sort_values("Item_Index").head(EXPECTED_ITEMS).copy()
        words = ordered["Word"].fillna("").astype(str).tolist()
        correct = sum(
            word == TASK_TARGETS[task][index] for index, word in enumerate(words)
        )
        ends = pd.to_numeric(ordered["Absolute_Word_End"], errors="coerce")
        completion_time = (
            float(ends.iloc[EXPECTED_ITEMS - 1])
            if len(ordered) == EXPECTED_ITEMS
            and np.isfinite(ends.iloc[EXPECTED_ITEMS - 1])
            else np.nan
        )
        rows.append(
            {
                "Participant_ID": participant_id,
                "Device_ID": device_id,
                "Device_Name": device_name,
                "Session": session,
                "Task": task,
                "Valid_Item_Count": len(ordered),
                "Correct_Item_Count": correct,
                "Overall_Accuracy": correct / EXPECTED_ITEMS,
                "Card_Completion_Time": completion_time,
            }
        )

    task_metrics = pd.DataFrame(rows)
    index = ["Participant_ID", "Device_ID", "Device_Name", "Session"]
    wide = task_metrics.pivot(
        index=index,
        columns="Task",
        values=[
            "Valid_Item_Count",
            "Correct_Item_Count",
            "Overall_Accuracy",
            "Card_Completion_Time",
        ],
    )
    wide.columns = [f"{measure}_{task}" for measure, task in wide.columns]
    wide = wide.reset_index()
    task_accuracy_columns = [
        f"Overall_Accuracy_{task}" for task in ("C", "W", "CW")
    ]
    task_correct_columns = [
        f"Correct_Item_Count_{task}" for task in ("C", "W", "CW")
    ]
    task_completion_columns = [
        f"Card_Completion_Time_{task}" for task in ("C", "W", "CW")
    ]
    missing_columns = sorted(
        set(task_accuracy_columns + task_correct_columns + task_completion_columns)
        .difference(wide.columns)
    )
    if missing_columns:
        raise RuntimeError(f"Cannot derive three-task scores: {missing_columns}")

    # Accuracy retains the prespecified denominator of 36. Three-condition total
    # completion time is available only when all three card times are present.
    wide["Overall_Accuracy_All_Three_Tasks"] = (
        wide[task_correct_columns].sum(axis=1) / (3 * EXPECTED_ITEMS)
    )
    wide["Total_Completion_Time_All_Three_Tasks"] = wide[
        task_completion_columns
    ].sum(axis=1, min_count=3)
    wide["Accuracy_Interference_W_Minus_CW"] = (
        wide["Overall_Accuracy_W"] - wide["Overall_Accuracy_CW"]
    )
    wide["Completion_Time_Interference_CW_Minus_W"] = (
        wide["Card_Completion_Time_CW"]
        - wide["Card_Completion_Time_W"]
    )
    return task_metrics, wide


def build_acoustic_levels(
    items: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:
    base_keys = ["Participant_ID", "Device_ID", "Device_Name", "Session"]
    parts: List[pd.DataFrame] = []

    overall = items.groupby(base_keys, as_index=False)[selected_features].mean(
        numeric_only=True
    )
    overall.insert(4, "Acoustic_Level", "Overall_All_Conditions")
    parts.append(overall)

    task_means = items[items["Task"].isin(["C", "W", "CW"])].groupby(
        base_keys + ["Task"], as_index=False
    )[selected_features].mean(numeric_only=True)
    for task in ("C", "W", "CW"):
        task_values = task_means[task_means["Task"] == task].drop(
            columns="Task"
        )
        task_values.insert(4, "Acoustic_Level", task)
        parts.append(task_values)

    return pd.concat(parts, ignore_index=True).sort_values(
        ["Device_ID", "Acoustic_Level", "Participant_ID", "Session"]
    ).reset_index(drop=True)


def prepare_lmm_items(
    items: pd.DataFrame,
    selected_features: List[str],
) -> pd.DataFrame:
    """Prepare consistently scaled W/CW item data for session LMMs."""
    columns = [
        "Participant_ID",
        "Device_ID",
        "Device_Name",
        "Session",
        "Task",
        "Age",
        *selected_features,
    ]
    work = items.loc[items["Task"].isin(["W", "CW"]), columns].copy()
    work["Task_Code"] = work["Task"].map({"W": -0.5, "CW": 0.5})
    work["Age"] = pd.to_numeric(work["Age"], errors="coerce")

    prepared = []
    for device_id, device in work.groupby("Device_ID", sort=True):
        device = device.copy()
        participant_age = device.groupby("Participant_ID")["Age"].first()
        age_mean = participant_age.mean()
        age_sd = participant_age.std(ddof=0)
        if not np.isfinite(age_sd) or age_sd <= 0:
            raise RuntimeError(f"Age has no usable variation for device {device_id}")
        device["Age_Z"] = (device["Age"] - age_mean) / age_sd

        # Use one scale per device across both sessions. This preserves session
        # differences while improving mixed-model numerical stability.
        for feature in selected_features:
            values = pd.to_numeric(device[feature], errors="coerce")
            mean = values.mean()
            sd = values.std(ddof=0)
            if np.isfinite(sd) and sd > 0:
                device[feature] = (values - mean) / sd
            else:
                device[feature] = np.nan
        prepared.append(device)
    return pd.concat(prepared, ignore_index=True)


def _fit_session_task_slopes(
    device_id: str,
    session: str,
    session_data: pd.DataFrame,
    selected_features: List[str],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Fit all feature LMMs for one device/session worker process."""
    slope_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    base = session_data[
        ["Participant_ID", "Task_Code", "Age_Z", *selected_features]
    ].copy()
    base["Participant_ID"] = base["Participant_ID"].astype(str)

    for feature in selected_features:
        model_data = base[
            ["Participant_ID", "Task_Code", "Age_Z", feature]
        ].rename(columns={feature: "Outcome_Z"}).dropna()
        n_subjects = model_data["Participant_ID"].nunique()
        n_items = len(model_data)
        fit = None
        method_used = ""
        error = ""
        warning_text = ""
        try:
            model = smf.mixedlm(
                "Outcome_Z ~ Age_Z * Task_Code",
                model_data,
                groups=model_data["Participant_ID"],
                re_formula="1 + Task_Code",
            )
            captured = []
            finite_candidates = []
            for method in ("lbfgs", "cg", "powell", "nm"):
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        candidate = model.fit(
                            reml=True,
                            method=method,
                            maxiter=1000,
                            disp=False,
                        )
                    captured.extend(str(item.message) for item in caught)
                    covariance = np.asarray(candidate.cov_re, dtype=float)
                    is_finite = (
                        np.isfinite(candidate.llf)
                        and np.isfinite(candidate.scale)
                        and np.isfinite(covariance).all()
                    )
                    if is_finite:
                        finite_candidates.append((candidate.llf, method, candidate))
                    if candidate.converged and is_finite:
                        fit = candidate
                        method_used = method
                        break
                except Exception as exc:
                    captured.append(f"{method}: {exc}")
            warning_text = " | ".join(dict.fromkeys(captured))
            if fit is None and finite_candidates:
                _, method_used, fit = max(finite_candidates, key=lambda item: item[0])
            if fit is None:
                raise RuntimeError("all optimizers failed")

            fixed_task = float(fit.fe_params.get("Task_Code", 0.0))
            fixed_age_task = float(
                fit.fe_params.get("Age_Z:Task_Code", 0.0)
            )
            ages = model_data.groupby("Participant_ID")["Age_Z"].first()
            random_effects = {}
            covariance = np.asarray(fit.cov_re, dtype=float)
            fixed_values = np.asarray(fit.fe_params, dtype=float)
            for participant_id in fit.model.group_labels:
                index = fit.model.row_indices[participant_id]
                fixed_design = fit.model.exog[index, :]
                random_design = fit.model.exog_re[index, :]
                residual = (
                    fit.model.endog[index] - fixed_design @ fixed_values
                )
                marginal_covariance = (
                    random_design @ covariance @ random_design.T
                    + float(fit.scale) * np.eye(len(index))
                )
                try:
                    weighted_residual = np.linalg.solve(
                        marginal_covariance, residual
                    )
                except np.linalg.LinAlgError:
                    weighted_residual = np.linalg.pinv(
                        marginal_covariance
                    ) @ residual
                random_effects[str(participant_id)] = (
                    covariance @ random_design.T @ weighted_residual
                )
            random_slope_variance = float(
                fit.cov_re.loc["Task_Code", "Task_Code"]
            )
            for participant_id, age_z in ages.items():
                effects = random_effects[str(participant_id)]
                random_task_slope = float(effects[1])
                model_task_slope = (
                    fixed_task + fixed_age_task * float(age_z) + random_task_slope
                )
                slope_rows.append(
                    {
                        "Participant_ID": str(participant_id),
                        "Device_ID": device_id,
                        "Device_Name": DEVICE_NAMES[device_id],
                        "Session": session,
                        "Feature": feature,
                        "Age_Z": float(age_z),
                        "Fixed_Task_Slope": fixed_task,
                        "Fixed_Age_x_Task": fixed_age_task,
                        "Random_Task_Slope_Deviation": random_task_slope,
                        "Model_Implied_Task_Slope": model_task_slope,
                    }
                )
        except Exception as exc:
            error = str(exc)

        diagnostic_rows.append(
            {
                "Device_ID": device_id,
                "Device_Name": DEVICE_NAMES[device_id],
                "Session": session,
                "Feature": feature,
                "N_Item_Observations": n_items,
                "N_Participants": n_subjects,
                "Converged": bool(
                    fit is not None
                    and fit.converged
                    and not error
                    and np.isfinite(fit.llf)
                ),
                "Optimizer": method_used,
                "Random_Task_Slope_Variance": random_slope_variance
                if fit is not None and not error
                else np.nan,
                "Warnings": warning_text,
                "Error": error,
            }
        )
    return slope_rows, diagnostic_rows


def fit_model_based_task_slopes(
    items: pd.DataFrame,
    selected_features: List[str],
    workers: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prepared = prepare_lmm_items(items, selected_features)
    work_units = []
    for device_id in DEVICE_ORDER:
        for session in SESSIONS:
            subset = prepared[
                (prepared["Device_ID"] == device_id)
                & (prepared["Session"] == session)
            ].copy()
            work_units.append((device_id, session, subset, selected_features))

    slope_rows: List[Dict[str, object]] = []
    diagnostic_rows: List[Dict[str, object]] = []
    if workers <= 1:
        for work_unit in work_units:
            slopes, diagnostics = _fit_session_task_slopes(*work_unit)
            slope_rows.extend(slopes)
            diagnostic_rows.extend(diagnostics)
    else:
        jobs = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for work_unit in work_units:
                jobs.append(executor.submit(_fit_session_task_slopes, *work_unit))
            for job in as_completed(jobs):
                slopes, diagnostics = job.result()
                slope_rows.extend(slopes)
                diagnostic_rows.extend(diagnostics)

    slopes = pd.DataFrame(slope_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    failed = diagnostics.loc[~diagnostics["Converged"]]
    if len(failed):
        examples = failed[["Device_Name", "Session", "Feature", "Error"]].head()
        raise RuntimeError(
            f"{len(failed)} session-specific mixed models did not converge:\n"
            f"{examples.to_string(index=False)}"
        )
    return (
        slopes.sort_values(
            ["Device_ID", "Feature", "Participant_ID", "Session"]
        ).reset_index(drop=True),
        diagnostics.sort_values(
            ["Device_ID", "Session", "Feature"]
        ).reset_index(drop=True),
    )


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


def bootstrap_icc_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> Tuple[float, float, int]:
    if len(x) < 3 or n_bootstrap <= 0:
        return np.nan, np.nan, 0
    estimates = []
    for _ in range(n_bootstrap):
        index = rng.integers(0, len(x), size=len(x))
        estimate = icc_3_1(x[index], y[index])
        if np.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) < max(100, n_bootstrap // 2):
        return np.nan, np.nan, len(estimates)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return float(lower), float(upper), len(estimates)


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


def paired_icc(
    values: pd.DataFrame,
    value_column: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> Dict[str, object]:
    wide = values.pivot_table(
        index="Participant_ID",
        columns="Session",
        values=value_column,
        aggfunc="mean",
    )
    if not set(SESSIONS).issubset(wide.columns):
        return {"N_Complete_Pairs": 0, "ICC_3_1": np.nan, "ICC_Category": "NA"}
    pair = wide[list(SESSIONS)].dropna()
    x = pair["session_1"].to_numpy(dtype=float)
    y = pair["session_2"].to_numpy(dtype=float)
    estimate = icc_3_1(x, y)
    ci_low, ci_high, valid_bootstrap = bootstrap_icc_ci(
        x, y, rng, n_bootstrap
    )
    return {
        "N_Complete_Pairs": len(pair),
        "ICC_3_1": estimate,
        "ICC_95CI_Low_Bootstrap": ci_low,
        "ICC_95CI_High_Bootstrap": ci_high,
        "Bootstrap_Valid_Replicates": valid_bootstrap,
        "ICC_Category": icc_category(estimate),
        "Session1_Mean": float(np.mean(x)) if len(x) else np.nan,
        "Session1_SD": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "Session2_Mean": float(np.mean(y)) if len(y) else np.nan,
        "Session2_SD": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
        "Mean_Change_S2_Minus_S1": float(np.mean(y - x)) if len(x) else np.nan,
    }


def traditional_reliability(
    subject_metrics: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    measures = {
        "Three-condition overall accuracy": "Overall_Accuracy_All_Three_Tasks",
        "C accuracy": "Overall_Accuracy_C",
        "W accuracy": "Overall_Accuracy_W",
        "CW accuracy": "Overall_Accuracy_CW",
        "Accuracy interference (W - CW)": "Accuracy_Interference_W_Minus_CW",
        "Three-condition total completion time": "Total_Completion_Time_All_Three_Tasks",
        "C card completion time": "Card_Completion_Time_C",
        "W card completion time": "Card_Completion_Time_W",
        "CW card completion time": "Card_Completion_Time_CW",
        "Completion-time interference (CW - W)": "Completion_Time_Interference_CW_Minus_W",
    }
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for device_id in DEVICE_ORDER:
        device_data = subject_metrics[subject_metrics["Device_ID"] == device_id]
        for measure, column in measures.items():
            row: Dict[str, object] = {
                "Device_ID": device_id,
                "Device_Name": DEVICE_NAMES[device_id],
                "Measure": measure,
                "Source_Column": column,
            }
            row.update(paired_icc(device_data, column, rng, n_bootstrap))
            rows.append(row)
    return pd.DataFrame(rows)


def acoustic_reliability(
    acoustic_levels: pd.DataFrame,
    model_task_slopes: pd.DataFrame,
    selected_features: List[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    rows: List[Dict[str, object]] = []
    for device_id in DEVICE_ORDER:
        device_data = acoustic_levels[acoustic_levels["Device_ID"] == device_id]
        for level in ["Overall_All_Conditions", "C", "W", "CW"]:
            level_data = device_data[device_data["Acoustic_Level"] == level]
            for feature in selected_features:
                row: Dict[str, object] = {
                    "Device_ID": device_id,
                    "Device_Name": DEVICE_NAMES[device_id],
                    "Acoustic_Level": level,
                    "Feature": feature,
                }
                row.update(paired_icc(level_data, feature, rng, n_bootstrap))
                rows.append(row)

        device_slopes = model_task_slopes[
            model_task_slopes["Device_ID"] == device_id
        ]
        for feature in selected_features:
            feature_slopes = device_slopes[
                device_slopes["Feature"] == feature
            ]
            row = {
                "Device_ID": device_id,
                "Device_Name": DEVICE_NAMES[device_id],
                "Acoustic_Level": "Model_Based_CW_vs_W_Task_Slope",
                "Feature": feature,
            }
            row.update(
                paired_icc(
                    feature_slopes,
                    "Model_Implied_Task_Slope",
                    rng,
                    n_bootstrap,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_acoustic(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (device_id, device_name, level), group in results.groupby(
        ["Device_ID", "Device_Name", "Acoustic_Level"], sort=True
    ):
        values = group["ICC_3_1"].dropna().to_numpy(dtype=float)
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        categories = group["ICC_Category"].value_counts()
        rows.append(
            {
                "Device_ID": device_id,
                "Device_Name": device_name,
                "Acoustic_Level": level,
                "N_Features": len(values),
                "Mean_ICC_3_1": float(np.mean(values)),
                "Median_ICC_3_1": median,
                "ICC_Q1": q1,
                "ICC_Q3": q3,
                "ICC_Min": float(np.min(values)),
                "ICC_Max": float(np.max(values)),
                "N_Poor": int(categories.get("poor", 0)),
                "N_Moderate": int(categories.get("moderate", 0)),
                "N_Good": int(categories.get("good", 0)),
                "N_Excellent": int(categories.get("excellent", 0)),
                "N_ICC_GE_0_50": int(np.sum(values >= 0.50)),
                "N_ICC_GE_0_75": int(np.sum(values >= 0.75)),
                "Median_Complete_Pairs": float(group["N_Complete_Pairs"].median()),
            }
        )
    return pd.DataFrame(rows)


def build_manuscript_table(
    traditional: pd.DataFrame,
    acoustic_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in acoustic_summary.iterrows():
        rows.append(
            {
                "Device_ID": row["Device_ID"],
                "Device_Name": row["Device_Name"],
                "Measure_Family": "Acoustic (32 features)",
                "Measure": row["Acoustic_Level"],
                "N_Measures": row["N_Features"],
                "N_Complete_Pairs": row["Median_Complete_Pairs"],
                "Mean_ICC_3_1": row["Mean_ICC_3_1"],
                "Median_ICC_3_1": row["Median_ICC_3_1"],
                "ICC_Q1": row["ICC_Q1"],
                "ICC_Q3": row["ICC_Q3"],
                "N_ICC_GE_0_50": row["N_ICC_GE_0_50"],
                "N_ICC_GE_0_75": row["N_ICC_GE_0_75"],
            }
        )
    for _, row in traditional.iterrows():
        rows.append(
            {
                "Device_ID": row["Device_ID"],
                "Device_Name": row["Device_Name"],
                "Measure_Family": "Traditional Stroop",
                "Measure": row["Measure"],
                "N_Measures": 1,
                "N_Complete_Pairs": row["N_Complete_Pairs"],
                "Mean_ICC_3_1": row["ICC_3_1"],
                "Median_ICC_3_1": row["ICC_3_1"],
                "ICC_Q1": np.nan,
                "ICC_Q3": np.nan,
                "N_ICC_GE_0_50": int(row["ICC_3_1"] >= 0.50)
                if np.isfinite(row["ICC_3_1"])
                else 0,
                "N_ICC_GE_0_75": int(row["ICC_3_1"] >= 0.75)
                if np.isfinite(row["ICC_3_1"])
                else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["Device_ID", "Measure_Family", "Measure"]
    ).reset_index(drop=True)


def build_main_text_table(
    traditional: pd.DataFrame,
    acoustic_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the compact four-information-column table for the main text."""
    traditional_lookup = traditional.set_index(["Device_ID", "Measure"])
    acoustic_lookup = acoustic_summary.set_index(
        ["Device_ID", "Acoustic_Level"]
    )
    levels = [
        (
            "Three-condition overall",
            "Three-condition overall accuracy",
            "Three-condition total completion time",
            "Overall_All_Conditions",
        ),
        (
            "C",
            "C accuracy",
            "C card completion time",
            "C",
        ),
        (
            "W",
            "W accuracy",
            "W card completion time",
            "W",
        ),
        (
            "CW",
            "CW accuracy",
            "CW card completion time",
            "CW",
        ),
        (
            "CW-W task effect",
            "Accuracy interference (W - CW)",
            "Completion-time interference (CW - W)",
            "Model_Based_CW_vs_W_Task_Slope",
        ),
    ]
    rows = []
    for device_id in DEVICE_ORDER:
        for level, accuracy_measure, time_measure, acoustic_level in levels:
            accuracy = traditional_lookup.loc[(device_id, accuracy_measure)]
            completion = traditional_lookup.loc[(device_id, time_measure)]
            acoustic = acoustic_lookup.loc[(device_id, acoustic_level)]
            rows.append(
                {
                    "Device_and_Measure_Level": f"{DEVICE_NAMES[device_id]} | {level}",
                    "Accuracy_ICC_3_1": accuracy["ICC_3_1"],
                    "Completion_Time_ICC_3_1": completion["ICC_3_1"],
                    "Acoustic_32_Feature_Median_ICC_3_1": acoustic[
                        "Median_ICC_3_1"
                    ],
                }
            )
    return pd.DataFrame(rows)


def write_notes(
    path: Path,
    feature_csv: Path,
    primary_results_csv: Path,
    selected_features: List[str],
    n_bootstrap: int,
) -> None:
    def public_path(value: Path) -> str:
        try:
            return value.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return str(value)

    text = f"""Task-specific test-retest reliability

Input feature table: {public_path(feature_csv)}
Primary feature-selection table: {public_path(primary_results_csv)}
Selected acoustic features: {len(selected_features)} (original CW-vs-W Age x Task effects with FDR-adjusted p <= .05)
Bootstrap replicates per ICC: {n_bootstrap}

All reliability analyses are performed separately within each device. No values are pooled across devices.

Traditional metric definitions
1. Task accuracy is the number of correctly paired recognized color words divided by 12. Missing responses count as incorrect.
2. Task card completion time is the end time of the 12th paired response relative to audio onset. It is missing when fewer than 12 paired responses are available.
3. Three-condition overall accuracy is total correct across C, W, and CW divided by 36.
4. Three-condition total completion time is the sum of the three card completion times and is missing unless all three are available. Its ICC equals the ICC of their mean because the two scores differ only by a constant factor.
5. Accuracy interference is W accuracy minus CW accuracy.
6. Completion-time interference is CW completion time minus W completion time.
7. Missing completion times are not imputed for reliability analysis; ICC uses complete session pairs.

Acoustic metric definitions
1. Overall_All_Conditions averages all available C, W, and CW item-level observations within participant, device, and session.
2. C, W, and CW average available item-level observations within the corresponding task.
3. The CW-versus-W measure is not computed by directly subtracting task means. For each device, session, and feature, all available W and CW items are fitted with Feature_Z ~ Age_Z * Task_Code + (1 + Task_Code | Participant_ID), where W = -0.5 and CW = +0.5.
4. The model-implied individual task slope equals the fixed Task effect plus the fixed Age x Task component at that participant's age plus the empirical-Bayes random Task slope. ICC compares these model-implied slopes across sessions.
5. Only the 32 prespecified significant features are analyzed; the remaining 58 features are not tested.

Reliability model
ICC(3,1), two-way mixed-effects consistency, single measurement, for session_1 versus session_2 within the same device. Bootstrap confidence intervals resample participants as paired observations.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)

    items = clean_input(pd.read_csv(args.feature_csv, encoding="utf-8-sig"))
    selected_features = load_selected_features(args.primary_results_csv, items)
    task_metrics, subject_behavior = derive_traditional_metrics(items)
    acoustic_levels = build_acoustic_levels(items, selected_features)
    model_task_slopes, model_diagnostics = fit_model_based_task_slopes(
        items, selected_features, args.workers
    )

    traditional = traditional_reliability(
        subject_behavior, args.bootstrap, args.seed
    )
    acoustic = acoustic_reliability(
        acoustic_levels,
        model_task_slopes,
        selected_features,
        args.bootstrap,
        args.seed,
    )
    acoustic_summary = summarize_acoustic(acoustic)
    manuscript_table = build_manuscript_table(traditional, acoustic_summary)
    main_text_table = build_main_text_table(traditional, acoustic_summary)

    task_metrics.to_csv(
        args.result_dir / "traditional_task_metrics_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    subject_behavior.to_csv(
        args.result_dir / "traditional_subject_session_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    traditional.to_csv(
        args.result_dir / "traditional_metric_retest_icc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    acoustic.to_csv(
        args.result_dir / "acoustic_32_feature_retest_icc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_task_slopes.to_csv(
        args.result_dir / "model_based_task_slopes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    model_diagnostics.to_csv(
        args.result_dir / "task_slope_lmm_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    acoustic_summary.to_csv(
        args.result_dir / "acoustic_32_feature_icc_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    manuscript_table.to_csv(
        args.result_dir / "manuscript_table_with_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_text_table.to_csv(
        args.result_dir / "table_s9_task_specific_reliability.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_notes(
        args.result_dir / "analysis_notes.txt",
        args.feature_csv,
        args.primary_results_csv,
        selected_features,
        args.bootstrap,
    )

    print("\nAcoustic reliability summary (32 features only)")
    print(acoustic_summary.to_string(index=False))
    print("\nTraditional Stroop reliability")
    print(
        traditional[
            [
                "Device_Name",
                "Measure",
                "N_Complete_Pairs",
                "ICC_3_1",
                "ICC_95CI_Low_Bootstrap",
                "ICC_95CI_High_Bootstrap",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs saved to: {args.result_dir}")


if __name__ == "__main__":
    main()
