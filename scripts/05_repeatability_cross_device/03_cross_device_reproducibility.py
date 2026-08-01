"""Compute cross-device reproducibility for Stroop speech features.

Scientific choice:
    The primary cross-device metric is ICC(2,1) absolute agreement because
    the question is whether devices are interchangeable. Pearson/Spearman
    correlations are also reported as supplementary association metrics, but
    correlation alone is not sufficient because it ignores systematic device
    bias and scale shifts.

Outputs are written under ``outputs/device_repeatability``.
"""

from __future__ import annotations

from itertools import combinations
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from common_paths import DEVICE_REPEATABILITY_OUTPUT_DIR, get_device_features_csv  # noqa: E402

FEATURE_CSV = get_device_features_csv()
RESULT_DIR = DEVICE_REPEATABILITY_OUTPUT_DIR
TASKS = ["Dots", "Hanzi", "STT"]
DEVICE_ORDER = ["1", "2", "3"]
DEVICE_NAMES = {
    "1": "ASUS_FX63VD_laptop",
    "2": "iPad_2021",
    "3": "Huawei_P30_phone",
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
TIMING_FEATURE_COLS = ["Word_Duration", "Gap_Before"]


def numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    candidates = [c for c in df.columns if c not in ID_COLS]
    return [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]


def clean_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Participant_ID", "Device_ID", "Device_Name", "Session", "Task"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")

    work = df.copy()
    for col in required:
        work[col] = work[col].astype(str)
    work = work[work["Device_ID"].isin(DEVICE_ORDER)].copy()
    work = work[work["Session"].isin(["session_1", "session_2"])].copy()
    work = work[work["Task"].isin(TASKS)].copy()
    return work


def aggregate_overall(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    return df.groupby(
        ["Participant_ID", "Session", "Device_ID", "Device_Name"],
        as_index=False,
    )[feature_cols].mean(numeric_only=True)


def aggregate_task(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    return df.groupby(
        ["Participant_ID", "Session", "Device_ID", "Device_Name", "Task"],
        as_index=False,
    )[feature_cols].mean(numeric_only=True)


def icc_2_1_matrix(data: np.ndarray) -> float:
    """
    Two-way random, absolute-agreement, single-measure ICC: ICC(2,1).

    Rows are targets, columns are devices/raters. Input must have no missing
    values and at least 2 targets x 2 devices.
    """
    data = np.asarray(data, dtype=float)
    n, k = data.shape
    if n < 2 or k < 2 or np.nanstd(data) == 0:
        return np.nan

    target_means = data.mean(axis=1, keepdims=True)
    rater_means = data.mean(axis=0, keepdims=True)
    grand_mean = data.mean()

    ss_target = k * np.sum((target_means - grand_mean) ** 2)
    ss_rater = n * np.sum((rater_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_target - ss_rater

    ms_target = ss_target / (n - 1)
    ms_rater = ss_rater / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_target + (k - 1) * ms_error + (k * (ms_rater - ms_error) / n)
    return np.nan if denom == 0 else float((ms_target - ms_error) / denom)


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
    t, p = stats.ttest_rel(y, x, nan_policy="omit")
    return float(t), float(p)


def icc_category(x) -> str:
    if pd.isna(x):
        return "NA"
    if x >= 0.90:
        return "excellent"
    if x >= 0.75:
        return "good"
    if x >= 0.50:
        return "moderate"
    return "poor"


def distribution_by_device(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    for (device_id, device_name, task), sub in df.groupby(["Device_ID", "Device_Name", "Task"]):
        for feature in feature_cols:
            x = sub[feature].dropna().to_numpy(dtype=float)
            row = {
                "Device_ID": device_id,
                "Device_Name": device_name,
                "Task": task,
                "Feature": feature,
                "N": len(x),
            }
            if len(x):
                q1, q3 = np.percentile(x, [25, 75])
                row.update({
                    "Mean": float(np.mean(x)),
                    "SD": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
                    "Median": float(np.median(x)),
                    "Q1": float(q1),
                    "Q3": float(q3),
                    "IQR": float(q3 - q1),
                    "Min": float(np.min(x)),
                    "Max": float(np.max(x)),
                })
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["Task", "Feature", "Device_ID"]).reset_index(drop=True)


def all3_icc(values: pd.DataFrame, feature_cols: List[str], group: str, task: str) -> pd.DataFrame:
    rows: List[Dict] = []
    index_cols = ["Participant_ID", "Session"]
    for feature in feature_cols:
        wide = values.pivot_table(index=index_cols, columns="Device_ID", values=feature, aggfunc="mean")
        present_devices = [d for d in DEVICE_ORDER if d in wide.columns]
        row = {
            "Group": group,
            "Task": task,
            "Feature": feature,
            "Devices": ",".join(present_devices),
        }
        if len(present_devices) >= 2:
            complete = wide[present_devices].dropna()
            row["N_complete_targets"] = len(complete)
            row["ICC_2_1_absolute_agreement"] = icc_2_1_matrix(complete.to_numpy(dtype=float)) if len(complete) >= 2 else np.nan
            row["ICC_Category"] = icc_category(row["ICC_2_1_absolute_agreement"])
        else:
            row["N_complete_targets"] = 0
            row["ICC_Category"] = "NA"
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_agreement(values: pd.DataFrame, feature_cols: List[str], group: str, task: str) -> pd.DataFrame:
    rows: List[Dict] = []
    index_cols = ["Participant_ID", "Session"]
    for d1, d2 in combinations(DEVICE_ORDER, 2):
        device_pair = f"{d1}_vs_{d2}"
        for feature in feature_cols:
            wide = values.pivot_table(index=index_cols, columns="Device_ID", values=feature, aggfunc="mean")
            if d1 not in wide.columns or d2 not in wide.columns:
                continue
            pair = wide[[d1, d2]].dropna()
            n = len(pair)
            row = {
                "Group": group,
                "Task": task,
                "Feature": feature,
                "Device_1": d1,
                "Device_1_Name": DEVICE_NAMES.get(d1, ""),
                "Device_2": d2,
                "Device_2_Name": DEVICE_NAMES.get(d2, ""),
                "Device_Pair": device_pair,
                "N": n,
            }
            if n >= 2:
                x = pair[d1].to_numpy(dtype=float)
                y = pair[d2].to_numpy(dtype=float)
                diff = y - x
                sd_diff = float(np.std(diff, ddof=1))
                mean_diff = float(np.mean(diff))
                icc = icc_2_1_matrix(pair[[d1, d2]].to_numpy(dtype=float))
                pearson_r, pearson_p = safe_corr(x, y, "pearson")
                spearman_rho, spearman_p = safe_corr(x, y, "spearman")
                t_value, t_p = paired_t(x, y)
                row.update({
                    "ICC_2_1_absolute_agreement": icc,
                    "ICC_Category": icc_category(icc),
                    "Pearson_r": pearson_r,
                    "Pearson_p": pearson_p,
                    "Spearman_rho": spearman_rho,
                    "Spearman_p": spearman_p,
                    "Device1_mean": float(np.mean(x)),
                    "Device1_sd": float(np.std(x, ddof=1)),
                    "Device2_mean": float(np.mean(y)),
                    "Device2_sd": float(np.std(y, ddof=1)),
                    "Mean_difference_D2_minus_D1": mean_diff,
                    "SD_difference": sd_diff,
                    "Paired_t": t_value,
                    "Paired_t_p": t_p,
                    "Bland_Altman_bias": mean_diff,
                    "Bland_Altman_lower_LOA": mean_diff - 1.96 * sd_diff,
                    "Bland_Altman_upper_LOA": mean_diff + 1.96 * sd_diff,
                })
            else:
                row["ICC_Category"] = "NA"
            rows.append(row)
    return pd.DataFrame(rows)


def compute_overall_level(overall_table: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all3 = all3_icc(overall_table, feature_cols, "overall_mean", "All")
    pairwise = pairwise_agreement(overall_table, feature_cols, "overall_mean", "All")
    return (
        all3.sort_values(["Feature"]).reset_index(drop=True),
        pairwise.sort_values(["Feature", "Device_Pair"]).reset_index(drop=True),
    )


def compute_task_level(task_table: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all3_parts = []
    pair_parts = []
    for task, sub in task_table.groupby("Task"):
        all3_parts.append(all3_icc(sub, feature_cols, "task_mean", str(task)))
        pair_parts.append(pairwise_agreement(sub, feature_cols, "task_mean", str(task)))
    all3 = pd.concat(all3_parts, ignore_index=True) if all3_parts else pd.DataFrame()
    pairwise = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    return (
        all3.sort_values(["Task", "Feature"]).reset_index(drop=True),
        pairwise.sort_values(["Task", "Feature", "Device_Pair"]).reset_index(drop=True),
    )


def remove_old_item_outputs() -> None:
    old_names = [
        "B3_HC_device_cross_device_ICC_all3_task_item.csv",
        "B3_HC_device_cross_device_pairwise_agreement_task_item.csv",
    ]
    for name in old_names:
        path = RESULT_DIR / name
        if path.exists():
            path.unlink()


def write_notes(
    path: Path,
    feature_cols: Sequence[str],
    overall_table: pd.DataFrame,
    task_table: pd.DataFrame,
) -> None:
    timing = [c for c in TIMING_FEATURE_COLS if c in feature_cols]
    opensmile_cols = [c for c in feature_cols if c not in TIMING_FEATURE_COLS]
    overall_counts = (
        overall_table.groupby(["Device_ID", "Device_Name", "Session"])["Participant_ID"]
        .nunique()
        .reset_index(name="N_participants")
        .to_string(index=False)
    )
    task_counts = (
        task_table.groupby(["Task", "Device_ID", "Session"])["Participant_ID"]
        .nunique()
        .reset_index(name="N_participants")
        .to_string(index=False)
    )
    lines = [
        "HC device Stroop cross-device validation notes",
        f"Input feature table: {FEATURE_CSV}",
        f"Numeric features tested: {len(feature_cols)}",
        f"OpenSMILE features: {len(opensmile_cols)}",
        f"Timing features: {len(timing)} ({', '.join(timing)})",
        "",
        "Primary metric: ICC(2,1), two-way random-effects absolute agreement, single measurement.",
        "Rationale: cross-device validation asks whether devices are interchangeable; absolute-agreement ICC penalizes systematic device bias.",
        "Pearson and Spearman correlations are reported as supplementary association metrics only.",
        "",
        "Matching unit for cross-device agreement: Participant_ID x Session.",
        "Both session_1 and session_2 are used as separate repeated targets, so 51 participants x 2 sessions = 102 targets when complete.",
        "",
        "No item-level ICC or pairwise agreement is computed in this script.",
        "Reason: trial order/content can differ across Stroop task sequences, so strict item-index matching is not appropriate.",
        "",
        "Device map:",
        "  1 = ASUS FX63VD laptop",
        "  2 = 2021 iPad",
        "  3 = Huawei P30 phone",
        "",
        "Participant counts by device/session in overall table:",
        overall_counts,
        "",
        "Participant counts by task/device/session in task table:",
        task_counts,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_CSV)
    feature_cols = numeric_feature_columns(df)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found.")

    clean = clean_feature_table(df)
    distribution = distribution_by_device(clean, feature_cols)
    overall_table = aggregate_overall(clean, feature_cols)
    task_table = aggregate_task(clean, feature_cols)
    all3_overall, pairwise_overall = compute_overall_level(overall_table, feature_cols)
    all3_task, pairwise_task = compute_task_level(task_table, feature_cols)

    remove_old_item_outputs()
    output_paths = {
        "distribution": RESULT_DIR / "feature_distribution_by_device.csv",
        "all3_overall": RESULT_DIR / "cross_device_all3_overall.csv",
        "pairwise_overall": RESULT_DIR / "cross_device_pairwise_overall.csv",
        "all3_task": RESULT_DIR / "cross_device_all3_by_task.csv",
        "pairwise_task": RESULT_DIR / "cross_device_pairwise_by_task.csv",
        "notes": RESULT_DIR / "cross_device_notes.txt",
    }
    distribution.to_csv(output_paths["distribution"], index=False, encoding="utf-8-sig")
    all3_overall.to_csv(output_paths["all3_overall"], index=False, encoding="utf-8-sig")
    pairwise_overall.to_csv(output_paths["pairwise_overall"], index=False, encoding="utf-8-sig")
    all3_task.to_csv(output_paths["all3_task"], index=False, encoding="utf-8-sig")
    pairwise_task.to_csv(output_paths["pairwise_task"], index=False, encoding="utf-8-sig")
    write_notes(output_paths["notes"], feature_cols, overall_table, task_table)

    for label, path in output_paths.items():
        print(f"{label} saved: {path}")
    print("removed old item-level B3 outputs if they existed")
    print(f"features tested: {len(feature_cols)}")


if __name__ == "__main__":
    main()
