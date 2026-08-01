"""Compute within-device test-retest reliability for Stroop features.

Main analysis:
    For each device separately:
        same Subject_ID x Feature
        overall mean across all Stroop tasks/trials in session_1 vs session_2
        ICC(3,1), two-way mixed-effects consistency, single measurement

Feature set:
    88 OpenSMILE eGeMAPSv02 Functionals + 2 timing features
    (Word_Duration, Gap_Before), total 90.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from common_paths import DEVICE_REPEATABILITY_OUTPUT_DIR, get_device_features_csv  # noqa: E402

FEATURE_CSV = get_device_features_csv()
RESULT_DIR = DEVICE_REPEATABILITY_OUTPUT_DIR

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


def build_overall_table(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    required = ["Subject_ID", "Device_ID", "Device_Name", "Session"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Feature table missing required columns: {missing}")

    work = df.copy()
    for col in required:
        work[col] = work[col].astype(str)
    work = work[work["Session"].isin(["session_1", "session_2"])].copy()

    return work.groupby(required, as_index=False)[feature_cols].mean(numeric_only=True)


def icc_3_1(x: np.ndarray, y: np.ndarray) -> float:
    """Two-way mixed, consistency, single-measure ICC: ICC(3,1)."""
    data = np.column_stack([x, y]).astype(float)
    n, k = data.shape
    if n < 2 or k != 2 or np.nanstd(data) == 0:
        return np.nan
    subject_means = data.mean(axis=1, keepdims=True)
    rater_means = data.mean(axis=0, keepdims=True)
    grand_mean = data.mean()
    ss_subject = k * np.sum((subject_means - grand_mean) ** 2)
    ss_error = np.sum((data - subject_means - rater_means + grand_mean) ** 2)
    ms_subject = ss_subject / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_subject + (k - 1) * ms_error
    return np.nan if denom == 0 else float((ms_subject - ms_error) / denom)


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


def reliability_from_pairs(values: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict] = []
    device_id = str(values["Device_ID"].iloc[0]) if len(values) else ""
    device_name = str(values["Device_Name"].iloc[0]) if len(values) else ""
    for feature in feature_cols:
        wide = values.pivot_table(index="Subject_ID", columns="Session", values=feature, aggfunc="mean")
        if "session_1" not in wide.columns or "session_2" not in wide.columns:
            continue
        pair = wide[["session_1", "session_2"]].dropna()
        n = len(pair)
        row = {
            "Device_ID": device_id,
            "Device_Name": device_name,
            "Analysis_Level": "device_overall_mean",
            "Feature": feature,
            "N": n,
        }
        if n >= 2:
            x = pair["session_1"].to_numpy(dtype=float)
            y = pair["session_2"].to_numpy(dtype=float)
            diff = y - x
            mean_pair = (x + y) / 2.0
            sd_diff = float(np.std(diff, ddof=1))
            mean_diff = float(np.mean(diff))
            icc = icc_3_1(x, y)
            pearson_r, pearson_p = safe_corr(x, y, "pearson")
            spearman_rho, spearman_p = safe_corr(x, y, "spearman")
            t_value, t_p = paired_t(x, y)
            sem = float(np.sqrt(max(0.0, 1.0 - icc)) * np.std(mean_pair, ddof=1)) if np.isfinite(icc) else np.nan
            mdc95 = float(1.96 * np.sqrt(2.0) * sem) if np.isfinite(sem) else np.nan
            row.update({
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
                "Mean_difference_S2_minus_S1": mean_diff,
                "SD_difference": sd_diff,
                "Paired_t": t_value,
                "Paired_t_p": t_p,
                "SEM": sem,
                "MDC95": mdc95,
                "Bland_Altman_bias": mean_diff,
                "Bland_Altman_lower_LOA": mean_diff - 1.96 * sd_diff,
                "Bland_Altman_upper_LOA": mean_diff + 1.96 * sd_diff,
            })
        else:
            row["ICC_Category"] = "NA"
        rows.append(row)
    return pd.DataFrame(rows)


def compute_overall(overall_table: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    parts = []
    for _, sub in overall_table.groupby("Device_ID"):
        parts.append(reliability_from_pairs(sub, feature_cols))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out.sort_values(["Device_ID", "Feature"]).reset_index(drop=True)


def write_notes(path: Path, feature_cols: List[str], overall_table: pd.DataFrame) -> None:
    timing = [c for c in TIMING_FEATURE_COLS if c in feature_cols]
    opensmile_cols = [c for c in feature_cols if c not in TIMING_FEATURE_COLS]
    counts = (
        overall_table.groupby(["Device_ID", "Device_Name", "Session"])["Subject_ID"]
        .nunique()
        .reset_index(name="N_subjects")
        .to_string(index=False)
    )
    lines = [
        "HC device Stroop within-device test-retest ICC notes",
        f"Input feature table: {FEATURE_CSV}",
        f"Overall session rows: {len(overall_table)}",
        f"Numeric features tested: {len(feature_cols)}",
        f"OpenSMILE features: {len(opensmile_cols)}",
        f"Timing features: {len(timing)} ({', '.join(timing)})",
        "",
        "Main result: within_device_feature_level_icc.csv",
        "  Each feature is first averaged across all Stroop tasks/trials for the same Subject_ID x Device_ID x Session.",
        "  ICC is computed within each device for session_1 vs session_2.",
        "",
        "No item-level or task-level ICC is computed in this script.",
        "Reason: trial order/content can differ across Stroop task sequences, so strict item-index matching is not appropriate.",
        "",
        "ICC model: ICC(3,1), two-way mixed-effects consistency, single measurement.",
        "",
        "Subject counts by device/session:",
        counts,
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def remove_old_outputs() -> None:
    old_names = [
        "B2_HC_device_retest_session_item_values.csv",
        "B2_HC_device_retest_ICC_by_device_task_item.csv",
        "B2_HC_device_retest_ICC_by_device_task.csv",
    ]
    for name in old_names:
        path = RESULT_DIR / name
        if path.exists():
            path.unlink()


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURE_CSV)
    feature_cols = numeric_feature_columns(df)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found.")

    overall_table = build_overall_table(df, feature_cols)
    overall = compute_overall(overall_table, feature_cols)

    remove_old_outputs()
    result_path = RESULT_DIR / "within_device_feature_level_icc.csv"
    notes_path = RESULT_DIR / "within_device_retest_notes.txt"
    overall.to_csv(result_path, index=False, encoding="utf-8-sig")
    write_notes(notes_path, feature_cols, overall_table)

    print(f"overall ICC saved: {result_path}")
    print(f"notes saved: {notes_path}")
    print("removed old item/task-level B2 outputs if they existed")
    print(f"features tested: {len(feature_cols)}")


if __name__ == "__main__":
    main()
