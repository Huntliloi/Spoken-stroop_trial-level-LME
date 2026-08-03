"""Reproduce substance-use cohort Tables 5, S10, and S11."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from common_paths import SUBSTANCE_USE_OUTPUT_DIR, get_substance_subject_scores_csv  # noqa: E402


INPUT_CSV = get_substance_subject_scores_csv()
OUTPUT_DIR = SUBSTANCE_USE_OUTPUT_DIR / "statistical_validation"
MEASURES = {
    "Mean |Z_dev|": "Mean_Abs_Deviation_Z",
    "RMS Z_dev": "RMS_Deviation_Z",
    "N_abn,1.96": "N_Features_AbsZ_GE_1_96",
    "N_abn,2.58": "N_Features_AbsZ_GE_2_58",
}


def delong_auc_ci(labels: np.ndarray, scores: np.ndarray, alpha: float = 0.95) -> tuple[float, float, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) < 2 or len(negative) < 2:
        return np.nan, np.nan, np.nan

    comparisons = (positive[:, None] > negative[None, :]).astype(float)
    comparisons += 0.5 * (positive[:, None] == negative[None, :])
    auc = float(comparisons.mean())
    v10 = comparisons.mean(axis=1)
    v01 = comparisons.mean(axis=0)
    variance = np.var(v10, ddof=1) / len(positive) + np.var(v01, ddof=1) / len(negative)
    standard_error = np.sqrt(max(variance, 0.0))
    z = stats.norm.ppf(0.5 + alpha / 2.0)
    return auc, max(0.0, auc - z * standard_error), min(1.0, auc + z * standard_error)


def prepare_data() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")
    if "Cognitive_Group" in frame.columns and "Group" not in frame.columns:
        frame = frame.rename(columns={"Cognitive_Group": "Group"})
    education_bonus = frame["Education"].le(12).astype(int)
    frame["MoCA_Corrected"] = np.minimum(frame["MoCA"] + education_bonus, 30)
    frame["MoCA_Group"] = np.where(frame["MoCA_Corrected"].lt(26), "MoCA-CI", "MoCA-CN")
    return frame


def compare_groups(frame: pd.DataFrame, group_column: str, positive_label: str, negative_label: str) -> pd.DataFrame:
    rows = []
    raw_p = []
    for measure, column in MEASURES.items():
        positive = frame.loc[frame[group_column].eq(positive_label), column].dropna().to_numpy(float)
        negative = frame.loc[frame[group_column].eq(negative_label), column].dropna().to_numpy(float)
        _, p_value = stats.mannwhitneyu(
            positive,
            negative,
            alternative="two-sided",
            method="asymptotic",
        )
        labels = frame[group_column].eq(positive_label).astype(int).to_numpy()
        scores = frame[column].to_numpy(float)
        auc, ci_low, ci_high = delong_auc_ci(labels, scores)
        rows.append({
            "Measure": measure,
            f"{positive_label}_N": len(positive),
            f"{positive_label}_Mean": np.mean(positive),
            f"{positive_label}_SD": np.std(positive, ddof=1),
            f"{negative_label}_N": len(negative),
            f"{negative_label}_Mean": np.mean(negative),
            f"{negative_label}_SD": np.std(negative, ddof=1),
            "P_Raw": p_value,
            "AUC": auc,
            "AUC_CI_Low": ci_low,
            "AUC_CI_High": ci_high,
        })
        raw_p.append(p_value)
    adjusted = multipletests(raw_p, method="fdr_bh")[1]
    for row, p_adjusted in zip(rows, adjusted):
        row["P_FDR_BH"] = p_adjusted
    return pd.DataFrame(rows)


def build_table_s11(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for analysis_set, subset in [
        ("Overall", frame),
        ("Drug", frame[frame["Group"].eq("Drug")]),
        ("HC", frame[frame["Group"].eq("HC")]),
    ]:
        result = compare_groups(subset, "MoCA_Group", "MoCA-CI", "MoCA-CN")
        result.insert(0, "Analysis_Set", analysis_set)
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def build_table5(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for measure, column in MEASURES.items():
        row = {"Measure": measure}
        for analysis_set, subset in [
            ("Overall", frame),
            ("Drug", frame[frame["Group"].eq("Drug")]),
            ("HC", frame[frame["Group"].eq("HC")]),
        ]:
            rho, p_value = stats.spearmanr(subset[column], subset["MoCA"], nan_policy="omit")
            row[f"{analysis_set}_Rho"] = rho
            row[f"{analysis_set}_P_Raw"] = p_value
        rows.append(row)

    result = pd.DataFrame(rows)
    for analysis_set in ["Overall", "Drug", "HC"]:
        result[f"{analysis_set}_P_FDR_BH"] = multipletests(
            result[f"{analysis_set}_P_Raw"], method="fdr_bh"
        )[1]
    return result


def main() -> None:
    frame = prepare_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table_s10_drug_vs_hc.csv": compare_groups(frame, "Group", "Drug", "HC"),
        "table_s11_moca_groups.csv": build_table_s11(frame),
        "table5_moca_correlations.csv": build_table5(frame),
    }
    for filename, result in outputs.items():
        path = OUTPUT_DIR / filename
        result.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
