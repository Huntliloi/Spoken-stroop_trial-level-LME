
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:
    stats = None



# Public repository path setup
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_SCRIPTS_DIR = _RepoPath(__file__).resolve().parents[2] / "scripts"
if str(_REPO_SCRIPTS_DIR) not in _repo_sys.path:
    _repo_sys.path.insert(0, str(_REPO_SCRIPTS_DIR))
from common_paths import *  # noqa: F403
ensure_output_dirs()
# =========================
# =========================
AUTO_FEATURE_CSV = str(get_imputed_acoustic_features_csv())  # noqa: F405
MANUAL_FEATURE_CSV = str(PUBLIC_PRAAT_MANUAL_FEATURE_CSV)  # noqa: F405
OUTPUT_DIR = str(VALIDATION_OUTPUT_DIR / "subject_task_feature_correlation")  # noqa: F405

TASK_ORDER = ["W", "C", "CW"]

ONLY_SHARED_FEATURES = True

KEY_COLS = ["Subject_ID", "Task", "Item_Index"]

NON_FEATURE_COLS = {
    "Subject_ID", "Task", "Item_Index",
    "Absolute_Word_Start", "Absolute_Word_End",
    "Auto_Word", "Manual_Word",
    "Word", "Label", "Source",
}

FEATURE_WHITELIST = None
# FEATURE_WHITELIST = [
#     "loudness_sma3_amean",
#     "F0semitoneFrom27.5Hz_sma3nz_amean",
# ]


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def maybe_pearson(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def maybe_spearman(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan

    xv = x[mask]
    yv = y[mask]

    if xv.nunique(dropna=True) < 2 or yv.nunique(dropna=True) < 2:
        return np.nan

    if stats is not None:
        return float(stats.spearmanr(xv, yv).correlation)

    xr = xv.rank(method="average")
    yr = yv.rank(method="average")
    return float(np.corrcoef(xr, yr)[0, 1])


def mae(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(x[mask] - y[mask])))


def rmse(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[mask] - y[mask]) ** 2)))


def mean_bias(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(x[mask] - y[mask]))


def fisher_ci_r(r_values: pd.Series, alpha: float = 0.05) -> Tuple[float, float]:
    r_values = pd.to_numeric(r_values, errors="coerce").dropna()
    r_values = r_values[(r_values > -0.999999) & (r_values < 0.999999)]
    n = len(r_values)
    if n < 2:
        return np.nan, np.nan
    z = np.arctanh(r_values)
    mean_z = np.mean(z)
    se = np.std(z, ddof=1) / np.sqrt(n)
    zcrit = 1.96 if stats is None else stats.norm.ppf(1 - alpha / 2)
    lo = mean_z - zcrit * se
    hi = mean_z + zcrit * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def fisher_mean_r(r_values: pd.Series) -> float:
    r_values = pd.to_numeric(r_values, errors="coerce").dropna()
    r_values = r_values[(r_values > -0.999999) & (r_values < 0.999999)]
    if len(r_values) == 0:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(r_values))))


def load_feature_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not all(col in df.columns for col in KEY_COLS):
        raise ValueError(f"{path} missing key columns; must include: {KEY_COLS}")
    df["Subject_ID"] = df["Subject_ID"].astype(str)
    df["Task"] = df["Task"].astype(str)
    df["Item_Index"] = pd.to_numeric(df["Item_Index"], errors="coerce").astype("Int64")
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if c in NON_FEATURE_COLS:
            continue
        if c in KEY_COLS:
            continue
        cols.append(c)
    if FEATURE_WHITELIST is not None:
        cols = [c for c in cols if c in FEATURE_WHITELIST]
    return cols


def align_tables(auto_df: pd.DataFrame, manual_df: pd.DataFrame):
    auto_features = get_feature_cols(auto_df)
    manual_features = get_feature_cols(manual_df)

    if ONLY_SHARED_FEATURES:
        shared_features = sorted(set(auto_features) & set(manual_features))
    else:
        shared_features = sorted(set(auto_features + manual_features))

    if len(shared_features) == 0:
        raise ValueError("No shared acoustic feature columns were found.")

    auto_keep = KEY_COLS + [c for c in shared_features if c in auto_df.columns]
    manual_keep = KEY_COLS + [c for c in shared_features if c in manual_df.columns]

    auto_sub = auto_df[auto_keep].copy()
    manual_sub = manual_df[manual_keep].copy()

    auto_renamed = auto_sub.rename(columns={c: f"{c}__auto" for c in shared_features if c in auto_sub.columns})
    manual_renamed = manual_sub.rename(columns={c: f"{c}__manual" for c in shared_features if c in manual_sub.columns})

    merged = pd.merge(
        auto_renamed,
        manual_renamed,
        on=KEY_COLS,
        how="inner",
        validate="one_to_one",
    )

    return merged, shared_features


def calc_feature_stats_one(df: pd.DataFrame, feature: str) -> Dict:
    auto_col = f"{feature}__auto"
    manual_col = f"{feature}__manual"

    if auto_col not in df.columns or manual_col not in df.columns:
        return {
            "feature": feature,
            "n": 0,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_bias_auto_minus_manual": np.nan,
            "auto_mean": np.nan,
            "manual_mean": np.nan,
            "auto_sd": np.nan,
            "manual_sd": np.nan,
        }

    x = pd.to_numeric(df[auto_col], errors="coerce")
    y = pd.to_numeric(df[manual_col], errors="coerce")
    mask = x.notna() & y.notna()

    if mask.sum() == 0:
        return {
            "feature": feature,
            "n": 0,
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "mean_bias_auto_minus_manual": np.nan,
            "auto_mean": np.nan,
            "manual_mean": np.nan,
            "auto_sd": np.nan,
            "manual_sd": np.nan,
        }

    xv = x[mask]
    yv = y[mask]

    return {
        "feature": feature,
        "n": int(mask.sum()),
        "pearson_r": maybe_pearson(xv, yv),
        "spearman_rho": maybe_spearman(xv, yv),
        "mae": mae(xv, yv),
        "rmse": rmse(xv, yv),
        "mean_bias_auto_minus_manual": mean_bias(xv, yv),
        "auto_mean": float(np.mean(xv)),
        "manual_mean": float(np.mean(yv)),
        "auto_sd": float(np.std(xv, ddof=1)) if len(xv) > 1 else 0.0,
        "manual_sd": float(np.std(yv, ddof=1)) if len(yv) > 1 else 0.0,
    }


def calc_overall_feature_stats(merged: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        rows.append(calc_feature_stats_one(merged, feat))
    out = pd.DataFrame(rows)
    out = out.sort_values(["pearson_r", "spearman_rho", "n"], ascending=[False, False, False])
    return out


def calc_task_feature_stats(merged: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    for task, g in merged.groupby("Task", dropna=False):
        for feat in features:
            row = calc_feature_stats_one(g, feat)
            row["Task"] = task
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["Task"] = pd.Categorical(out["Task"], categories=TASK_ORDER, ordered=True)
        out = out.sort_values(["Task", "pearson_r"], ascending=[True, False])
    return out


def calc_subject_feature_stats(merged: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []
    group_cols = ["Subject_ID", "Task"]
    for (subject_id, task), g in merged.groupby(group_cols, dropna=False):
        for feat in features:
            row = calc_feature_stats_one(g, feat)
            row["Subject_ID"] = subject_id
            row["Task"] = task
            rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["Task"] = pd.Categorical(out["Task"], categories=TASK_ORDER, ordered=True)
        out = out.sort_values(["Task", "Subject_ID", "feature"])
    return out


def summarize_feature_across_subjects(subject_stats: pd.DataFrame) -> pd.DataFrame:
    """Summarize subject-level feature agreement within each task."""
    rows = []
    if len(subject_stats) == 0:
        return pd.DataFrame()

    for (task, feat), g in subject_stats.groupby(["Task", "feature"], dropna=False):
        rs = pd.to_numeric(g["pearson_r"], errors="coerce").dropna()
        rhos = pd.to_numeric(g["spearman_rho"], errors="coerce").dropna()

        row = {
            "Task": task,
            "feature": feat,
            "n_subjects": int(rs.shape[0]),
            "pearson_r_mean": float(rs.mean()) if len(rs) else np.nan,
            "pearson_r_median": float(rs.median()) if len(rs) else np.nan,
            "pearson_r_q1": float(rs.quantile(0.25)) if len(rs) else np.nan,
            "pearson_r_q3": float(rs.quantile(0.75)) if len(rs) else np.nan,
            "pearson_r_fisher_mean": fisher_mean_r(rs),
        }

        lo, hi = fisher_ci_r(rs)
        row["pearson_r_ci_lower"] = lo
        row["pearson_r_ci_upper"] = hi

        row["spearman_rho_n_subjects"] = int(rhos.shape[0])
        row["spearman_rho_mean"] = float(rhos.mean()) if len(rhos) else np.nan
        row["spearman_rho_median"] = float(rhos.median()) if len(rhos) else np.nan
        row["spearman_rho_q1"] = float(rhos.quantile(0.25)) if len(rhos) else np.nan
        row["spearman_rho_q3"] = float(rhos.quantile(0.75)) if len(rhos) else np.nan
        row["spearman_rho_fisher_mean"] = fisher_mean_r(rhos)
        rho_lo, rho_hi = fisher_ci_r(rhos)
        row["spearman_rho_ci_lower"] = rho_lo
        row["spearman_rho_ci_upper"] = rho_hi

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out["Task"] = pd.Categorical(out["Task"], categories=TASK_ORDER, ordered=True)
        out = out.sort_values(["Task", "pearson_r_fisher_mean"], ascending=[True, False])
    return out


def build_match_overview(auto_df: pd.DataFrame, manual_df: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    auto_counts = (
        auto_df.groupby(KEY_COLS[:2], dropna=False)
        .size()
        .reset_index(name="n_auto_rows")
    )
    manual_counts = (
        manual_df.groupby(KEY_COLS[:2], dropna=False)
        .size()
        .reset_index(name="n_manual_rows")
    )
    matched_counts = (
        merged.groupby(KEY_COLS[:2], dropna=False)
        .size()
        .reset_index(name="n_matched_rows")
    )

    out = auto_counts.merge(manual_counts, on=KEY_COLS[:2], how="outer").merge(matched_counts, on=KEY_COLS[:2], how="outer")
    out["n_auto_rows"] = out["n_auto_rows"].fillna(0).astype(int)
    out["n_manual_rows"] = out["n_manual_rows"].fillna(0).astype(int)
    out["n_matched_rows"] = out["n_matched_rows"].fillna(0).astype(int)
    out["Task"] = pd.Categorical(out["Task"], categories=TASK_ORDER, ordered=True)
    out = out.sort_values(["Task", "Subject_ID"])
    return out


def main():
    ensure_dir(OUTPUT_DIR)

    print("Loading automatic feature table...")
    auto_df = load_feature_table(AUTO_FEATURE_CSV)
    print("Loading Praat/manual feature table...")
    manual_df = load_feature_table(MANUAL_FEATURE_CSV)

    print("Aligning automatic and manual item-level rows...")
    merged, shared_features = align_tables(auto_df, manual_df)

    print(f"Comparable shared features: {len(shared_features)}")
    print(f"Matched rows: {len(merged)}")

    print("Computing subject-task-feature correlations...")
    subject_stats = calc_subject_feature_stats(merged, shared_features)

    print("Summarizing correlations across subjects...")
    group_subject_summary = summarize_feature_across_subjects(subject_stats)

    group_subject_summary.to_csv(Path(OUTPUT_DIR) / "subject_task_feature_correlation_summary.csv", index=False, encoding="utf-8-sig")

    print("\nDone. Results saved to:", OUTPUT_DIR)
    print(" - subject_task_feature_correlation_summary.csv")


if __name__ == "__main__":
    main()






