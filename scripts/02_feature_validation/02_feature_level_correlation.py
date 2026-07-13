# -*- coding: utf-8 -*-
'\nFeature-level correlation analysis between automatic and manual Stroop feature extraction.\n\n  ：\n     “  ”                 trial-level        。\n       feature            ；   Task  ，    Task × feature         。\n\n  subject-task-feature      ：\n       ：   Subject_ID × Task × feature  ，   12   trial         。\n       ：   feature  ，        Subject_ID × Task × Item_Index          。\n               158      × 3  Task × 12   trial，    feature    5688   auto-manual   。\n\n    ：\n    1) aligned_feature_values.csv\n       Automatic table Manual table  KEY_COLS          。\n\n    2) match_overview.csv\n          Subject_ID × Task    、  、  Rows。\n\n    3) feature_level_overall_correlation.csv\n       ALL     feature-level   。   = 1   feature。\n\n    4) feature_level_task_correlation.csv\n       Task-specific feature-level   。   = 1   Task × feature。\n\n    5) feature_level_correlation_long_for_plot.csv\n            ，     4B：Group = ALL/Hanzi/Dots/STT，Metric = Pearson r/Spearman rho。\n\n    6) feature_level_distribution_summary.csv\n         Group    90             。\n\n    7) feature_level_correlation_analysis.xlsx\n          Excel。\n'

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
OUTPUT_DIR = str(VALIDATION_OUTPUT_DIR / "feature_level_correlation")  # noqa: F405

TASK_ORDER = ["Hanzi", "Dots", "STT"]

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

CORR_THRESHOLDS = [0.3, 0.5, 0.7, 0.8]


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def to_numeric_pair(x, y) -> Tuple[pd.Series, pd.Series, pd.Series]:
    '     ，    auto/manual       mask。'
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = x.notna() & y.notna()
    return x, y, mask


def safe_pearson(x, y) -> Tuple[float, float]:
    '\n       Pearson r   p value。\n                     ，   nan。\n    '
    x, y, mask = to_numeric_pair(x, y)
    if mask.sum() < 3:
        return np.nan, np.nan

    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)

    if np.nanstd(xv) == 0 or np.nanstd(yv) == 0:
        return np.nan, np.nan

    if stats is not None:
        res = stats.pearsonr(xv, yv)
        return float(res.statistic), float(res.pvalue)

    return float(np.corrcoef(xv, yv)[0, 1]), np.nan


def safe_spearman(x, y) -> Tuple[float, float]:
    '\n       Spearman rho   p value。\n                     ，   nan。\n    '
    x, y, mask = to_numeric_pair(x, y)
    if mask.sum() < 3:
        return np.nan, np.nan

    xv = x[mask].to_numpy(dtype=float)
    yv = y[mask].to_numpy(dtype=float)

    if pd.Series(xv).nunique(dropna=True) < 2 or pd.Series(yv).nunique(dropna=True) < 2:
        return np.nan, np.nan

    if stats is not None:
        res = stats.spearmanr(xv, yv)
        return float(res.correlation), float(res.pvalue)

    xr = pd.Series(xv).rank(method="average").to_numpy()
    yr = pd.Series(yv).rank(method="average").to_numpy()
    if np.nanstd(xr) == 0 or np.nanstd(yr) == 0:
        return np.nan, np.nan
    return float(np.corrcoef(xr, yr)[0, 1]), np.nan


def mae(x, y) -> float:
    x, y, mask = to_numeric_pair(x, y)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(x[mask] - y[mask])))


def rmse(x, y) -> float:
    x, y, mask = to_numeric_pair(x, y)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[mask] - y[mask]) ** 2)))


def mean_bias_auto_minus_manual(x, y) -> float:
    x, y, mask = to_numeric_pair(x, y)
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(x[mask] - y[mask]))


def fisher_ci_single_r(r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    '\n       Pearson r   Fisher z     。\n      ：    n       。     trial   ，  CI         ；\n          ，                   bootstrap。\n    '
    if pd.isna(r) or n < 4:
        return np.nan, np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    zcrit = 1.96 if stats is None else stats.norm.ppf(1 - alpha / 2)
    lo = z - zcrit * se
    hi = z + zcrit * se
    return float(np.tanh(lo)), float(np.tanh(hi))


def fisher_mean_r(r_values: pd.Series) -> float:
    r_values = pd.to_numeric(r_values, errors="coerce").dropna()
    r_values = r_values[(r_values > -0.999999) & (r_values < 0.999999)]
    if len(r_values) == 0:
        return np.nan
    return float(np.tanh(np.mean(np.arctanh(r_values))))


def load_feature_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in KEY_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing key columns: {missing}；must include {KEY_COLS}")

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


def align_tables(auto_df: pd.DataFrame, manual_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    '\n      Subject_ID, Task, Item_Index           。\n              feature__auto   feature__manual。\n    '
    auto_features = get_feature_cols(auto_df)
    manual_features = get_feature_cols(manual_df)

    if ONLY_SHARED_FEATURES:
        shared_features = sorted(set(auto_features) & set(manual_features))
    else:
        shared_features = sorted(set(auto_features + manual_features))

    if len(shared_features) == 0:
        raise ValueError('English documentation.')

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

    out = (
        auto_counts
        .merge(manual_counts, on=KEY_COLS[:2], how="outer")
        .merge(matched_counts, on=KEY_COLS[:2], how="outer")
    )

    out["n_auto_rows"] = out["n_auto_rows"].fillna(0).astype(int)
    out["n_manual_rows"] = out["n_manual_rows"].fillna(0).astype(int)
    out["n_matched_rows"] = out["n_matched_rows"].fillna(0).astype(int)

    out["Task"] = pd.Categorical(out["Task"], categories=TASK_ORDER, ordered=True)
    out = out.sort_values(["Task", "Subject_ID"])
    return out


def calc_feature_level_one(df: pd.DataFrame, feature: str, group: str = "ALL") -> Dict:
    '\n    feature-level      。\n       df                 ：\n        -   ：   Subject_ID × Task × Item_Index\n        -  Task：    Task    Subject_ID × Item_Index\n       feature         。\n    '
    auto_col = f"{feature}__auto"
    manual_col = f"{feature}__manual"

    base = {
        "Group": group,
        "feature": feature,
        "n_pairs": 0,
        "n_subjects": 0,
        "n_tasks": 0,
        "n_items": 0,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "pearson_ci_lower": np.nan,
        "pearson_ci_upper": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "mae": np.nan,
        "rmse": np.nan,
        "mean_bias_auto_minus_manual": np.nan,
        "auto_mean": np.nan,
        "manual_mean": np.nan,
        "auto_sd": np.nan,
        "manual_sd": np.nan,
        "auto_min": np.nan,
        "auto_max": np.nan,
        "manual_min": np.nan,
        "manual_max": np.nan,
    }

    if auto_col not in df.columns or manual_col not in df.columns:
        return base

    x = pd.to_numeric(df[auto_col], errors="coerce")
    y = pd.to_numeric(df[manual_col], errors="coerce")
    mask = x.notna() & y.notna()

    if mask.sum() == 0:
        return base

    valid = df.loc[mask, KEY_COLS].copy()
    xv = x[mask]
    yv = y[mask]

    pearson_r, pearson_p = safe_pearson(xv, yv)
    spearman_rho, spearman_p = safe_spearman(xv, yv)
    ci_lo, ci_hi = fisher_ci_single_r(pearson_r, int(mask.sum()))

    out = base.copy()
    out.update({
        "n_pairs": int(mask.sum()),
        "n_subjects": int(valid["Subject_ID"].nunique()),
        "n_tasks": int(valid["Task"].nunique()),
        "n_items": int(valid["Item_Index"].nunique()),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "pearson_ci_lower": ci_lo,
        "pearson_ci_upper": ci_hi,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "mae": mae(xv, yv),
        "rmse": rmse(xv, yv),
        "mean_bias_auto_minus_manual": mean_bias_auto_minus_manual(xv, yv),
        "auto_mean": float(np.mean(xv)),
        "manual_mean": float(np.mean(yv)),
        "auto_sd": float(np.std(xv, ddof=1)) if len(xv) > 1 else 0.0,
        "manual_sd": float(np.std(yv, ddof=1)) if len(yv) > 1 else 0.0,
        "auto_min": float(np.min(xv)),
        "auto_max": float(np.max(xv)),
        "manual_min": float(np.min(yv)),
        "manual_max": float(np.max(yv)),
    })
    return out


def calc_overall_feature_level_stats(merged: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    'ALL   ：   feature     matched subject-task-trial          。'
    rows = [calc_feature_level_one(merged, feat, group="ALL") for feat in features]
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["spearman_rho", "pearson_r", "n_pairs"], ascending=[False, False, False])
    return out


def calc_task_feature_level_stats(merged: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    ' Task  ：   Task × feature       。'
    rows = []
    for task in TASK_ORDER:
        g = merged.loc[merged["Task"] == task].copy()
        if len(g) == 0:
            continue
        for feat in features:
            rows.append(calc_feature_level_one(g, feat, group=task))

    other_tasks = [t for t in sorted(merged["Task"].dropna().unique()) if t not in TASK_ORDER]
    for task in other_tasks:
        g = merged.loc[merged["Task"] == task].copy()
        for feat in features:
            rows.append(calc_feature_level_one(g, feat, group=task))

    out = pd.DataFrame(rows)
    if len(out):
        out["Group"] = pd.Categorical(out["Group"], categories=["ALL"] + TASK_ORDER + other_tasks, ordered=True)
        out = out.sort_values(["Group", "spearman_rho", "pearson_r"], ascending=[True, False, False])
    return out


def build_long_for_plot(overall_stats: pd.DataFrame, task_stats: pd.DataFrame) -> pd.DataFrame:
    '\n         ，    ：\n        Group: ALL/Hanzi/Dots/STT\n        Metric: Pearson r / Spearman rho\n        value:     \n    '
    all_stats = pd.concat([overall_stats, task_stats], ignore_index=True)

    keep_cols = ["Group", "feature", "n_pairs", "pearson_r", "spearman_rho"]
    all_stats = all_stats[[c for c in keep_cols if c in all_stats.columns]].copy()

    long = all_stats.melt(
        id_vars=["Group", "feature", "n_pairs"],
        value_vars=["pearson_r", "spearman_rho"],
        var_name="Metric",
        value_name="correlation",
    )
    long["Metric"] = long["Metric"].map({
        "pearson_r": "Pearson r",
        "spearman_rho": "Spearman rho",
    })
    long["Group"] = pd.Categorical(long["Group"], categories=["ALL"] + TASK_ORDER, ordered=True)
    long = long.sort_values(["Metric", "Group", "feature"])
    return long


def summarize_correlation_distribution(overall_stats: pd.DataFrame, task_stats: pd.DataFrame) -> pd.DataFrame:
    '\n        Group    feature-level       。\n       Group      n_features      ：\n        ALL: 90   feature\n        Hanzi/Dots/STT:   90   feature\n    '
    all_stats = pd.concat([overall_stats, task_stats], ignore_index=True)
    rows = []

    for group, g in all_stats.groupby("Group", dropna=False):
        row = {"Group": group, "n_features": int(g["feature"].nunique())}

        for col, label in [("pearson_r", "pearson"), ("spearman_rho", "spearman")]:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()

            row[f"{label}_n_valid"] = int(vals.shape[0])
            row[f"{label}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{label}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{label}_q1"] = float(vals.quantile(0.25)) if len(vals) else np.nan
            row[f"{label}_q3"] = float(vals.quantile(0.75)) if len(vals) else np.nan
            row[f"{label}_min"] = float(vals.min()) if len(vals) else np.nan
            row[f"{label}_max"] = float(vals.max()) if len(vals) else np.nan
            row[f"{label}_fisher_mean"] = fisher_mean_r(vals)

            for thr in CORR_THRESHOLDS:
                row[f"{label}_prop_gt_{thr}"] = float((vals > thr).mean()) if len(vals) else np.nan
                row[f"{label}_n_gt_{thr}"] = int((vals > thr).sum()) if len(vals) else 0

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out):
        out["Group"] = pd.Categorical(out["Group"], categories=["ALL"] + TASK_ORDER, ordered=True)
        out = out.sort_values("Group")
    return out


def main():
    ensure_dir(OUTPUT_DIR)

    print('        ...')
    auto_df = load_feature_table(AUTO_FEATURE_CSV)
    print(f"Automatic feature rows: {len(auto_df)}")

    print('         ...')
    manual_df = load_feature_table(MANUAL_FEATURE_CSV)
    print(f"Manual feature rows: {len(manual_df)}")

    print('  Subject_ID × Task × Item_Index        ...')
    merged, shared_features = align_tables(auto_df, manual_df)

    print(f"Comparable shared features: {len(shared_features)}")
    print(f"Matched paired rows: {len(merged)}")
    print('  ：feature-level ALL    ，                   Rows。')

    match_overview = build_match_overview(auto_df, manual_df, merged)

    print('   ALL feature-level   ：   feature       ...')
    overall_stats = calc_overall_feature_level_stats(merged, shared_features)

    print('   Task-specific feature-level   ：   Task × feature       ...')
    task_stats = calc_task_feature_level_stats(merged, shared_features)

    print('         ...')
    long_for_plot = build_long_for_plot(overall_stats, task_stats)

    print('        ...')
    dist_summary = summarize_correlation_distribution(overall_stats, task_stats)

    out_dir = Path(OUTPUT_DIR)
    merged.to_csv(out_dir / "aligned_feature_values.csv", index=False, encoding="utf-8-sig")
    match_overview.to_csv(out_dir / "match_overview.csv", index=False, encoding="utf-8-sig")
    overall_stats.to_csv(out_dir / "feature_level_overall_correlation.csv", index=False, encoding="utf-8-sig")
    task_stats.to_csv(out_dir / "feature_level_task_correlation.csv", index=False, encoding="utf-8-sig")
    long_for_plot.to_csv(out_dir / "feature_level_correlation_long_for_plot.csv", index=False, encoding="utf-8-sig")
    dist_summary.to_csv(out_dir / "feature_level_distribution_summary.csv", index=False, encoding="utf-8-sig")

    xlsx_path = out_dir / "feature_level_correlation_analysis.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        match_overview.to_excel(writer, sheet_name="match_overview", index=False)
        overall_stats.to_excel(writer, sheet_name="overall_feature_level", index=False)
        task_stats.to_excel(writer, sheet_name="task_feature_level", index=False)
        long_for_plot.to_excel(writer, sheet_name="long_for_plot", index=False)
        dist_summary.to_excel(writer, sheet_name="distribution_summary", index=False)

    print('\n  Done.    ：', OUTPUT_DIR)
    print('English documentation.')
    print(" - aligned_feature_values.csv")
    print(" - match_overview.csv")
    print(" - feature_level_overall_correlation.csv")
    print(" - feature_level_task_correlation.csv")
    print(" - feature_level_correlation_long_for_plot.csv")
    print(" - feature_level_distribution_summary.csv")
    print(" - feature_level_correlation_analysis.xlsx")

    if len(dist_summary):
        print('English documentation.')
        show_cols = [
            "Group", "n_features",
            "spearman_mean", "spearman_median", "spearman_prop_gt_0.5",
            "pearson_mean", "pearson_median", "pearson_prop_gt_0.5",
        ]
        show_cols = [c for c in show_cols if c in dist_summary.columns]
        print(dist_summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()






