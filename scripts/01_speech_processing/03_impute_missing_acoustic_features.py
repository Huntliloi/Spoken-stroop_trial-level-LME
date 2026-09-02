import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd



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
INPUT_CSV = str(ACOUSTIC_FEATURES_CSV)  # noqa: F405
OUTPUT_CSV = str(IMPUTED_ACOUSTIC_FEATURES_CSV)  # noqa: F405

BASIC_COLS = [
    "Subject_ID", "Task", "Item_Index",
    "Absolute_Word_Start", "Absolute_Word_End",
    "Duration", "Gap_Before", "Auto_Word"
]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in BASIC_COLS]


def summarize_missing(df: pd.DataFrame, feature_cols: List[str], title: str):
    total_missing = int(df[feature_cols].isna().sum().sum())
    total_cells = int(len(df) * len(feature_cols))
    pct = 100 * total_missing / total_cells if total_cells > 0 else 0
    print(f"\n[{title}]")
    print(f"Rows: {len(df)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Missing values: {total_missing} / {total_cells} ({pct:.3f}%)")


def impute_opensmile_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    feature_cols = get_feature_columns(df)
    summarize_missing(df, feature_cols, "Before imputation")

    original_missing_mask = df[feature_cols].isna().copy()

    df["Item_Index"] = pd.to_numeric(df["Item_Index"], errors="coerce")
    df = df.sort_values(["Subject_ID", "Task", "Item_Index"]).reset_index(drop=True)

    # =========
    # =========
    def interp_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy().sort_values("Item_Index")

        g[feature_cols] = g[feature_cols].interpolate(
            method="linear",
            axis=0,
            limit_direction="both"
        )

        g[feature_cols] = g[feature_cols].ffill().bfill()

        return g

    df = (
        df.groupby(["Subject_ID", "Task"], group_keys=False)
        .apply(interp_group)
        .reset_index(drop=True)
    )

    # =========
    # =========
    for task, idx in df.groupby("Task").groups.items():
        sub = df.loc[idx, feature_cols]
        medians = sub.median(axis=0, skipna=True)
        df.loc[idx, feature_cols] = sub.fillna(medians)

    # =========
    # =========
    global_medians = df[feature_cols].median(axis=0, skipna=True)
    df[feature_cols] = df[feature_cols].fillna(global_medians)

    summarize_missing(df, feature_cols, "After imputation")

    # =========
    # =========
    final_missing_mask = df[feature_cols].isna()

    df["n_missing_features_before"] = original_missing_mask.sum(axis=1).values

    df["n_missing_features_after"] = final_missing_mask.sum(axis=1).values

    df["any_feature_imputed"] = (df["n_missing_features_before"] > 0).astype(int)

    return df


def main():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input file not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    if df.empty:
        raise ValueError("Input table is empty.")

    required_cols = ["Subject_ID", "Task", "Item_Index"]
    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    df_imputed = impute_opensmile_features(df)

    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_imputed.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nImputation complete. Saved to:\n{OUTPUT_CSV}")


if __name__ == "__main__":
    main()



