# -*- coding: utf-8 -*-
'\n  -       ：         \n\n  ：\n1.   Task       （   Subject_ID × Task   ）\n2.          trial-level      \n3.          Subject_ID × Task   \n4.    Task      ：\n   - STT_minus_Dots\n   - STT_minus_Hanzi\n   - Dots_minus_Hanzi\n5.          ×        delta    Spearman   \n6.     pair          FDR-BH   \n7.    coupling_*.csv，            \n\n  ：\n-         ，   RESULT_DIR         。\n- SPEECH_CSV            。\n'

import os
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests



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

BEHAVIOR_CSV = str(BEHAVIORAL_METRICS_CSV)  # noqa: F405

SPEECH_CSV = str(get_imputed_acoustic_features_csv())  # noqa: F405

RESULT_DIR = str(BEHAVIOR_OUTPUT_DIR / "coupling_results")  # noqa: F405
os.makedirs(RESULT_DIR, exist_ok=True)


# =========================
# =========================

TASK_PAIRS = {
    "STT_minus_Dots": ("STT", "Dots"),
    "STT_minus_Hanzi": ("STT", "Hanzi"),
    "Dots_minus_Hanzi": ("Dots", "Hanzi"),
}

BEHAVIOR_METRICS = [
    "Overall_Accuracy",
    "Total_Hesitation_Duration",
]

ALPHA = 0.05

NON_SPEECH_FEATURE_COLS = {
    "Subject_ID", "Task", "Item_Index", "Auto_Word", "Word",
    "Word_Start", "Word_End",
    "Absolute_Word_Start", "Absolute_Word_End",
    "Absolute_Word_Start_x", "Absolute_Word_End_x",
    "Absolute_Word_Start_y", "Absolute_Word_End_y",
    "Valid_Trial_Count", "Correct_Trial_Count",
    "Overall_Accuracy", "Total_Hesitation_Duration",
    "n_missing_features_before", "n_missing_features_after", "any_feature_imputed",
}


# =========================
# =========================

def normalize_task_name(x):
    x0 = str(x).strip()
    xl = x0.lower()
    if xl == "stt":
        return "STT"
    if xl in ["dots", "dot"]:
        return "Dots"
    if xl in ["hanzi", "hz"]:
        return "Hanzi"
    return x0


def standardize_basic_columns(df):
    '   Subject_ID / Task / Item_Index      。'
    df = df.copy()
    rename_map = {}
    for c in df.columns:
        cl = str(c).strip()
        if cl in ["subject_id", "Subject_ID", "subject", "ID", "Participant_ID", "Subject_ID"]:
            rename_map[c] = "Subject_ID"
        elif cl in ["task", "Task"]:
            rename_map[c] = "Task"
        elif cl in ["Item_Index", "item", "trial", "Trial", "item_index"]:
            rename_map[c] = "Item_Index"
    df = df.rename(columns=rename_map)

    if "Subject_ID" not in df.columns:
        raise ValueError('      Subject_ID  。')
    if "Task" not in df.columns:
        raise ValueError('      Task  。')

    df["Subject_ID"] = df["Subject_ID"].astype(str).str.strip()
    df["Task"] = df["Task"].astype(str).str.strip().apply(normalize_task_name)
    return df


def infer_speech_feature_cols(speech_df):
    'English documentation.'
    feature_cols = []
    for c in speech_df.columns:
        if c in NON_SPEECH_FEATURE_COLS:
            continue
        if "Unnamed" in str(c):
            continue
        if pd.api.types.is_numeric_dtype(speech_df[c]):
            feature_cols.append(c)
    return feature_cols


def aggregate_speech_to_task_level(speech_df, feature_cols):
    'trial-level        -> Subject_ID × Task   Mean。'
    return (
        speech_df
        .groupby(["Subject_ID", "Task"], as_index=False)[feature_cols]
        .mean()
    )


def make_delta(df_task_level, pair, value_cols):
    '  Task  ：pair[0] - pair[1]。'
    t1, t2 = pair
    tmp = df_task_level[df_task_level["Task"].isin([t1, t2])].copy()

    wide = tmp.pivot(index="Subject_ID", columns="Task", values=value_cols)
    wide.columns = [f"{m}__{t}" for m, t in wide.columns]
    wide = wide.reset_index()

    out = wide[["Subject_ID"]].copy()
    for m in value_cols:
        c1 = f"{m}__{t1}"
        c2 = f"{m}__{t2}"
        if c1 in wide.columns and c2 in wide.columns:
            out[f"Delta_{m}"] = wide[c1] - wide[c2]
    return out


def safe_spearman(x, y):
    '     Spearman   ，            。'
    dat = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(dat) < 3:
        return np.nan, np.nan, len(dat), "insufficient_n"
    if dat["x"].nunique() < 2 or dat["y"].nunique() < 2:
        return np.nan, np.nan, len(dat), "constant_input"
    rho, p = spearmanr(dat["x"], dat["y"])
    return rho, p, len(dat), "ok"


def apply_fdr(df, p_col="p_raw", alpha=0.05):
    df = df.copy()
    df["p_fdr_bh"] = np.nan
    df["Significant_FDR"] = False

    valid = df[p_col].notna()
    if valid.any():
        reject, p_corr, _, _ = multipletests(
            df.loc[valid, p_col].astype(float).values,
            alpha=alpha,
            method="fdr_bh"
        )
        df.loc[valid, "p_fdr_bh"] = p_corr
        df.loc[valid, "Significant_FDR"] = reject
    return df


# =========================
# =========================

def main():
    print('1.                ...')
    behavior_df = pd.read_csv(BEHAVIOR_CSV, encoding="utf-8-sig")
    speech_df = pd.read_csv(SPEECH_CSV, encoding="utf-8-sig")

    behavior_df = standardize_basic_columns(behavior_df)
    speech_df = standardize_basic_columns(speech_df)

    missing_behavior = [m for m in BEHAVIOR_METRICS if m not in behavior_df.columns]
    if missing_behavior:
        raise ValueError(f"Behavior table missing metric columns: {missing_behavior}")

    feature_cols = infer_speech_feature_cols(speech_df)
    if not feature_cols:
        raise ValueError('           ，    SPEECH_CSV。')

    print(f"   -> Behavior rows: {len(behavior_df)}")
    print(f"   -> Automatic speech trial-level rows: {len(speech_df)}")
    print(f"   -> Detected speech feature count: {len(feature_cols)}")

    print('2.         Subject_ID × Task   ...')
    speech_task_df = aggregate_speech_to_task_level(speech_df, feature_cols)
    print(f"   -> Aggregated speech task-level rows: {len(speech_task_df)}")

    print('3.     Task    ...')
    summary_rows = []

    for pair_name, pair_tuple in TASK_PAIRS.items():
        print(f"\n=== {pair_name}: {pair_tuple[0]} - {pair_tuple[1]} ===")

        delta_behavior = make_delta(
            behavior_df[["Subject_ID", "Task"] + BEHAVIOR_METRICS],
            pair_tuple,
            BEHAVIOR_METRICS
        )
        delta_speech = make_delta(
            speech_task_df[["Subject_ID", "Task"] + feature_cols],
            pair_tuple,
            feature_cols
        )

        merged = pd.merge(delta_behavior, delta_speech, on="Subject_ID", how="inner")
        print(f"   -> Subjects available for correlation: {merged['Subject_ID'].nunique()}")

        rows = []
        for beh in BEHAVIOR_METRICS:
            xcol = f"Delta_{beh}"
            if xcol not in merged.columns:
                continue

            for feat in feature_cols:
                ycol = f"Delta_{feat}"
                if ycol not in merged.columns:
                    continue

                rho, p, n, status = safe_spearman(merged[xcol], merged[ycol])
                rows.append({
                    "Pair": pair_name,
                    "Task_1": pair_tuple[0],
                    "Task_2": pair_tuple[1],
                    "Behavior_Metric": beh,
                    "Speech_Feature": feat,
                    "N": n,
                    "Spearman_rho": rho,
                    "p_raw": p,
                    "Status": status,
                })

        pair_df = pd.DataFrame(rows)
        pair_df = apply_fdr(pair_df, p_col="p_raw", alpha=ALPHA)
        pair_df["Significant_Raw"] = pair_df["p_raw"] < ALPHA
        pair_df = pair_df.sort_values(["p_fdr_bh", "p_raw"], na_position="last")

        out_path = os.path.join(RESULT_DIR, f"coupling_{pair_name}.csv")
        pair_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"   -> Saved: {out_path}")
        print(f"   -> Raw p < .05 count: {int(pair_df['Significant_Raw'].sum())}")
        print(f"   -> FDR-significant count: {int(pair_df['Significant_FDR'].sum())}")

        summary_rows.append({
            "Pair": pair_name,
            "Task_1": pair_tuple[0],
            "Task_2": pair_tuple[1],
            "N_Subjects": merged["Subject_ID"].nunique(),
            "N_Speech_Features": len(feature_cols),
            "N_Behavior_Metrics": len(BEHAVIOR_METRICS),
            "N_Tests": len(pair_df),
            "Raw_Significant_Count": int(pair_df["Significant_Raw"].sum()),
            "FDR_Significant_Count": int(pair_df["Significant_FDR"].sum()),
            "FDR_Method": "Benjamini-Hochberg",
            "Alpha": ALPHA,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULT_DIR, "coupling_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\nDone. Summary table saved: {summary_path}")


if __name__ == "__main__":
    main()





