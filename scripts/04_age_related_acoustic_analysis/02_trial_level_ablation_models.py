import os
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter("ignore", ConvergenceWarning)



# Public repository path setup
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_SCRIPTS_DIR = _RepoPath(__file__).resolve().parents[2] / "scripts"
if str(_REPO_SCRIPTS_DIR) not in _repo_sys.path:
    _repo_sys.path.insert(0, str(_REPO_SCRIPTS_DIR))
from common_paths import *  # noqa: F403
ensure_output_dirs()
FEATURE_PATH = str(get_imputed_acoustic_features_csv())  # noqa: F405
INFO_PATH = str(PARTICIPANT_INFO_XLSX)  # noqa: F405

COMPARISON = "Dots_vs_Hanzi"
# "STT_vs_Hanzi"
# "STT_vs_Dots"
# "Dots_vs_Hanzi"

TARGET_FEATURES = "ALL"   # or provide a list, for example ["Duration", "Gap_Before"]
ALPHA = 0.05

COMPARISON_CONFIG = {
    "STT_vs_Hanzi": {
        "tasks": ["Hanzi", "STT"],
        "task_code_map": {"Hanzi": 0, "STT": 1},
        "base_task": "Hanzi",
        "contrast_task": "STT",
        "output_dir": "Result_Utterance_vs_Trial_STT_vs_Hanzi",
        "utterance_output_file": "Utterance_Level_STT_vs_Hanzi.csv",
        "trial_output_file": "Trial_Level_STT_vs_Hanzi.csv",
        "merged_output_file": "Utterance_vs_Trial_Comparison_STT_vs_Hanzi.csv",
        "summary_output_file": "Summary_STT_vs_Hanzi.csv",
    },
    "STT_vs_Dots": {
        "tasks": ["Dots", "STT"],
        "task_code_map": {"Dots": 0, "STT": 1},
        "base_task": "Dots",
        "contrast_task": "STT",
        "output_dir": "Result_Utterance_vs_Trial_STT_vs_Dots",
        "utterance_output_file": "Utterance_Level_STT_vs_Dots.csv",
        "trial_output_file": "Trial_Level_STT_vs_Dots.csv",
        "merged_output_file": "Utterance_vs_Trial_Comparison_STT_vs_Dots.csv",
        "summary_output_file": "Summary_STT_vs_Dots.csv",
    },
    "Dots_vs_Hanzi": {
        "tasks": ["Hanzi", "Dots"],
        "task_code_map": {"Hanzi": 0, "Dots": 1},
        "base_task": "Hanzi",
        "contrast_task": "Dots",
        "output_dir": "Result_Utterance_vs_Trial_Dots_vs_Hanzi",
        "utterance_output_file": "Utterance_Level_Dots_vs_Hanzi.csv",
        "trial_output_file": "Trial_Level_Dots_vs_Hanzi.csv",
        "merged_output_file": "Utterance_vs_Trial_Comparison_Dots_vs_Hanzi.csv",
        "summary_output_file": "Summary_Dots_vs_Hanzi.csv",
    }
}

cfg = COMPARISON_CONFIG[COMPARISON]
OUTPUT_DIR = str(LME_OUTPUT_DIR / cfg["output_dir"])  # noqa: F405
os.makedirs(OUTPUT_DIR, exist_ok=True)


def translate_opensmile_feature(name):
    if name == "Gap_Before":
        return "Hesitation duration (response latency)"
    if name in ("Word_Duration", "Duration"):
        return "Word duration (articulation lengthening)"

    stat_cn = ""
    if "amean" in name:
        stat_cn = "Mean"
    elif "stddevNorm" in name:
        stat_cn = "Coefficient of variation"
    elif "stddev" in name:
        stat_cn = "Standard deviation"
    elif "percentile20" in name:
        stat_cn = "20th percentile"
    elif "percentile50" in name:
        stat_cn = "Median"
    elif "percentile80" in name:
        stat_cn = "80th percentile"
    elif "pctlrange" in name:
        stat_cn = "Percentile range"
    elif "meanSegLen" in name:
        stat_cn = "Mean segment length"
    elif "meanRisingSlope" in name:
        stat_cn = "Mean rising slope"
    elif "meanFallingSlope" in name:
        stat_cn = "Mean falling slope"
    elif "PerSecond" in name:
        stat_cn = "Events per second"

    base_cn = name.split("_")[0]
    desc = ""
    if "F0semitone" in name:
        base_cn = "F0 semitone"; desc = "pitch"
    elif "jitter" in name.lower():
        base_cn = "Jitter"; desc = "roughness"
    elif "F1" in name:
        base_cn = "First formant"; desc = "mouth opening"
    elif "F2" in name:
        base_cn = "Second formant"; desc = "tongue position"
    elif "F3" in name:
        base_cn = "Third formant"; desc = "timbre"
    elif "loudness" in name.lower():
        base_cn = "Loudness"; desc = "intensity"
    elif "shimmer" in name.lower():
        base_cn = "Shimmer"; desc = "hoarseness"
    elif "HNR" in name:
        base_cn = "Harmonics-to-noise ratio"; desc = "voice clarity"
    elif "alphaRatio" in name:
        base_cn = "Alpha ratio"; desc = "energy distribution"
    elif "hammarbergIndex" in name.lower():
        base_cn = "Hammarberg index"; desc = "breathiness"
    elif "spectralSlope" in name:
        base_cn = "Spectral slope"; desc = 'timbre  '
    elif "mfcc" in name.lower():
        num = re.search(r"mfcc(\d+)", name.lower())
        idx = num.group(1) if num else ""
        base_cn = f"MFCC coefficient {idx}"; desc = 'spectral-envelope representation'
    elif "spectralFlux" in name:
        base_cn = "Spectral flux"; desc = 'timbre  '

    if stat_cn:
        res = f"{base_cn}-{stat_cn}"
        if desc:
            res += f" [{desc}]"
        return res
    return name


def clean_feature_name(name):
    return f'Q("{name}")' if re.search(r'[.\-:]', name) else name


def apply_fdr(df, p_col="Interaction_PValue_Raw", alpha=0.05):
    df = df.copy()
    df["FDR_PValue"] = np.nan
    df["Significant_FDR"] = False

    valid_mask = df[p_col].notna()
    if valid_mask.sum() == 0:
        return df

    pvals = df.loc[valid_mask, p_col].astype(float).values
    _, pvals_fdr, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    df.loc[valid_mask, "FDR_PValue"] = pvals_fdr
    df.loc[valid_mask, "Significant_FDR"] = df.loc[valid_mask, "FDR_PValue"] < alpha
    return df


if __name__ == "__main__":
    print("1. Reading participant information and acoustic features...")
    df_info = pd.read_excel(INFO_PATH)
    df_info["Subject_ID"] = df_info["Subject_ID"].astype(str)

    df_audio = pd.read_csv(FEATURE_PATH)
    df_audio["Subject_ID"] = df_audio["Subject_ID"].astype(str)

    df = pd.merge(df_audio, df_info, left_on="Subject_ID", right_on="Subject_ID", how="inner")
    print(f"   -> Merged rows: {len(df)} rows")

    print("2. Preprocessing data...")
    df_trial = df[df["Task"].isin(cfg["tasks"])].copy()
    df_trial.dropna(subset=["Age"], inplace=True)
    df_trial["Task_Code"] = df_trial["Task"].map(cfg["task_code_map"])

    age_scaler = StandardScaler()
    df_trial["Age_Z"] = age_scaler.fit_transform(df_trial[["Age"]])

    meta_cols = [
        "Subject_ID", "Task", "Item_Index", "Word", "Subject_ID", "Name", "Sex", "Age", "Education_Years", "Task_Code",
        "Age_Z", "Index", "IQCODE",
        "Speech_Test_Noise_dBA", "Participant_Fee",
        "Absolute_Word_Start", "Absolute_Word_End",
        "n_missing_features_before", "n_missing_features_after", "any_feature_imputed"
    ]
    numeric_df = df_trial.select_dtypes(include=np.number)
    candidate_features = [c for c in numeric_df.columns if c not in meta_cols and "Unnamed" not in c]

    run_list = candidate_features if TARGET_FEATURES == "ALL" else [f for f in TARGET_FEATURES if f in df_trial.columns]
    if not run_list:
        raise ValueError("No modelable features were found. Check FEATURE_PATH or TARGET_FEATURES.")

    feat_scaler = StandardScaler()
    df_trial[run_list] = feat_scaler.fit_transform(df_trial[run_list])

    df_utterance = df_trial.groupby(["Subject_ID", "Task_Code", "Age_Z"], as_index=False)[run_list].mean()

    print(f"   -> Selected {len(run_list)} features")
    print(f"   -> Trial-level sample size: {len(df_trial)}")
    print(f"   -> Utterance-level sample size: {len(df_utterance)}")

    utterance_results = []
    trial_results = []

    print('\n===     ：    Utterance-level vs Trial-by-Trial ===')
    print(f"Comparison: {COMPARISON} ({cfg['base_task']} vs {cfg['contrast_task']})")
    print('FDR   ：                 p      Benjamini-Hochberg   ')

    for i, feature in enumerate(run_list):
        print(f"[{i+1}/{len(run_list)}] {feature}")
        cleaned_feature = clean_feature_name(feature)

        u_row = {
            "Feature": feature,
            "English_Description": translate_opensmile_feature(feature),
            "Model": "Utterance-level",
            "N": np.nan,
            "Interaction_Coef": np.nan,
            "Interaction_PValue_Raw": np.nan,
            "Age_Main_PValue": np.nan,
            "Task_Main_PValue": np.nan,
            "Status": "failed"
        }
        try:
            dat_u = df_utterance.dropna(subset=[feature]).copy()
            if not dat_u.empty and dat_u["Task_Code"].nunique() == 2:
                mod_u = smf.ols(f"{cleaned_feature} ~ Age_Z * Task_Code", data=dat_u).fit()
                u_row["N"] = len(dat_u)
                u_row["Interaction_Coef"] = mod_u.params.get("Age_Z:Task_Code", np.nan)
                u_row["Interaction_PValue_Raw"] = mod_u.pvalues.get("Age_Z:Task_Code", np.nan)
                u_row["Age_Main_PValue"] = mod_u.pvalues.get("Age_Z", np.nan)
                u_row["Task_Main_PValue"] = mod_u.pvalues.get("Task_Code", np.nan)
                u_row["Status"] = "ok"
            else:
                u_row["Status"] = "insufficient_data"
        except Exception as e:
            u_row["Status"] = f"failed: {e}"
        utterance_results.append(u_row)

        t_row = {
            "Feature": feature,
            "English_Description": translate_opensmile_feature(feature),
            "Model": "Trial-by-Trial",
            "N": np.nan,
            "Interaction_Coef": np.nan,
            "Interaction_PValue_Raw": np.nan,
            "Age_Main_PValue": np.nan,
            "Task_Main_PValue": np.nan,
            "Status": "failed"
        }
        try:
            dat_t = df_trial.dropna(subset=[feature]).copy()
            if not dat_t.empty and dat_t["Task_Code"].nunique() == 2:
                mod_t = smf.mixedlm(
                    f"{cleaned_feature} ~ Age_Z * Task_Code",
                    dat_t,
                    groups=dat_t["Subject_ID"],
                    re_formula="~Task_Code"
                ).fit(method=["lbfgs"])
                t_row["N"] = len(dat_t)
                t_row["Interaction_Coef"] = mod_t.params.get("Age_Z:Task_Code", np.nan)
                t_row["Interaction_PValue_Raw"] = mod_t.pvalues.get("Age_Z:Task_Code", np.nan)
                t_row["Age_Main_PValue"] = mod_t.pvalues.get("Age_Z", np.nan)
                t_row["Task_Main_PValue"] = mod_t.pvalues.get("Task_Code", np.nan)
                t_row["Status"] = "ok"
            else:
                t_row["Status"] = "insufficient_data"
        except Exception as e:
            t_row["Status"] = f"failed: {e}"
        trial_results.append(t_row)

    utterance_df = pd.DataFrame(utterance_results)
    trial_df = pd.DataFrame(trial_results)

    utterance_df = apply_fdr(utterance_df, p_col="Interaction_PValue_Raw", alpha=ALPHA)
    trial_df = apply_fdr(trial_df, p_col="Interaction_PValue_Raw", alpha=ALPHA)

    utterance_df["Significant_Raw"] = utterance_df["Interaction_PValue_Raw"] < ALPHA
    trial_df["Significant_Raw"] = trial_df["Interaction_PValue_Raw"] < ALPHA

    utterance_df = utterance_df.sort_values(["FDR_PValue", "Interaction_PValue_Raw"], na_position="last")
    trial_df = trial_df.sort_values(["FDR_PValue", "Interaction_PValue_Raw"], na_position="last")

    merged_df = pd.merge(
        utterance_df[
            ["Feature", "English_Description", "Interaction_Coef", "Interaction_PValue_Raw", "FDR_PValue", "Significant_FDR", "Status"]
        ].rename(columns={
            "Interaction_Coef": "Utterance_Interaction_Coef",
            "Interaction_PValue_Raw": "Utterance_P_Raw",
            "FDR_PValue": "Utterance_P_FDR",
            "Significant_FDR": "Utterance_Significant_FDR",
            "Status": "Utterance_Status",
        }),
        trial_df[
            ["Feature", "English_Description", "Interaction_Coef", "Interaction_PValue_Raw", "FDR_PValue", "Significant_FDR", "Status"]
        ].rename(columns={
            "English_Description": "English_Description_Trial",
            "Interaction_Coef": "Trial_Interaction_Coef",
            "Interaction_PValue_Raw": "Trial_P_Raw",
            "FDR_PValue": "Trial_P_FDR",
            "Significant_FDR": "Trial_Significant_FDR",
            "Status": "Trial_Status",
        }),
        on="Feature",
        how="outer"
    )

    if "English_Description" not in merged_df.columns:
        merged_df["English_Description"] = merged_df.get("English_Description_Trial", np.nan)
    else:
        merged_df["English_Description"] = merged_df["English_Description"].fillna(merged_df.get("English_Description_Trial", np.nan))

    if "English_Description_Trial" in merged_df.columns:
        merged_df.drop(columns=["English_Description_Trial"], inplace=True)

    merged_df["More_Sensitive_Model"] = pd.NA

    valid_mask = (
            merged_df["Utterance_P_FDR"].notna()
            & merged_df["Trial_P_FDR"].notna()
    )

    merged_df.loc[
        valid_mask & (merged_df["Trial_P_FDR"] < merged_df["Utterance_P_FDR"]),
        "More_Sensitive_Model"
    ] = "Trial-by-Trial"

    merged_df.loc[
        valid_mask & (merged_df["Trial_P_FDR"] > merged_df["Utterance_P_FDR"]),
        "More_Sensitive_Model"
    ] = "Utterance-level"

    merged_df.loc[
        valid_mask & (merged_df["Trial_P_FDR"] == merged_df["Utterance_P_FDR"]),
        "More_Sensitive_Model"
    ] = "Equal"

    merged_df = merged_df.sort_values(["Trial_P_FDR", "Utterance_P_FDR"], na_position="last")

    summary_df = pd.DataFrame([{
        "Comparison": COMPARISON,
        "Base_Task": cfg["base_task"],
        "Contrast_Task": cfg["contrast_task"],
        "N_Features_Tested": len(run_list),
        "Utterance_Significant_Raw_Count": int(utterance_df["Significant_Raw"].sum()),
        "Utterance_Significant_FDR_Count": int(utterance_df["Significant_FDR"].sum()),
        "Trial_Significant_Raw_Count": int(trial_df["Significant_Raw"].sum()),
        "Trial_Significant_FDR_Count": int(trial_df["Significant_FDR"].sum()),
        "FDR_Method": "Benjamini-Hochberg",
        "Alpha": ALPHA,
    }])

    utterance_path = os.path.join(OUTPUT_DIR, cfg["utterance_output_file"])
    trial_path = os.path.join(OUTPUT_DIR, cfg["trial_output_file"])
    merged_path = os.path.join(OUTPUT_DIR, cfg["merged_output_file"])
    summary_path = os.path.join(OUTPUT_DIR, cfg["summary_output_file"])

    utterance_df.to_csv(utterance_path, index=False, encoding="utf-8-sig")
    trial_df.to_csv(trial_path, index=False, encoding="utf-8-sig")
    merged_df.to_csv(merged_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print('\n✅   ！')
    print(f"Utterance-level results: {utterance_path}")
    print(f"Trial-by-trial results: {trial_path}")
    print(f"Utterance vs trial comparison: {merged_path}")
    print(f"Summary table: {summary_path}")
    print('\n=== FDR          ===')
    print(f"Utterance-level: {int(utterance_df['Significant_FDR'].sum())}")
    print(f"Trial-by-Trial: {int(trial_df['Significant_FDR'].sum())}")





