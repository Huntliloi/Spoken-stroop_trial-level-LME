"""Compute WD deviation scores from healthy LME norms for CW_vs_W.

Set SCORING_GROUP to:
    "WD_CI"  : Wilson disease participants with cognitive impairment.
    "WD_CN"  : Wilson disease participants without cognitive impairment.
    "WD_ALL" : Both WD groups combined.

The composite deviation score uses features with FDR-corrected
Age_Z x Task_Code p < 0.01 in the healthy LME models.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")



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
SCORING_GROUP = "WD_CI"  # options: "WD_CI", "WD_CN", "WD_ALL"


# =========================
# Paths and parameters
# =========================
BASE_DIR = OUTPUT_DIR / "wd_external_validation"  # noqa: F405

HEALTHY_FEATURES_PATH = get_imputed_acoustic_features_csv()  # noqa: F405
HEALTHY_INFO_PATH = PARTICIPANT_INFO_XLSX  # noqa: F405
WD_INFO_PATH = PUBLIC_WD_INFO_XLSX  # noqa: F405

WD_FEATURE_PATHS = {
    "WD_CI": PUBLIC_WD_CI_FEATURE_CSV,  # noqa: F405
    "WD_CN": PUBLIC_WD_CN_FEATURE_CSV,  # noqa: F405
}

TASK_A = "CW"
TASK_B = "W"
TASK_PAIR_LABEL = f"{TASK_A}_vs_{TASK_B}"

FDR_ALPHA_FOR_COMPOSITE = 0.01
MIN_HEALTHY_SUBJECTS_PER_MODEL = 20
MIN_WD_SUBJECTS_PER_FEATURE = 1

RANDOM_STATE = 2026
np.random.seed(RANDOM_STATE)


META_COLS = {
    "Subject_ID", "Subject", "subject", "SubjectFolder", "Session", "Session_ID",
    "Task", "Task_Name", "Task_Type", "Task_Code",
    "Item", "Item_Index", "Word", "Text",
    "Start", "End", "Onset", "Offset",
    "Absolute_Word_Start", "Absolute_Word_End",
    "File", "File_Path", "Audio_Path", "Segment_Path",
    "ASR_Text", "Recognized_Text", "Target_Text",
    "Correct", "Correctness",
    "Age", "Age_Z", "Sex", "Gender", "Education", "CMMS",
    "Group", "Cognitive_Group", "Word_Duration",
}


def drop_duplicate_columns(df, source_name="dataframe"):
    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        print(f"{source_name}: duplicated columns removed: {duplicated}")
    return df.loc[:, ~df.columns.duplicated()].copy()


def get_1d_column(df, col):
    values = df[col]
    if isinstance(values, pd.DataFrame):
        values = values.iloc[:, 0]
    return values


def normalize_subject_id(x):
    if pd.isna(x):
        return np.nan
    text = str(x).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def find_col(df, candidates):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def ensure_subject_task_columns(df, source_name):
    df = drop_duplicate_columns(df, source_name)
    df = df.copy()

    subject_col = find_col(
        df,
        ["Subject_ID", "Subject", "subject", "SubjectFolder", "ID", '  Subject_ID', "Subject_ID"],
    )
    if subject_col is None:
        raise ValueError(f"{source_name}: cannot find subject column.")
    df["Subject_ID"] = get_1d_column(df, subject_col).map(normalize_subject_id)

    task_col = find_col(df, ["Task", "Task_Name", "Task_Type", "task", "Task"])
    if task_col is None:
        raise ValueError(f"{source_name}: cannot find task column.")
    df["Task"] = get_1d_column(df, task_col).astype(str).str.strip()

    if "Duration" not in df.columns and "Word_Duration" in df.columns:
        df["Duration"] = pd.to_numeric(get_1d_column(df, "Word_Duration"), errors="coerce")
    if "Word_Duration" not in df.columns and "Duration" in df.columns:
        df["Word_Duration"] = pd.to_numeric(get_1d_column(df, "Duration"), errors="coerce")

    return drop_duplicate_columns(df, source_name)


def load_healthy_info():
    info = pd.read_excel(HEALTHY_INFO_PATH)
    info = drop_duplicate_columns(info, "healthy info")

    subject_col = find_col(info, ["Subject_ID", "Subject", "ID"])
    age_col = find_col(info, ["Age"])
    sex_col = find_col(info, ["Sex", "Gender"])
    edu_col = find_col(info, ["Education", "Education_Years", "Education_Level"])

    if subject_col is None or age_col is None:
        raise ValueError("Healthy info table must contain subject ID and age columns.")

    keep = {subject_col: "Subject_ID", age_col: "Age"}
    if sex_col is not None:
        keep[sex_col] = "Sex"
    if edu_col is not None:
        keep[edu_col] = "Education"

    out = info[list(keep)].rename(columns=keep)
    out = drop_duplicate_columns(out, "healthy info selected")
    out["Subject_ID"] = get_1d_column(out, "Subject_ID").map(normalize_subject_id)
    out["Age"] = pd.to_numeric(get_1d_column(out, "Age"), errors="coerce")
    return out.drop_duplicates("Subject_ID")


def load_wd_info():
    if not WD_INFO_PATH.exists():
        return pd.DataFrame(columns=["Subject_ID"])

    info = pd.read_excel(WD_INFO_PATH)
    info = drop_duplicate_columns(info, "WD info")

    subject_col = find_col(info, ["Subject_ID", "Subject", "ID"])
    if subject_col is None:
        return pd.DataFrame(columns=["Subject_ID"])

    rename = {subject_col: "Subject_ID"}
    for names, target in [
        (["WD_Group", "Cognitive_Group", "Group"], "Cognitive_Group"),
        (["Sex", "Gender"], "Sex"),
        (["Age"], "Age"),
        (["Education", "Education_Level", "Education_Years"], "Education"),
        (["CMMS"], "CMMS"),
    ]:
        col = find_col(info, names)
        if col is not None:
            rename[col] = target

    out = info[list(rename)].rename(columns=rename)
    out = drop_duplicate_columns(out, "WD info selected")
    out["Subject_ID"] = get_1d_column(out, "Subject_ID").map(normalize_subject_id)
    for col in ["Age", "CMMS"]:
        if col in out.columns:
            out[col] = pd.to_numeric(get_1d_column(out, col), errors="coerce")
    return out.drop_duplicates("Subject_ID")


def load_wd_features():
    if SCORING_GROUP not in {"WD_CI", "WD_CN", "WD_ALL"}:
        raise ValueError('SCORING_GROUP must be "WD_CI", "WD_CN", or "WD_ALL".')

    frames = []
    groups = ["WD_CI", "WD_CN"] if SCORING_GROUP == "WD_ALL" else [SCORING_GROUP]

    for group in groups:
        path = WD_FEATURE_PATHS[group]
        if not path.exists():
            raise FileNotFoundError(f"Missing WD feature file for {group}: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = ensure_subject_task_columns(df, str(path))
        df["Cognitive_Group"] = group
        frames.append(df)

    wd = pd.concat(frames, ignore_index=True)
    wd = drop_duplicate_columns(wd, "WD features concatenated")

    duplicate_keys = [c for c in ["Subject_ID", "Task", "Item_Index", "Word", "Segment_Path"] if c in wd.columns]
    if duplicate_keys:
        wd = wd.drop_duplicates(subset=duplicate_keys, keep="first")

    return wd


def get_numeric_feature_columns(healthy, wd):
    common_cols = [c for c in healthy.columns if c in wd.columns]
    numeric_cols = []

    for col in common_cols:
        if col in META_COLS:
            continue
        h_num = pd.to_numeric(get_1d_column(healthy, col), errors="coerce")
        w_num = pd.to_numeric(get_1d_column(wd, col), errors="coerce")
        if h_num.notna().sum() >= 10 and w_num.notna().sum() >= 1:
            numeric_cols.append(col)

    if "Duration" in healthy.columns and "Duration" in wd.columns and "Duration" not in numeric_cols:
        numeric_cols.insert(0, "Duration")

    return list(dict.fromkeys(numeric_cols))


def prepare_data():
    healthy = pd.read_csv(HEALTHY_FEATURES_PATH, encoding="utf-8-sig")
    healthy = ensure_subject_task_columns(healthy, str(HEALTHY_FEATURES_PATH))

    healthy_info = load_healthy_info()
    healthy = healthy.merge(healthy_info, on="Subject_ID", how="left", suffixes=("", "_info"))
    healthy = drop_duplicate_columns(healthy, "healthy merged")
    if "Age_info" in healthy.columns:
        if "Age" in healthy.columns:
            healthy["Age"] = get_1d_column(healthy, "Age").fillna(get_1d_column(healthy, "Age_info"))
        else:
            healthy["Age"] = get_1d_column(healthy, "Age_info")
        healthy = healthy.drop(columns=["Age_info"])

    wd = load_wd_features()
    wd_info = load_wd_info()
    wd = wd.merge(wd_info, on="Subject_ID", how="left", suffixes=("", "_info"))
    wd = drop_duplicate_columns(wd, "WD merged")

    for col in ["Age", "Sex", "Education", "CMMS"]:
        info_col = f"{col}_info"
        if info_col in wd.columns:
            if col in wd.columns:
                wd[col] = get_1d_column(wd, col).where(
                    get_1d_column(wd, col).notna(),
                    get_1d_column(wd, info_col),
                )
            else:
                wd[col] = get_1d_column(wd, info_col)
            wd = wd.drop(columns=[info_col])

    healthy = healthy[get_1d_column(healthy, "Task").isin([TASK_A, TASK_B])].copy()
    wd = wd[get_1d_column(wd, "Task").isin([TASK_A, TASK_B])].copy()

    healthy["Age"] = pd.to_numeric(get_1d_column(healthy, "Age"), errors="coerce")
    wd["Age"] = pd.to_numeric(get_1d_column(wd, "Age"), errors="coerce")

    healthy_subject_age = healthy.drop_duplicates("Subject_ID")["Age"]
    age_mean = healthy_subject_age.mean()
    age_sd = healthy_subject_age.std(ddof=0)
    if not np.isfinite(age_sd) or age_sd == 0:
        raise ValueError("Healthy age SD is zero or missing; cannot compute Age_Z.")

    healthy["Age_Z"] = (healthy["Age"] - age_mean) / age_sd
    wd["Age_Z"] = (wd["Age"] - age_mean) / age_sd

    healthy["Task_Code"] = np.where(healthy["Task"] == TASK_A, 1, 0)
    wd["Task_Code"] = np.where(wd["Task"] == TASK_A, 1, 0)

    return healthy, wd


def fit_lme_for_feature(feature, healthy):
    dat = healthy[["Subject_ID", "Task_Code", "Age_Z", feature]].copy()
    dat = drop_duplicate_columns(dat, f"LME data {feature}")
    dat[feature] = pd.to_numeric(get_1d_column(dat, feature), errors="coerce")
    dat = dat.dropna()

    if dat["Subject_ID"].nunique() < MIN_HEALTHY_SUBJECTS_PER_MODEL:
        return None, None
    if dat["Task_Code"].nunique() < 2:
        return None, None

    scaler = StandardScaler()
    dat["Feature_Z"] = scaler.fit_transform(dat[[feature]])

    formula = "Feature_Z ~ Age_Z * Task_Code"
    try:
        model = smf.mixedlm(formula, dat, groups=dat["Subject_ID"], re_formula="~Task_Code")
        result = model.fit(method="lbfgs", reml=False, maxiter=1000, disp=False)
    except Exception:
        try:
            model = smf.mixedlm(formula, dat, groups=dat["Subject_ID"], re_formula="1")
            result = model.fit(method="powell", reml=False, maxiter=1000, disp=False)
        except Exception:
            return None, None

    return result, {
        "scaler": scaler,
        "interaction_p": result.pvalues.get("Age_Z:Task_Code", np.nan),
    }


def subject_task_means(df, feature, scaler):
    dat = df[["Subject_ID", "Task", "Task_Code", "Age_Z", feature]].copy()
    dat = drop_duplicate_columns(dat, f"subject means {feature}")
    dat[feature] = pd.to_numeric(get_1d_column(dat, feature), errors="coerce")
    dat = dat.dropna(subset=["Subject_ID", "Task", "Task_Code", "Age_Z", feature])
    if dat.empty:
        return dat

    dat["Feature_Z"] = scaler.transform(dat[[feature]])
    return (
        dat.groupby(["Subject_ID", "Task", "Task_Code"], as_index=False)
        .agg(Feature_Z=("Feature_Z", "mean"), Age_Z=("Age_Z", "mean"), N_Items=(feature, "count"))
    )


def calculate_feature_deviation(feature, model_result, scaler, healthy, wd):
    h_sub = subject_task_means(healthy, feature, scaler)
    w_sub = subject_task_means(wd, feature, scaler)

    def wide_deviation(sub_df):
        if sub_df.empty:
            return pd.DataFrame()

        pred_dat = sub_df.copy()
        pred_dat["Pred_Z"] = model_result.predict(pred_dat)

        obs = sub_df.pivot(index="Subject_ID", columns="Task", values="Feature_Z")
        pred = pred_dat.pivot(index="Subject_ID", columns="Task", values="Pred_Z")
        n_items = sub_df.pivot(index="Subject_ID", columns="Task", values="N_Items")

        if TASK_A not in obs.columns or TASK_B not in obs.columns:
            return pd.DataFrame()

        out = pd.DataFrame(index=obs.index)
        out["Observed_Diff_Z"] = obs[TASK_A] - obs[TASK_B]
        out["Predicted_Diff_Z"] = pred[TASK_A] - pred[TASK_B]
        out["Residual_Diff_Z"] = out["Observed_Diff_Z"] - out["Predicted_Diff_Z"]
        out[f"N_{TASK_A}"] = n_items[TASK_A]
        out[f"N_{TASK_B}"] = n_items[TASK_B]
        return out.reset_index()

    h_dev = wide_deviation(h_sub)
    w_dev = wide_deviation(w_sub)
    if h_dev.empty or w_dev.empty:
        return None

    residual_mean = h_dev["Residual_Diff_Z"].mean()
    residual_sd = h_dev["Residual_Diff_Z"].std(ddof=1)
    if not np.isfinite(residual_sd) or residual_sd == 0:
        return None

    w_dev["Feature"] = feature
    w_dev["Deviation_Z"] = (w_dev["Residual_Diff_Z"] - residual_mean) / residual_sd
    w_dev["Abs_Deviation_Z"] = w_dev["Deviation_Z"].abs()
    w_dev["Healthy_Residual_Mean"] = residual_mean
    w_dev["Healthy_Residual_SD"] = residual_sd
    return w_dev


def main():
    print(f"Scoring group: {SCORING_GROUP}")
    healthy, wd = prepare_data()
    feature_cols = get_numeric_feature_columns(healthy, wd)
    print(f"Candidate numeric features: {len(feature_cols)}")

    model_rows = []
    wd_feature_rows = []

    for i, feature in enumerate(feature_cols, start=1):
        print(f"[{i}/{len(feature_cols)}] {feature}")
        model_result, model_info = fit_lme_for_feature(feature, healthy)
        if model_result is None:
            model_rows.append({
                "Feature": feature,
                "Status": "model_failed_or_insufficient_data",
                "Interaction_PValue": np.nan,
            })
            continue

        w_dev = calculate_feature_deviation(feature, model_result, model_info["scaler"], healthy, wd)
        if w_dev is None or w_dev["Subject_ID"].nunique() < MIN_WD_SUBJECTS_PER_FEATURE:
            model_rows.append({
                "Feature": feature,
                "Status": "deviation_failed_or_no_wd_data",
                "Interaction_PValue": model_info["interaction_p"],
            })
            continue

        model_rows.append({
            "Feature": feature,
            "Status": "ok",
            "Interaction_PValue": model_info["interaction_p"],
            "N_Healthy_Subjects": healthy.dropna(subset=[feature])["Subject_ID"].nunique(),
            "N_WD_Subjects": w_dev["Subject_ID"].nunique(),
            "Coef_Intercept": model_result.params.get("Intercept", np.nan),
            "Coef_Age_Z": model_result.params.get("Age_Z", np.nan),
            "Coef_Task_Code": model_result.params.get("Task_Code", np.nan),
            "Coef_Age_Z_x_Task_Code": model_result.params.get("Age_Z:Task_Code", np.nan),
        })
        wd_feature_rows.append(w_dev)

    model_summary = pd.DataFrame(model_rows)
    ok_mask = model_summary["Status"].eq("ok") & model_summary["Interaction_PValue"].notna()
    model_summary["Interaction_FDR_PValue"] = np.nan
    model_summary["Interaction_FDR_Significant"] = False

    if ok_mask.any():
        reject, p_fdr, _, _ = multipletests(
            model_summary.loc[ok_mask, "Interaction_PValue"].values,
            alpha=FDR_ALPHA_FOR_COMPOSITE,
            method="fdr_bh",
        )
        model_summary.loc[ok_mask, "Interaction_FDR_PValue"] = p_fdr
        model_summary.loc[ok_mask, "Interaction_FDR_Significant"] = reject

    selected_features = model_summary.loc[
        model_summary["Interaction_FDR_PValue"].lt(FDR_ALPHA_FOR_COMPOSITE),
        "Feature",
    ].tolist()

    print(f"FDR p < {FDR_ALPHA_FOR_COMPOSITE} selected features: {len(selected_features)}")
    if not selected_features:
        raise RuntimeError(
            f"No features survived Age_Z x Task_Code FDR p < {FDR_ALPHA_FOR_COMPOSITE}. "
            "Composite deviation score was not computed."
        )
    if not wd_feature_rows:
        raise RuntimeError("No WD feature-level deviation scores were computed.")

    wd_feature_scores = pd.concat(wd_feature_rows, ignore_index=True)
    wd_feature_scores = wd_feature_scores[wd_feature_scores["Feature"].isin(selected_features)].copy()

    wd_meta_cols = [
        c for c in ["Subject_ID", "Cognitive_Group", "Sex", "Age", "Education", "CMMS"]
        if c in wd.columns
    ]
    wd_meta = wd[wd_meta_cols].drop_duplicates("Subject_ID") if wd_meta_cols else pd.DataFrame()

    subject_scores = (
        wd_feature_scores.groupby("Subject_ID", as_index=False)
        .agg(
            N_Features=("Feature", "nunique"),
            Mean_Deviation_Z=("Deviation_Z", "mean"),
            Mean_Abs_Deviation_Z=("Abs_Deviation_Z", "mean"),
            RMS_Deviation_Z=("Deviation_Z", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            Max_Abs_Deviation_Z=("Abs_Deviation_Z", "max"),
            N_Features_AbsZ_GE_1_96=("Abs_Deviation_Z", lambda x: int((x >= 1.96).sum())),
            N_Features_AbsZ_GE_2_58=("Abs_Deviation_Z", lambda x: int((x >= 2.58).sum())),
        )
    )

    if not wd_meta.empty:
        subject_scores = subject_scores.merge(wd_meta, on="Subject_ID", how="left")

    out_dir = BASE_DIR / f"LME_deviation_scores_{TASK_PAIR_LABEL}_{SCORING_GROUP}_FDR01"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"healthy_LME_model_summary_{TASK_PAIR_LABEL}_{SCORING_GROUP}_FDR01.csv"
    feature_path = out_dir / f"WD_feature_level_deviation_scores_{TASK_PAIR_LABEL}_{SCORING_GROUP}_FDR01.csv"
    subject_path = out_dir / f"WD_subject_level_deviation_index_{TASK_PAIR_LABEL}_{SCORING_GROUP}_FDR01.csv"
    selected_path = out_dir / f"selected_features_{TASK_PAIR_LABEL}_{SCORING_GROUP}_FDR01.txt"

    model_summary.sort_values(["Interaction_FDR_PValue", "Interaction_PValue"], na_position="last").to_csv(
        model_path, index=False, encoding="utf-8-sig"
    )
    wd_feature_scores.to_csv(feature_path, index=False, encoding="utf-8-sig")
    subject_scores.sort_values("Mean_Abs_Deviation_Z", ascending=False).to_csv(
        subject_path, index=False, encoding="utf-8-sig"
    )
    selected_path.write_text("\n".join(selected_features), encoding="utf-8")

    print("Done.")
    print(f"Model summary: {model_path}")
    print(f"Feature-level scores: {feature_path}")
    print(f"Subject-level scores: {subject_path}")
    print(f"Selected features: {selected_path}")


if __name__ == "__main__":
    main()





