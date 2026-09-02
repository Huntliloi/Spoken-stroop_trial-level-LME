"""Compute substance-use cohort deviation scores from healthy LME norms.

This script follows S.2_compute_WD_LME_deviation_scores_CW_vs_W.py,
but scores new Drug/HC subjects with item-level LME random-slope BLUPs:
  1. Load healthy Stroop feature norms.
  2. Fit one healthy LME per numeric feature:
     Feature_Z ~ Age_Z * Task_Code + (1 + Task_Code | Subject_ID).
  3. Use Age_Z x Task_Code FDR p < 0.01 features for the composite score.
  4. Estimate each Drug/HC subject's Task_Code random slope from all item-level
     CW/W items using the healthy LME variance structure.
  5. Standardize that random slope against the healthy random-slope distribution.
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# =========================
# Change this to score one group or both groups.
# =========================
SCORING_GROUP = "ALL"  # options: "Drug", "HC", "ALL"


# =========================
# Paths and parameters
# =========================
SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from common_paths import (  # noqa: E402
    PARTICIPANT_INFO_XLSX,
    PUBLIC_SUBSTANCE_DRUG_FEATURE_CSV,
    PUBLIC_SUBSTANCE_HC_FEATURE_CSV,
    PUBLIC_SUBSTANCE_INFO_CSV,
    SUBSTANCE_USE_OUTPUT_DIR,
    get_imputed_acoustic_features_csv,
)

RESULT_DIR = SUBSTANCE_USE_OUTPUT_DIR

HEALTHY_FEATURES_PATH = get_imputed_acoustic_features_csv()
HEALTHY_INFO_PATH = PARTICIPANT_INFO_XLSX
SUBJECT_INFO_CSV_PATH = PUBLIC_SUBSTANCE_INFO_CSV

FEATURE_PATHS = {
    "Drug": PUBLIC_SUBSTANCE_DRUG_FEATURE_CSV,
    "HC": PUBLIC_SUBSTANCE_HC_FEATURE_CSV,
}

TASK_A = "CW"
TASK_B = "W"
TASK_PAIR_LABEL = f"{TASK_A}_vs_{TASK_B}"
SCORING_METHOD_LABEL = "RandomSlopeBLUP"

FDR_ALPHA_FOR_COMPOSITE = 0.01
MIN_HEALTHY_SUBJECTS_PER_MODEL = 20
MIN_SCORING_SUBJECTS_PER_FEATURE = 1

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
    "Age", "Age_Z", "Sex", "Gender", "Education",
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
        ["Subject_ID", "Subject", "subject", "SubjectFolder", "ID"],
    )
    if subject_col is None:
        raise ValueError(f"{source_name}: cannot find subject column.")
    df["Subject_ID"] = get_1d_column(df, subject_col).map(normalize_subject_id)

    task_col = find_col(df, ["Task", "Task_Name", "Task_Type", "task"])
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
    edu_col = find_col(info, ["Education"])

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


def load_subject_info():
    if SUBJECT_INFO_CSV_PATH.exists():
        info = pd.read_csv(SUBJECT_INFO_CSV_PATH, dtype={"subject_id": str}, encoding="utf-8-sig")
    else:
        return pd.DataFrame(columns=["Subject_ID"])

    info = drop_duplicate_columns(info, "Drug/HC subject info")

    rename = {}
    subject_col = find_col(info, ["subject_id", "Subject_ID", "Subject", "ID"])
    group_col = find_col(info, ["group", "Group", "Cognitive_Group"])
    sex_col = find_col(info, ["sex", "Sex", "Gender"])
    age_col = find_col(info, ["age", "Age"])
    edu_col = find_col(info, ["education", "Education"])
    cmms_col = find_col(info, ["CMMS"])
    moca_col = find_col(info, ["MoCA", "MOCA"])

    if subject_col is None:
        return pd.DataFrame(columns=["Subject_ID"])

    rename[subject_col] = "Subject_ID"
    for col, target in [
        (group_col, "Cognitive_Group"),
        (sex_col, "Sex"),
        (age_col, "Age"),
        (edu_col, "Education"),
        (cmms_col, "CMMS"),
        (moca_col, "MoCA"),
    ]:
        if col is not None:
            rename[col] = target

    out = info[list(rename)].rename(columns=rename)
    out = drop_duplicate_columns(out, "Drug/HC subject info selected")
    out["Subject_ID"] = get_1d_column(out, "Subject_ID").map(normalize_subject_id)
    for col in ["Age", "Education", "CMMS", "MoCA"]:
        if col in out.columns:
            out[col] = pd.to_numeric(get_1d_column(out, col), errors="coerce")
    return out.drop_duplicates("Subject_ID")


def load_scoring_features():
    if SCORING_GROUP not in {"Drug", "HC", "ALL"}:
        raise ValueError('SCORING_GROUP must be "Drug", "HC", or "ALL".')

    frames = []
    groups = ["Drug", "HC"] if SCORING_GROUP == "ALL" else [SCORING_GROUP]

    for group in groups:
        path = FEATURE_PATHS[group]
        if not path.exists():
            raise FileNotFoundError(f"Missing feature file for {group}: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        df = ensure_subject_task_columns(df, str(path))
        df["Cognitive_Group"] = group
        frames.append(df)

    scoring = pd.concat(frames, ignore_index=True)
    scoring = drop_duplicate_columns(scoring, "scoring features concatenated")

    duplicate_keys = [c for c in ["Subject_ID", "Task", "Item_Index", "Word", "Segment_Path"] if c in scoring.columns]
    if duplicate_keys:
        scoring = scoring.drop_duplicates(subset=duplicate_keys, keep="first")

    return scoring


def get_numeric_feature_columns(healthy, scoring):
    common_cols = [c for c in healthy.columns if c in scoring.columns]
    numeric_cols = []

    for col in common_cols:
        if col in META_COLS:
            continue
        h_num = pd.to_numeric(get_1d_column(healthy, col), errors="coerce")
        s_num = pd.to_numeric(get_1d_column(scoring, col), errors="coerce")
        if h_num.notna().sum() >= 10 and s_num.notna().sum() >= 1:
            numeric_cols.append(col)

    if "Duration" in healthy.columns and "Duration" in scoring.columns and "Duration" not in numeric_cols:
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

    scoring = load_scoring_features()
    subject_info = load_subject_info()
    scoring = scoring.merge(subject_info, on="Subject_ID", how="left", suffixes=("", "_info"))
    scoring = drop_duplicate_columns(scoring, "scoring merged")

    for col in ["Cognitive_Group", "Sex", "Age", "Education", "CMMS", "MoCA"]:
        info_col = f"{col}_info"
        if info_col in scoring.columns:
            if col in scoring.columns:
                scoring[col] = get_1d_column(scoring, col).where(
                    get_1d_column(scoring, col).notna(),
                    get_1d_column(scoring, info_col),
                )
            else:
                scoring[col] = get_1d_column(scoring, info_col)
            scoring = scoring.drop(columns=[info_col])

    healthy = healthy[get_1d_column(healthy, "Task").isin([TASK_A, TASK_B])].copy()
    scoring = scoring[get_1d_column(scoring, "Task").isin([TASK_A, TASK_B])].copy()

    healthy["Age"] = pd.to_numeric(get_1d_column(healthy, "Age"), errors="coerce")
    scoring["Age"] = pd.to_numeric(get_1d_column(scoring, "Age"), errors="coerce")

    healthy_subject_age = healthy.drop_duplicates("Subject_ID")["Age"]
    age_mean = healthy_subject_age.mean()
    age_sd = healthy_subject_age.std(ddof=0)
    if not np.isfinite(age_sd) or age_sd == 0:
        raise ValueError("Healthy age SD is zero or missing; cannot compute Age_Z.")

    healthy["Age_Z"] = (healthy["Age"] - age_mean) / age_sd
    scoring["Age_Z"] = (scoring["Age"] - age_mean) / age_sd

    healthy["Task_Code"] = np.where(healthy["Task"] == TASK_A, 1, 0)
    scoring["Task_Code"] = np.where(scoring["Task"] == TASK_A, 1, 0)

    return healthy, scoring


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


def random_effect_design(dat, random_effect_names):
    cols = []
    for name in random_effect_names:
        key = str(name)
        if key in {"Group", "Intercept", "const", "1"}:
            cols.append(np.ones(len(dat)))
        elif key in dat.columns:
            cols.append(pd.to_numeric(get_1d_column(dat, key), errors="coerce").to_numpy(dtype=float))
        else:
            raise ValueError(f"Cannot build random-effect design column: {name}")
    return np.column_stack(cols)


def estimate_subject_random_effects(model_result, dat, feature, scaler, source_name):
    dat = dat[["Subject_ID", "Task", "Task_Code", "Age_Z", feature]].copy()
    dat = drop_duplicate_columns(dat, f"random effects {source_name} {feature}")
    dat[feature] = pd.to_numeric(get_1d_column(dat, feature), errors="coerce")
    dat = dat.dropna(subset=["Subject_ID", "Task", "Task_Code", "Age_Z", feature])
    if dat.empty:
        return pd.DataFrame()

    dat["Feature_Z"] = scaler.transform(dat[[feature]])
    dat["Fixed_Pred_Z"] = model_result.predict(dat)
    dat["Fixed_Residual_Z"] = dat["Feature_Z"] - dat["Fixed_Pred_Z"]

    cov_re = model_result.cov_re
    if cov_re is None or cov_re.empty:
        return pd.DataFrame()

    random_effect_names = list(cov_re.index)
    if "Task_Code" not in random_effect_names:
        return pd.DataFrame()

    task_idx = random_effect_names.index("Task_Code")
    G = np.asarray(cov_re, dtype=float)
    sigma2 = float(model_result.scale)
    if not np.isfinite(sigma2) or sigma2 <= 0:
        return pd.DataFrame()

    rows = []
    for sid, sub in dat.groupby("Subject_ID", sort=False):
        if sub["Task_Code"].nunique() < 2:
            continue

        Z = random_effect_design(sub, random_effect_names)
        resid = sub["Fixed_Residual_Z"].to_numpy(dtype=float)
        V = Z @ G @ Z.T + sigma2 * np.eye(len(sub))
        try:
            b_hat = G @ Z.T @ np.linalg.solve(V, resid)
        except np.linalg.LinAlgError:
            b_hat = G @ Z.T @ np.linalg.pinv(V) @ resid

        row = {
            "Subject_ID": sid,
            "Random_Task_Slope_Z": float(b_hat[task_idx]),
            f"N_{TASK_A}": int((sub["Task"] == TASK_A).sum()),
            f"N_{TASK_B}": int((sub["Task"] == TASK_B).sum()),
        }
        intercept_idx = next(
            (i for i, name in enumerate(random_effect_names) if str(name) in {"Group", "Intercept", "const", "1"}),
            None,
        )
        if intercept_idx is not None:
            row["Random_Intercept_Z"] = float(b_hat[intercept_idx])
        rows.append(row)

    return pd.DataFrame(rows)


def calculate_feature_deviation(feature, model_result, scaler, healthy, scoring):
    h_dev = estimate_subject_random_effects(model_result, healthy, feature, scaler, "healthy")
    s_dev = estimate_subject_random_effects(model_result, scoring, feature, scaler, "scoring")
    if h_dev.empty or s_dev.empty:
        return None

    slope_mean = h_dev["Random_Task_Slope_Z"].mean()
    slope_sd = h_dev["Random_Task_Slope_Z"].std(ddof=1)
    if not np.isfinite(slope_sd) or slope_sd == 0:
        return None

    s_dev["Feature"] = feature
    s_dev["Deviation_Z"] = (s_dev["Random_Task_Slope_Z"] - slope_mean) / slope_sd
    s_dev["Abs_Deviation_Z"] = s_dev["Deviation_Z"].abs()
    s_dev["Healthy_Random_Task_Slope_Mean"] = slope_mean
    s_dev["Healthy_Random_Task_Slope_SD"] = slope_sd
    s_dev["Scoring_Method"] = SCORING_METHOD_LABEL
    return s_dev


def main():
    print(f"Scoring group: {SCORING_GROUP}")
    healthy, scoring = prepare_data()
    feature_cols = get_numeric_feature_columns(healthy, scoring)
    print(f"Candidate numeric features: {len(feature_cols)}")

    model_rows = []
    scoring_feature_rows = []

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

        s_dev = calculate_feature_deviation(feature, model_result, model_info["scaler"], healthy, scoring)
        if s_dev is None or s_dev["Subject_ID"].nunique() < MIN_SCORING_SUBJECTS_PER_FEATURE:
            model_rows.append({
                "Feature": feature,
                "Status": "deviation_failed_or_no_scoring_data",
                "Interaction_PValue": model_info["interaction_p"],
            })
            continue

        model_rows.append({
            "Feature": feature,
            "Status": "ok",
            "Interaction_PValue": model_info["interaction_p"],
            "N_Healthy_Subjects": healthy.dropna(subset=[feature])["Subject_ID"].nunique(),
            "N_Scoring_Subjects": s_dev["Subject_ID"].nunique(),
            "Coef_Intercept": model_result.params.get("Intercept", np.nan),
            "Coef_Age_Z": model_result.params.get("Age_Z", np.nan),
            "Coef_Task_Code": model_result.params.get("Task_Code", np.nan),
            "Coef_Age_Z_x_Task_Code": model_result.params.get("Age_Z:Task_Code", np.nan),
        })
        scoring_feature_rows.append(s_dev)

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
    if not scoring_feature_rows:
        raise RuntimeError("No feature-level deviation scores were computed.")

    scoring_feature_scores = pd.concat(scoring_feature_rows, ignore_index=True)
    scoring_feature_scores = scoring_feature_scores[scoring_feature_scores["Feature"].isin(selected_features)].copy()

    meta_cols = [
        c for c in ["Subject_ID", "Cognitive_Group", "Sex", "Age", "Education", "CMMS", "MoCA"]
        if c in scoring.columns
    ]
    scoring_meta = scoring[meta_cols].drop_duplicates("Subject_ID") if meta_cols else pd.DataFrame()

    subject_scores = (
        scoring_feature_scores.groupby("Subject_ID", as_index=False)
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

    if not scoring_meta.empty:
        subject_scores = subject_scores.merge(scoring_meta, on="Subject_ID", how="left")
    subject_scores = subject_scores.rename(columns={"Cognitive_Group": "Group"})

    out_dir = RESULT_DIR / "deviation_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"healthy_LME_model_summary_{TASK_PAIR_LABEL}_{SCORING_METHOD_LABEL}_{SCORING_GROUP}_FDR01.csv"
    feature_path = out_dir / f"feature_level_deviation_scores_{TASK_PAIR_LABEL}_{SCORING_METHOD_LABEL}_{SCORING_GROUP}_FDR01.csv"
    subject_path = RESULT_DIR / "subject_level_deviation_indices.csv"
    selected_path = out_dir / f"selected_features_{TASK_PAIR_LABEL}_{SCORING_METHOD_LABEL}_{SCORING_GROUP}_FDR01.txt"

    model_summary.sort_values(["Interaction_FDR_PValue", "Interaction_PValue"], na_position="last").to_csv(
        model_path, index=False, encoding="utf-8-sig"
    )
    scoring_feature_scores.to_csv(feature_path, index=False, encoding="utf-8-sig")
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
