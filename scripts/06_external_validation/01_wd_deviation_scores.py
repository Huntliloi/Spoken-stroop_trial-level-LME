"""Evaluate W, CW, and CW-minus-W acoustic deviations in Wilson disease.

The same 25 prespecified acoustic features are scored for all three effects
from one healthy-reference item-level mixed model per feature:

    Feature_Z ~ Age_Z * Task_Code + (1 + Task_Code | Subject_ID)

W is represented by the participant random intercept, CW by the random
intercept plus random task slope, and CW-minus-W by the random task slope.
Each effect is standardized against its matching empirical BLUP distribution
in the healthy reference cohort. Subject-level summaries are limited to
Mean |Z_dev| and RMS Z_dev.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PUBLIC_DATA_DIR = REPO_ROOT / "data" / "public"
PUBLIC_WD_DIR = PUBLIC_DATA_DIR / "wd"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "wd_external_validation"
DEFAULT_HEALTHY_FEATURES = PUBLIC_DATA_DIR / "features" / "Automated_feature.csv"
DEFAULT_HEALTHY_INFO = PUBLIC_DATA_DIR / "participant_info_HC_158.xlsx"
DEFAULT_WD_CI_FEATURES = PUBLIC_WD_DIR / "features" / "Stroop_features_WD_CI.csv"
DEFAULT_WD_CN_FEATURES = PUBLIC_WD_DIR / "features" / "Stroop_features_WD_CN.csv"
DEFAULT_WD_INFO = PUBLIC_WD_DIR / "WD_participant_info.xlsx"

TASK_ALIASES = {"W": "W", "CW": "CW"}
TASK_CODE = {"W": 0, "CW": 1}
CONDITIONS = ("W", "CW", "CW-W")
GROUPS = ("WD-CI", "WD-CN")
EXPECTED_HEALTHY_N = 158
EXPECTED_GROUP_COUNTS = {"WD-CI": 12, "WD-CN": 31}

SELECTED_FEATURES = [
    "Gap_Before",
    "F0semitoneFrom27.5Hz_sma3nz_stddevNorm",
    "F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2",
    "F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope",
    "loudness_sma3_stddevNorm",
    "loudness_sma3_percentile20.0",
    "loudness_sma3_pctlrange0-2",
    "spectralFlux_sma3_stddevNorm",
    "mfcc1_sma3_amean",
    "mfcc1_sma3_stddevNorm",
    "jitterLocal_sma3nz_amean",
    "shimmerLocaldB_sma3nz_amean",
    "HNRdBACF_sma3nz_amean",
    "F1amplitudeLogRelF0_sma3nz_amean",
    "F3amplitudeLogRelF0_sma3nz_amean",
    "F3amplitudeLogRelF0_sma3nz_stddevNorm",
    "hammarbergIndexV_sma3nz_stddevNorm",
    "spectralFluxV_sma3nz_stddevNorm",
    "mfcc1V_sma3nz_amean",
    "mfcc1V_sma3nz_stddevNorm",
    "alphaRatioUV_sma3nz_amean",
    "hammarbergIndexUV_sma3nz_amean",
    "slopeUV500-1500_sma3nz_amean",
    "spectralFluxUV_sma3nz_amean",
    "StddevVoicedSegmentLengthSec",
]

METRICS = {
    "Mean |Z_dev|": "Mean_Abs_Deviation_Z",
    "RMS Z_dev": "RMS_Deviation_Z",
}
GROUP_COLORS = {"WD-CI": "#ff1f1f", "WD-CN": "#42bde3"}


def normalize_subject_id(value: object) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def normalize_items(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    subject_column = find_column(
        frame, ["Subject_ID", "Subject", "subject", "SubjectFolder", "ID"]
    )
    task_column = find_column(frame, ["Task", "Task_Name", "Task_Type"])
    if subject_column is None or task_column is None:
        raise ValueError(f"{source}: subject ID or task column is missing.")
    frame["Subject_ID"] = frame[subject_column].map(normalize_subject_id)
    frame["Task"] = frame[task_column].astype(str).str.strip().map(TASK_ALIASES)
    return frame[frame["Task"].isin(TASK_CODE)].copy()


def load_age_table(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_excel(path)
    subject_column = find_column(
        frame, ["Subject_ID", "Subject", "ID"]
    )
    age_column = find_column(frame, ["Age"])
    if subject_column is None or age_column is None:
        raise ValueError(f"{label}: subject ID or age column is missing.")
    result = frame[[subject_column, age_column]].copy()
    result.columns = ["Subject_ID", "Age"]
    result["Subject_ID"] = result["Subject_ID"].map(normalize_subject_id)
    result["Age"] = pd.to_numeric(result["Age"], errors="coerce")
    return result.dropna().drop_duplicates("Subject_ID")


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    healthy = normalize_items(read_csv(args.healthy_features), "healthy features")
    healthy_age = load_age_table(args.healthy_info, "healthy information")
    healthy = healthy.merge(
        healthy_age, on="Subject_ID", how="inner", validate="many_to_one"
    )
    if healthy["Subject_ID"].nunique() != EXPECTED_HEALTHY_N:
        raise ValueError(
            f"Expected {EXPECTED_HEALTHY_N} healthy participants, found "
            f"{healthy['Subject_ID'].nunique()}."
        )

    wd_frames = []
    for path, group in (
        (args.wd_ci_features, "WD-CI"),
        (args.wd_cn_features, "WD-CN"),
    ):
        frame = normalize_items(read_csv(path), f"{group} features")
        frame["Clinical_Group"] = group
        wd_frames.append(frame)
    wd = pd.concat(wd_frames, ignore_index=True)
    wd_age = load_age_table(args.wd_info, "WD information")
    wd = wd.merge(wd_age, on="Subject_ID", how="inner", validate="many_to_one")
    counts = (
        wd[["Subject_ID", "Clinical_Group"]]
        .drop_duplicates()["Clinical_Group"]
        .value_counts()
        .to_dict()
    )
    if counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            f"Expected WD group counts {EXPECTED_GROUP_COUNTS}, found {counts}."
        )

    missing_healthy = sorted(set(SELECTED_FEATURES).difference(healthy.columns))
    missing_wd = sorted(set(SELECTED_FEATURES).difference(wd.columns))
    if missing_healthy or missing_wd:
        raise ValueError(
            f"Missing prespecified features; healthy={missing_healthy}, WD={missing_wd}."
        )
    return healthy, wd


def random_effect_design(frame: pd.DataFrame, names: list[object]) -> np.ndarray:
    columns = []
    for name in names:
        key = str(name)
        if key in {"Group", "Intercept", "const", "1"}:
            columns.append(np.ones(len(frame)))
        elif key in frame:
            columns.append(pd.to_numeric(frame[key], errors="coerce").to_numpy(float))
        else:
            raise ValueError(f"Cannot construct random-effect column: {key}")
    return np.column_stack(columns)


def estimate_subject_effects(
    model_result, frame: pd.DataFrame, feature: str, feature_mean: float, feature_sd: float
) -> pd.DataFrame:
    data = frame[["Subject_ID", "Task", "Task_Code", "Age_Z", feature]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna()
    data["Feature_Z"] = (data[feature] - feature_mean) / feature_sd
    data["Fixed_Residual_Z"] = data["Feature_Z"] - model_result.predict(data)

    covariance = model_result.cov_re
    names = list(covariance.index)
    intercept_index = next(
        (
            index
            for index, name in enumerate(names)
            if str(name) in {"Group", "Intercept", "const", "1"}
        ),
        None,
    )
    name_strings = [str(name) for name in names]
    if intercept_index is None or "Task_Code" not in name_strings:
        raise RuntimeError("The fitted model lacks a random intercept or random task slope.")
    task_index = name_strings.index("Task_Code")
    g_matrix = np.asarray(covariance, dtype=float)
    sigma_squared = float(model_result.scale)
    if not np.isfinite(sigma_squared) or sigma_squared <= 0:
        raise RuntimeError("The fitted model has an invalid residual variance.")

    rows = []
    for subject_id, subset in data.groupby("Subject_ID", sort=False):
        if set(subset["Task"]) != {"W", "CW"}:
            continue
        design = random_effect_design(subset, names)
        residual = subset["Fixed_Residual_Z"].to_numpy(float)
        variance = design @ g_matrix @ design.T + sigma_squared * np.eye(len(subset))
        try:
            effects = g_matrix @ design.T @ np.linalg.solve(variance, residual)
        except np.linalg.LinAlgError:
            effects = g_matrix @ design.T @ np.linalg.pinv(variance) @ residual
        random_intercept = float(effects[intercept_index])
        random_slope = float(effects[task_index])
        rows.append(
            {
                "Subject_ID": subject_id,
                "W": random_intercept,
                "CW": random_intercept + random_slope,
                "CW-W": random_slope,
                "N_W_Items": int((subset["Task"] == "W").sum()),
                "N_CW_Items": int((subset["Task"] == "CW").sum()),
            }
        )
    return pd.DataFrame(rows)


def fit_feature_model(feature: str, healthy: pd.DataFrame):
    data = healthy[["Subject_ID", "Task", "Task_Code", "Age_Z", feature]].copy()
    data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data = data.dropna()
    feature_mean = float(data[feature].mean())
    feature_sd = float(data[feature].std(ddof=0))
    if not np.isfinite(feature_sd) or feature_sd <= 0:
        raise RuntimeError(f"Invalid healthy feature SD for {feature}.")
    data["Feature_Z"] = (data[feature] - feature_mean) / feature_sd

    last_result = None
    messages = []
    for optimizer in ("lbfgs", "powell"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(
                    "Feature_Z ~ Age_Z * Task_Code",
                    data,
                    groups=data["Subject_ID"],
                    re_formula="~Task_Code",
                )
                result = model.fit(
                    method=optimizer, reml=False, maxiter=2000, disp=False
                )
            messages.extend(str(item.message) for item in caught)
            last_result = (result, optimizer)
            if bool(getattr(result, "converged", False)):
                break
        except Exception as error:
            messages.append(f"{optimizer}: {error}")
    if last_result is None:
        raise RuntimeError(f"Could not fit the healthy reference model for {feature}.")
    return (
        last_result[0],
        last_result[1],
        feature_mean,
        feature_sd,
        " | ".join(dict.fromkeys(messages)),
    )


def compute_deviations(
    healthy: pd.DataFrame, wd: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    healthy = healthy.copy()
    wd = wd.copy()
    healthy_ages = healthy[["Subject_ID", "Age"]].drop_duplicates("Subject_ID")
    age_mean = float(healthy_ages["Age"].mean())
    age_sd = float(healthy_ages["Age"].std(ddof=0))
    if not np.isfinite(age_sd) or age_sd <= 0:
        raise RuntimeError("Healthy age SD is invalid.")
    for frame in (healthy, wd):
        frame["Age_Z"] = (frame["Age"] - age_mean) / age_sd
        frame["Task_Code"] = frame["Task"].map(TASK_CODE).astype(int)

    feature_rows = []
    model_rows = []
    for index, feature in enumerate(SELECTED_FEATURES, start=1):
        print(f"[{index:02d}/{len(SELECTED_FEATURES)}] {feature}")
        model, optimizer, feature_mean, feature_sd, model_warnings = fit_feature_model(
            feature, healthy
        )
        healthy_effects = estimate_subject_effects(
            model, healthy, feature, feature_mean, feature_sd
        )
        wd_effects = estimate_subject_effects(model, wd, feature, feature_mean, feature_sd)
        if healthy_effects["Subject_ID"].nunique() != EXPECTED_HEALTHY_N:
            raise RuntimeError(f"Incomplete healthy random effects for {feature}.")
        if wd_effects["Subject_ID"].nunique() != sum(EXPECTED_GROUP_COUNTS.values()):
            raise RuntimeError(f"Incomplete WD random effects for {feature}.")

        model_row = {
            "Feature": feature,
            "Optimizer": optimizer,
            "Converged": bool(getattr(model, "converged", False)),
            "N_Healthy_Subjects": healthy_effects["Subject_ID"].nunique(),
            "N_WD_Subjects": wd_effects["Subject_ID"].nunique(),
            "Feature_Mean": feature_mean,
            "Feature_SD_DDof0": feature_sd,
            "Age_Mean": age_mean,
            "Age_SD_DDof0": age_sd,
            "Warnings": model_warnings,
        }
        for condition in CONDITIONS:
            reference_mean = float(healthy_effects[condition].mean())
            reference_sd = float(healthy_effects[condition].std(ddof=1))
            if not np.isfinite(reference_sd) or reference_sd <= 0:
                raise RuntimeError(
                    f"Invalid healthy {condition} random-effect SD for {feature}."
                )
            scores = wd_effects[
                ["Subject_ID", "N_W_Items", "N_CW_Items", condition]
            ].copy()
            scores["Feature"] = feature
            scores["Condition"] = condition
            scores["Deviation_Z"] = (
                scores[condition] - reference_mean
            ) / reference_sd
            scores["Abs_Deviation_Z"] = scores["Deviation_Z"].abs()
            scores["Healthy_Random_Effect_Mean"] = reference_mean
            scores["Healthy_Random_Effect_SD"] = reference_sd
            feature_rows.append(scores.drop(columns=[condition]))
            model_row[f"{condition}_Reference_Mean"] = reference_mean
            model_row[f"{condition}_Reference_SD_DDof1"] = reference_sd
        model_rows.append(model_row)

    feature_scores = pd.concat(feature_rows, ignore_index=True)
    group_map = wd[["Subject_ID", "Clinical_Group", "Age"]].drop_duplicates(
        "Subject_ID"
    )
    feature_scores = feature_scores.merge(
        group_map, on="Subject_ID", how="left", validate="many_to_one"
    )
    subject_scores = (
        feature_scores.groupby(
            ["Subject_ID", "Clinical_Group", "Age", "Condition"], as_index=False
        )
        .agg(
            N_Features=("Feature", "nunique"),
            Mean_Abs_Deviation_Z=("Abs_Deviation_Z", "mean"),
            RMS_Deviation_Z=(
                "Deviation_Z",
                lambda values: float(np.sqrt(np.mean(np.square(values)))),
            ),
        )
    )
    if not subject_scores["N_Features"].eq(len(SELECTED_FEATURES)).all():
        raise RuntimeError("Not every subject-condition summary contains all 25 features.")
    return pd.DataFrame(model_rows), feature_scores, subject_scores


def format_mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.3f} ± {values.std(ddof=1):.3f}"


def compare_groups(subject_scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for condition in CONDITIONS:
        subset = subject_scores[subject_scores["Condition"].eq(condition)]
        for measure, column in METRICS.items():
            wd_ci = subset.loc[subset["Clinical_Group"].eq("WD-CI"), column]
            wd_cn = subset.loc[subset["Clinical_Group"].eq("WD-CN"), column]
            test = mannwhitneyu(wd_ci, wd_cn, alternative="two-sided", method="asymptotic")
            rows.append(
                {
                    "Condition": condition,
                    "Measure": measure,
                    "WD_CI_N": len(wd_ci),
                    "WD_CI_Mean": wd_ci.mean(),
                    "WD_CI_SD": wd_ci.std(ddof=1),
                    "WD_CN_N": len(wd_cn),
                    "WD_CN_Mean": wd_cn.mean(),
                    "WD_CN_SD": wd_cn.std(ddof=1),
                    "Mann_Whitney_U": float(test.statistic),
                    "P_Value": float(test.pvalue),
                }
            )
    statistics = pd.DataFrame(rows)
    statistics["FDR_P_Value"] = multipletests(
        statistics["P_Value"].to_numpy(), method="fdr_bh"
    )[1]

    article_rows = []
    for _, row in statistics.iterrows():
        article_rows.append(
            {
                "Condition": row["Condition"],
                "Measure": row["Measure"],
                "WD-CI": f"{row['WD_CI_Mean']:.3f} ± {row['WD_CI_SD']:.3f}",
                "WD-CN": f"{row['WD_CN_Mean']:.3f} ± {row['WD_CN_SD']:.3f}",
                "p value": row["FDR_P_Value"],
            }
        )
    return statistics, pd.DataFrame(article_rows)


def violin_boxplot(
    axis: plt.Axes,
    subject_scores: pd.DataFrame,
    column: str,
    panel_label: str,
    title: str,
) -> None:
    centers = np.arange(len(CONDITIONS), dtype=float)
    offsets = {"WD-CI": -0.19, "WD-CN": 0.19}
    for condition_index, condition in enumerate(CONDITIONS):
        for group in GROUPS:
            values = subject_scores.loc[
                subject_scores["Condition"].eq(condition)
                & subject_scores["Clinical_Group"].eq(group),
                column,
            ].dropna().to_numpy(float)
            position = centers[condition_index] + offsets[group]
            violin = axis.violinplot(
                values,
                positions=[position],
                widths=0.34,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(GROUP_COLORS[group])
                body.set_edgecolor(GROUP_COLORS[group])
                body.set_alpha(0.25)
                body.set_linewidth(1.4)
            axis.boxplot(
                values,
                positions=[position],
                widths=0.17,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": GROUP_COLORS[group],
                    "edgecolor": "black",
                    "linewidth": 1.5,
                },
                medianprops={"color": "black", "linewidth": 1.6},
                whiskerprops={"color": "black", "linewidth": 1.4},
                capprops={"color": "black", "linewidth": 1.4},
            )
            axis.scatter(
                position,
                values.mean(),
                s=38,
                facecolor="white",
                edgecolor="black",
                linewidth=1.1,
                zorder=5,
            )

    axis.set_xticks(centers, CONDITIONS)
    axis.set_xlim(-0.65, len(CONDITIONS) - 0.35)
    axis.set_title(title, fontsize=18, fontweight="bold", pad=14)
    axis.grid(axis="y", linestyle=(0, (2, 2)), linewidth=0.8, color="#c9c9c9")
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.5)
    axis.spines["bottom"].set_linewidth(1.5)
    axis.tick_params(axis="both", labelsize=13, width=1.4, length=5)
    axis.text(
        -0.10,
        1.04,
        f"({panel_label})",
        transform=axis.transAxes,
        fontsize=20,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def plot_figure_6(subject_scores: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.labelweight": "bold",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.4))
    violin_boxplot(
        axes[0],
        subject_scores,
        "Mean_Abs_Deviation_Z",
        "A",
        r"Mean $|Z_{dev}|$",
    )
    violin_boxplot(
        axes[1], subject_scores, "RMS_Deviation_Z", "B", r"RMS $Z_{dev}$"
    )
    axes[0].set_ylabel("Age-referenced speech deviation", fontsize=16)
    axes[1].set_ylabel("")
    legend_handles = [
        Patch(
            facecolor=GROUP_COLORS[group],
            edgecolor="black",
            alpha=0.55,
            label=group,
        )
        for group in GROUPS
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93), w_pad=3.2)
    for suffix, options in (
        ("png", {"dpi": 600}),
        ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("pdf", {}),
    ):
        path = output_dir / f"Figure_6_W_CW_CW_minus_W.{suffix}"
        figure.savefig(path, bbox_inches="tight", facecolor="white", **options)
        print(f"Saved: {path}")
    plt.close(figure)


def write_notes(path: Path) -> None:
    path.write_text(
        """WD W/CW/CW-minus-W deviation analysis

1. The same 25 prespecified acoustic features are used for W, CW, and CW-minus-W.
2. Each healthy reference model is Feature_Z ~ Age_Z * Task_Code + (1 + Task_Code | Subject_ID), with W coded 0 and CW coded 1.
3. W is the participant random intercept, CW is the random intercept plus random task slope, and CW-minus-W is the random task slope.
4. Each feature-level effect is standardized against the matching empirical healthy BLUP distribution using its mean and sample SD.
5. Subject-level indices are limited to Mean |Z_dev| and RMS Z_dev.
6. Two-sided asymptotic Mann-Whitney U tests compare WD-CI and WD-CN. Benjamini-Hochberg correction is applied across all six condition-by-index tests.
7. Figure 6 shows violin distributions, embedded boxplots, and white mean markers for both groups under W, CW, and CW-minus-W.
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute W, CW, and CW-minus-W acoustic deviation indices and "
            "generate the WD group-comparison table and Figure 6."
        )
    )
    parser.add_argument("--healthy-features", type=Path, default=DEFAULT_HEALTHY_FEATURES)
    parser.add_argument("--healthy-info", type=Path, default=DEFAULT_HEALTHY_INFO)
    parser.add_argument("--wd-ci-features", type=Path, default=DEFAULT_WD_CI_FEATURES)
    parser.add_argument("--wd-cn-features", type=Path, default=DEFAULT_WD_CN_FEATURES)
    parser.add_argument("--wd-info", type=Path, default=DEFAULT_WD_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "healthy_features",
        "healthy_info",
        "wd_ci_features",
        "wd_cn_features",
        "wd_info",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    healthy, wd = load_inputs(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_summary, feature_scores, subject_scores = compute_deviations(healthy, wd)
    statistics, article_table = compare_groups(subject_scores)

    outputs = {
        "healthy_reference_LME_models.csv": model_summary,
        "WD_feature_level_deviation_scores.csv": feature_scores,
        "wd_subject_condition_deviation_indices.csv": subject_scores,
        "wd_ci_vs_wd_cn_statistics.csv": statistics,
        "table3_wd_deviation_indices_by_condition.csv": article_table,
    }
    for filename, frame in outputs.items():
        destination = args.output_dir / filename
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        print(f"Saved: {destination}")
    plot_figure_6(subject_scores, args.output_dir)
    write_notes(args.output_dir / "analysis_notes.txt")
    print(f"Saved: {args.output_dir / 'analysis_notes.txt'}")


if __name__ == "__main__":
    main()
