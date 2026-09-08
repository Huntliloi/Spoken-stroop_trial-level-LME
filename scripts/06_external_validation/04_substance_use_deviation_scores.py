"""Evaluate W, CW, and CW-minus-W acoustic deviations in substance use.

This script applies the same 25-feature healthy-reference LME procedure as the
expanded WD S.2 analysis. It calculates only Mean |Z_dev| and RMS Z_dev for W,
CW, and CW-minus-W, evaluates their associations with MoCA, updates the
categorical group comparisons, and generates the expanded Figure 7.
"""

from __future__ import annotations

import argparse
import importlib.util
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
PUBLIC_DATA_DIR = REPO_ROOT / "data" / "public"
PUBLIC_SUBSTANCE_DIR = PUBLIC_DATA_DIR / "substance_use"
REFERENCE_S2 = SCRIPT_DIR / "01_wd_deviation_scores.py"
DEFAULT_HEALTHY_FEATURES = PUBLIC_DATA_DIR / "features" / "Automated_feature.csv"
DEFAULT_HEALTHY_INFO = PUBLIC_DATA_DIR / "participant_info_HC_158.xlsx"
DEFAULT_DRUG_FEATURES = (
    PUBLIC_SUBSTANCE_DIR / "features" / "Stroop_features_Drug.csv"
)
DEFAULT_HC_FEATURES = PUBLIC_SUBSTANCE_DIR / "features" / "Stroop_features_HC.csv"
DEFAULT_PARTICIPANT_INFO = (
    PUBLIC_SUBSTANCE_DIR / "participant_info_substance_use_58.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "substance_use_external_validation"

EXPECTED_GROUP_COUNTS = {"Drug": 28, "HC": 30}
CONDITIONS = ("W", "CW", "CW-W")
METRICS = {
    "Mean |Z_dev|": "Mean_Abs_Deviation_Z",
    "RMS Z_dev": "RMS_Deviation_Z",
}
MOCA_COLORS = {"MoCA-CI": "#ff1f1f", "MoCA-CN": "#42bde3"}


def load_module(path: Path, module_name: str):
    if not path.exists():
        raise FileNotFoundError(path)
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


reference = load_module(REFERENCE_S2, "expanded_wd_s2_reference")


def load_participant_info(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, encoding="utf-8-sig")
    subject_column = reference.find_column(
        frame, ["Subject_ID", "subject_id", "Subject", "ID"]
    )
    group_column = reference.find_column(frame, ["Group", "group"])
    age_column = reference.find_column(frame, ["Age", "age"])
    education_column = reference.find_column(
        frame, ["Education", "education", "Education_Level"]
    )
    moca_column = reference.find_column(frame, ["MoCA", "moca"])
    required = {
        "Subject_ID": subject_column,
        "Group": group_column,
        "Age": age_column,
        "Education": education_column,
        "MoCA": moca_column,
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(f"Participant information lacks columns: {missing}")
    result = frame[list(required.values())].copy()
    result.columns = list(required)
    result["Subject_ID"] = result["Subject_ID"].map(reference.normalize_subject_id)
    result["Group"] = result["Group"].astype(str).str.strip()
    for column in ("Age", "Education", "MoCA"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["Subject_ID", "Group", "Age", "Education", "MoCA"]].isna().any().any():
        raise ValueError("Participant information contains missing required values.")
    result["MoCA_Corrected"] = np.minimum(
        result["MoCA"] + result["Education"].le(12).astype(int), 30
    )
    result["MoCA_Group"] = np.where(
        result["MoCA_Corrected"].lt(26), "MoCA-CI", "MoCA-CN"
    )
    return result.drop_duplicates("Subject_ID")


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    healthy = reference.normalize_items(
        reference.read_csv(args.healthy_features), "healthy features"
    )
    healthy_age = reference.load_age_table(args.healthy_info, "healthy information")
    healthy = healthy.merge(
        healthy_age, on="Subject_ID", how="inner", validate="many_to_one"
    )
    if healthy["Subject_ID"].nunique() != reference.EXPECTED_HEALTHY_N:
        raise ValueError(
            f"Expected {reference.EXPECTED_HEALTHY_N} healthy participants, found "
            f"{healthy['Subject_ID'].nunique()}."
        )

    scoring = pd.concat(
        [
            reference.normalize_items(
                reference.read_csv(args.drug_features), "Drug features"
            ),
            reference.normalize_items(reference.read_csv(args.hc_features), "HC features"),
        ],
        ignore_index=True,
    )
    scoring = scoring.drop(
        columns=[
            "Group",
            "Age",
            "Education",
            "MoCA",
            "MoCA_Corrected",
            "MoCA_Group",
        ],
        errors="ignore",
    )
    participant_info = load_participant_info(args.participant_info)
    scoring = scoring.merge(
        participant_info,
        on="Subject_ID",
        how="inner",
        validate="many_to_one",
    )
    counts = (
        scoring[["Subject_ID", "Group"]]
        .drop_duplicates()["Group"]
        .value_counts()
        .to_dict()
    )
    if counts != EXPECTED_GROUP_COUNTS:
        raise ValueError(
            f"Expected substance-use group counts {EXPECTED_GROUP_COUNTS}, found {counts}."
        )
    scoring["Clinical_Group"] = scoring["Group"]

    missing_healthy = sorted(
        set(reference.SELECTED_FEATURES).difference(healthy.columns)
    )
    missing_scoring = sorted(
        set(reference.SELECTED_FEATURES).difference(scoring.columns)
    )
    if missing_healthy or missing_scoring:
        raise ValueError(
            f"Missing prespecified features; healthy={missing_healthy}, "
            f"substance-use={missing_scoring}."
        )
    return healthy, scoring


def compute_deviations(
    healthy: pd.DataFrame, scoring: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    healthy = healthy.copy()
    scoring = scoring.copy()
    healthy_ages = healthy[["Subject_ID", "Age"]].drop_duplicates("Subject_ID")
    age_mean = float(healthy_ages["Age"].mean())
    age_sd = float(healthy_ages["Age"].std(ddof=0))
    if not np.isfinite(age_sd) or age_sd <= 0:
        raise RuntimeError("Healthy age SD is invalid.")
    for frame in (healthy, scoring):
        frame["Age_Z"] = (frame["Age"] - age_mean) / age_sd
        frame["Task_Code"] = frame["Task"].map(reference.TASK_CODE).astype(int)

    feature_rows = []
    model_rows = []
    expected_n = sum(EXPECTED_GROUP_COUNTS.values())
    for index, feature in enumerate(reference.SELECTED_FEATURES, start=1):
        print(f"[{index:02d}/{len(reference.SELECTED_FEATURES)}] {feature}")
        model, optimizer, feature_mean, feature_sd, model_warnings = (
            reference.fit_feature_model(feature, healthy)
        )
        healthy_effects = reference.estimate_subject_effects(
            model, healthy, feature, feature_mean, feature_sd
        )
        scoring_effects = reference.estimate_subject_effects(
            model, scoring, feature, feature_mean, feature_sd
        )
        if healthy_effects["Subject_ID"].nunique() != reference.EXPECTED_HEALTHY_N:
            raise RuntimeError(f"Incomplete healthy random effects for {feature}.")
        if scoring_effects["Subject_ID"].nunique() != expected_n:
            raise RuntimeError(
                f"Expected {expected_n} substance-use random effects for {feature}, "
                f"found {scoring_effects['Subject_ID'].nunique()}."
            )

        model_row = {
            "Feature": feature,
            "Optimizer": optimizer,
            "Converged": bool(getattr(model, "converged", False)),
            "N_Healthy_Subjects": healthy_effects["Subject_ID"].nunique(),
            "N_Scoring_Subjects": scoring_effects["Subject_ID"].nunique(),
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
            scores = scoring_effects[
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
    metadata = scoring[
        [
            "Subject_ID",
            "Group",
            "Age",
            "Education",
            "MoCA",
            "MoCA_Corrected",
            "MoCA_Group",
        ]
    ].drop_duplicates("Subject_ID")
    feature_scores = feature_scores.merge(
        metadata, on="Subject_ID", how="left", validate="many_to_one"
    )
    subject_scores = (
        feature_scores.groupby(
            [
                "Subject_ID",
                "Group",
                "Age",
                "Education",
                "MoCA",
                "MoCA_Corrected",
                "MoCA_Group",
                "Condition",
            ],
            as_index=False,
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
    if not subject_scores["N_Features"].eq(len(reference.SELECTED_FEATURES)).all():
        raise RuntimeError("Not every subject-condition summary contains all 25 features.")
    return pd.DataFrame(model_rows), feature_scores, subject_scores


def compare_groups(
    frame: pd.DataFrame,
    group_column: str,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        condition_data = frame[frame["Condition"].eq(condition)]
        for measure, column in METRICS.items():
            positive = condition_data.loc[
                condition_data[group_column].eq(positive_label), column
            ].dropna()
            negative = condition_data.loc[
                condition_data[group_column].eq(negative_label), column
            ].dropna()
            test = stats.mannwhitneyu(
                positive, negative, alternative="two-sided", method="asymptotic"
            )
            rows.append(
                {
                    "Condition": condition,
                    "Measure": measure,
                    f"{positive_label}_N": len(positive),
                    f"{positive_label}_Mean": positive.mean(),
                    f"{positive_label}_SD": positive.std(ddof=1),
                    f"{negative_label}_N": len(negative),
                    f"{negative_label}_Mean": negative.mean(),
                    f"{negative_label}_SD": negative.std(ddof=1),
                    "Mann_Whitney_U": float(test.statistic),
                    "P_Raw": float(test.pvalue),
                }
            )
    result = pd.DataFrame(rows)
    result["P_FDR_BH"] = multipletests(result["P_Raw"], method="fdr_bh")[1]
    return result


def build_table_4(subject_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for condition in CONDITIONS:
        condition_data = subject_scores[subject_scores["Condition"].eq(condition)]
        for measure, column in METRICS.items():
            row = {"Condition": condition, "Measure": measure}
            rho, p_value = stats.spearmanr(
                condition_data[column], condition_data["MoCA"], nan_policy="omit"
            )
            row["N"] = len(condition_data)
            row["Rho"] = float(rho)
            row["P_Raw"] = float(p_value)
            rows.append(row)
    result = pd.DataFrame(rows)
    result["P_FDR_BH"] = multipletests(result["P_Raw"], method="fdr_bh")[1]
    return result[["Condition", "Measure", "N", "Rho", "P_Raw", "P_FDR_BH"]]


def violin_boxplot(
    axis: plt.Axes,
    frame: pd.DataFrame,
    metric_column: str,
    group_column: str,
    groups: tuple[str, str],
    colors: dict[str, str],
    panel_label: str,
    title: str,
    show_legend: bool,
) -> None:
    centers = np.arange(len(CONDITIONS), dtype=float)
    offsets = {groups[0]: -0.19, groups[1]: 0.19}
    for condition_index, condition in enumerate(CONDITIONS):
        for group in groups:
            values = frame.loc[
                frame["Condition"].eq(condition) & frame[group_column].eq(group),
                metric_column,
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
                body.set_facecolor(colors[group])
                body.set_edgecolor(colors[group])
                body.set_alpha(0.25)
                body.set_linewidth(1.2)
            axis.boxplot(
                values,
                positions=[position],
                widths=0.17,
                patch_artist=True,
                showfliers=False,
                boxprops={
                    "facecolor": colors[group],
                    "edgecolor": "black",
                    "linewidth": 1.3,
                },
                medianprops={"color": "black", "linewidth": 1.4},
                whiskerprops={"color": "black", "linewidth": 1.2},
                capprops={"color": "black", "linewidth": 1.2},
            )
            axis.scatter(
                position,
                values.mean(),
                s=30,
                facecolor="white",
                edgecolor="black",
                linewidth=1.0,
                zorder=5,
            )

    axis.set_xticks(centers, CONDITIONS)
    axis.set_xlim(-0.65, len(CONDITIONS) - 0.35)
    axis.set_title(title, fontsize=16, fontweight="bold", pad=10)
    axis.grid(axis="y", linestyle=(0, (2, 2)), linewidth=0.7, color="#c9c9c9")
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(1.3)
    axis.spines["bottom"].set_linewidth(1.3)
    axis.tick_params(axis="both", labelsize=11, width=1.2, length=4)
    axis.text(
        -0.11,
        1.04,
        f"({panel_label})",
        transform=axis.transAxes,
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    if show_legend:
        handles = [
            Patch(
                facecolor=colors[group],
                edgecolor="black",
                alpha=0.55,
                label=group,
            )
            for group in groups
        ]
        axis.legend(handles=handles, loc="upper right", frameon=False, fontsize=10)


def plot_figure_7(subject_scores: pd.DataFrame, output_dir: Path) -> None:
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
        "MoCA_Group",
        ("MoCA-CI", "MoCA-CN"),
        MOCA_COLORS,
        "A",
        r"Mean $|Z_{dev}|$",
        True,
    )
    violin_boxplot(
        axes[1],
        subject_scores,
        "RMS_Deviation_Z",
        "MoCA_Group",
        ("MoCA-CI", "MoCA-CN"),
        MOCA_COLORS,
        "B",
        r"RMS $Z_{dev}$",
        False,
    )
    axes[0].set_ylabel("Age-referenced speech deviation", fontsize=14)
    axes[1].set_ylabel("")
    figure.tight_layout(w_pad=2.8)
    for suffix, options in (
        ("png", {"dpi": 600}),
        ("tif", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
        ("pdf", {}),
    ):
        destination = output_dir / f"Figure_7_W_CW_CW_minus_W.{suffix}"
        figure.savefig(destination, bbox_inches="tight", facecolor="white", **options)
        print(f"Saved: {destination}")
    plt.close(figure)


def write_notes(path: Path) -> None:
    path.write_text(
        """Substance-use W/CW/CW-minus-W deviation analysis

1. The same 25 prespecified acoustic features and healthy-reference LME specification used in the WD analysis are applied without cohort-specific feature reselection.
2. W is the participant random intercept, CW is the random intercept plus random task slope, and CW-minus-W is the random task slope.
3. Subject-level indices are limited to Mean |Z_dev| and RMS Z_dev.
4. Table 4 reports Spearman correlations with raw MoCA in the full cohort. Benjamini-Hochberg correction is applied across the six condition-by-index correlations.
5. For categorical analyses, one point is added to MoCA for education of 12 years or less, capped at 30; corrected scores below 26 define MoCA-CI.
6. Table S12 compares MoCA-CI with MoCA-CN across six indices. Table S13 reports the secondary exploratory Drug-versus-HC comparison. BH correction is applied across the six tests in each table.
7. Figure 7 compares MoCA-CI with MoCA-CN. Panels A and B display Mean |Z_dev| and RMS Z_dev, respectively, for W, CW, and CW-minus-W.
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute W, CW, and CW-minus-W acoustic deviation indices, "
            "MoCA correlations, group comparisons, and Figure 7."
        )
    )
    parser.add_argument("--healthy-features", type=Path, default=DEFAULT_HEALTHY_FEATURES)
    parser.add_argument("--healthy-info", type=Path, default=DEFAULT_HEALTHY_INFO)
    parser.add_argument("--drug-features", type=Path, default=DEFAULT_DRUG_FEATURES)
    parser.add_argument("--hc-features", type=Path, default=DEFAULT_HC_FEATURES)
    parser.add_argument("--participant-info", type=Path, default=DEFAULT_PARTICIPANT_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "healthy_features",
        "healthy_info",
        "drug_features",
        "hc_features",
        "participant_info",
        "output_dir",
    ):
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    healthy, scoring = load_inputs(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_summary, feature_scores, subject_scores = compute_deviations(
            healthy, scoring
        )
    table_4 = build_table_4(subject_scores)
    table_s12 = compare_groups(
        subject_scores, "MoCA_Group", "MoCA-CI", "MoCA-CN"
    )
    table_s13 = compare_groups(subject_scores, "Group", "Drug", "HC")

    outputs = {
        "healthy_reference_LME_models.csv": model_summary,
        "substance_use_feature_level_deviation_scores.csv": feature_scores,
        "subject_condition_deviation_indices.csv": subject_scores,
        "table4_moca_correlations_by_condition.csv": table_4,
        "table_s12_moca_groups_by_condition.csv": table_s12,
        "table_s13_drug_vs_hc_by_condition.csv": table_s13,
    }
    for filename, frame in outputs.items():
        destination = args.output_dir / filename
        frame.to_csv(destination, index=False, encoding="utf-8-sig")
        print(f"Saved: {destination}")
    plot_figure_7(subject_scores, args.output_dir)
    write_notes(args.output_dir / "analysis_notes.txt")
    print(f"Saved: {args.output_dir / 'analysis_notes.txt'}")


if __name__ == "__main__":
    main()
