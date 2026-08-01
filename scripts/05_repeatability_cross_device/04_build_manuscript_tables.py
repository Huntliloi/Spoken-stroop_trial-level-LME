"""Build manuscript Tables 4, S7, and S8 from feature-level ICC results."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from common_paths import DEVICE_REPEATABILITY_OUTPUT_DIR, RESULTS_DIR  # noqa: E402


PUBLIC_RESULTS = RESULTS_DIR / "03_repeatability_cross_device"
OUTPUT_DIR = DEVICE_REPEATABILITY_OUTPUT_DIR


def input_path(filename: str) -> Path:
    generated = OUTPUT_DIR / filename
    return generated if generated.exists() else PUBLIC_RESULTS / filename


def category_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["ICC_Category"].value_counts()
    return {name.title(): int(counts.get(name, 0)) for name in ["excellent", "good", "moderate", "poor"]}


def build_table4() -> pd.DataFrame:
    frame = pd.read_csv(input_path("within_device_feature_level_icc.csv"))
    labels = {
        1: "ASUS FX63VD (PC)",
        2: "Apple iPad 2021 (iPad)",
        3: "HUAWEI P30 (smartphone)",
    }
    rows = []
    for device_id, group in frame.groupby("Device_ID", sort=True):
        rows.append({
            "Device": labels[int(device_id)],
            "Mean_ICC": group["ICC_3_1_consistency"].mean(),
            "Median_ICC": group["ICC_3_1_consistency"].median(),
            **category_counts(group),
        })
    return pd.DataFrame(rows)


def summarize_cross_device(label: str, frame: pd.DataFrame, correlations: bool) -> dict[str, object]:
    row = {
        "Comparison": label,
        "Mean_ICC": frame["ICC_2_1_absolute_agreement"].mean(),
        "Median_ICC": frame["ICC_2_1_absolute_agreement"].median(),
        **category_counts(frame),
        "Median_Pearson_r": np.nan,
        "Median_Spearman_rho": np.nan,
    }
    if correlations:
        row["Median_Pearson_r"] = frame["Pearson_r"].median()
        row["Median_Spearman_rho"] = frame["Spearman_rho"].median()
    return row


def build_table_s7() -> pd.DataFrame:
    all3 = pd.read_csv(input_path("cross_device_all3_overall.csv"))
    pairwise = pd.read_csv(input_path("cross_device_pairwise_overall.csv"))
    rows = [summarize_cross_device("All three devices", all3, False)]
    for code, label in [
        ("1_vs_2", "Laptop vs iPad"),
        ("1_vs_3", "Laptop vs smartphone"),
        ("2_vs_3", "iPad vs smartphone"),
    ]:
        rows.append(summarize_cross_device(label, pairwise[pairwise["Device_Pair"].eq(code)], True))
    return pd.DataFrame(rows)


def build_table_s8() -> pd.DataFrame:
    all3 = pd.read_csv(input_path("cross_device_all3_overall.csv"))
    return all3.loc[
        all3["ICC_2_1_absolute_agreement"].ge(0.50),
        ["Feature", "ICC_2_1_absolute_agreement", "ICC_Category"],
    ].sort_values("ICC_2_1_absolute_agreement", ascending=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "table4_within_device_retest_summary.csv": build_table4(),
        "table_s7_cross_device_summary.csv": build_table_s7(),
        "table_s8_cross_device_features_icc_ge_0_50.csv": build_table_s8(),
    }
    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
