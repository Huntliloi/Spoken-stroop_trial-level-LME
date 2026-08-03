import os
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import soundfile as sf

import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False



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

TOKEN_CSV = str(TOKEN_RESULTS_CSV)  # noqa: F405

AUDIO_ROOT = str(AUDIO_ROOT)  # noqa: F405
OUTPUT_DIR = str(FIGURE_OUTPUT_DIR / "waveforms")  # noqa: F405


# =========================
# =========================

TASK_FILTER = None  # for example ["hanzi", "dots", "STT"]

SUBJECT_FILTER = ["xxxxxxxx001", "xxxxxxxx002"]
# SUBJECT_FILTER = None

PLOT_RAW_COMPARISON = False

SHOW_TOKEN_TEXT = True


# =========================
# =========================

FIGSIZE = (16, 5)

X_LIM = (0, 10)

# Y_LIM = (-0.4, 0.4)
Y_LIM = None

LABEL_FONTSIZE = 36

REFINED_LINE_COLOR = "tab:red"
REFINED_TEXT_COLOR = "tab:red"

RAW_LINE_COLOR = "tab:blue"
RAW_TEXT_COLOR = "tab:blue"


# =========================
# =========================

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_task_for_filename(task: str) -> str:
    task = str(task).strip()
    if task.lower() == "dots":
        return "dots"
    if task.lower() == "hanzi":
        return "hanzi"
    if task.lower() == "stt":
        return "STT"
    return task


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_word(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    mapping = {'\u7ea2\u8272': '\u7ea2', '\u9ec4\u8272': '\u9ec4', '\u84dd\u8272': '\u84dd', '\u7eff\u8272': '\u7eff'}
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s


def infer_audio_path(subject_id: str, task: str, audio_root: str) -> Optional[Path]:
    """
    Infer the public example wav path for a subject and task.

    Expected layout:
    data/example/audio/<anonymized_subject_folder>/session_1/<subject_id>_stroop_<task>.wav
    """
    root = Path(audio_root)
    task_norm = normalize_task_for_filename(task)

    candidates = list(root.glob(f"*_{subject_id}"))
    if not candidates:
        return None

    subj_dir = candidates[0]

    wav_name = f"{subject_id}_stroop_{task_norm}.wav"
    wav_path = subj_dir / "session_1" / "04_Stroop" / wav_name
    if wav_path.exists():
        return wav_path

    wav_name_alt = f"{subject_id}_stroop_{task_norm.lower()}.wav"
    wav_path_alt = subj_dir / "session_1" / "04_Stroop" / wav_name_alt
    if wav_path_alt.exists():
        return wav_path_alt

    return None


def load_audio(audio_path: Path):
    y, sr = sf.read(str(audio_path))
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y, sr


def make_token_label(row: pd.Series) -> str:
    txt = normalize_word(row.get("token_text", ""))

    color_to_letter = {
        '\u7ea2': "R",
        '\u9ec4': "Y",
        '\u84dd': "B",
        '\u7eff': "G",
    }

    if SHOW_TOKEN_TEXT:
        return color_to_letter.get(txt, txt)

    idx = row.get("token_index", np.nan)
    if pd.notna(idx):
        try:
            return str(int(idx))
        except Exception:
            return str(idx)

    return color_to_letter.get(txt, txt)


def plot_waveform_with_timestamps(
    y: np.ndarray,
    sr: int,
    refined_tokens: pd.DataFrame,
    outpath: Path,
    title: str,
    raw_tokens: Optional[pd.DataFrame] = None,
):
    duration = len(y) / sr
    t = np.arange(len(y)) / sr

    plt.figure(figsize=FIGSIZE, dpi=1200)
    plt.plot(t, y, linewidth=0.6, color="black")
    plt.axhline(0, linewidth=0.6, color="gray", alpha=0.6)

    # =========================
    # =========================
    if Y_LIM is not None:
        ymin, ymax = Y_LIM
        plt.ylim(ymin, ymax)
    else:
        ymin, ymax = plt.ylim()

    yrange = ymax - ymin

    # =========================
    # =========================
    if raw_tokens is not None and len(raw_tokens) > 0:
        for _, row in raw_tokens.iterrows():
            start_t = row["start_time_raw"]
            end_t = row["end_time_raw"]
            label = make_token_label(row)

            if pd.notna(start_t):
                plt.axvline(
                    start_t,
                    color=RAW_LINE_COLOR,
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.45,
                )

            if pd.notna(end_t):
                plt.axvline(
                    end_t,
                    color=RAW_LINE_COLOR,
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.45,
                )

            if pd.notna(start_t) and pd.notna(end_t):
                mid = (start_t + end_t) / 2
            elif pd.notna(start_t):
                mid = start_t
            elif pd.notna(end_t):
                mid = end_t
            else:
                continue

            plt.text(
                mid,
                ymin + 0.10 * yrange,
                label,
                color=RAW_TEXT_COLOR,
                fontsize=LABEL_FONTSIZE - 1,
                ha="center",
                va="bottom",
                rotation=0,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.55,
                    pad=1.2,
                ),
            )

    # =========================
    # =========================
    for _, row in refined_tokens.iterrows():
        start_t = row["start_time"]
        end_t = row["end_time"]
        label = make_token_label(row)

        if pd.notna(start_t):
            plt.axvline(
                start_t,
                color=REFINED_LINE_COLOR,
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
            )

        if pd.notna(end_t):
            plt.axvline(
                end_t,
                color=REFINED_LINE_COLOR,
                linestyle=":",
                linewidth=1.2,
                alpha=0.85,
            )

        if pd.notna(start_t) and pd.notna(end_t):
            mid = (start_t + end_t) / 2
        elif pd.notna(start_t):
            mid = start_t
        elif pd.notna(end_t):
            mid = end_t
        else:
            continue

        plt.text(
            mid,
            ymax - 0.08 * yrange,
            label,
            color=REFINED_TEXT_COLOR,
            fontsize=LABEL_FONTSIZE,
            ha="center",
            va="top",
            rotation=0,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.7,
                pad=1.5,
            ),
        )

    # =========================
    # =========================
    if X_LIM is not None:
        plt.xlim(X_LIM)
    else:
        plt.xlim(0, duration)

    plt.xlabel("Time (s)", fontsize=36)
    plt.ylabel("Amplitude", fontsize=36)
    plt.title(title)
    plt.xticks(fontsize=36)
    plt.yticks(fontsize=36)
    plt.tight_layout()

    plt.savefig(outpath)
    plt.close()


# =========================
# =========================

def main():
    ensure_dir(OUTPUT_DIR)

    df = pd.read_csv(TOKEN_CSV)

    subject_col = find_first_existing_col(df, ["subject_id"])
    task_col = find_first_existing_col(df, ["task"])
    token_idx_col = find_first_existing_col(df, ["token_index", "Item_Index", "idx"])
    token_text_col = find_first_existing_col(
        df,
        ["token_text", "word", "label", "aligned_text", "word_auto"],
    )

    refined_start_col = find_first_existing_col(
        df,
        ["start_time", "auto_start_time", "start_refined", "refined_start_time"],
    )
    refined_end_col = find_first_existing_col(
        df,
        ["end_time", "auto_end_time", "end_refined", "refined_end_time"],
    )

    raw_start_col = find_first_existing_col(
        df,
        ["start_time_raw", "raw_start_time", "start_raw"],
    )
    raw_end_col = find_first_existing_col(
        df,
        ["end_time_raw", "raw_end_time", "end_raw"],
    )

    required = {
        "subject_id": subject_col,
        "task": task_col,
        "token_index": token_idx_col,
        "refined_start": refined_start_col,
        "refined_end": refined_end_col,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"TOKEN_CSV Missing required columns: {missing}")

    work = df.copy()
    work["subject_id"] = work[subject_col].astype(str)
    work["task"] = work[task_col].astype(str)
    work["token_index"] = pd.to_numeric(work[token_idx_col], errors="coerce")

    if token_text_col is not None:
        work["token_text"] = work[token_text_col].map(normalize_word)
    else:
        work["token_text"] = ""

    work["start_time"] = pd.to_numeric(work[refined_start_col], errors="coerce")
    work["end_time"] = pd.to_numeric(work[refined_end_col], errors="coerce")

    if raw_start_col is not None:
        work["start_time_raw"] = pd.to_numeric(work[raw_start_col], errors="coerce")
    else:
        work["start_time_raw"] = np.nan

    if raw_end_col is not None:
        work["end_time_raw"] = pd.to_numeric(work[raw_end_col], errors="coerce")
    else:
        work["end_time_raw"] = np.nan

    if TASK_FILTER is not None:
        work = work[work["task"].isin(TASK_FILTER)].copy()

    if SUBJECT_FILTER is not None:
        subject_filter_str = {str(x) for x in SUBJECT_FILTER}
        work = work[work["subject_id"].isin(subject_filter_str)].copy()

    work = work[work["start_time"].notna() | work["end_time"].notna()].copy()

    grouped = work.groupby(["subject_id", "task"], dropna=False)
    summary_rows: List[Dict] = []

    for (subject_id, task), g in grouped:
        g = g.sort_values("token_index").copy()
        audio_path = infer_audio_path(subject_id, task, AUDIO_ROOT)

        if audio_path is None or not audio_path.exists():
            summary_rows.append(
                {
                    "subject_id": subject_id,
                    "task": task,
                    "n_tokens": len(g),
                    "status": "audio_not_found",
                    "audio_path": None,
                    "output_png": None,
                }
            )
            continue

        try:
            y, sr = load_audio(audio_path)
        except Exception as e:
            summary_rows.append(
                {
                    "subject_id": subject_id,
                    "task": task,
                    "n_tokens": len(g),
                    "status": f"audio_load_error: {e}",
                    "audio_path": str(audio_path),
                    "output_png": None,
                }
            )
            continue

        raw_tokens = None
        if PLOT_RAW_COMPARISON and (
            "start_time_raw" in g.columns or "end_time_raw" in g.columns
        ):
            raw_tokens = g.copy()

        outname = f"{subject_id}_{normalize_task_for_filename(task)}_waveform_refined.png"
        outpath = Path(OUTPUT_DIR) / outname
        title = f"{subject_id} | {task} | refined timestamps from token_results.csv"

        try:
            plot_waveform_with_timestamps(
                y=y,
                sr=sr,
                refined_tokens=g,
                raw_tokens=raw_tokens,
                outpath=outpath,
                title=title,
            )
            status = "ok"
        except Exception as e:
            status = f"plot_error: {e}"
            outpath = None

        summary_rows.append(
            {
                "subject_id": subject_id,
                "task": task,
                "n_tokens": len(g),
                "status": status,
                "audio_path": str(audio_path),
                "output_png": str(outpath) if outpath is not None else None,
            }
        )

    summary = pd.DataFrame(summary_rows)

    # summary.to_csv(Path(OUTPUT_DIR) / "waveform_plot_summary.csv", index=False, encoding="utf-8-sig")

    print('Done. Results saved to:', OUTPUT_DIR)


if __name__ == "__main__":
    main()



