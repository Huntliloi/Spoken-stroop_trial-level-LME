
# Public repository path setup
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_SCRIPTS_DIR = _RepoPath(__file__).resolve().parents[2] / "scripts"
if str(_REPO_SCRIPTS_DIR) not in _repo_sys.path:
    _repo_sys.path.insert(0, str(_REPO_SCRIPTS_DIR))
from common_paths import *  # noqa: F403
ensure_output_dirs()
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import pearsonr, spearmanr
except Exception as e:
    raise ImportError("scipy is required. Install it with: pip install scipy") from e


TOKEN_CSV = str(TOKEN_RESULTS_CSV)  # noqa: F405
TEXTGRID_DIR = str(MANUAL_TEXTGRID_ROOT)  # noqa: F405
OUTPUT_DIR = str(VALIDATION_OUTPUT_DIR / "token_vs_textgrid_correlation")  # noqa: F405

AUTO_START_COL = "start_time"
AUTO_END_COL = "end_time"

TASK_FILTER = None
SUBJECT_FILTER = None


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


def normalize_word(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    mapping = {'\u7ea2\u8272': '\u7ea2', '\u9ec4\u8272': '\u9ec4', '\u84dd\u8272': '\u84dd', '\u7eff\u8272': '\u7eff'}
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def read_text_fallback(path: Path) -> str:
    encodings = ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "gb18030", "ansi"]
    last_err = None
    for enc in encodings:
        try:
            return path.read_text(encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot read TextGrid: {path}\nLast error: {last_err}")


def list_all_textgrids(textgrid_dir: str) -> List[Path]:
    base = Path(textgrid_dir)
    files = []
    files.extend(base.rglob("*.TextGrid"))
    files.extend(base.rglob("*.textgrid"))
    uniq = []
    seen = set()
    for p in files:
        k = str(p.resolve()) if p.exists() else str(p)
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return sorted(uniq)


def infer_textgrid_path(subject_id: str, task: str, textgrid_dir: str) -> Optional[Path]:
    task_norm = normalize_task_for_filename(task)
    all_tgs = list_all_textgrids(textgrid_dir)

    exact_names = {
        f"{subject_id}_stroop_{task_norm}.TextGrid",
        f"{subject_id}_stroop_{task_norm.lower()}.TextGrid",
        f"{subject_id}_stroop_{task_norm}.textgrid",
        f"{subject_id}_stroop_{task_norm.lower()}.textgrid",
    }

    for p in all_tgs:
        if p.name in exact_names:
            return p

    task_key = task_norm.lower()
    for p in all_tgs:
        name_low = p.name.lower()
        if subject_id in p.name and task_key in name_low:
            return p

    return None


def parse_textgrid_word_intervals(textgrid_path: Path) -> List[Dict]:
    text = read_text_fallback(textgrid_path)

    item_positions = list(re.finditer(r'^\s*item \[\d+\]:', text, flags=re.M))
    blocks: List[str] = []
    if item_positions:
        for i, m in enumerate(item_positions):
            start = m.start()
            end = item_positions[i + 1].start() if i + 1 < len(item_positions) else len(text)
            blocks.append(text[start:end])
    else:
        blocks = [text]

    target_block = None
    for block in blocks:
        if re.search(r'class\s*=\s*"IntervalTier"', block) and re.search(r'name\s*=\s*"word"', block):
            target_block = block
            break

    if target_block is None:
        for block in blocks:
            if re.search(r'class\s*=\s*"IntervalTier"', block):
                target_block = block
                break

    if target_block is None:
        return []

    pattern = re.compile(
        r'intervals \[\d+\]:\s*'
        r'xmin = ([0-9eE+\-\.]+)\s*'
        r'xmax = ([0-9eE+\-\.]+)\s*'
        r'text = "(.*?)"',
        flags=re.S
    )

    out: List[Dict] = []
    for m in pattern.finditer(target_block):
        xmin = float(m.group(1))
        xmax = float(m.group(2))
        word = normalize_word(m.group(3))
        if word == "":
            continue
        out.append({
            "manual_start_time": xmin,
            "manual_end_time": xmax,
            "word_manual": word,
        })

    return out


def safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if len(x) < 2:
        return np.nan, np.nan, np.nan, np.nan

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, np.nan, np.nan, np.nan

    try:
        pear_r, pear_p = pearsonr(x, y)
    except Exception:
        pear_r, pear_p = np.nan, np.nan

    try:
        spear_r, spear_p = spearmanr(x, y)
    except Exception:
        spear_r, spear_p = np.nan, np.nan

    return float(pear_r), float(pear_p), float(spear_r), float(spear_p)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(x[ok] - y[ok])))


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))


def build_auto_tokens(df_group: pd.DataFrame) -> pd.DataFrame:
    out = df_group.copy()
    out = out.sort_values("token_index").reset_index(drop=True)
    out["word_auto"] = out["token_text"].map(normalize_word)
    out["auto_start_time"] = pd.to_numeric(out["auto_start_time"], errors="coerce")
    out["auto_end_time"] = pd.to_numeric(out["auto_end_time"], errors="coerce")
    return out[["token_index", "word_auto", "auto_start_time", "auto_end_time"]].copy()


def pair_auto_manual(auto_tokens: pd.DataFrame, manual_tokens: List[Dict]) -> pd.DataFrame:
    manual_df = pd.DataFrame(manual_tokens).copy()
    if manual_df.empty:
        return pd.DataFrame()

    manual_df = manual_df.reset_index(drop=True)
    manual_df["token_index"] = np.arange(1, len(manual_df) + 1)

    n = min(len(auto_tokens), len(manual_df))
    auto_use = auto_tokens.iloc[:n].copy().reset_index(drop=True)
    man_use = manual_df.iloc[:n].copy().reset_index(drop=True)

    paired = pd.DataFrame({
        "token_index": np.arange(1, n + 1),
        "word_auto": auto_use["word_auto"].values,
        "word_manual": man_use["word_manual"].values,
        "auto_start_time": auto_use["auto_start_time"].values,
        "auto_end_time": auto_use["auto_end_time"].values,
        "manual_start_time": man_use["manual_start_time"].values,
        "manual_end_time": man_use["manual_end_time"].values,
    })

    paired["label_match"] = paired["word_auto"] == paired["word_manual"]
    paired["auto_duration"] = paired["auto_end_time"] - paired["auto_start_time"]
    paired["manual_duration"] = paired["manual_end_time"] - paired["manual_start_time"]
    paired["start_abs_error"] = np.abs(paired["auto_start_time"] - paired["manual_start_time"])
    paired["end_abs_error"] = np.abs(paired["auto_end_time"] - paired["manual_end_time"])
    paired["duration_abs_error"] = np.abs(paired["auto_duration"] - paired["manual_duration"])

    return paired


def summarize_pairing(subject_id: str, task: str, textgrid_path: Optional[Path], paired: pd.DataFrame,
                      n_auto: int, n_manual: int) -> Dict:
    row: Dict = {
        "subject_id": subject_id,
        "task": task,
        "textgrid_path": str(textgrid_path) if textgrid_path is not None else None,
        "n_auto_tokens": n_auto,
        "n_manual_tokens": n_manual,
        "n_paired_tokens": len(paired),
        "label_match_rate": np.nan,
        "start_pearson_r": np.nan,
        "start_pearson_p": np.nan,
        "start_spearman_rho": np.nan,
        "start_spearman_p": np.nan,
        "end_pearson_r": np.nan,
        "end_pearson_p": np.nan,
        "end_spearman_rho": np.nan,
        "end_spearman_p": np.nan,
        "duration_pearson_r": np.nan,
        "duration_pearson_p": np.nan,
        "duration_spearman_rho": np.nan,
        "duration_spearman_p": np.nan,
        "start_mae": np.nan,
        "start_rmse": np.nan,
        "end_mae": np.nan,
        "end_rmse": np.nan,
        "duration_mae": np.nan,
        "duration_rmse": np.nan,
        "status": "ok",
    }

    if paired.empty:
        row["status"] = "no_pairs"
        return row

    row["label_match_rate"] = float(paired["label_match"].mean())

    xs = paired["auto_start_time"].values
    ys = paired["manual_start_time"].values
    row["start_pearson_r"], row["start_pearson_p"], row["start_spearman_rho"], row["start_spearman_p"] = safe_corr(xs, ys)
    row["start_mae"] = mae(xs, ys)
    row["start_rmse"] = rmse(xs, ys)

    xe = paired["auto_end_time"].values
    ye = paired["manual_end_time"].values
    row["end_pearson_r"], row["end_pearson_p"], row["end_spearman_rho"], row["end_spearman_p"] = safe_corr(xe, ye)
    row["end_mae"] = mae(xe, ye)
    row["end_rmse"] = rmse(xe, ye)

    xd = paired["auto_duration"].values
    yd = paired["manual_duration"].values
    row["duration_pearson_r"], row["duration_pearson_p"], row["duration_spearman_rho"], row["duration_spearman_p"] = safe_corr(xd, yd)
    row["duration_mae"] = mae(xd, yd)
    row["duration_rmse"] = rmse(xd, yd)

    if n_auto != n_manual:
        row["status"] = "token_count_mismatch"
    elif row["label_match_rate"] < 1.0:
        row["status"] = "label_mismatch"
    else:
        row["status"] = "ok"

    return row


def summarize_pooled(df: pd.DataFrame, name: str) -> Dict:
    row = {
        "group_name": name,
        "n_pairs": len(df),
        "label_match_rate": np.nan,
        "start_pearson_r": np.nan,
        "start_pearson_p": np.nan,
        "start_spearman_rho": np.nan,
        "start_spearman_p": np.nan,
        "end_pearson_r": np.nan,
        "end_pearson_p": np.nan,
        "end_spearman_rho": np.nan,
        "end_spearman_p": np.nan,
        "duration_pearson_r": np.nan,
        "duration_pearson_p": np.nan,
        "duration_spearman_rho": np.nan,
        "duration_spearman_p": np.nan,
        "start_mae": np.nan,
        "start_rmse": np.nan,
        "end_mae": np.nan,
        "end_rmse": np.nan,
        "duration_mae": np.nan,
        "duration_rmse": np.nan,
    }
    if df.empty:
        return row

    row["label_match_rate"] = float(df["label_match"].mean())

    xs = df["auto_start_time"].values
    ys = df["manual_start_time"].values
    row["start_pearson_r"], row["start_pearson_p"], row["start_spearman_rho"], row["start_spearman_p"] = safe_corr(xs, ys)
    row["start_mae"] = mae(xs, ys)
    row["start_rmse"] = rmse(xs, ys)

    xe = df["auto_end_time"].values
    ye = df["manual_end_time"].values
    row["end_pearson_r"], row["end_pearson_p"], row["end_spearman_rho"], row["end_spearman_p"] = safe_corr(xe, ye)
    row["end_mae"] = mae(xe, ye)
    row["end_rmse"] = rmse(xe, ye)

    xd = df["auto_duration"].values
    yd = df["manual_duration"].values
    row["duration_pearson_r"], row["duration_pearson_p"], row["duration_spearman_rho"], row["duration_spearman_p"] = safe_corr(xd, yd)
    row["duration_mae"] = mae(xd, yd)
    row["duration_rmse"] = rmse(xd, yd)

    return row


def main():
    ensure_dir(OUTPUT_DIR)

    token_path = Path(TOKEN_CSV)
    if not token_path.exists():
        raise FileNotFoundError(f"TOKEN_CSV does not exist: {TOKEN_CSV}")

    df = pd.read_csv(TOKEN_CSV)
    if df.empty:
        raise ValueError(f"token_results.csv is empty: {TOKEN_CSV}")

    subject_col = find_first_existing_col(df, ["subject_id"])
    task_col = find_first_existing_col(df, ["task"])
    token_idx_col = find_first_existing_col(df, ["token_index", "Item_Index", "idx"])
    token_text_col = find_first_existing_col(df, ["token_text", "word", "label"])

    if subject_col is None or task_col is None or token_idx_col is None:
        raise ValueError("token_results.csv Missing required columns: subject_id / task / token_index")

    if AUTO_START_COL not in df.columns or AUTO_END_COL not in df.columns:
        raise ValueError(f"token_results.csv does not contain columns: {AUTO_START_COL}, {AUTO_END_COL}")

    work = df.copy()
    work["subject_id"] = work[subject_col].astype(str)
    work["task"] = work[task_col].astype(str)
    work["token_index"] = pd.to_numeric(work[token_idx_col], errors="coerce")
    work["token_text"] = work[token_text_col].map(normalize_word) if token_text_col is not None else ""
    work["auto_start_time"] = pd.to_numeric(work[AUTO_START_COL], errors="coerce")
    work["auto_end_time"] = pd.to_numeric(work[AUTO_END_COL], errors="coerce")

    if TASK_FILTER is not None:
        work = work[work["task"].isin(TASK_FILTER)].copy()

    if SUBJECT_FILTER is not None:
        subject_filter_str = {str(x) for x in SUBJECT_FILTER}
        work = work[work["subject_id"].isin(subject_filter_str)].copy()

    if work.empty:
        raise ValueError("Filtered work table is empty. Check TOKEN_CSV, TASK_FILTER, SUBJECT_FILTER, and timestamp columns.")

    grouped = work.groupby(["subject_id", "task"], dropna=False)

    all_paired = []
    subject_task_rows = []
    missing_rows = []

    for (subject_id, task), g in grouped:
        auto_tokens = build_auto_tokens(g)
        textgrid_path = infer_textgrid_path(subject_id, task, TEXTGRID_DIR)

        if textgrid_path is None:
            missing_rows.append({
                "subject_id": subject_id,
                "task": task,
                "reason": "textgrid_not_found",
            })
            subject_task_rows.append({
                "subject_id": subject_id,
                "task": task,
                "textgrid_path": None,
                "n_auto_tokens": len(auto_tokens),
                "n_manual_tokens": 0,
                "n_paired_tokens": 0,
                "label_match_rate": np.nan,
                "start_pearson_r": np.nan,
                "start_pearson_p": np.nan,
                "start_spearman_rho": np.nan,
                "start_spearman_p": np.nan,
                "end_pearson_r": np.nan,
                "end_pearson_p": np.nan,
                "end_spearman_rho": np.nan,
                "end_spearman_p": np.nan,
                "duration_pearson_r": np.nan,
                "duration_pearson_p": np.nan,
                "duration_spearman_rho": np.nan,
                "duration_spearman_p": np.nan,
                "start_mae": np.nan,
                "start_rmse": np.nan,
                "end_mae": np.nan,
                "end_rmse": np.nan,
                "duration_mae": np.nan,
                "duration_rmse": np.nan,
                "status": "textgrid_not_found",
            })
            continue

        manual_tokens = parse_textgrid_word_intervals(textgrid_path)
        paired = pair_auto_manual(auto_tokens, manual_tokens)

        if not paired.empty:
            paired.insert(0, "subject_id", subject_id)
            paired.insert(1, "task", task)
            paired.insert(2, "textgrid_path", str(textgrid_path))
            all_paired.append(paired)

        subject_task_rows.append(
            summarize_pairing(
                subject_id=subject_id,
                task=task,
                textgrid_path=textgrid_path,
                paired=paired,
                n_auto=len(auto_tokens),
                n_manual=len(manual_tokens),
            )
        )

    paired_df = pd.concat(all_paired, ignore_index=True) if all_paired else pd.DataFrame()
    subject_task_df = pd.DataFrame(subject_task_rows)
    missing_df = pd.DataFrame(missing_rows)

    pooled_rows = [summarize_pooled(paired_df, "ALL")]

    if not paired_df.empty:
        for task, sub in paired_df.groupby("task", dropna=False):
            pooled_rows.append(summarize_pooled(sub, f"TASK={task}"))

        for subject_id, sub in paired_df.groupby("subject_id", dropna=False):
            pooled_rows.append(summarize_pooled(sub, f"SUBJECT={subject_id}"))

    pooled_df = pd.DataFrame(pooled_rows)

    paired_path = Path(OUTPUT_DIR) / "paired_token_level_from_textgrid.csv"
    summary_path = Path(OUTPUT_DIR) / "subject_task_correlation_summary.csv"
    pooled_path = Path(OUTPUT_DIR) / "pooled_correlation_summary.csv"
    missing_path = Path(OUTPUT_DIR) / "missing_textgrids.csv"

    paired_df.to_csv(paired_path, index=False, encoding="utf-8-sig")
    subject_task_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pooled_df.to_csv(pooled_path, index=False, encoding="utf-8-sig")
    missing_df.to_csv(missing_path, index=False, encoding="utf-8-sig")

    print("Done.")
    print("paired token-level:", paired_path)
    print("subject-task summary:", summary_path)
    print("pooled summary:", pooled_path)
    print("missing textgrids:", missing_path)


if __name__ == "__main__":
    main()



