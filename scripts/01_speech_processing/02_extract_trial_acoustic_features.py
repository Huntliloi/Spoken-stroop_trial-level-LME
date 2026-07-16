import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import opensmile



# Public repository path setup
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_SCRIPTS_DIR = _RepoPath(__file__).resolve().parents[2] / "scripts"
if str(_REPO_SCRIPTS_DIR) not in _repo_sys.path:
    _repo_sys.path.insert(0, str(_REPO_SCRIPTS_DIR))
from common_paths import *  # noqa: F403
ensure_output_dirs()
# =========================================================
# =========================================================
AUDIO_ROOT = str(AUDIO_ROOT)  # noqa: F405
TOKEN_RESULTS_CSV = str(TOKEN_RESULTS_CSV)  # noqa: F405

FINAL_OUTPUT_CSV_FILE_PATH = str(ACOUSTIC_FEATURES_CSV)  # noqa: F405

AUTO_START_COL = "start_time"
AUTO_END_COL = "end_time"

# AUTO_START_COL = "start_time_raw"
# AUTO_END_COL = "end_time_raw"

PROCESS_TASKS = ["Hanzi", "STT", "Dots"]
TASK_FILENAME_MAP = {"Hanzi": "hanzi", "STT": "STT", "Dots": "dots"}
TASK_FROM_CSV_MAP = {"hanzi": "Hanzi", "dots": "Dots", "stt": "STT", "STT": "STT"}

VALID_KEYWORDS = {'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.'}

OPENSMILE_FEATURE_SET = opensmile.FeatureSet.eGeMAPSv02
OPENSMILE_FEATURE_LEVEL = opensmile.FeatureLevel.Functionals

SUBJECT_COLS = ["subject_id", "Subject_ID", "subject"]
TASK_COLS = ["task", "Task"]
ITEM_COLS = ["token_index", "Item_Index", "trial_index", "idx"]
TOKEN_TEXT_COLS = ["token_text", "word", "label", "aligned_text"]

audio_file_cache: Dict[tuple, Optional[str]] = {}


def find_first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_word(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    mapping = {'English documentation.': 'English documentation.', 'English documentation.': 'English documentation.', 'English documentation.': 'English documentation.', 'English documentation.': 'English documentation.'}
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s


def normalize_task_for_report(x: str) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in TASK_FROM_CSV_MAP:
        return TASK_FROM_CSV_MAP[s]
    sl = s.lower()
    if sl in TASK_FROM_CSV_MAP:
        return TASK_FROM_CSV_MAP[sl]
    return None


def find_audio_file(subject_id: str, task_report_name: str) -> Optional[str]:
    """
    Locate an anonymized task wav file under AUDIO_ROOT.

    Expected public filenames:
    xxxxxxxx001_stroop_dots.wav
    xxxxxxxx001_stroop_STT.wav
    xxxxxxxx001_stroop_hanzi.wav
    """
    key = (subject_id, task_report_name)
    if key in audio_file_cache:
        return audio_file_cache[key]

    filename_part = TASK_FILENAME_MAP.get(task_report_name, "")
    target_name = f"{subject_id}_stroop_{filename_part}.wav"

    candidates = list(Path(AUDIO_ROOT).glob(f"*_{subject_id}/session_1/04_Stroop/{target_name}"))
    if candidates:
        audio_file_cache[key] = str(candidates[0])
        return str(candidates[0])

    for root, _, files in os.walk(AUDIO_ROOT):
        if target_name in files:
            p = os.path.join(root, target_name)
            audio_file_cache[key] = p
            return p

    audio_file_cache[key] = None
    return None


def load_audio(audio_path: str):
    signal, sr = sf.read(audio_path)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return signal, sr


def slice_signal(signal: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start_idx = max(0, int(round(start_s * sr)))
    end_idx = min(len(signal), int(round(end_s * sr)))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)
    seg = signal[start_idx:end_idx]
    return seg.astype(np.float32, copy=False)


def prepare_timestamp_df() -> pd.DataFrame:
    print('\n---    1/3:   token_results.csv          Duration, Gap_Before ---')

    if not os.path.exists(TOKEN_RESULTS_CSV):
        raise FileNotFoundError(f"token_results.csv not found: {TOKEN_RESULTS_CSV}")

    df = pd.read_csv(TOKEN_RESULTS_CSV)

    subject_col = find_first_existing_col(df, SUBJECT_COLS)
    task_col = find_first_existing_col(df, TASK_COLS)
    item_col = find_first_existing_col(df, ITEM_COLS)
    word_col = find_first_existing_col(df, TOKEN_TEXT_COLS)

    missing = []
    if subject_col is None:
        missing.append("subject_id")
    if task_col is None:
        missing.append("task")
    if item_col is None:
        missing.append("token_index")
    if word_col is None:
        missing.append("token_text")
    if AUTO_START_COL not in df.columns:
        missing.append(AUTO_START_COL)
    if AUTO_END_COL not in df.columns:
        missing.append(AUTO_END_COL)
    if missing:
        raise ValueError(f"token_results.csv Missing required columns: {missing}")

    work = df.copy()
    work["Subject_ID"] = work[subject_col].astype(str)
    work["Task"] = work[task_col].map(normalize_task_for_report)
    work["Item_Index"] = pd.to_numeric(work[item_col], errors="coerce")
    work["Auto_Word"] = work[word_col].map(normalize_word)
    work["Absolute_Word_Start"] = pd.to_numeric(work[AUTO_START_COL], errors="coerce")
    work["Absolute_Word_End"] = pd.to_numeric(work[AUTO_END_COL], errors="coerce")

    work = work[work["Task"].isin(PROCESS_TASKS)].copy()
    work = work[work["Auto_Word"].isin(VALID_KEYWORDS)].copy()
    work = work[work["Absolute_Word_Start"].notna() & work["Absolute_Word_End"].notna()].copy()
    work = work[work["Absolute_Word_End"] > work["Absolute_Word_Start"]].copy()
    work = work[work["Item_Index"].notna()].copy()

    work["Item_Index"] = work["Item_Index"].astype(int)

    work = work.sort_values(["Subject_ID", "Task", "Item_Index"]).copy()
    work = work.groupby(["Subject_ID", "Task"], as_index=False, group_keys=False).head(12).copy()

    work["Item_Index"] = work.groupby(["Subject_ID", "Task"]).cumcount() + 1

    # Duration / Gap_Before
    work["Duration"] = work["Absolute_Word_End"] - work["Absolute_Word_Start"]
    work["Gap_Before"] = 0.0

    for (sid, task), idx in work.groupby(["Subject_ID", "Task"]).groups.items():
        sub = work.loc[idx].sort_values("Item_Index")
        prev_end = None
        gaps = []
        for _, row in sub.iterrows():
            if prev_end is None:
                gaps.append(float(row["Absolute_Word_Start"]))
            else:
                gaps.append(float(row["Absolute_Word_Start"] - prev_end))
            prev_end = float(row["Absolute_Word_End"])
        work.loc[sub.index, "Gap_Before"] = gaps

    out_cols = [
        "Subject_ID", "Task", "Item_Index",
        "Absolute_Word_Start", "Absolute_Word_End",
        "Duration", "Gap_Before", "Auto_Word"
    ]
    work = work[out_cols].copy()

    print(f"Automatic timestamp collection complete; rows: {len(work)} records。")
    if not work.empty:
        print('Task  :')
        for task, count in work.groupby("Task").size().items():
            print(f"  {task}: {count} records")

    return work


def extract_opensmile_features_in_memory(df_timestamps: pd.DataFrame) -> pd.DataFrame:
    print('\n---    2/3:      token_results.csv             OpenSMILE   (   、     ) ---')
    print('  OpenSMILE   : eGeMAPSv02 (Functionals)')

    smile_test = opensmile.Smile(
        feature_set=OPENSMILE_FEATURE_SET,
        feature_level=OPENSMILE_FEATURE_LEVEL,
    )

    feature_columns = []
    test_done = False

    for _, row in df_timestamps.iterrows():
        sid = str(row["Subject_ID"])
        task = row["Task"]
        audio_path = find_audio_file(sid, task)
        if not audio_path or not os.path.exists(audio_path):
            continue
        try:
            signal, sr = load_audio(audio_path)
            seg = slice_signal(signal, sr, float(row["Absolute_Word_Start"]), float(row["Absolute_Word_End"]))
            if len(seg) == 0:
                continue
            test_features = smile_test.process_signal(seg, sr)
            if not test_features.empty:
                feature_columns = test_features.columns.tolist()
                print(f"Test audio {os.path.basename(audio_path)} feature extraction succeeded:")
                print(f"  feature shape: {test_features.shape}")
                print(f"  feature count: {len(feature_columns)}")
                print(f"  first 10 features: {feature_columns[:10]}")
                print(f"  last 10 features: {feature_columns[-10:]}")
                test_done = True
                break
        except Exception as e:
            print(f"test segment feature extraction failed: {e}")

    if not test_done or not feature_columns:
        print('  ：          ，       。')
        return pd.DataFrame(columns=["Subject_ID", "Task", "Item_Index"])

    smile = opensmile.Smile(
        feature_set=OPENSMILE_FEATURE_SET,
        feature_level=OPENSMILE_FEATURE_LEVEL,
        num_workers=1,
    )

    all_features_data = []

    loaded_audio_cache: Dict[str, tuple] = {}

    for _, row in tqdm(df_timestamps.iterrows(), total=len(df_timestamps), desc='  OpenSMILE  '):
        sid = str(row["Subject_ID"])
        task = row["Task"]
        item_idx = int(row["Item_Index"])
        start_s = float(row["Absolute_Word_Start"])
        end_s = float(row["Absolute_Word_End"])

        audio_path = find_audio_file(sid, task)

        feature_row = {
            "Subject_ID": sid,
            "Task": task,
            "Item_Index": item_idx,
        }

        if not audio_path or not os.path.exists(audio_path):
            print(f"Warning：original audio not found Subject={sid}, Task={task}")
            for col in feature_columns:
                feature_row[col] = np.nan
            all_features_data.append(feature_row)
            continue

        try:
            if audio_path not in loaded_audio_cache:
                loaded_audio_cache[audio_path] = load_audio(audio_path)

            signal, sr = loaded_audio_cache[audio_path]
            seg = slice_signal(signal, sr, start_s, end_s)

            if len(seg) == 0:
                print(f"Warning：empty audio slice Subject={sid}, Task={task}, Item={item_idx}")
                for col in feature_columns:
                    feature_row[col] = np.nan
                all_features_data.append(feature_row)
                continue

            features_df_single = smile.process_signal(seg, sr)

            if not features_df_single.empty:
                features = features_df_single.iloc[0].values
                for col_name, feature_val in zip(feature_columns, features):
                    feature_row[col_name] = feature_val
            else:
                print(f"Warning：feature extraction returned empty output Subject={sid}, Task={task}, Item={item_idx}")
                for col in feature_columns:
                    feature_row[col] = np.nan

        except Exception as e:
            print(f"Error：processing Subject={sid}, Task={task}, Item={item_idx} Failed: {e}")
            for col in feature_columns:
                feature_row[col] = np.nan

        all_features_data.append(feature_row)

    features_df = pd.DataFrame(all_features_data)

    expected_columns = ["Subject_ID", "Task", "Item_Index"] + feature_columns
    for col in expected_columns:
        if col not in features_df.columns:
            features_df[col] = np.nan
    features_df = features_df[expected_columns]

    return features_df


def merge_and_save(df_timestamps: pd.DataFrame, features_df: pd.DataFrame):
    print('\n---    3/3:            CSV ---')

    df_timestamps = df_timestamps.copy()
    features_df = features_df.copy()

    df_timestamps["Subject_ID"] = df_timestamps["Subject_ID"].astype(str)
    features_df["Subject_ID"] = features_df["Subject_ID"].astype(str)

    final_df = pd.merge(
        df_timestamps,
        features_df,
        on=["Subject_ID", "Task", "Item_Index"],
        how="left"
    )

    ordered_front = [
        "Subject_ID", "Task", "Item_Index",
        "Absolute_Word_Start", "Absolute_Word_End",
        "Duration", "Gap_Before"
    ]
    other_cols = [c for c in final_df.columns if c not in ordered_front]
    final_df = final_df[ordered_front + other_cols]

    os.makedirs(os.path.dirname(FINAL_OUTPUT_CSV_FILE_PATH), exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT_CSV_FILE_PATH, index=False, encoding="utf-8-sig")

    opensmile_cols = [col for col in features_df.columns if col not in ["Subject_ID", "Task", "Item_Index"] and col in final_df.columns]
    print(f"\nFinal CSV saved to: {FINAL_OUTPUT_CSV_FILE_PATH}")
    print('\n===      ===')
    print(f"Total rows: {len(final_df)}")
    print(f"Total columns: {len(final_df.columns)}")
    print(f"  basic features: {len(final_df.columns) - len(opensmile_cols)} columns")
    print(f"  OpenSMILE features: {len(opensmile_cols)} columns")

    if opensmile_cols:
        missing_values = final_df[opensmile_cols].isna().sum().sum()
        total_cells = len(final_df) * len(opensmile_cols)
        missing_percentage = (missing_values / total_cells) * 100 if total_cells > 0 else 0
        print(f"OpenSMILE featuresMissing values: {missing_values} / {total_cells} ({missing_percentage:.1f}%)")

    basic_cols = [col for col in final_df.columns if col not in opensmile_cols]
    print(f"\nBasic feature columns: {basic_cols}")


def process_stroop_data():
    print("Starting Stroop processing: token_results.csv -> in-memory OpenSMILE extraction -> CSV.")
    print("This script does not read TextGrid files or save segmented audio clips.")

    df_timestamps = prepare_timestamp_df()
    if df_timestamps.empty:
        print("No valid automatic timestamp rows were found. Check token_results.csv.")
        return

    features_df = extract_opensmile_features_in_memory(df_timestamps)
    if features_df.empty:
        print("Warning: no OpenSMILE features were extracted.")
        return

    merge_and_save(df_timestamps, features_df)


if __name__ == "__main__":
    try:
        import opensmile
        print(f"opensmile-python version: {opensmile.__version__}")
        print('OpenSMILE    : eGeMAPSv02')
        print('OpenSMILE     : Functionals')
    except ImportError:
        print('  ：    opensmile-python  。   pip install opensmile')
        raise

    process_stroop_data()




