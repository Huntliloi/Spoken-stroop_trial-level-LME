"""Extract item-level Stroop features for the substance-use cohort.

This script follows the shared public speech-processing workflow:
  1. Qwen3 ASR + forced alignment + boundary refinement.
  2. Build an item-level Stroop segment table from token_results.csv.
  3. Write item-level wav segments.
  4. Extract openSMILE eGeMAPSv02 Functionals features.

Feature output matches the reference: 88 openSMILE columns plus the two timing
features Word_Duration and Gap_Before.
"""

import importlib.util
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.dont_write_bytecode = True

try:
    import opensmile
except Exception as exc:
    raise ImportError("Please install opensmile-python first: pip install opensmile") from exc


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from common_paths import REPO_ROOT, SUBSTANCE_USE_OUTPUT_DIR  # noqa: E402

DATA_ROOT = Path(os.environ.get(
    "SUBSTANCE_USE_AUDIO_ROOT",
    REPO_ROOT / "data" / "not_shared" / "substance_use" / "audio",
))
RESULT_DIR = SUBSTANCE_USE_OUTPUT_DIR / "feature_pipeline"

QWEN_PIPELINE_PATH = SCRIPTS_DIR / "01_speech_processing" / "01_qwen3_asr_alignment_pipeline.py"

ASR_MODEL = Path(os.environ.get("QWEN3_ASR_MODEL", "Qwen3-ASR-0.6B"))
ALIGNER_MODEL = Path(os.environ.get("QWEN3_ALIGNER_MODEL", "Qwen3-ForcedAligner-0.6B"))

DEVICE = os.environ.get("QWEN_DEVICE", "cuda:0")
DTYPE = os.environ.get("QWEN_DTYPE", "float32")
LOCAL_FILES_ONLY = os.environ.get("QWEN_LOCAL_FILES_ONLY", "1") == "1"
EXPECTED_TOKENS_PER_TASK = 12

# Set RUN_ASR=False if token_results.csv already exists and only features need rebuild.
RUN_ASR = os.environ.get("RUN_SUBSTANCE_ASR", "0") == "1"
RUN_FEATURES = os.environ.get("RUN_SUBSTANCE_FEATURES", "1") == "1"

AUTO_START_COL = "start_time"
AUTO_END_COL = "end_time"

PROCESS_TASKS = ["W", "CW", "C"]
TASK_FILENAME_MAP = {"W": "W", "CW": "CW", "C": "C"}
TASK_FROM_CSV_MAP = {"w": "W", "c": "C", "cw": "CW", "CW": "CW"}
VALID_WORDS = {"\u7ea2", "\u9ec4", "\u84dd", "\u7eff"}

OPENSMILE_FEATURE_SET = opensmile.FeatureSet.eGeMAPSv02
OPENSMILE_FEATURE_LEVEL = opensmile.FeatureLevel.Functionals


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    audio_dir: Path
    asr_output_dir: Path
    segment_dir: Path
    segment_csv: Path
    feature_csv: Path


DATASETS = [
    DatasetConfig(
        name="Drug",
        audio_dir=DATA_ROOT / "drug",
        asr_output_dir=RESULT_DIR / "asr_outputs_drug",
        segment_dir=RESULT_DIR / "stroop_audio_drug",
        segment_csv=RESULT_DIR / "Stroop_segments_drug.csv",
        feature_csv=RESULT_DIR / "Stroop_features_drug.csv",
    ),
    DatasetConfig(
        name="HC",
        audio_dir=DATA_ROOT / "HC",
        asr_output_dir=RESULT_DIR / "asr_outputs_HC",
        segment_dir=RESULT_DIR / "stroop_audio_HC",
        segment_csv=RESULT_DIR / "Stroop_segments_HC.csv",
        feature_csv=RESULT_DIR / "Stroop_features_HC.csv",
    ),
]


def load_qwen_pipeline():
    if not QWEN_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Missing reference script: {QWEN_PIPELINE_PATH}")

    spec = importlib.util.spec_from_file_location("qwen3_stroop_pipeline_add_ref", QWEN_PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load reference script: {QWEN_PIPELINE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_task_for_report(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in TASK_FROM_CSV_MAP:
        return TASK_FROM_CSV_MAP[s]
    return TASK_FROM_CSV_MAP.get(s.lower())


def normalize_word(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    mapping = {
        "\u7ea2": "\u7ea2",
        "\u9ec4": "\u9ec4",
        "\u84dd": "\u84dd",
        "\u7eff": "\u7eff",
        "\u7ea2\u8272": "\u7ea2",
        "\u9ec4\u8272": "\u9ec4",
        "\u84dd\u8272": "\u84dd",
        "\u7eff\u8272": "\u7eff",
        "\u041a\u044c": "\u7ea2",
        "\u041b\u0426": "\u9ec4",
        "\u0420\u0416": "\u84dd",
        "\u0422\u042c": "\u7eff",
    }
    return mapping.get(s, s)


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", str(name))


def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    signal, sr = sf.read(str(audio_path))
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return signal.astype(np.float32, copy=False), int(sr)


def slice_signal(signal: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    start_idx = max(0, int(round(float(start_s) * sr)))
    end_idx = min(len(signal), int(round(float(end_s) * sr)))
    if end_idx <= start_idx:
        return np.array([], dtype=np.float32)
    return signal[start_idx:end_idx].astype(np.float32, copy=False)


def find_stroop_audio_files(qwen_pipe, audio_root: Path) -> List[Path]:
    root = Path(audio_root)
    if not root.exists():
        raise FileNotFoundError(f"Audio root does not exist: {root}")

    wav_files: List[Path] = []
    for wav_path in sorted(root.rglob("*.wav")):
        name = wav_path.name.lower()
        if "stroop" not in name:
            continue
        task = qwen_pipe.detect_task_from_filename(wav_path.name)
        if task in {"W", "C", "CW"}:
            wav_files.append(wav_path)
    return wav_files


def find_audio_file(cfg: DatasetConfig, row: pd.Series) -> Optional[Path]:
    token_path = row.get("audio_path")
    if isinstance(token_path, str) and token_path.strip():
        p = Path(token_path)
        if p.exists():
            return p

    subject_folder = str(row.get("subject_folder", "")).strip()
    subject_id = str(row.get("subject_id", "")).strip()
    task = normalize_task_for_report(row.get("task"))
    if not subject_id or task is None:
        return None

    target_name = f"{subject_id}_stroop_{TASK_FILENAME_MAP[task]}.wav"

    if subject_folder:
        for p in [
            cfg.audio_dir / subject_folder / "04_Stroop" / target_name,
            cfg.audio_dir / subject_folder / "session_1" / "04_Stroop" / target_name,
        ]:
            if p.exists():
                return p

    hits = list(cfg.audio_dir.rglob(target_name))
    if hits:
        return hits[0]

    return None


def run_asr_for_dataset(qwen_pipe, model, cfg: DatasetConfig) -> None:
    cfg.asr_output_dir.mkdir(parents=True, exist_ok=True)

    qwen_pipe.AUDIO_ROOT = str(cfg.audio_dir)
    qwen_pipe.OUTPUT_DIR = str(cfg.asr_output_dir)
    qwen_pipe.ASR_MODEL = str(ASR_MODEL)
    qwen_pipe.ALIGNER_MODEL = str(ALIGNER_MODEL)
    qwen_pipe.DEVICE = DEVICE
    qwen_pipe.DTYPE = DTYPE
    qwen_pipe.LOCAL_FILES_ONLY = LOCAL_FILES_ONLY
    qwen_pipe.EXPECTED_TOKENS_PER_TASK = EXPECTED_TOKENS_PER_TASK

    wav_files = find_stroop_audio_files(qwen_pipe, cfg.audio_dir)
    print(f"\n[{cfg.name}] ASR input wav files: {len(wav_files)}")
    if not wav_files:
        raise RuntimeError(f"No Stroop wav files found under {cfg.audio_dir}")

    utterance_records: List[Dict] = []
    token_records: List[Dict] = []

    for wav_path in tqdm(wav_files, desc=f"{cfg.name} ASR"):
        record, token_rows = qwen_pipe.process_one_file(model, wav_path, str(cfg.audio_dir))
        utterance_records.append(record)
        token_records.extend(token_rows)
        qwen_pipe.save_json_record(str(cfg.asr_output_dir), record)

    qwen_pipe.write_tables(str(cfg.asr_output_dir), utterance_records, token_records)

    n_review = sum(1 for x in utterance_records if x.get("needs_review"))
    print(f"[{cfg.name}] ASR done. utterances={len(utterance_records)}, needs_review={n_review}")


def prepare_segments(cfg: DatasetConfig) -> pd.DataFrame:
    token_csv = cfg.asr_output_dir / "token_results.csv"
    if not token_csv.exists():
        raise FileNotFoundError(f"Missing token_results.csv: {token_csv}")
    if token_csv.stat().st_size <= 5:
        raise RuntimeError(f"{token_csv} is empty. Re-run ASR.")

    df = pd.read_csv(token_csv)
    required = ["subject_id", "subject_folder", "task", "token_index", "token_text", AUTO_START_COL, AUTO_END_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{token_csv} missing columns: {missing}")

    work = df.copy()
    work["Subject_ID"] = work["subject_id"].astype(str)
    work["Task"] = work["task"].map(normalize_task_for_report)
    work["Item_Index"] = pd.to_numeric(work["token_index"], errors="coerce")
    work["Word"] = work["token_text"].map(normalize_word)
    work["Absolute_Word_Start"] = pd.to_numeric(work[AUTO_START_COL], errors="coerce")
    work["Absolute_Word_End"] = pd.to_numeric(work[AUTO_END_COL], errors="coerce")

    work = work[work["Task"].isin(PROCESS_TASKS)].copy()
    work = work[work["Word"].isin(VALID_WORDS)].copy()
    work = work[work["Item_Index"].notna()].copy()
    work = work[work["Absolute_Word_Start"].notna() & work["Absolute_Word_End"].notna()].copy()
    work = work[work["Absolute_Word_End"] > work["Absolute_Word_Start"]].copy()
    work["Item_Index"] = work["Item_Index"].astype(int)

    work = work.sort_values(["Subject_ID", "Task", "Item_Index"]).copy()
    work = work.groupby(["Subject_ID", "Task"], as_index=False, group_keys=False).head(EXPECTED_TOKENS_PER_TASK)
    work["Item_Index"] = work.groupby(["Subject_ID", "Task"]).cumcount() + 1

    audio_paths: List[Optional[str]] = []
    segment_paths: List[str] = []
    durations: List[float] = []
    gaps: List[float] = []
    prev_end_by_task: Dict[Tuple[str, str], float] = {}

    for _, row in work.iterrows():
        sid = str(row["Subject_ID"])
        task = str(row["Task"])
        item_idx = int(row["Item_Index"])
        word = str(row["Word"])
        start_s = float(row["Absolute_Word_Start"])
        end_s = float(row["Absolute_Word_End"])

        audio_path = find_audio_file(cfg, row)
        audio_paths.append(str(audio_path) if audio_path else "")

        safe_word = sanitize_filename(word)
        segment_path = cfg.segment_dir / sid / task / f"{sid}_{task}_{item_idx:02d}_{safe_word}.wav"
        segment_paths.append(str(segment_path))

        durations.append(end_s - start_s)

        key = (sid, task)
        prev_end = prev_end_by_task.get(key)
        gaps.append(start_s if prev_end is None else start_s - prev_end)
        prev_end_by_task[key] = end_s

    return pd.DataFrame({
        "Subject_ID": work["Subject_ID"].to_numpy(),
        "Task": work["Task"].to_numpy(),
        "Item_Index": work["Item_Index"].to_numpy(),
        "Word": work["Word"].to_numpy(),
        "Absolute_Word_Start": work["Absolute_Word_Start"].to_numpy(),
        "Absolute_Word_End": work["Absolute_Word_End"].to_numpy(),
        "Word_Duration": durations,
        "Gap_Before": gaps,
        "Audio_Path": audio_paths,
        "Segment_Path": segment_paths,
    })


def write_segment_audio(segments: pd.DataFrame) -> None:
    loaded_audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}

    for _, row in tqdm(segments.iterrows(), total=len(segments), desc="write segments"):
        audio_path = str(row["Audio_Path"])
        segment_path = Path(str(row["Segment_Path"]))
        if not audio_path or not Path(audio_path).exists():
            continue

        try:
            if audio_path not in loaded_audio_cache:
                loaded_audio_cache[audio_path] = load_audio(Path(audio_path))
            signal, sr = loaded_audio_cache[audio_path]
            seg = slice_signal(signal, sr, float(row["Absolute_Word_Start"]), float(row["Absolute_Word_End"]))
            if len(seg) == 0:
                continue

            segment_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(segment_path), seg, sr)
        except Exception as exc:
            print(f"Failed to write segment {segment_path}: {type(exc).__name__}: {exc}")


def get_feature_columns_from_first_valid_segment(segments: pd.DataFrame) -> List[str]:
    smile = opensmile.Smile(feature_set=OPENSMILE_FEATURE_SET, feature_level=OPENSMILE_FEATURE_LEVEL)

    for _, row in segments.iterrows():
        audio_path = str(row["Audio_Path"])
        if not audio_path or not Path(audio_path).exists():
            continue
        try:
            signal, sr = load_audio(Path(audio_path))
            seg = slice_signal(signal, sr, float(row["Absolute_Word_Start"]), float(row["Absolute_Word_End"]))
            if len(seg) == 0:
                continue
            feat = smile.process_signal(seg, sr)
            if not feat.empty:
                return feat.columns.tolist()
        except Exception:
            continue

    raise RuntimeError("No valid segment found for OpenSMILE feature-column discovery.")


def extract_features(segments: pd.DataFrame) -> pd.DataFrame:
    feature_columns = get_feature_columns_from_first_valid_segment(segments)
    smile = opensmile.Smile(
        feature_set=OPENSMILE_FEATURE_SET,
        feature_level=OPENSMILE_FEATURE_LEVEL,
        num_workers=1,
    )

    loaded_audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}
    rows: List[Dict] = []

    for _, row in tqdm(segments.iterrows(), total=len(segments), desc="OpenSMILE"):
        feature_row = row.to_dict()
        audio_path = str(row["Audio_Path"])

        try:
            if not audio_path or not Path(audio_path).exists():
                raise FileNotFoundError(audio_path)

            if audio_path not in loaded_audio_cache:
                loaded_audio_cache[audio_path] = load_audio(Path(audio_path))

            signal, sr = loaded_audio_cache[audio_path]
            seg = slice_signal(signal, sr, float(row["Absolute_Word_Start"]), float(row["Absolute_Word_End"]))
            if len(seg) == 0:
                raise ValueError("empty segment")

            feat = smile.process_signal(seg, sr)
            if feat.empty:
                raise ValueError("empty opensmile result")

            for col, val in zip(feature_columns, feat.iloc[0].values):
                feature_row[col] = val

        except Exception as exc:
            print(
                f"OpenSMILE failed: Subject={row['Subject_ID']}, Task={row['Task']}, "
                f"Item={row['Item_Index']}: {type(exc).__name__}: {exc}"
            )
            for col in feature_columns:
                feature_row[col] = np.nan

        rows.append(feature_row)

    front_cols = [
        "Subject_ID", "Task", "Item_Index", "Word",
        "Absolute_Word_Start", "Absolute_Word_End", "Word_Duration",
        "Gap_Before", "Audio_Path", "Segment_Path",
    ]
    final_df = pd.DataFrame(rows)
    for col in front_cols + feature_columns:
        if col not in final_df.columns:
            final_df[col] = np.nan
    return final_df[front_cols + feature_columns]


def run_features_for_dataset(cfg: DatasetConfig) -> None:
    print(f"\n[{cfg.name}] Build segments and OpenSMILE features")
    segments = prepare_segments(cfg)
    print(f"[{cfg.name}] valid segment rows: {len(segments)}")

    write_segment_audio(segments)

    cfg.segment_csv.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(cfg.segment_csv, index=False, encoding="utf-8-sig")
    print(f"[{cfg.name}] segment table saved: {cfg.segment_csv}")

    features = extract_features(segments)
    features.to_csv(cfg.feature_csv, index=False, encoding="utf-8-sig")
    print(f"[{cfg.name}] feature table saved: {cfg.feature_csv}")

    opensmile_cols = [c for c in features.columns if c not in segments.columns]
    missing = features[opensmile_cols].isna().sum().sum() if opensmile_cols else 0
    total = len(features) * len(opensmile_cols) if opensmile_cols else 0
    print(f"[{cfg.name}] rows={len(features)}, opensmile_cols={len(opensmile_cols)}, missing={missing}/{total}")


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("Substance-use cohort Stroop feature pipeline")
    print(f"DATA_ROOT: {DATA_ROOT}")
    print(f"RESULT_DIR: {RESULT_DIR}")
    print(f"RUN_ASR={RUN_ASR}, RUN_FEATURES={RUN_FEATURES}")
    print(f"OpenSMILE: {opensmile.__version__}, eGeMAPSv02 Functionals")

    qwen_pipe = None
    model = None

    if RUN_ASR:
        pending_asr = []
        for cfg in DATASETS:
            token_csv = cfg.asr_output_dir / "token_results.csv"
            if token_csv.exists() and token_csv.stat().st_size > 5:
                print(f"[{cfg.name}] existing ASR token table found, skip ASR: {token_csv}")
            else:
                pending_asr.append(cfg)

        if pending_asr:
            qwen_pipe = load_qwen_pipeline()
            qwen_pipe.LOCAL_FILES_ONLY = LOCAL_FILES_ONLY
            model = qwen_pipe.load_model(str(ASR_MODEL), str(ALIGNER_MODEL), DEVICE, DTYPE)

        for cfg in pending_asr:
            run_asr_for_dataset(qwen_pipe, model, cfg)

    if RUN_FEATURES:
        for cfg in DATASETS:
            run_features_for_dataset(cfg)

    print("\nAll done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
