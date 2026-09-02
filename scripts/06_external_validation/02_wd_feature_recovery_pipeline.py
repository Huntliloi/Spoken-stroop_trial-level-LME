"""Rebuild additional WD Stroop item-level features from private audio.

The WD audio is not public. This script documents the recovery pipeline used to
produce the public WD_CI and WD_CN feature tables. It can be run by setting
environment variables that point to private WD audio and locally available Qwen
models; generated files are written to ``outputs/wd_recovered_pipeline``.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.dont_write_bytecode = True

try:
    import opensmile
except Exception as exc:  # pragma: no cover - dependency message for users
    raise ImportError("Please install opensmile-python first: pip install opensmile") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from common_paths import OUTPUT_DIR, REPO_ROOT  # noqa: E402


BASE_DIR = OUTPUT_DIR / "wd_recovered_pipeline"
WD_AUDIO_ROOT = Path(os.environ.get("WD_AUDIO_ROOT", REPO_ROOT / "data" / "not_shared" / "wd_audio"))
QWEN_PIPELINE_PATH = Path(
    os.environ.get(
        "QWEN_STROOP_PIPELINE",
        REPO_ROOT / "scripts" / "01_speech_processing" / "01_qwen3_asr_alignment_pipeline.py",
    )
)
ASR_MODEL = os.environ.get("QWEN3_ASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
ALIGNER_MODEL = os.environ.get("QWEN3_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")

DEVICE = os.environ.get("QWEN_DEVICE", "cuda:0")
DTYPE = os.environ.get("QWEN_DTYPE", "float32")
LOCAL_FILES_ONLY = os.environ.get("QWEN_LOCAL_FILES_ONLY", "0") == "1"
RUN_ASR = os.environ.get("RUN_WD_ASR", "0") == "1"
RUN_FEATURES = os.environ.get("RUN_WD_FEATURES", "1") == "1"

EXPECTED_TOKENS_PER_TASK = 12
AUTO_START_COL = "start_time"
AUTO_END_COL = "end_time"
PROCESS_TASKS = ["W", "CW", "C"]
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
        name="WD_CI",
        audio_dir=Path(os.environ.get("WD_CI_AUDIO_ROOT", WD_AUDIO_ROOT / "WD_CI")),
        asr_output_dir=BASE_DIR / "asr_outputs_WD_CI_add",
        segment_dir=BASE_DIR / "stroop_audio_WD_CI_add",
        segment_csv=BASE_DIR / "Stroop_segments_WD_CI_add.csv",
        feature_csv=BASE_DIR / "Stroop_features_WD_CI_add.csv",
    ),
    DatasetConfig(
        name="WD_CN",
        audio_dir=Path(os.environ.get("WD_CN_AUDIO_ROOT", WD_AUDIO_ROOT / "WD_CN")),
        asr_output_dir=BASE_DIR / "asr_outputs_WD_CN_add",
        segment_dir=BASE_DIR / "stroop_audio_WD_CN_add",
        segment_csv=BASE_DIR / "Stroop_segments_WD_CN_add.csv",
        feature_csv=BASE_DIR / "Stroop_features_WD_CN_add.csv",
    ),
]


def load_qwen_pipeline():
    if not QWEN_PIPELINE_PATH.exists():
        raise FileNotFoundError(f"Missing Qwen Stroop pipeline script: {QWEN_PIPELINE_PATH}")

    spec = importlib.util.spec_from_file_location("qwen3_stroop_pipeline_public", QWEN_PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Qwen Stroop pipeline script: {QWEN_PIPELINE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_task_for_report(value) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text in TASK_FROM_CSV_MAP:
        return TASK_FROM_CSV_MAP[text]
    return TASK_FROM_CSV_MAP.get(text.lower())


def normalize_word(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    mapping = {
        "\u7d05": "\u7ea2",
        "\u9ec3": "\u9ec4",
        "\u85cd": "\u84dd",
        "\u7da0": "\u7eff",
        "\u7ea2\u8272": "\u7ea2",
        "\u9ec4\u8272": "\u9ec4",
        "\u84dd\u8272": "\u84dd",
        "\u7eff\u8272": "\u7eff",
        "red": "\u7ea2",
        "yellow": "\u9ec4",
        "blue": "\u84dd",
        "green": "\u7eff",
    }
    return mapping.get(text, text)


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", str(name))


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
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


def find_wd_audio_files(qwen_pipe, audio_root: Path) -> list[Path]:
    root = Path(audio_root)
    if not root.exists():
        raise FileNotFoundError(f"Audio root does not exist: {root}")

    wav_files: list[Path] = []
    for wav_path in sorted(root.rglob("*.wav")):
        if "stroop" not in wav_path.name.lower():
            continue
        task = qwen_pipe.detect_task_from_filename(wav_path.name)
        if task in {"W", "C", "CW"}:
            wav_files.append(wav_path)
    return wav_files


def run_asr_for_dataset(qwen_pipe, model, cfg: DatasetConfig) -> None:
    print(f"\n[{cfg.name}] Running Qwen ASR and alignment")
    cfg.asr_output_dir.mkdir(parents=True, exist_ok=True)

    qwen_pipe.AUDIO_ROOT = str(cfg.audio_dir)
    qwen_pipe.OUTPUT_DIR = str(cfg.asr_output_dir)
    qwen_pipe.ASR_MODEL = str(ASR_MODEL)
    qwen_pipe.ALIGNER_MODEL = str(ALIGNER_MODEL)
    qwen_pipe.DEVICE = DEVICE
    qwen_pipe.DTYPE = DTYPE
    qwen_pipe.LOCAL_FILES_ONLY = LOCAL_FILES_ONLY
    qwen_pipe.EXPECTED_TOKENS_PER_TASK = EXPECTED_TOKENS_PER_TASK

    utterance_records = []
    token_records = []
    for wav_path in tqdm(find_wd_audio_files(qwen_pipe, cfg.audio_dir), desc=f"{cfg.name} ASR"):
        try:
            record, token_rows = qwen_pipe.process_one_file(model, wav_path, str(cfg.audio_dir))
            utterance_records.append(record)
            token_records.extend(token_rows)
            qwen_pipe.save_json_record(str(cfg.asr_output_dir), record)
        except Exception as exc:
            print(f"Failed ASR/alignment for {wav_path}: {type(exc).__name__}: {exc}")

    qwen_pipe.write_tables(str(cfg.asr_output_dir), utterance_records, token_records)
    print(f"[{cfg.name}] ASR outputs saved to {cfg.asr_output_dir}")


def load_token_results(cfg: DatasetConfig) -> pd.DataFrame:
    candidates = [
        cfg.asr_output_dir / "token_results.csv",
        cfg.asr_output_dir / "token_results_refined.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, dtype={"subject_id": str}, encoding="utf-8-sig")
    raise FileNotFoundError(f"No token result table found in {cfg.asr_output_dir}")


def find_audio_file(cfg: DatasetConfig, row: pd.Series) -> Optional[Path]:
    token_path = row.get("audio_path")
    if isinstance(token_path, str) and token_path.strip():
        path = Path(token_path)
        if path.exists():
            return path

    subject_id = str(row["Subject_ID"])
    task = str(row["Task"])
    task_key = {"W": "W", "C": "C", "CW": "CW"}[task]
    patterns = [
        f"*{subject_id}*stroop*{task_key}*.wav",
        f"*stroop*{task_key}*.wav",
    ]
    for pattern in patterns:
        matches = sorted((cfg.audio_dir / subject_id).rglob(pattern)) if (cfg.audio_dir / subject_id).exists() else []
        matches.extend(sorted(cfg.audio_dir.rglob(pattern)))
        if matches:
            return matches[0]
    return None


def prepare_segments(cfg: DatasetConfig) -> pd.DataFrame:
    tokens = load_token_results(cfg)
    required = ["subject_id", "task", "word", AUTO_START_COL, AUTO_END_COL]
    missing = [col for col in required if col not in tokens.columns]
    if missing:
        raise ValueError(f"Missing token columns in {cfg.asr_output_dir}: {missing}")

    work = tokens.copy()
    work["Subject_ID"] = work["subject_id"].astype(str)
    work["Task"] = work["task"].map(normalize_task_for_report)
    work["Word"] = work["word"].map(normalize_word)
    work = work[work["Task"].isin(PROCESS_TASKS)]
    work = work[work["Word"].isin(VALID_WORDS)].copy()
    work[AUTO_START_COL] = pd.to_numeric(work[AUTO_START_COL], errors="coerce")
    work[AUTO_END_COL] = pd.to_numeric(work[AUTO_END_COL], errors="coerce")
    work = work.dropna(subset=[AUTO_START_COL, AUTO_END_COL])
    work = work[work[AUTO_END_COL] > work[AUTO_START_COL]]
    work = work.sort_values(["Subject_ID", "Task", AUTO_START_COL, AUTO_END_COL])

    rows = []
    for (subject_id, task), group in work.groupby(["Subject_ID", "Task"], sort=True):
        group = group.head(EXPECTED_TOKENS_PER_TASK).copy()
        audio_path = find_audio_file(cfg, group.iloc[0])
        for item_index, (_, row) in enumerate(group.iterrows(), start=1):
            segment_name = f"{sanitize_filename(subject_id)}_{task}_{item_index:02d}_{row['Word']}.wav"
            segment_path = cfg.segment_dir / sanitize_filename(subject_id) / task / segment_name
            rows.append(
                {
                    "Subject_ID": subject_id,
                    "Task": task,
                    "Item_Index": item_index,
                    "Word": row["Word"],
                    "Absolute_Word_Start": float(row[AUTO_START_COL]),
                    "Absolute_Word_End": float(row[AUTO_END_COL]),
                    "Word_Duration": float(row[AUTO_END_COL] - row[AUTO_START_COL]),
                    "Gap_Before": np.nan,
                    "Audio_Path": str(audio_path) if audio_path else "",
                    "Segment_Path": str(segment_path),
                }
            )

    segments = pd.DataFrame(rows)
    if not segments.empty:
        previous_end = segments.groupby(["Subject_ID", "Task"])["Absolute_Word_End"].shift()
        segments["Gap_Before"] = segments["Absolute_Word_Start"] - previous_end
        segments.loc[segments["Item_Index"] == 1, "Gap_Before"] = np.nan
    return segments


def write_segment_audio(segments: pd.DataFrame) -> None:
    loaded_audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    for _, row in tqdm(segments.iterrows(), total=len(segments), desc="write segments"):
        audio_path = str(row["Audio_Path"])
        segment_path = Path(str(row["Segment_Path"]))
        if not audio_path or not Path(audio_path).exists():
            continue
        try:
            if audio_path not in loaded_audio_cache:
                loaded_audio_cache[audio_path] = load_audio(Path(audio_path))
            signal, sr = loaded_audio_cache[audio_path]
            seg = slice_signal(signal, sr, row["Absolute_Word_Start"], row["Absolute_Word_End"])
            if len(seg) == 0:
                continue
            segment_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(segment_path), seg, sr)
        except Exception as exc:
            print(f"Failed to write segment {segment_path}: {type(exc).__name__}: {exc}")


def extract_features(segments: pd.DataFrame) -> pd.DataFrame:
    smile = opensmile.Smile(
        feature_set=OPENSMILE_FEATURE_SET,
        feature_level=OPENSMILE_FEATURE_LEVEL,
        num_workers=1,
    )
    rows = []
    loaded_audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    feature_columns: list[str] = []

    for _, row in tqdm(segments.iterrows(), total=len(segments), desc="OpenSMILE"):
        feature_row = row.to_dict()
        audio_path = str(row["Audio_Path"])
        try:
            if not audio_path or not Path(audio_path).exists():
                raise FileNotFoundError(audio_path)
            if audio_path not in loaded_audio_cache:
                loaded_audio_cache[audio_path] = load_audio(Path(audio_path))
            signal, sr = loaded_audio_cache[audio_path]
            seg = slice_signal(signal, sr, row["Absolute_Word_Start"], row["Absolute_Word_End"])
            if len(seg) == 0:
                raise ValueError("empty segment")
            feat = smile.process_signal(seg, sr)
            if feat.empty:
                raise ValueError("empty opensmile result")
            if not feature_columns:
                feature_columns = feat.columns.tolist()
            for col in feature_columns:
                feature_row[col] = feat.iloc[0].get(col, np.nan)
        except Exception as exc:
            print(
                f"OpenSMILE failed: Subject={row['Subject_ID']}, Task={row['Task']}, "
                f"Item={row['Item_Index']}: {type(exc).__name__}: {exc}"
            )
            for col in feature_columns:
                feature_row[col] = np.nan
        rows.append(feature_row)

    front_cols = [
        "Subject_ID",
        "Task",
        "Item_Index",
        "Word",
        "Absolute_Word_Start",
        "Absolute_Word_End",
        "Word_Duration",
        "Gap_Before",
        "Audio_Path",
        "Segment_Path",
    ]
    final_df = pd.DataFrame(rows)
    for col in front_cols + feature_columns:
        if col not in final_df.columns:
            final_df[col] = np.nan
    return final_df[front_cols + feature_columns]


def run_features_for_dataset(cfg: DatasetConfig) -> None:
    print(f"\n[{cfg.name}] Building segments and OpenSMILE features")
    segments = prepare_segments(cfg)
    print(f"[{cfg.name}] valid segment rows: {len(segments)}")

    write_segment_audio(segments)

    cfg.segment_csv.parent.mkdir(parents=True, exist_ok=True)
    segments.to_csv(cfg.segment_csv, index=False, encoding="utf-8-sig")
    print(f"[{cfg.name}] segment table saved: {cfg.segment_csv}")

    features = extract_features(segments)
    features.to_csv(cfg.feature_csv, index=False, encoding="utf-8-sig")
    print(f"[{cfg.name}] feature table saved: {cfg.feature_csv}")


def main() -> None:
    print("Recovered WD Stroop pipeline")
    print(f"Output directory: {BASE_DIR}")
    print(f"RUN_ASR={RUN_ASR}, RUN_FEATURES={RUN_FEATURES}")
    print(f"OpenSMILE: {opensmile.__version__}, eGeMAPSv02 Functionals")

    if RUN_ASR:
        qwen_pipe = load_qwen_pipeline()
        qwen_pipe.LOCAL_FILES_ONLY = LOCAL_FILES_ONLY
        model = qwen_pipe.load_model(str(ASR_MODEL), str(ALIGNER_MODEL), DEVICE, DTYPE)
        for cfg in DATASETS:
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
