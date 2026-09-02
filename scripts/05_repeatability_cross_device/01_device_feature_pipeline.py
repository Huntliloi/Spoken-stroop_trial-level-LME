"""Extract item-level features for the device-repeatability cohort.

Private audio and metadata are not included. By default, the script looks under
``data/not_shared/device_repeatability`` and writes generated files under
``outputs/device_repeatability``. Paths can also be supplied with command-line
arguments.

Extracted acoustic features:
    - 88 OpenSMILE eGeMAPSv02 Functionals
    - 2 timing features: Word_Duration and Gap_Before

The metadata CSV must contain: subject_id, participant_id, device_id, age, sex,
follow_up, session, and stroop_relative_path.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
from tqdm import tqdm


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
from common_paths import DEVICE_REPEATABILITY_OUTPUT_DIR, REPO_ROOT  # noqa: E402

QWEN_PIPELINE_PATH = SCRIPTS_DIR / "01_speech_processing" / "01_qwen3_asr_alignment_pipeline.py"

DEFAULT_AUDIO_ROOT = Path(os.environ.get(
    "DEVICE_AUDIO_ROOT",
    REPO_ROOT / "data" / "not_shared" / "device_repeatability" / "audio",
))
DEFAULT_METADATA_CSV = Path(os.environ.get(
    "DEVICE_METADATA_CSV",
    REPO_ROOT / "data" / "not_shared" / "device_repeatability" / "participant_metadata.csv",
))
DEFAULT_RESULT_DIR = DEVICE_REPEATABILITY_OUTPUT_DIR
DEFAULT_ASR_CACHE_DIR = DEFAULT_RESULT_DIR / "B1_HC_device_stroop_asr_cache"
DEFAULT_FEATURE_CSV = DEFAULT_RESULT_DIR / "Stroop_features_device_repeatability.csv"
DEFAULT_ASR_INPUT_MANIFEST = DEFAULT_ASR_CACHE_DIR / "asr_input_manifest.csv"

ASR_MODEL = Path(os.environ.get("QWEN3_ASR_MODEL", "Qwen3-ASR-0.6B"))
ALIGNER_MODEL = Path(os.environ.get("QWEN3_ALIGNER_MODEL", "Qwen3-ForcedAligner-0.6B"))

PROCESS_TASKS = ["W", "CW", "C"]
TASK_FROM_CSV_MAP = {"w": "W", "c": "C", "cw": "CW", "CW": "CW"}
EXPECTED_TOKENS_PER_TASK = 12

DEVICE_NAMES = {
    "1": "ASUS_FX63VD_laptop",
    "2": "iPad_2021",
    "3": "Huawei_P30_phone",
}

OPENSMILE_FEATURE_SET = opensmile.FeatureSet.eGeMAPSv02
OPENSMILE_FEATURE_LEVEL = opensmile.FeatureLevel.Functionals

ID_COLS = [
    "Subject_ID",
    "Participant_ID",
    "Device_ID",
    "Device_Name",
    "Age",
    "Sex",
    "Follow_Up",
    "Session",
    "Task",
    "Item_Index",
    "Word",
    "Absolute_Word_Start",
    "Absolute_Word_End",
    "Audio_Path",
]

TIMING_FEATURE_COLS = ["Word_Duration", "Gap_Before"]


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
    return str(x).strip()


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


def metadata_session(follow_up, folder_session: str) -> str:
    s = str(follow_up).strip()
    if s in {"1", "1.0"}:
        return "session_1"
    if s in {"2", "2.0"}:
        return "session_2"
    return str(folder_session)


def load_metadata(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV does not exist: {metadata_csv}")
    meta = pd.read_csv(metadata_csv, dtype=str, encoding="utf-8-sig").fillna("")
    required = [
        "subject_id",
        "participant_id",
        "device_id",
        "age",
        "sex",
        "follow_up",
        "session",
        "stroop_relative_path",
    ]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"{metadata_csv} missing required columns: {missing}")

    meta["Subject_ID"] = meta["subject_id"].astype(str)
    meta["Participant_ID"] = meta["participant_id"].astype(str)
    meta["Device_ID"] = meta["device_id"].astype(str)
    meta["Device_Name"] = meta["Device_ID"].map(DEVICE_NAMES).fillna("unknown_device")
    meta["Age"] = meta["age"].astype(str)
    meta["Sex"] = meta["sex"].astype(str)
    meta["Follow_Up"] = meta["follow_up"].astype(str)
    meta["Session"] = [metadata_session(fu, fs) for fu, fs in zip(meta["Follow_Up"], meta["session"])]
    meta["Stroop_Relative_Path"] = meta["stroop_relative_path"].str.replace("\\", "/", regex=False)
    return meta


def find_hc_device_audio_files(qwen_pipe, audio_root: Path, meta: pd.DataFrame) -> List[Path]:
    wav_files: List[Path] = []
    for rel in sorted(meta["Stroop_Relative_Path"].unique()):
        stroop_dir = audio_root / Path(rel.replace("/", "\\"))
        if not stroop_dir.is_dir():
            print(f"Missing 04_Stroop directory from metadata: {stroop_dir}")
            continue
        for wav_path in sorted(stroop_dir.glob("*.wav")):
            name = wav_path.name.lower()
            if "stroop" not in name:
                continue
            task = qwen_pipe.detect_task_from_filename(wav_path.name)
            if task in {"W", "C", "CW"}:
                wav_files.append(wav_path)
    return wav_files


def build_asr_input_manifest(wav_files: List[Path], audio_root: Path) -> pd.DataFrame:
    rows = []
    root = audio_root.resolve()
    for wav_path in sorted(wav_files):
        stat = wav_path.stat()
        rows.append({
            "relative_audio_path": str(wav_path.resolve().relative_to(root)).replace("\\", "/"),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    return pd.DataFrame(rows)


def validate_asr_cache(qwen_pipe, args, meta: pd.DataFrame) -> None:
    token_csv = args.asr_cache_dir / "token_results.csv"
    manifest_csv = args.asr_cache_dir / "asr_input_manifest.csv"
    if not token_csv.exists():
        raise FileNotFoundError(f"Missing token cache: {token_csv}. Run without --skip-asr.")
    if not manifest_csv.exists():
        raise FileNotFoundError(
            f"Missing ASR input manifest: {manifest_csv}. "
            "The audio selection rule changed; run B.1 once without --skip-asr."
        )

    expected_wavs = find_hc_device_audio_files(qwen_pipe, args.audio_root, meta)
    expected = build_asr_input_manifest(expected_wavs, args.audio_root)
    cached = pd.read_csv(manifest_csv, dtype={"relative_audio_path": str})
    sort_col = ["relative_audio_path"]
    expected = expected.sort_values(sort_col).reset_index(drop=True)
    cached = cached.sort_values(sort_col).reset_index(drop=True)
    if not expected.equals(cached):
        raise RuntimeError(
            "ASR cache does not match the current selected audio files. "
            "Run B.1 without --skip-asr to rebuild ASR, token, and feature outputs."
        )


def run_asr(qwen_pipe, args, meta: pd.DataFrame) -> None:
    args.asr_cache_dir.mkdir(parents=True, exist_ok=True)

    qwen_pipe.AUDIO_ROOT = str(args.audio_root)
    qwen_pipe.OUTPUT_DIR = str(args.asr_cache_dir)
    qwen_pipe.ASR_MODEL = str(args.asr_model)
    qwen_pipe.ALIGNER_MODEL = str(args.aligner_model)
    qwen_pipe.DEVICE = args.device
    qwen_pipe.DTYPE = args.dtype
    qwen_pipe.LOCAL_FILES_ONLY = args.local_files_only
    qwen_pipe.EXPECTED_TOKENS_PER_TASK = EXPECTED_TOKENS_PER_TASK

    wav_files = find_hc_device_audio_files(qwen_pipe, args.audio_root, meta)
    print(f"ASR input wav files: {len(wav_files)}")
    if not wav_files:
        raise RuntimeError(f"No Stroop wav files found under {args.audio_root}")

    model = qwen_pipe.load_model(str(args.asr_model), str(args.aligner_model), args.device, args.dtype)
    utterance_records: List[Dict] = []
    token_records: List[Dict] = []

    for wav_path in tqdm(wav_files, desc="HC device Stroop ASR"):
        record, token_rows = qwen_pipe.process_one_file(model, wav_path, str(args.audio_root))
        utterance_records.append(record)
        token_records.extend(token_rows)
        qwen_pipe.save_json_record(str(args.asr_cache_dir), record)

    qwen_pipe.write_tables(str(args.asr_cache_dir), utterance_records, token_records)
    build_asr_input_manifest(wav_files, args.audio_root).to_csv(
        args.asr_cache_dir / "asr_input_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    n_review = sum(1 for x in utterance_records if x.get("needs_review"))
    print(f"ASR done: utterances={len(utterance_records)}, tokens={len(token_records)}, needs_review={n_review}")


def audio_path_to_stroop_relative(audio_path: Path, audio_root: Path) -> str:
    rel = audio_path.resolve().relative_to(audio_root.resolve())
    parts = rel.parts
    if "04_Stroop" not in parts:
        raise ValueError(f"Audio path is not under 04_Stroop: {audio_path}")
    idx = parts.index("04_Stroop")
    return "/".join(parts[: idx + 1])


def prepare_token_table(token_csv: Path, audio_root: Path, meta: pd.DataFrame) -> pd.DataFrame:
    if not token_csv.exists():
        raise FileNotFoundError(f"Missing token_results.csv: {token_csv}")
    df = pd.read_csv(token_csv)

    required = ["task", "token_index", "token_text", "audio_path", "start_time", "end_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{token_csv} missing columns: {missing}")

    meta_key_cols = [
        "Stroop_Relative_Path",
        "Subject_ID",
        "Participant_ID",
        "Device_ID",
        "Device_Name",
        "Age",
        "Sex",
        "Follow_Up",
        "Session",
    ]
    meta_map = meta[meta_key_cols].drop_duplicates("Stroop_Relative_Path")

    work = df.copy()
    work["Audio_Path"] = work["audio_path"].astype(str)
    work["Stroop_Relative_Path"] = [
        audio_path_to_stroop_relative(Path(p), audio_root) for p in work["Audio_Path"]
    ]
    work = work.merge(meta_map, on="Stroop_Relative_Path", how="left")
    if work["Subject_ID"].isna().any():
        bad = work.loc[work["Subject_ID"].isna(), "Stroop_Relative_Path"].drop_duplicates().head(10).tolist()
        raise ValueError(f"Some token rows could not be matched to metadata, examples: {bad}")

    work["Task"] = work["task"].map(normalize_task_for_report)
    work["Item_Index"] = pd.to_numeric(work["token_index"], errors="coerce")
    work["Word"] = work["token_text"].map(normalize_word)
    work["Absolute_Word_Start"] = pd.to_numeric(work["start_time"], errors="coerce")
    work["Absolute_Word_End"] = pd.to_numeric(work["end_time"], errors="coerce")

    work = work[work["Task"].isin(PROCESS_TASKS)].copy()
    work = work[work["Session"].isin(["session_1", "session_2"])].copy()
    work = work[work["Item_Index"].notna()].copy()
    work = work[work["Absolute_Word_Start"].notna() & work["Absolute_Word_End"].notna()].copy()
    work = work[work["Absolute_Word_End"] > work["Absolute_Word_Start"]].copy()
    work["Item_Index"] = work["Item_Index"].astype(int)

    sort_cols = ["Subject_ID", "Session", "Task", "Item_Index"]
    work = work.sort_values(sort_cols).copy()
    work = work.groupby(["Subject_ID", "Session", "Task"], as_index=False, group_keys=False).head(EXPECTED_TOKENS_PER_TASK)
    work["Item_Index"] = work.groupby(["Subject_ID", "Session", "Task"]).cumcount() + 1
    work["Word_Duration"] = work["Absolute_Word_End"] - work["Absolute_Word_Start"]

    gaps = []
    prev_end_by_key: Dict[Tuple[str, str, str], float] = {}
    for _, row in work.iterrows():
        key = (str(row["Subject_ID"]), str(row["Session"]), str(row["Task"]))
        start_s = float(row["Absolute_Word_Start"])
        prev_end = prev_end_by_key.get(key)
        gaps.append(start_s if prev_end is None else start_s - prev_end)
        prev_end_by_key[key] = float(row["Absolute_Word_End"])
    work["Gap_Before"] = gaps

    return work[ID_COLS + TIMING_FEATURE_COLS].copy()


def discover_feature_columns(tokens: pd.DataFrame) -> List[str]:
    smile = opensmile.Smile(feature_set=OPENSMILE_FEATURE_SET, feature_level=OPENSMILE_FEATURE_LEVEL)
    for _, row in tokens.iterrows():
        audio_path = Path(str(row["Audio_Path"]))
        if not audio_path.exists():
            continue
        try:
            signal, sr = load_audio(audio_path)
            seg = slice_signal(signal, sr, float(row["Absolute_Word_Start"]), float(row["Absolute_Word_End"]))
            if len(seg) == 0:
                continue
            feat = smile.process_signal(seg, sr)
            if not feat.empty:
                return feat.columns.tolist()
        except Exception:
            continue
    raise RuntimeError("No valid audio segment found for OpenSMILE feature-column discovery.")


def extract_features(tokens: pd.DataFrame) -> pd.DataFrame:
    feature_columns = discover_feature_columns(tokens)
    print(f"OpenSMILE feature columns: {len(feature_columns)}")

    smile = opensmile.Smile(
        feature_set=OPENSMILE_FEATURE_SET,
        feature_level=OPENSMILE_FEATURE_LEVEL,
        num_workers=1,
    )

    loaded_audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}
    rows: List[Dict] = []

    for _, row in tqdm(tokens.iterrows(), total=len(tokens), desc="OpenSMILE"):
        feature_row = row.to_dict()
        audio_path = str(row["Audio_Path"])

        try:
            p = Path(audio_path)
            if not p.exists():
                raise FileNotFoundError(audio_path)
            if audio_path not in loaded_audio_cache:
                loaded_audio_cache[audio_path] = load_audio(p)
            signal, sr = loaded_audio_cache[audio_path]
            seg = slice_signal(signal, sr, float(row["Absolute_Word_Start"]), float(row["Absolute_Word_End"]))
            if len(seg) == 0:
                raise ValueError("empty segment")
            feat = smile.process_signal(seg, sr)
            if feat.empty:
                raise ValueError("empty OpenSMILE result")
            for col, val in zip(feature_columns, feat.iloc[0].values):
                feature_row[col] = val
        except Exception as exc:
            print(
                f"OpenSMILE failed: Subject={row['Subject_ID']}, Session={row['Session']}, "
                f"Task={row['Task']}, Item={row['Item_Index']}: {type(exc).__name__}: {exc}"
            )
            for col in feature_columns:
                feature_row[col] = np.nan

        rows.append(feature_row)

    final_df = pd.DataFrame(rows)
    final_cols = ID_COLS + TIMING_FEATURE_COLS + feature_columns
    for col in final_cols:
        if col not in final_df.columns:
            final_df[col] = np.nan
    return final_df[final_cols]


def parse_args():
    parser = argparse.ArgumentParser(description="Run HC device Stroop ASR alignment and feature extraction.")
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--asr-cache-dir", type=Path, default=DEFAULT_ASR_CACHE_DIR)
    parser.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    parser.add_argument("--asr-model", type=Path, default=ASR_MODEL)
    parser.add_argument("--aligner-model", type=Path, default=ALIGNER_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--skip-asr", action="store_true", help="Reuse asr-cache-dir/token_results.csv and only rebuild features.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    args.asr_cache_dir.mkdir(parents=True, exist_ok=True)
    args.feature_csv.parent.mkdir(parents=True, exist_ok=True)

    print("HC device Stroop pipeline")
    print(f"audio_root: {args.audio_root}")
    print(f"metadata_csv: {args.metadata_csv}")
    print(f"asr_cache_dir: {args.asr_cache_dir}")
    print(f"feature_csv: {args.feature_csv}")
    print(f"skip_asr: {args.skip_asr}")

    meta = load_metadata(args.metadata_csv)
    qwen_pipe = load_qwen_pipeline()
    if not args.skip_asr:
        run_asr(qwen_pipe, args, meta)
    else:
        validate_asr_cache(qwen_pipe, args, meta)

    token_csv = args.asr_cache_dir / "token_results.csv"
    tokens = prepare_token_table(token_csv, args.audio_root, meta)
    print(f"valid token rows for feature extraction: {len(tokens)}")
    if tokens.empty:
        raise RuntimeError("No valid token rows after filtering.")

    features = extract_features(tokens)
    features.to_csv(args.feature_csv, index=False, encoding="utf-8-sig")
    n_opensmile = len([c for c in features.columns if c not in set(ID_COLS + TIMING_FEATURE_COLS)])
    print(f"feature table saved: {args.feature_csv}")
    print(f"rows={len(features)}, OpenSMILE={n_opensmile}, timing={len(TIMING_FEATURE_COLS)}, total_features={n_opensmile + len(TIMING_FEATURE_COLS)}")


if __name__ == "__main__":
    main()
