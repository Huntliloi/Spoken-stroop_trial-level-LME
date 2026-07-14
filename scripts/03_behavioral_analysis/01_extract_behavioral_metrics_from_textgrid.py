import os
import re
import pandas as pd
from tqdm import tqdm
from praatio import tgio


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
AUDIO_ROOT = str(AUDIO_ROOT)  # noqa: F405
TEXTGRID_ROOT = str(MANUAL_TEXTGRID_ROOT)  # noqa: F405

FINAL_OUTPUT_CSV_FILE_PATH = str(BEHAVIORAL_METRICS_CSV)  # noqa: F405

VALID_KEYWORDS = {'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.'}
TASK_FILENAME_MAP = {'Hanzi': 'hanzi', 'STT': 'STT', 'Dots': 'dots'}
PROCESS_TASKS = ['Hanzi', 'STT', 'Dots']

TASK_TARGET_SEQUENCES = {
    'STT':   ['English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.'],
    'Dots':  ['English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.'],
    'Hanzi': ['English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.', 'English documentation.']
}


# =========================
# =========================
def get_all_subject_ids(audio_root):
    'English documentation.'
    subject_ids = set()
    for root, _, files in os.walk(audio_root):
        for f in files:
            if f.endswith('.wav') and '_stroop_' in f:
                m = re.match(r'(\d{11})_stroop_.*\.wav', f)
                if m:
                    subject_ids.add(m.group(1))
    return sorted(subject_ids)


def load_word_entries_from_textgrid(textgrid_path, tier_name='word'):
    '\n      TextGrid word   interval\n      : [(start, end, label), ...]\n    '
    tg = tgio.openTextgrid(textgrid_path)

    if not hasattr(tg, 'tierDict') or not tg.tierDict:
        raise ValueError(f"TextGrid parsing failed or content is empty: {textgrid_path}")

    if tier_name not in tg.tierDict:
        raise ValueError(f"TextGrid tier not found '{tier_name}': {textgrid_path}")

    word_tier = tg.tierDict[tier_name]
    entries = []

    for entry in word_tier.entryList:
        if len(entry) == 3:
            start_time, end_time, text = entry
            text = str(text).strip()
            entries.append((float(start_time), float(end_time), text))

    return entries


def extract_first_12_valid_words(entries):
    '\n       12    （ / / / ）\n      : [(start, end, label), ...]\n    '
    valid_entries = []
    for start, end, label in entries:
        if label in VALID_KEYWORDS:
            valid_entries.append((start, end, label))
            if len(valid_entries) == 12:
                break
    return valid_entries


def compute_overall_accuracy(valid_entries, target_sequence):
    '\n         ： 12           \n        12 ，    12   \n    '
    correct_count = 0
    for i in range(min(len(valid_entries), 12)):
        produced_label = valid_entries[i][2]
        target_label = target_sequence[i]
        if produced_label == target_label:
            correct_count += 1

    overall_accuracy = correct_count / 12.0
    return correct_count, overall_accuracy


def compute_total_hesitation_duration(valid_entries):
    '\n              ：\n    0   1       +            \n                   \n\n        ：\n    gap1 = first_valid_start - 0\n    gap2 = valid2_start - valid1_end\n    gap3 = valid3_start - valid2_end\n    ...\n    gap12 = valid12_start - valid11_end\n       =   gap  \n\n     ：\n    -        、 、 、   ，             ，\n               gap \n    '
    if len(valid_entries) == 0:
        return 0.0

    total_gap = 0.0

    first_start = valid_entries[0][0]
    total_gap += max(0.0, first_start)

    for i in range(1, len(valid_entries)):
        prev_end = valid_entries[i - 1][1]
        curr_start = valid_entries[i][0]
        gap = curr_start - prev_end
        total_gap += max(0.0, gap)

    return total_gap


def compute_task_level_features(entries, target_sequence):
    '\n        Task      ：\n    1. Overall_Accuracy\n    2. Total_Hesitation_Duration\n    '
    valid_entries = extract_first_12_valid_words(entries)

    valid_count = len(valid_entries)
    correct_count, overall_accuracy = compute_overall_accuracy(valid_entries, target_sequence)
    total_hesitation_duration = compute_total_hesitation_duration(valid_entries)

    return {
        'Valid_Trial_Count': valid_count,
        'Correct_Trial_Count': correct_count,
        'Overall_Accuracy': overall_accuracy,
        'Total_Hesitation_Duration': total_hesitation_duration
    }


# =========================
# =========================
def process_new_features():
    print('       ：      +      （       ）')

    subject_ids = get_all_subject_ids(AUDIO_ROOT)
    print(f"Found {len(subject_ids)} subjects。")

    all_results = []

    for subject_id_str in tqdm(subject_ids, desc='English documentation.'):
        for task in PROCESS_TASKS:
            textgrid_filename = f"{subject_id_str}_stroop_{TASK_FILENAME_MAP[task]}.TextGrid"
            textgrid_path = os.path.join(TEXTGRID_ROOT, textgrid_filename)

            if not os.path.exists(textgrid_path):
                print(f"Warning：TextGrid file not found: {textgrid_path}")
                continue

            try:
                entries = load_word_entries_from_textgrid(textgrid_path, tier_name='word')
                result = compute_task_level_features(
                    entries=entries,
                    target_sequence=TASK_TARGET_SEQUENCES[task]
                )

                row = {
                    'Subject_ID': subject_id_str,
                    'Task': task,
                    'Valid_Trial_Count': result['Valid_Trial_Count'],
                    'Correct_Trial_Count': result['Correct_Trial_Count'],
                    'Overall_Accuracy': result['Overall_Accuracy'],
                    'Total_Hesitation_Duration': result['Total_Hesitation_Duration']
                }
                all_results.append(row)

            except Exception as e:
                print(f"Warning: error while processing {textgrid_path}: {e}")

    if not all_results:
        print("No behavioral metrics were generated. Check the TextGrid directory and tier names.")
        return

    result_df = pd.DataFrame(all_results)
    result_df['Subject_ID'] = result_df['Subject_ID'].astype(str)
    result_df = result_df.sort_values(by=['Subject_ID', 'Task']).reset_index(drop=True)

    os.makedirs(os.path.dirname(FINAL_OUTPUT_CSV_FILE_PATH), exist_ok=True)
    result_df.to_csv(FINAL_OUTPUT_CSV_FILE_PATH, index=False, encoding='utf-8-sig')

    print('\n  Done.')
    print(f"Results saved to: {FINAL_OUTPUT_CSV_FILE_PATH}")
    print('English documentation.')
    print(result_df.head(10))


if __name__ == "__main__":
    try:
        import praatio
        print(f"praatio version: {praatio.__version__}")
    except ImportError:
        print('  ：    praatio。    : pip install praatio')
    except Exception as e:
        print(f"Error while checking praatio version: {e}")

    process_new_features()



