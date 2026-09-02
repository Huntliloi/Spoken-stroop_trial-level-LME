import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


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
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

sns.set_theme(style="ticks", context="paper")

JOURNAL_PALETTE = ['#5A93C4', '#E18A54', '#78AB85']

# =========================
# =========================
CSV_PATH = str(get_behavioral_metrics_csv())  # noqa: F405
OUTPUT_DIR = str(BEHAVIOR_OUTPUT_DIR / "behavior_stats_plots")  # noqa: F405

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# =========================
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')

task_order = ['W', 'C', 'CW']
df = df[df['Task'].isin(task_order)].copy()
df['Task'] = pd.Categorical(df['Task'], categories=task_order, ordered=True)


# =========================
# =========================
def descriptive_stats_by_task(data, value_col, task_col='Task'):
    results = []
    for task in task_order:
        sub = data[data[task_col] == task][value_col].dropna()
        if len(sub) == 0:
            results.append({
                'Task': task, 'N': 0, 'Mean': np.nan, 'SD': np.nan,
                'Median': np.nan, 'Q1': np.nan, 'Q3': np.nan,
                'IQR': np.nan, 'Min': np.nan, 'Max': np.nan
            })
            continue
        q1 = sub.quantile(0.25)
        q3 = sub.quantile(0.75)
        results.append({
            'Task': task, 'N': len(sub), 'Mean': sub.mean(), 'SD': sub.std(ddof=1),
            'Median': sub.median(), 'Q1': q1, 'Q3': q3,
            'IQR': q3 - q1, 'Min': sub.min(), 'Max': sub.max()
        })
    return pd.DataFrame(results)


# =========================
# =========================
acc_stats = descriptive_stats_by_task(df, 'Overall_Accuracy')
hes_stats = descriptive_stats_by_task(df, 'Total_Hesitation_Duration')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

print("\n==============================")
print("Overall_Accuracy descriptive statistics")
print("==============================")
print(acc_stats)

print("\n==============================")
print("Total_Hesitation_Duration descriptive statistics")
print("==============================")
print(hes_stats)

acc_stats.to_csv(os.path.join(OUTPUT_DIR, 'Overall_Accuracy_descriptive_stats.csv'), index=False, encoding='utf-8-sig')
hes_stats.to_csv(os.path.join(OUTPUT_DIR, 'Total_Hesitation_Duration_descriptive_stats.csv'), index=False,
                 encoding='utf-8-sig')


# =========================
# =========================
def plot_violin(data, y_col, y_label, title, save_name_base):
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.violinplot(
        data=data,
        x='Task',
        y=y_col,
        hue='Task',  # use hue with palette
        palette=JOURNAL_PALETTE,
        order=task_order,
        inner=None,
        cut=0,
        linewidth=1.5,  # thicker outline
        alpha=0.8,  # slightly transparent fill
        legend=False,
        ax=ax
    )

    sns.boxplot(
        data=data,
        x='Task',
        y=y_col,
        order=task_order,
        width=0.12,  # narrower box
        showcaps=False,  # hide whisker caps
        boxprops={'facecolor': 'white', 'edgecolor': 'black', 'zorder': 3, 'linewidth': 1.5},
        whiskerprops={'linewidth': 1.5, 'color': 'black'},
        medianprops={'color': 'black', 'linewidth': 2},  # thicker median line
        showfliers=False,  # hide boxplot outliers because stripplot shows points
        ax=ax
    )

    sns.stripplot(
        data=data,
        x='Task',
        y=y_col,
        order=task_order,
        color='#333333',  # dark gray rather than pure black
        size=5.5,  # slightly larger points
        alpha=0.6,
        jitter=0.15,
        edgecolor='white',  # white outline
        linewidth=0.6,
        zorder=2,
        ax=ax
    )

    ax.set_xlabel('Task', fontsize=24, fontweight='bold', labelpad=10)
    ax.set_ylabel(y_label, fontsize=24, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=22, fontweight='bold', pad=15)

    ax.tick_params(axis='both', which='major', labelsize=20, width=1.5, length=6)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    sns.despine(trim=False, offset=5)

    plt.tight_layout()

    save_path_png = os.path.join(OUTPUT_DIR, f"{save_name_base}.png")
    save_path_pdf = os.path.join(OUTPUT_DIR, f"{save_name_base}.pdf")

    plt.savefig(save_path_png, dpi=1200, bbox_inches='tight')
    plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight', transparent=True)  # vector graphic

    plt.close(fig)


# =========================
# =========================
plot_violin(
    data=df,
    y_col='Overall_Accuracy',
    y_label='Overall Accuracy',
    title='Overall Accuracy across Tasks',  # simplified title
    save_name_base='violin_overall_accuracy_HQ'
)

plot_violin(
    data=df,
    y_col='Total_Hesitation_Duration',
    y_label='Total Hesitation Duration (s)',
    title='Total Hesitation Duration across Tasks',
    save_name_base='violin_total_hesitation_duration_HQ'
)

print(f"\nHigh-resolution figures (PNG 600 dpi and vector PDF) and descriptive statistics were saved to:\n{OUTPUT_DIR}")



