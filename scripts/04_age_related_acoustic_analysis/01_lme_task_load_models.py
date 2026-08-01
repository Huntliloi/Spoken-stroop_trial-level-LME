import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import patsy
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests  # for FDR-BH correction
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# Public repository path setup
from pathlib import Path as _RepoPath
import sys as _repo_sys
_REPO_SCRIPTS_DIR = _RepoPath(__file__).resolve().parents[2] / "scripts"
if str(_REPO_SCRIPTS_DIR) not in _repo_sys.path:
    _repo_sys.path.insert(0, str(_REPO_SCRIPTS_DIR))
from common_paths import *  # noqa: F403
ensure_output_dirs()
FEATURE_PATH = str(get_imputed_acoustic_features_csv())  # noqa: F405
INFO_PATH = str(PARTICIPANT_INFO_XLSX)  # noqa: F405

COMPARISON = "STT_vs_Hanzi"
# "STT_vs_Hanzi"
# "STT_vs_Dots"
# "Dots_vs_Hanzi"

TARGET_FEATURES = 'ALL'

COMPARISON_CONFIG = {
    "STT_vs_Hanzi": {
        "tasks": ["Hanzi", "STT"],
        "task_code_map": {"Hanzi": 0, "STT": 1},
        "base_task": "Hanzi",
        "contrast_task": "STT",
        "output_file": "plot_paper_STT_vs_Hanzi.csv",
        "plot_dir": "result_LME_STT_vs_Hanzi",
        "interference_label": "Stroop Interference (STT-Hanzi)"
    },
    "STT_vs_Dots": {
        "tasks": ["Dots", "STT"],
        "task_code_map": {"Dots": 0, "STT": 1},
        "base_task": "Dots",
        "contrast_task": "STT",
        "output_file": "plot_paper_STT_vs_Dots.csv",
        "plot_dir": "result_LME_STT_vs_Dots",
        "interference_label": "Stroop Interference (STT-Dots)"
    },
    "Dots_vs_Hanzi": {
        "tasks": ["Hanzi", "Dots"],
        "task_code_map": {"Hanzi": 0, "Dots": 1},
        "base_task": "Hanzi",
        "contrast_task": "Dots",
        "output_file": "plot_paper_Dots_vs_Hanzi.csv",
        "plot_dir": "result_LME_Dots_vs_Hanzi",
        "interference_label": "Stroop Interference (Dots-Hanzi)"
    }
}

if COMPARISON not in COMPARISON_CONFIG:
    raise ValueError(f"Invalid COMPARISON: {COMPARISON}")

cfg = COMPARISON_CONFIG[COMPARISON]
OUTPUT_FILE = str(LME_OUTPUT_DIR / cfg["output_file"])  # noqa: F405
PLOT_DIR = str(LME_OUTPUT_DIR / cfg["plot_dir"])  # noqa: F405


def translate_opensmile_feature(name):
    if name == 'Gap_Before':
        return "Hesitation duration (response latency)"
    if name == 'Word_Duration':
        return "Word duration (articulation lengthening)"
    if name == 'Duration':
        return "Word duration (articulation lengthening)"

    stat_cn = ""
    if "amean" in name:
        stat_cn = "Mean"
    elif "stddevNorm" in name:
        stat_cn = "Coefficient of variation"
    elif "stddev" in name:
        stat_cn = "Standard deviation"
    elif "percentile20" in name:
        stat_cn = "20th percentile"
    elif "percentile50" in name:
        stat_cn = "Median"
    elif "percentile80" in name:
        stat_cn = "80th percentile"
    elif "pctlrange" in name:
        stat_cn = "Percentile range"
    elif "meanSegLen" in name:
        stat_cn = "Mean segment length"
    elif "meanRisingSlope" in name:
        stat_cn = "Mean rising slope"
    elif "meanFallingSlope" in name:
        stat_cn = "Mean falling slope"
    elif "PerSecond" in name:
        stat_cn = "Events per second"

    base_cn = name.split('_')[0]
    desc = ""

    if "F0semitone" in name:
        base_cn = "F0 semitone";
        desc = "pitch"
    elif "jitter" in name.lower():
        base_cn = "Jitter";
        desc = "roughness"
    elif "F1" in name:
        base_cn = "First formant";
        desc = "mouth opening"
    elif "F2" in name:
        base_cn = "Second formant";
        desc = "tongue position"
    elif "F3" in name:
        base_cn = "Third formant";
        desc = "timbre"
    elif "loudness" in name.lower():
        base_cn = "Loudness";
        desc = "intensity"
    elif "shimmer" in name.lower():
        base_cn = "Shimmer";
        desc = "hoarseness"
    elif "HNR" in name:
        base_cn = "Harmonics-to-noise ratio";
        desc = "voice clarity"
    elif "alphaRatio" in name:
        base_cn = "Alpha ratio";
        desc = "energy distribution"
    elif "hammarbergIndex" in name.lower():
        base_cn = "Hammarberg index";
        desc = "breathiness"
    elif "spectralSlope" in name:
        base_cn = "Spectral slope";
        desc = 'timbre  '
    elif "mfcc" in name.lower():
        num = re.search(r'mfcc(\d+)', name.lower())
        idx = num.group(1) if num else ""
        base_cn = f"MFCC coefficient {idx}";
        desc = 'spectral-envelope representation'
    elif "spectralFlux" in name:
        base_cn = "Spectral flux";
        desc = 'timbre  '
    elif "spectralCentroid" in name:
        base_cn = 'Spectral centroid';
        desc = 'timbre  '
    elif "spectralEntropy" in name:
        base_cn = 'Spectral entropy';
        desc = 'timbre  '
    elif "spectralVariance" in name:
        base_cn = 'Spectral variance';
        desc = 'spectral spread'
    elif "spectralSkewness" in name:
        base_cn = 'Spectral skewness';
        desc = 'energy distribution   '
    elif "spectralKurtosis" in name:
        base_cn = 'Spectral kurtosis';
        desc = 'energy distribution   '
    elif "voicingFinalUnclipped" in name:
        base_cn = 'Final voicing probability';
        desc = 'voicing stability'
    elif "VoicedSegments" in name:
        base_cn = 'Voiced-segment rate';
        desc = 'phonation timing'
    elif "UnvoicedSegment" in name:
        base_cn = 'Unvoiced-segment duration';
        desc = 'pause timing'

    if stat_cn:
        res = f"{base_cn}-{stat_cn}"
        if desc:
            res += f" [{desc}]"
        return res
    else:
        if "VoicedSegmentsPerSecond" in name:
            return '       [    ]'
        if "MeanVoicedSegmentLength" in name:
            return '        [     ]'
        if "StddevVoicedSegmentLength" in name:
            return '     Standard deviation [     ]'
        if "MeanUnvoicedSegmentLength" in name:
            return '        [    ]'
        if "StddevUnvoicedSegmentLength" in name:
            return '     Standard deviation [     ]'
        return name


def plot_fitted_lines(results, df_model, feature_name, scaler, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    age_z_min, age_z_max = df_model['Age_Z'].min(), df_model['Age_Z'].max()
    age_z_seq = np.linspace(age_z_min, age_z_max, 100)

    new_data_base = pd.DataFrame({'Age_Z': age_z_seq, 'Task_Code': 0})
    new_data_contrast = pd.DataFrame({'Age_Z': age_z_seq, 'Task_Code': 1})

    pred_base = results.predict(new_data_base)
    pred_contrast = results.predict(new_data_contrast)

    rhs_formula = results.model.formula.split('~')[1].strip()
    design_matrix_base = patsy.dmatrix(rhs_formula, new_data_base, return_type='dataframe')
    design_matrix_contrast = patsy.dmatrix(rhs_formula, new_data_contrast, return_type='dataframe')

    cov_params = results.cov_params()
    aligned_cov_params_base = cov_params.loc[design_matrix_base.columns, design_matrix_base.columns]
    aligned_cov_params_contrast = cov_params.loc[design_matrix_contrast.columns, design_matrix_contrast.columns]

    pred_var_base = np.diag(design_matrix_base @ aligned_cov_params_base @ design_matrix_base.T)
    pred_se_base = np.sqrt(pred_var_base)
    pred_var_contrast = np.diag(design_matrix_contrast @ aligned_cov_params_contrast @ design_matrix_contrast.T)
    pred_se_contrast = np.sqrt(pred_var_contrast)

    z = 1.96
    ci_lower_base, ci_upper_base = pred_base - z * pred_se_base, pred_base + z * pred_se_base
    ci_lower_contrast, ci_upper_contrast = pred_contrast - z * pred_se_contrast, pred_contrast + z * pred_se_contrast

    age_real = age_z_seq * scaler.scale_[0] + scaler.mean_[0]

    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ax.plot(age_real, pred_base, color='royalblue', linewidth=2.5, label=cfg["base_task"])
    ax.fill_between(age_real, ci_lower_base, ci_upper_base, color='royalblue', alpha=0.2,
                    label=f'{cfg["base_task"]} 95% CI')
    ax.plot(age_real, pred_contrast, color='darkorange', linewidth=2.5, label=cfg["contrast_task"])
    ax.fill_between(age_real, ci_lower_contrast, ci_upper_contrast, color='darkorange', alpha=0.2,
                    label=f'{cfg["contrast_task"]} 95% CI')

    ax.set_title('Task Separation by Age', fontsize=28, weight='bold', pad=20)
    ax.set_xlabel("Age (years)", fontsize=26)
    ax.set_ylabel("Feature Value (Z-scored)", fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.legend(loc='best', fontsize=24)
    ax.grid(True, linestyle=':', alpha=0.6)

    safe_name = feature_name.replace('/', '_').replace('\\', '_')
    save_path = os.path.join(save_dir, f"Fit_{safe_name}_Tasks_CI.png")
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close()


def plot_interference_effect(results, df_model, feature_name, scaler, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    age_z_min, age_z_max = df_model['Age_Z'].min(), df_model['Age_Z'].max()
    age_z_seq = np.linspace(age_z_min, age_z_max, 100)

    new_data_base = pd.DataFrame({'Age_Z': age_z_seq, 'Task_Code': 0})
    new_data_contrast = pd.DataFrame({'Age_Z': age_z_seq, 'Task_Code': 1})

    pred_base = results.predict(new_data_base)
    pred_contrast = results.predict(new_data_contrast)
    pred_diff = pred_contrast - pred_base

    rhs_formula = results.model.formula.split('~')[1].strip()
    design_matrix_base = patsy.dmatrix(rhs_formula, new_data_base, return_type='dataframe')
    design_matrix_contrast = patsy.dmatrix(rhs_formula, new_data_contrast, return_type='dataframe')
    design_matrix_diff = design_matrix_contrast - design_matrix_base

    cov_params = results.cov_params()
    aligned_cov_params = cov_params.loc[design_matrix_diff.columns, design_matrix_diff.columns]
    pred_var_diff = np.diag(design_matrix_diff @ aligned_cov_params @ design_matrix_diff.T)
    pred_se_diff = np.sqrt(pred_var_diff)

    z = 1.96
    ci_lower_diff, ci_upper_diff = pred_diff - z * pred_se_diff, pred_diff + z * pred_se_diff
    age_real = age_z_seq * scaler.scale_[0] + scaler.mean_[0]

    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ax.plot(age_real, pred_diff, color='crimson', linewidth=2.5, label='Interference Effect')
    ax.fill_between(age_real, ci_lower_diff, ci_upper_diff, color='crimson', alpha=0.2, label='95% CI')
    ax.legend(loc='upper left', fontsize=24)

    try:
        interaction_term = 'Age_Z:Task_Code'
        coef = results.params[interaction_term]
        p_val = results.pvalues[interaction_term]
        stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        stats_text = (f"$\\beta_3$ = {coef:.3f}\n"
                      f"P-value = {p_val:.3f} {stars}")
        ax.text(0.95, 0.1, stats_text, transform=ax.transAxes, fontsize=24,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', fc='#f0f0f0', ec='gray', alpha=0.9))
    except Exception as e:
        print(f"      [Warning] Cannot add statistical indicators: {e}")

    ax.axhline(0, color='grey', linestyle='--', linewidth=1)
    ax.set_title('Stroop Interference Effect by Age', fontsize=28, weight='bold', pad=20)
    ax.set_xlabel("Age (years)", fontsize=26)
    ax.set_ylabel(cfg["interference_label"], fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.grid(True, linestyle=':', alpha=0.6)

    safe_name = feature_name.replace('/', '_').replace('\\', '_')
    save_path = os.path.join(save_dir, f"Fit_{safe_name}_Interference_CI.png")
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    sns.set(style="whitegrid", font_scale=1.1)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    print("1. Reading participant information and acoustic features...")
    df_info = pd.read_excel(INFO_PATH)
    df_info['Subject_ID'] = df_info['Subject_ID'].astype(str)

    df_audio = pd.read_csv(FEATURE_PATH)
    df_audio['Subject_ID'] = df_audio['Subject_ID'].astype(str)

    df = pd.merge(df_audio, df_info, left_on='Subject_ID', right_on='Subject_ID', how='inner')
    print(f"   -> Merged rows: {len(df)} rows")

    print(f"2. Preprocessing (keeping {cfg['tasks'][0]} and {cfg['tasks'][1]})...")
    df_model = df[df['Task'].isin(cfg['tasks'])].copy()
    df_model.dropna(subset=['Age'], inplace=True)
    df_model['Task_Code'] = df_model['Task'].map(cfg['task_code_map'])

    age_scaler = StandardScaler()
    df_model['Age_Z'] = age_scaler.fit_transform(df_model[['Age']])

    meta_cols = [
        'Subject_ID', 'Task', 'Item_Index', 'Word', 'Subject_ID', 'Name', 'Sex', 'Age', 'Education_Years', 'Task_Code',
        'Age_Z', 'Index', 'IQCODE',
        'Speech_Test_Noise_dBA', 'Participant_Fee',

        'Absolute_Word_Start', 'Absolute_Word_End',
        'n_missing_features_before', 'n_missing_features_after', 'any_feature_imputed'
    ]
    numeric_df = df_model.select_dtypes(include=np.number)
    all_acoustic_cols = [c for c in numeric_df.columns if c not in meta_cols and 'Unnamed' not in c]


    def clean_feature_name(name):
        if re.search(r'[.\-:]', name):
            return f'Q("{name}")'
        return name


    if all_acoustic_cols:
        feat_scaler = StandardScaler()
        df_model[all_acoustic_cols] = feat_scaler.fit_transform(df_model[all_acoustic_cols])

    if TARGET_FEATURES == 'ALL':
        run_list = all_acoustic_cols
    else:
        run_list = [f for f in TARGET_FEATURES if f in df_model.columns]

    print(f"   -> Features to analyze: {len(run_list)} features.")

    results_list = []
    print('\n===    Statsmodels          ===')

    for i, feature in enumerate(run_list):
        print(f"\n[{i + 1}/{len(run_list)}] Analyzing feature: {feature}")

        model_data = df_model.dropna(subset=[feature]).copy()
        if model_data.empty:
            print(f"   -> [Warning] feature '{feature}' has no valid data and was skipped.")
            continue

        try:
            cleaned_feature = clean_feature_name(feature)
            formula = f"{cleaned_feature} ~ Age_Z * Task_Code"
            model = smf.mixedlm(formula, model_data,
                                groups=model_data["Subject_ID"],
                                re_formula="~Task_Code")

            results = model.fit(method=["lbfgs"])

            interaction_term = 'Age_Z:Task_Code'
            coef = results.params[interaction_term]
            std_err = results.bse[interaction_term]
            p_val = results.pvalues[interaction_term]
            z_score = results.tvalues[interaction_term]
            is_sig = p_val < 0.05
            ci_lower = coef - 1.96 * std_err
            ci_upper = coef + 1.96 * std_err

            translation = translate_opensmile_feature(feature)

            print(f"   -> [Interaction effect] Coef: {coef:.3f}, P-value: {p_val:.3f} {'★' if is_sig else ''}")

            print('   ->      Task  Age  ...')
            p_val_base = results.pvalues['Age_Z']
            coef_base = results.params['Age_Z']
            is_sig_base = p_val_base < 0.05
            print(
                f"      [{cfg['base_task']}] Age effect Coef: {coef_base:.3f}, P-value: {p_val_base:.3f} {'★' if is_sig_base else ''}")

            try:
                hypothesis = "Age_Z + `Age_Z:Task_Code` = 0"
                contrast_age_effect_test = results.t_test(hypothesis)
                summary_df = contrast_age_effect_test.summary_frame()
                coef_contrast = summary_df['coef'].iloc[0]
                p_val_contrast = summary_df['P>|z|'].iloc[0]
                is_sig_contrast = p_val_contrast < 0.05
                p_val_contrast_str = "< 0.001" if p_val_contrast < 0.001 else f"{p_val_contrast:.3f}"
                print(
                    f"      [{cfg['contrast_task']}] Age effect (t_testmethod): Coef: {coef_contrast:.3f}, P-value: {p_val_contrast_str} {'★' if is_sig_contrast else ''}")

            except Exception as test_e:
                print(f"      [Warning] t_test methodFailed ('{test_e}')，switching to manual calculation...")
                try:
                    coef_contrast = results.params['Age_Z'] + results.params['Age_Z:Task_Code']
                    cov_matrix = results.cov_params()
                    var_age_z = cov_matrix.loc['Age_Z', 'Age_Z']
                    var_interaction = cov_matrix.loc['Age_Z:Task_Code', 'Age_Z:Task_Code']
                    cov_age_interaction = cov_matrix.loc['Age_Z', 'Age_Z:Task_Code']
                    se_contrast = np.sqrt(var_age_z + var_interaction + 2 * cov_age_interaction)
                    z_score_contrast = coef_contrast / se_contrast
                    from scipy.stats import norm

                    p_val_contrast = 2 * (1 - norm.cdf(np.abs(z_score_contrast)))
                    is_sig_contrast = p_val_contrast < 0.05
                    p_val_contrast_str = "< 0.001" if p_val_contrast < 0.001 else f"{p_val_contrast:.3f}"
                    print(
                        f"      [{cfg['contrast_task']}] Age effect (manual calculation): Coef: {coef_contrast:.3f}, P-value: {p_val_contrast_str} {'★' if is_sig_contrast else ''}")

                except Exception as manual_e:
                    print(f"      [Error] manual calculation {cfg['contrast_task']} Age effectalso failed: {manual_e}")
                    p_val_contrast = float('nan')
                    coef_contrast = float('nan')
                    is_sig_contrast = False

            results_list.append({
                'Feature': feature,
                'English_Description': translation,
                'Interaction_Coef': coef,
                'Z_Score': z_score,
                'HDI_2.5%': ci_lower,
                'HDI_97.5%': ci_upper,
                'Interaction_PValue': p_val,  # raw interaction p value
                'Significant_Interaction': is_sig,
                f'{cfg["base_task"]}_Age_Coef': coef_base,
                f'{cfg["base_task"]}_Age_PValue': p_val_base,
                f'{cfg["base_task"]}_Age_Significant': is_sig_base,
                f'{cfg["contrast_task"]}_Age_Coef': coef_contrast,
                f'{cfg["contrast_task"]}_Age_PValue': p_val_contrast,
                f'{cfg["contrast_task"]}_Age_Significant': is_sig_contrast,
                'model_results': results  # temporarily store the model for plotting
            })

        except Exception as e:
            print(f"   -> [Error] modeling failed: {e}")

    if results_list:
        res_df = pd.DataFrame(results_list)

        print('\n===      FDR-BH        ===')
        valid_p_mask = res_df['Interaction_PValue'].notna()
        res_df['Interaction_FDR_PValue'] = np.nan
        res_df['Significant_Interaction_FDR'] = False

        if valid_p_mask.any():
            reject, pvals_corrected, _, _ = multipletests(
                res_df.loc[valid_p_mask, 'Interaction_PValue'],
                alpha=0.05,
                method='fdr_bh'
            )
            res_df.loc[valid_p_mask, 'Interaction_FDR_PValue'] = pvals_corrected
            res_df.loc[valid_p_mask, 'Significant_Interaction_FDR'] = reject

        fdr_sig_df = res_df[res_df['Significant_Interaction_FDR'] == True]
        print(f"FDR-significant feature count: {len(fdr_sig_df)}")

        if len(fdr_sig_df) > 0:
            print(f"\n=== Plotting FDR-significant features ===")
            for idx, row in fdr_sig_df.iterrows():
                feat = row['Feature']
                mod_res = row['model_results']
                print(f"   -> Plotting: {feat} ...")
                try:
                    plot_fitted_lines(mod_res, df_model, feat, age_scaler, PLOT_DIR)
                    plot_interference_effect(mod_res, df_model, feat, age_scaler, PLOT_DIR)
                except Exception as plot_e:
                    print(f"      [Warning] plotting failed: {plot_e}")

        output_columns = [
            'Feature', 'English_Description',
            'Interaction_Coef', 'Z_Score', 'HDI_2.5%', 'HDI_97.5%',
            'Interaction_PValue', 'Significant_Interaction',  # raw p value and significance flag
            'Interaction_FDR_PValue', 'Significant_Interaction_FDR',  # FDR-corrected p value and significance flag
            f'{cfg["base_task"]}_Age_Coef', f'{cfg["base_task"]}_Age_PValue', f'{cfg["base_task"]}_Age_Significant',
            f'{cfg["contrast_task"]}_Age_Coef', f'{cfg["contrast_task"]}_Age_PValue',
            f'{cfg["contrast_task"]}_Age_Significant'
        ]

        res_df = res_df.sort_values(by='Interaction_FDR_PValue', ascending=True)

        final_columns = [col for col in output_columns if col in res_df.columns]
        res_df_final = res_df[final_columns]

        res_df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\nAnalysis complete. Formatted results with interaction p values and FDR-corrected results saved to: {OUTPUT_FILE}")

    else:
        print('No valid model results were produced.')





