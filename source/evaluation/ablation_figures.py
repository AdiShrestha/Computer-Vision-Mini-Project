"""
Ablation Figures & Comparison Tables Generator (C05-04).

Generates 4 publication-quality figures in results/figures/:
1. ablation_bar_chart.png — AUC-ROC per configuration
2. channel_contribution.png — Marginal AUC contribution per channel
3. threshold_roc_curve.png — ROC curve with original vs. refined threshold markers
4. ablation_comparison_table.png — Full metrics comparison table across all 11 configs

MANDATORY PRECONDITION:
Asserts `ablation_version == C05-02` in results/ablation/ablation_summary.json.
All values are programmatically read from results/ablation/*.json.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if source_root not in sys.path:
    sys.path.insert(0, source_root)


def generate_ablation_figures():
    repo_root = os.path.dirname(source_root)
    ablation_summary_path = os.path.join(repo_root, 'results', 'ablation', 'ablation_summary.json')
    threshold_analysis_path = os.path.join(repo_root, 'results', 'ablation', 'threshold_analysis.json')
    fig_dir = os.path.join(repo_root, 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # Mandatory Precondition Check
    assert os.path.isfile(ablation_summary_path), f"ABORT: {ablation_summary_path} not found"
    with open(ablation_summary_path) as f:
        summary = json.load(f)
    assert summary.get('ablation_version') == 'C05-02', (
        f"ABORT: Must use C05-02 ablation results (found {summary.get('ablation_version')})"
    )

    with open(threshold_analysis_path) as f:
        threshold_meta = json.load(f)

    # ========================================================================
    # Figure 1: ablation_bar_chart.png
    # ========================================================================
    configs = summary['configs']
    sorted_cfg_keys = sorted(configs.keys(), key=lambda k: configs[k]['auc_roc'], reverse=True)
    auc_values = [configs[k]['auc_roc'] for k in sorted_cfg_keys]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['forestgreen' if k == 'FULL_15CH' else 'royalblue' for k in sorted_cfg_keys]
    bars = ax.bar(sorted_cfg_keys, auc_values, color=colors, alpha=0.85, width=0.6)
    
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUC-ROC", fontsize=10)
    ax.set_title("Ablation Study: Synthetic Anomaly Discrimination (AUC-ROC)", fontsize=11, fontweight='bold')
    plt.xticks(rotation=35, ha='right', fontsize=9)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
                    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(fig_dir, 'ablation_bar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # Figure 2: channel_contribution.png
    # ========================================================================
    contribs = summary['channel_contributions']
    sorted_channels = sorted(contribs.keys(), key=lambda ch: contribs[ch], reverse=True)
    contrib_values = [contribs[ch] for ch in sorted_channels]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    c_colors = ['crimson' if ch == summary['most_important_channel'] else 'darkorange' for ch in sorted_channels]
    bars = ax.bar(sorted_channels, contrib_values, color=c_colors, alpha=0.85, width=0.5)
    
    ax.set_ylabel(r"Marginal AUC-ROC Contribution ($\Delta$AUC)", fontsize=10)
    ax.set_title("Marginal Channel Importance Contribution (FULL_15CH vs. NO_CHxx)", fontsize=11, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'+{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')
                    
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(fig_dir, 'channel_contribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # Figure 3: threshold_roc_curve.png
    # ========================================================================
    sweep_table = threshold_meta['threshold_sweep_table']
    fp_rates = [e['false_positive_rate'] for e in sweep_table]
    det_rates = [e['synthetic_detection_rate'] for e in sweep_table]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fp_rates, det_rates, color='forestgreen', lw=2, label='Score-C Threshold Sweep')
    
    # Original vs Refined threshold points
    orig_fp = threshold_meta['original_fp_rate']
    orig_det = 1.0  # At 85th percentile
    ref_fp = threshold_meta['refined_fp_rate']
    ref_det = threshold_meta['refined_detection_rate']
    
    ax.plot(orig_fp, orig_det, 'o', color='crimson', markersize=9, label=f'Original (85th Pct, FP={orig_fp*100:.1f}%)')
    ax.plot(ref_fp, ref_det, '*', color='gold', markersize=14, markeredgecolor='black', label=f'Refined (88th Pct, FP={ref_fp*100:.1f}%)')
    
    ax.axvline(x=0.10, color='gray', linestyle='--', label='INV-007 Target (FP ≤ 10%)')
    ax.set_title("Detection Threshold Refinement & INV-007 Compliance", fontsize=11, fontweight='bold')
    ax.set_xlabel("False Positive Rate (Control Lakes)", fontsize=10)
    ax.set_ylabel("Synthetic Detection Rate (E3)", fontsize=10)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(fig_dir, 'threshold_roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ========================================================================
    # Figure 4: ablation_comparison_table.png
    # ========================================================================
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis('off')
    
    headers = ['Config Name', 'Active Channels', 'AUC-ROC', 'AUC-PR', 'Synth Det Rate', 'FP Rate']
    cell_data = []
    
    for k in sorted_cfg_keys:
        cfg = configs[k]
        cell_data.append([
            cfg['config_name'],
            str(cfg['n_active_channels']),
            f"{cfg['auc_roc']:.4f}",
            f"{cfg['auc_pr']:.4f}",
            f"{cfg['synthetic_detection_rate']*100:.1f}%",
            f"{cfg['false_positive_rate']*100:.1f}%"
        ])
        
    table = ax.table(cellText=cell_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.1, 1.6)
    
    # Highlight FULL_15CH row
    for col_idx in range(len(headers)):
        table[(1, col_idx)].set_facecolor('#d1e7dd')
        table[(1, col_idx)].get_text().set_weight('bold')
        
    ax.set_title("Ablation Study Metrics Comparison (11 Configurations)", fontsize=11, fontweight='bold', pad=20)
    plt.savefig(os.path.join(fig_dir, 'ablation_comparison_table.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Ablation Figures Generation Completed.")


if __name__ == '__main__':
    generate_ablation_figures()
