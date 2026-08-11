"""
Reality Gate — Three-State Verification Engine.

Checks that assembled feature matrices represent real satellite observations with expected HKH properties,
blocking encoder training on FAIL.

States: PASS, WARNING, FAIL
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_feature_data(feature_dir, registry):
    """Load all feature matrices and compute statistics."""
    stats = {}
    for lake in registry['lakes']:
        lake_id = lake['id']
        npz_path = feature_dir / lake_id / 'feature_matrix.npz'
        if not npz_path.exists():
            stats[lake_id] = {'missing': True}
            continue
        data = np.load(npz_path, allow_pickle=True)
        features = data['features']
        dates = data['dates']
        T, C = features.shape

        # Per-channel gap rates
        channel_gaps = np.isnan(features).mean(axis=0)  # (C,)
        overall_gap = np.isnan(features).mean()

        # Per-channel std
        with np.errstate(all='ignore'):
            channel_stds = np.nanstd(features, axis=0)

        # Monsoon gap rate (June-September, optical channels 0..4)
        monsoon_mask = np.array([
            d[5:7] in ('06', '07', '08', '09')
            for d in dates
        ])
        optical_channels = [0, 1, 2, 3, 4]  # CH-01, CH-02a-d
        if monsoon_mask.any():
            monsoon_optical_gaps = float(np.isnan(features[monsoon_mask][:, optical_channels]).mean())
        else:
            monsoon_optical_gaps = 0.0

        # Temporal coverage: fraction of dates with at least one active observation (not all NaN)
        has_observation = (~np.isnan(features)).any(axis=1)
        temporal_coverage = float(has_observation.mean())

        stats[lake_id] = {
            'missing': False,
            'shape': list(features.shape),
            'channel_gaps': channel_gaps.tolist(),
            'overall_gap': float(overall_gap),
            'channel_stds': channel_stds.tolist(),
            'monsoon_optical_gap_rate': float(monsoon_optical_gaps),
            'role': lake['role'],
            'temporal_coverage': temporal_coverage
        }
    return stats


def run_gate(stats, registry):
    """Run all 5 Reality Gate checks."""
    checks = {}
    valid_stats = [s for s in stats.values() if not s.get('missing')]
    n_lakes = len(valid_stats)

    # CHECK 1: Gap statistics
    all_gap_rates = [s['overall_gap'] for s in valid_stats]
    if all(g < 0.02 for g in all_gap_rates):
        checks['gap_statistics'] = {
            'verdict': 'FAIL',
            'reason': f"All {n_lakes} lakes have <2% gaps. Physically impossible for real HKH data.",
            'data': {'min_gap': min(all_gap_rates), 'max_gap': max(all_gap_rates)}
        }
    elif any(g < 0.02 for g in all_gap_rates):
        low_gap_lakes = [lid for lid, s in stats.items() if not s.get('missing') and s['overall_gap'] < 0.02]
        checks['gap_statistics'] = {
            'verdict': 'WARNING',
            'reason': f"{len(low_gap_lakes)} lakes have <2% gaps: {low_gap_lakes}.",
            'data': {'low_gap_lakes': low_gap_lakes}
        }
    else:
        checks['gap_statistics'] = {
            'verdict': 'PASS',
            'reason': f"Gap rates range from {min(all_gap_rates):.3f} to {max(all_gap_rates):.3f}.",
            'data': {'min_gap': min(all_gap_rates), 'max_gap': max(all_gap_rates)}
        }

    # CHECK 2: Distribution variance
    all_stds = np.array([s['channel_stds'] for s in valid_stats])
    per_channel_std_ratio = all_stds.max(axis=0) / (all_stds.min(axis=0) + 1e-10)
    channels_with_high_ratio = int(np.sum(per_channel_std_ratio > 2.0))
    n_channels = all_stds.shape[1]
    if channels_with_high_ratio == 0:
        checks['distribution_variance'] = {
            'verdict': 'FAIL',
            'reason': "No channel has >2x std variation across lakes. Data is suspiciously uniform.",
            'data': {'std_ratios': per_channel_std_ratio.tolist()}
        }
    elif channels_with_high_ratio < n_channels // 2:
        checks['distribution_variance'] = {
            'verdict': 'WARNING',
            'reason': f"Only {channels_with_high_ratio}/{n_channels} channels show >2x std variation.",
            'data': {'std_ratios': per_channel_std_ratio.tolist()}
        }
    else:
        checks['distribution_variance'] = {
            'verdict': 'PASS',
            'reason': f"{channels_with_high_ratio}/{n_channels} channels show >2x std variation across lakes.",
            'data': {'std_ratios': per_channel_std_ratio.tolist()}
        }

    # CHECK 3: Temporal coverage
    coverages = [s['temporal_coverage'] for s in valid_stats]
    lakes_above_80 = sum(1 for c in coverages if c >= 0.80)
    if lakes_above_80 < 13:
        checks['temporal_coverage'] = {
            'verdict': 'FAIL',
            'reason': f"Only {lakes_above_80}/20 lakes have >=80% temporal coverage.",
            'data': {'lakes_above_80': lakes_above_80}
        }
    elif lakes_above_80 < 17:
        checks['temporal_coverage'] = {
            'verdict': 'WARNING',
            'reason': f"{lakes_above_80}/20 lakes have >=80% coverage (expected >=17).",
            'data': {'lakes_above_80': lakes_above_80}
        }
    else:
        checks['temporal_coverage'] = {
            'verdict': 'PASS',
            'reason': f"{lakes_above_80}/20 lakes have >=80% temporal coverage.",
            'data': {'lakes_above_80': lakes_above_80}
        }

    # CHECK 4: Sensor coverage (13 channels)
    lakes_with_all_channels = 0
    for s in valid_stats:
        # A channel is present if it has >=5% non-NaN data (g < 0.95) over the 8-year timeline
        present = sum(1 for g in s['channel_gaps'] if g < 0.95)
        if present >= 13:
            lakes_with_all_channels += 1
    if lakes_with_all_channels < 14:
        checks['sensor_coverage'] = {
            'verdict': 'FAIL',
            'reason': f"Only {lakes_with_all_channels}/20 lakes have all 13 channels present.",
            'data': {'lakes_with_all_channels': lakes_with_all_channels}
        }
    elif lakes_with_all_channels < 17:
        checks['sensor_coverage'] = {
            'verdict': 'WARNING',
            'reason': f"{lakes_with_all_channels}/20 lakes have all 13 channels.",
            'data': {'lakes_with_all_channels': lakes_with_all_channels}
        }
    else:
        checks['sensor_coverage'] = {
            'verdict': 'PASS',
            'reason': f"{lakes_with_all_channels}/20 lakes have all 13 channels present.",
            'data': {'lakes_with_all_channels': lakes_with_all_channels}
        }

    # CHECK 5: Cloud contamination presence
    monsoon_gaps = [s['monsoon_optical_gap_rate'] for s in valid_stats]
    lakes_with_monsoon_gaps = sum(1 for g in monsoon_gaps if g > 0.15)
    frac_with_gaps = lakes_with_monsoon_gaps / len(monsoon_gaps) if monsoon_gaps else 0.0
    if frac_with_gaps < 0.40:
        if all(g < 0.02 for g in monsoon_gaps):
            checks['cloud_contamination'] = {
                'verdict': 'FAIL',
                'reason': "Zero monsoon optical gaps across all lakes. Impossible for real HKH data.",
                'data': {'lakes_with_monsoon_gaps': lakes_with_monsoon_gaps}
            }
        else:
            checks['cloud_contamination'] = {
                'verdict': 'WARNING',
                'reason': f"Only {frac_with_gaps*100:.0f}% of lakes have >15% monsoon gaps (expected >=60%).",
                'data': {'frac_with_gaps': frac_with_gaps}
            }
    else:
        checks['cloud_contamination'] = {
            'verdict': 'PASS',
            'reason': f"{frac_with_gaps*100:.0f}% of lakes have >15% monsoon optical gaps.",
            'data': {'lakes_with_monsoon_gaps': lakes_with_monsoon_gaps}
        }

    # Overall verdict
    verdicts = [c['verdict'] for c in checks.values()]
    if 'FAIL' in verdicts:
        overall = 'FAIL'
        action = 'STOP — do NOT proceed to encoder training. Diagnose the failed checks.'
    elif 'WARNING' in verdicts:
        overall = 'WARNING'
        action = 'Human review recommended for WARNING checks before proceeding.'
    else:
        overall = 'PASS'
        action = 'Proceed to encoder training (C07-05).'

    return {
        'timestamp': datetime.now().isoformat(),
        'n_lakes': n_lakes,
        'n_channels': 13,
        'checks': checks,
        'overall_verdict': overall,
        'action': action,
        'per_lake_stats': stats
    }


def write_report(gate_result, report_path):
    """Write human-readable Reality Gate report."""
    lines = [
        f"# Reality Gate Report — {gate_result['timestamp'][:10]}",
        "",
        f"**Overall Verdict: {gate_result['overall_verdict']}**",
        f"**Action: {gate_result['action']}**",
        "",
        f"Lakes processed: {gate_result['n_lakes']}",
        f"Channels: {gate_result['n_channels']}",
        "",
    ]
    for check_name, check in gate_result['checks'].items():
        lines.append(f"## {check_name.upper().replace('_', ' ')}")
        lines.append(f"**Verdict: {check['verdict']}**")
        lines.append(f"Reason: {check['reason']}")
        lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    reg_path = PROJECT_ROOT / 'source' / 'data' / 'registry' / 'lake_registry.json'
    with open(reg_path, 'r', encoding='utf-8') as f:
        registry = json.load(f)

    feature_dir = PROJECT_ROOT / 'data' / 'features_real'
    result_dir = PROJECT_ROOT / 'results' / 'reality_gate'
    result_dir.mkdir(parents=True, exist_ok=True)

    stats = load_feature_data(feature_dir, registry)
    gate_result = run_gate(stats, registry)

    with open(result_dir / 'reality_gate_data.json', 'w', encoding='utf-8') as f:
        json.dump(gate_result, f, indent=2, default=str)

    write_report(gate_result, result_dir / 'reality_gate_report.md')

    print(f"\n{'='*60}")
    print(f"REALITY GATE OVERALL: {gate_result['overall_verdict']}")
    print(f"ACTION: {gate_result['action']}")
    for name, check in gate_result['checks'].items():
        status_icon = '✅' if check['verdict'] == 'PASS' else '⚠️' if check['verdict'] == 'WARNING' else '❌'
        print(f"  {status_icon} {name}: {check['verdict']}")
    print(f"{'='*60}")

    if gate_result['overall_verdict'] == 'FAIL':
        sys.exit(1)


if __name__ == '__main__':
    main()
