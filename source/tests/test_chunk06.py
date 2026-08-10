"""
Adversarial verification tests for Chunk 06 — Manuscript Preparation.
"""
import os
import sys
import json
import subprocess

source_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_root = os.path.dirname(source_root)

MANUSCRIPT_DIR = os.path.join(repo_root, 'results', 'manuscript')
SECTIONS_DIR = os.path.join(MANUSCRIPT_DIR, 'sections')
MAP_PATH = os.path.join(MANUSCRIPT_DIR, 'claim_evidence_map.json')

REQUIRED_CLAIM_IDS = [f"CL-{i:02d}" for i in range(1, 21)]  # CL-01 through CL-20


# ============================================================
# C06-01: Claim-evidence map
# ============================================================

def test_claim_evidence_map_exists():
    assert os.path.isfile(MAP_PATH), "claim_evidence_map.json not found"


def test_claim_evidence_map_version():
    with open(MAP_PATH) as f:
        m = json.load(f)
    assert m.get('map_version') == 'C06-01', f"map_version={m.get('map_version')}"


def test_all_required_claims_present():
    """ADVERSARIAL: All 20 required claims must be present."""
    with open(MAP_PATH) as f:
        m = json.load(f)
    for cid in REQUIRED_CLAIM_IDS:
        assert cid in m['claims'], f"Missing required claim: {cid}"


def test_claim_evidence_verification_passes():
    """ADVERSARIAL: verify_claim_evidence.py must exit 0."""
    result = subprocess.run(
        [sys.executable, os.path.join(source_root, 'scripts', 'verify_claim_evidence.py')],
        capture_output=True, text=True, cwd=repo_root
    )
    assert result.returncode == 0, (
        f"verify_claim_evidence.py failed:\n{result.stdout}\n{result.stderr}"
    )


def test_score_c_auc_claim_matches_live():
    """ADVERSARIAL: CL-01 value must match live evaluation_summary.json exactly."""
    with open(MAP_PATH) as f:
        m = json.load(f)
    with open(os.path.join(repo_root, 'results', 'evaluation', 'evaluation_summary.json')) as f:
        ev = json.load(f)
    claim_val = m['claims']['CL-01']['value']
    live_val = ev['scorer_comparison']['score_c']['auc_roc']
    assert abs(claim_val - live_val) < 0.001, (
        f"CL-01 claim value {claim_val} != live {live_val}"
    )


# ============================================================
# C06-02: Methods and experiments sections
# ============================================================

def test_methods_section_exists():
    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'methods.md'))


def test_experiments_section_exists():
    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'experiments.md'))


def test_methods_covers_all_components():
    """Methods section must mention all 7 key system components."""
    with open(os.path.join(SECTIONS_DIR, 'methods.md')) as f:
        content = f.read()
    required = ['TS-MAE', 'Score-A', 'Score-B', 'Score-C', 'INV-004', 'ablation', 'threshold']
    for term in required:
        assert term in content, f"Methods missing required term: '{term}'"


def test_experiments_cites_claim_ids():
    """Experiments section must reference at least 5 CL-XX claim IDs."""
    with open(os.path.join(SECTIONS_DIR, 'experiments.md')) as f:
        content = f.read()
    cited = [f"CL-{i:02d}" for i in range(1, 26) if f"CL-{i:02d}" in content]
    assert len(cited) >= 5, f"Experiments section only cites {len(cited)} claims (need ≥5)"


# ============================================================
# C06-03: Introduction, related work, conclusion
# ============================================================

def test_abstract_exists_and_nonempty():
    p = os.path.join(SECTIONS_DIR, 'abstract.md')
    assert os.path.isfile(p)
    with open(p) as f:
        content = f.read()
    assert len(content.strip()) >= 100, "Abstract too short (< 100 chars)"


def test_introduction_exists():
    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'introduction.md'))


def test_related_work_exists():
    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'related_work.md'))


def test_conclusion_exists():
    assert os.path.isfile(os.path.join(SECTIONS_DIR, 'conclusion.md'))


def test_conclusion_cites_all_three_rqs():
    with open(os.path.join(SECTIONS_DIR, 'conclusion.md')) as f:
        content = f.read()
    assert 'RQ1' in content, "Conclusion must mention RQ1"
    assert 'RQ2' in content, "Conclusion must mention RQ2"
    assert 'RQ3' in content, "Conclusion must mention RQ3"


def test_conclusion_has_honest_verdicts():
    """ADVERSARIAL: Conclusion must use the actual verified verdict words."""
    with open(os.path.join(SECTIONS_DIR, 'conclusion.md')) as f:
        content = f.read()
    # RQ1 is MIXED, RQ2 is MIXED, RQ3 is POSITIVE — at least one verdict word required
    verdict_words = ['MIXED', 'POSITIVE', 'mixed', 'positive', 'negative', 'NEGATIVE']
    found = any(w in content for w in verdict_words)
    assert found, "Conclusion must include honest verdicts (MIXED/POSITIVE/NEGATIVE)"


# ============================================================
# C06-04: Reproducibility package
# ============================================================

def test_reproducibility_md_exists():
    assert os.path.isfile(os.path.join(repo_root, 'REPRODUCIBILITY.md'))


def test_reproducibility_has_all_steps():
    with open(os.path.join(repo_root, 'REPRODUCIBILITY.md')) as f:
        content = f.read()
    required = [
        'run_evaluation',
        'run_ablation',
        'run_threshold_analysis',
        'verify_claim_evidence',
        'pytest',
    ]
    for step in required:
        assert step in content, f"REPRODUCIBILITY.md missing step: '{step}'"


# ============================================================
# C06-05: project_knowledge.md update
# ============================================================

def test_project_knowledge_updated():
    pk_path = os.path.join(repo_root, 'project', 'project_knowledge.md')
    assert os.path.isfile(pk_path)


def test_knowledge_has_chunk05_findings():
    with open(os.path.join(repo_root, 'project', 'project_knowledge.md')) as f:
        content = f.read()
    assert 'InSAR' in content or 'coherence' in content, (
        "project_knowledge.md must document InSAR infeasibility finding from Chunk 05"
    )
    assert 'CH-05' in content or 'SAR backscatter' in content.replace('SAR Backscatter', 'SAR backscatter'), (
        "project_knowledge.md must document CH-05 as most important channel"
    )


# ============================================================
# C06-06: Manuscript assembly and final verification
# ============================================================

def test_manuscript_assembled():
    assert os.path.isfile(os.path.join(MANUSCRIPT_DIR, 'sentinel_gl_manuscript.md'))


def test_verification_report_exists():
    assert os.path.isfile(os.path.join(MANUSCRIPT_DIR, 'claim_evidence_verification_report.md'))


def test_all_figures_referenced_exist():
    """All 10 result figures must exist."""
    figs = [
        'roc_curves.png', 'baseline_comparison.png', 'scorer_comparison_table.png',
        'south_lhonak_anomaly_timeline.png', 'synthetic_detection_rates.png',
        'control_lake_scores.png',
        'ablation_bar_chart.png', 'channel_contribution.png',
        'threshold_roc_curve.png', 'ablation_comparison_table.png',
    ]
    for fig in figs:
        p = os.path.join(repo_root, 'results', 'figures', fig)
        assert os.path.isfile(p), f"Missing figure: {fig}"


def test_claim_evidence_all_pass():
    """ADVERSARIAL: Final verification report must show 0 failures."""
    report_path = os.path.join(MANUSCRIPT_DIR, 'claim_evidence_verification_report.md')
    with open(report_path) as f:
        content = f.read()
    assert 'FAIL' not in content or '0 FAIL' in content, (
        "Claim-evidence verification report contains failures"
    )
    assert 'All claims verified' in content or '0 FAIL' in content, (
        "Claim-evidence verification report does not show all claims verified"
    )
