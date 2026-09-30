from pathlib import Path
import json,hashlib
import argparse
a=argparse.ArgumentParser();a.add_argument("legacy_root",type=Path);a.add_argument("output",type=Path);args=a.parse_args()
old=args.legacy_root.resolve();out=args.output;out.mkdir(parents=True,exist_ok=True)
selections={
'source/data/preprocessing/preprocess_optical.py':[(1,95)],
'source/data/channels/extract_sar.py':[(1,160)],
'source/data/insar/insar_feasibility.py':[(1,130)],
'source/scripts/run_ablation.py':[(54,155)],
'source/scripts/run_bootstrap_ci.py':[(52,73),(96,145)],
'source/scripts/cloud_stratified_eval.py':[(84,128)],
'source/evaluation/figures.py':[(55,68),(127,142)],
'source/models/anomaly/score_b.py':[(1,140)],
'source/models/anomaly/score_c.py':[(1,175)],
'source/scripts/train_ts_mae.py':[(120,230)],
}
text='# Legacy evidence excerpts\n\nThese excerpts are quarantined forensic evidence, not active code or scientific results. Line numbers refer to the SHA-256 identified working-tree file recorded on 2026-09-30. Complete per-file inventory and review index accompany this document. The old Git tags preserve full original source.\n'
for name,ranges in selections.items():
 p=old/name;raw=p.read_bytes();lines=raw.decode().splitlines();text+='\n## '+name+'\n\nSHA-256: `'+hashlib.sha256(raw).hexdigest()+'`\n'
 for lo,hi in ranges:
  text+='\n```text\n'+'\n'.join(f'{i}: {lines[i-1]}'.rstrip() for i in range(lo,min(hi,len(lines))+1))+'\n```\n'
(out/'legacy_evidence_excerpts.md').write_text(text)
