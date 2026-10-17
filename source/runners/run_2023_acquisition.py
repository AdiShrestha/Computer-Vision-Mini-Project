"""Bounded full-2023 ROI acquisition."""
import argparse
from pathlib import Path
from sentinel_gl.ingestion.expansion2023 import Budget,run_optical,run_sar
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--modality',choices=['optical','sar'],required=True);p.add_argument('--workers',type=int,default=3);a=p.parse_args()
    b=Budget(a.root/'data/raw/expansion_2023/transfer_ledger.json')
    if a.modality=='optical':run_optical(a.root,b,a.workers)
    else:run_sar(a.root,b)
