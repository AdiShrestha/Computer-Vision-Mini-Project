#!/usr/bin/env python3
"""Extract analytical fixture bytes into a new, explicitly nonresearch directory."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sentinel_gl.admission import AdmissionContext
from sentinel_gl.features import FeatureExtractionPipeline
from sentinel_gl.integrity import loads_strict, file_hash, verified_file
import numpy as np


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--records',required=True)
    parser.add_argument('--records-sha256',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--lake-id',required=True)
    parser.add_argument('--start-date',required=True)
    parser.add_argument('--window-days',type=int,required=True)
    args=parser.parse_args(argv)
    if 'fixtures' not in args.output_dir.parts:
        raise ValueError('fixture output must have a fixtures namespace')
    if args.output_dir.exists() or any(p.is_symlink() for p in (args.output_dir,*args.output_dir.parents)):
        raise ValueError('fixture output must be new and not a symlink')
    manifest=loads_strict(verified_file(args.input_root,args.manifest,args.manifest_sha256).read_text())
    records=loads_strict(verified_file(args.input_root,args.records,args.records_sha256).read_text())
    if set(records) != {'optical','sar','era5'} or any(type(v) is not list for v in records.values()):
        raise ValueError('explicit fixture modality lists required')
    ctx=AdmissionContext(args.input_root,manifest,mode='fixture')
    panel=FeatureExtractionPipeline(args.window_days,admission=ctx).extract_panel(args.lake_id,'analytical-fixture',args.start_date,
        optical_observations=records['optical'],sar_observations=records['sar'],era5_observations=records['era5'])
    args.output_dir.mkdir(parents=True,exist_ok=False)
    np.savez(args.output_dir/'panel.npz',values=panel.values,mask=panel.mask)
    summary={**panel.to_dict(),'research_admissible':False,'artifact_sha256':file_hash(args.output_dir/'panel.npz')}
    with (args.output_dir/'summary.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
    print(json.dumps({'origin':'fixture','research_admissible':False,'observed_count':int(panel.mask.sum())}))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
