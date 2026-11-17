"""Constructed fixture-only full fit/inference regression; no research metrics."""
import json
from pathlib import Path
import numpy as np
import pytest
from sentinel_gl.panel_v2 import load_panel_v2
from sentinel_gl.training_v2 import run_training_v2,TrainingConfigV2
from sentinel_gl.inference_v2 import run_inference_v2
from test_v2_pipeline_engineering import _fixture_panel,_sha

@pytest.mark.parametrize('single_quadrant',[False,True])
def test_first_missing_or_isolated_scene_survives_full_fit_and_inference(tmp_path,single_quadrant):
    panel,_=_fixture_panel(tmp_path)
    root=panel.root;manifest=panel.manifest_path;doc=json.loads(manifest.read_text())
    first_opt=min((r for r in doc['observations'] if r['sensor']=='SENTINEL_2_OPTICAL'),key=lambda r:r['acquired_at'])
    # Remove simultaneous radar context so the first optical fold really has none.
    doc['observations']=[r for r in doc['observations'] if not (
        r['sensor']=='SENTINEL_1_IW_GRD' and r['acquired_at']==first_opt['acquired_at'])]
    row=next(r for r in doc['observations'] if r['observation_id']==first_opt['observation_id'])
    path=root/row['values_file']
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    keep=np.zeros((499,499),dtype=bool)
    if single_quadrant:keep[:249,:249]=True
    arrays['observed_valid'] &= keep[None,:,:]
    arrays['values'][~arrays['observed_valid']]=np.nan
    arrays['qa_clear'] &= keep
    arrays['scl_classes'][~keep]=0
    np.savez_compressed(path,**arrays)
    row['sha256']=_sha(path);receipt_path=root/row['receipt_file'];receipt=json.loads(receipt_path.read_text());receipt['sha256']=row['sha256']
    receipt_path.write_text(json.dumps(receipt));row['receipt_sha256']=_sha(receipt_path)
    manifest.write_text(json.dumps(doc));panel=load_panel_v2(root,manifest,mode='fixture')
    config=TrainingConfigV2(d_model=8,n_heads=2,encoder_layers=1,decoder_layers=1,
        encoder_ff=16,decoder_ff=16,warmup_epochs=1,minimum_epochs=1,earliest_checkpoint_epoch=1,max_epochs=2,patience=1)
    output=root/'CONSTRUCTED_ENGINEERING_RUN'
    report=run_training_v2(panel,output/'training',config=config)
    assert any(r['observation_id']==row['observation_id'] for r in report['unavailable_reconstruction_folds'])
    run_inference_v2(panel,output/'training',output/'inference')
    scores=[json.loads(x) for x in (output/'inference/scores.v2.jsonl').read_text().splitlines()]
    first=[r for r in scores if r['observation_id']==row['observation_id'] and r['method']=='tmae_score_a']
    assert len(first)==5
    assert all(r['score'] is None and r['status']=='UNAVAILABLE_NO_VISIBLE_CONTEXT_OR_TARGET_SUPPORT' for r in first)
    assert any(r['method']=='tmae_score_a' and r['score'] is not None for r in scores)
