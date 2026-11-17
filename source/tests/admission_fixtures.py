"""Analytical bytes for integrity controls; never provider observations.
# FABRICATION-DISCLOSURE: TEST-FIXTURE-ONLY
"""
from pathlib import Path
import json
import numpy as np
from sentinel_gl.integrity import digest, file_hash
from sentinel_gl.admission import AdmissionContext


def measurement_fixture(root, *, modality='optical', suffix='a', day='2023-01-02', fields=None, units=None):
    root = Path(root);root.mkdir(parents=True, exist_ok=True)
    ids = {k: suffix + '-' + k for k in ('metadata', 'product', 'query', 'response', 'roi', 'qa')}
    record = {'schema_version': 1, 'measurement_id': suffix, 'lake_id': 'ANALYTICAL-SITE',
              'status': 'MEASURED', 'origin': 'fixture', 'product': 'ANALYTICAL-' + modality,
              'version': '1', 'endpoint': 'fixture://analytical-grid', 'crs': 'LOCAL-ANALYTICAL-METRES',
              'support_start': day + 'T00:00:00Z', 'support_end': day + 'T12:00:00Z',
              'available_at': day + 'T13:00:00Z', 'availability_policy': 'retrieval-time',
              **{k + '_id': v for k, v in ids.items()}}
    defaults = {
        'optical': ({'B03': np.array([[.3,.0]]), 'B08': np.array([[.1,.0]]), 'B11': np.array([[.1,.0]]), 'water': np.array([[True,False]]), 'pixel_area_m2': np.array([[100.,100.]])}, {'B03':'1','B08':'1','B11':'1','pixel_area_m2':'m2'}),
        'sar': ({'VV':np.array([[-10.,-20.]]),'VH':np.array([[-20.,-30.]])}, {'VV':'dB','VH':'dB'}),
        'weather': ({'temperature':np.array([[273.15,275.15]]),'precipitation':np.array([[.001,.003]])}, {'temperature':'K','precipitation':'m'}),
    }
    default_fields, default_units = defaults[modality]
    record['units'] = default_units if units is None else units
    fields = default_fields if fields is None else fields
    payloads = {
        'metadata': {k: record[k] for k in ('product','version','endpoint','units','crs','support_start','support_end','available_at','availability_policy')},
        'query': {'endpoint':record['endpoint'],'roi_id':ids['roi'],'product':record['product']},
        'response': {'query_id':ids['query'],'product_id':ids['product'],'status':'SUCCESS'},
        'roi': {'type':'Polygon','coordinates':[[[0,0],[2,0],[2,1],[0,1],[0,0]]],'crs':record['crs']},
        'qa': {'status':'ACCEPTED','policy':'analytical-mask-v1'},
    }
    artifacts={}
    for k,v in ids.items():
        path=root/(v + ('.npz' if k=='product' else '.json'))
        if k == 'product':
            np.savez(path, valid=np.array([[True,True]]), roi=np.array([[True,True]]), **fields)
        else:path.write_text(json.dumps(payloads[k],allow_nan=False))
        artifacts[v]={'artifact_id':v,'path':path.name,'sha256':file_hash(path),'origin':'fixture','parents':[ids[x] for x in ('metadata','query','response','roi','qa')] if k=='product' else []}
    manifest={'schema_version':1,'artifacts':artifacts,'measurements':{suffix:digest(record)}}
    return AdmissionContext(root,manifest,mode='fixture'),record
