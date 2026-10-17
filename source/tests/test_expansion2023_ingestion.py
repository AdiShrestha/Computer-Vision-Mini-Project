"""Constructed fixtures only; no provider observations or live requests."""
from pathlib import Path
import json
import numpy as np
import pytest
from sentinel_gl.ingestion import expansion2023 as x

def test_exact_http_range_coalescing_cache_and_seek(monkeypatch,tmp_path):
    payload=bytes(range(256))*1024;calls=[]
    def fetch(url,budget,**kwargs):
        start,end=map(int,kwargs['headers']['Range'][6:].split('-'));calls.append((start,end))
        data=payload[start:end+1]
        return data,{'http_status':206,'content_range':f'bytes {start}-{end}/{len(payload)}','received_bytes':len(data)}
    monkeypatch.setattr(x,'fetch',fetch)
    store=x.RangeStore('https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/constructed.tif',len(payload),None)
    assert store.read(123,200000)==payload[123:200123]
    assert len(calls)==1
    file=x.RangeFile(store);file.seek(70000);assert file.read(1000)==payload[70000:71000]
    file.seek(-100,2);assert file.read(100)==payload[-100:]
    assert len(calls)==1

def test_wrong_range_status_is_rejected(monkeypatch):
    def fetch(*a,**k):return b'a'*65536,{'http_status':200,'content_range':None,'received_bytes':65536}
    monkeypatch.setattr(x,'fetch',fetch)
    store=x.RangeStore('https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/constructed.tif',65536,None)
    with pytest.raises(ValueError,match='range status'):store.read(0,10)

def test_two_budget_instances_do_not_overwrite_prior_receipts(tmp_path):
    p=tmp_path/'ledger.json';one=x.Budget(p,100);two=x.Budget(p,100)
    one.add({'received_bytes':7,'sha256':'fixture-1'})
    two.add({'received_bytes':9,'sha256':'fixture-2'})
    result=json.loads(p.read_text());assert result['received_bytes']==16 and len(result['requests'])==2

@pytest.mark.parametrize('url',['http://catalogue.dataspace.copernicus.eu/x',
    'https://not-approved.example/x','https://secret@download.dataspace.copernicus.eu/x'])
def test_endpoint_allowlist_remains_exact(url):
    with pytest.raises(ValueError,match='allowlist'):x.check_url(url)

@pytest.mark.parametrize('baseline',['05.09','05.10','05.11'])
def test_native_boa_baselines_are_explicitly_joined(monkeypatch,tmp_path,baseline):
    product=f'''<root><PRODUCT_URI>CONSTRUCTED.SAFE</PRODUCT_URI><PROCESSING_BASELINE>{baseline}</PROCESSING_BASELINE>
    <BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE><Spectral_Information physicalBand="B3" bandId="2"/>
    <Spectral_Information physicalBand="B8" bandId="7"/><Spectral_Information physicalBand="B11" bandId="11"/>
    <BOA_ADD_OFFSET band_id="2">-1000</BOA_ADD_OFFSET><BOA_ADD_OFFSET band_id="7">-1000</BOA_ADD_OFFSET>
    <BOA_ADD_OFFSET band_id="11">-1000</BOA_ADD_OFFSET></root>'''.encode()
    granule=b'<root><SENSING_TIME>2023-01-01T00:00:00Z</SENSING_TIME></root>'
    def fetch(url,*a,**k):return (product if url.endswith('/p') else granule),{'received_bytes':1}
    monkeypatch.setattr(x,'fetch',fetch)
    item={'properties':{'s2:processing_baseline':baseline,'s2:product_uri':'CONSTRUCTED.SAFE'},
        'assets':{'product_metadata':{'href':'https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/p'},
                  'granule_metadata':{'href':'https://e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com/g'}}}
    result=x.optical_metadata(item,tmp_path,None)
    assert result['baseline']==baseline
    for values in result['radiometry'].values():assert values=={'scale':.0001,'offset':-.1}

def test_native_utc_preserves_subsecond_precision():
    assert x._utc_native('2023-09-01T12:14:08.715352')=='2023-09-01T12:14:08.715352Z'

def test_observation_merge_preserves_other_sensor(tmp_path):
    def row(i,sensor):return {'observation_id':i,'site_id':'CONSTRUCTED','acquired_at':'2023-01-01T00:00:00Z','sensor':sensor}
    x.save_observations(tmp_path,[row('s1','SENTINEL_1_IW_GRD')])
    x.save_observations(tmp_path,[row('s2','SENTINEL_2_OPTICAL')])
    out=json.loads((tmp_path/'data/derived/expansion_2023/observations.v2.json').read_text())
    assert {r['observation_id'] for r in out['observations']}=={'s1','s2'}

def test_response_cap_abort_preserves_actual_partial_transfer(monkeypatch,tmp_path):
    class Response:
        status_code=206;headers={};closed=False
        def iter_content(self,n):yield b'1234';yield b'5678'
        def close(self):self.closed=True
    response=Response()
    monkeypatch.setattr(x.requests,'request',lambda *a,**k:response)
    ledger=tmp_path/'ledger.json';budget=x.Budget(ledger,100)
    with pytest.raises(ValueError,match='Bounded response'):
        x.fetch('https://earth-search.aws.element84.com/constructed',budget,limit=6)
    row=json.loads(ledger.read_text())['requests'][0]
    assert row['received_bytes']==8 and row['body_status']=='PARTIAL_ABORTED'
    assert row['sha256']==x.hashlib.sha256(b'12345678').hexdigest() and response.closed

def test_actual_over_budget_receipt_is_not_silently_discarded(tmp_path):
    ledger=tmp_path/'ledger.json';budget=x.Budget(ledger,5)
    with pytest.raises(ValueError,match='receipt retained'):budget.add({'received_bytes':6})
    assert json.loads(ledger.read_text())['received_bytes']==6

def test_resume_complete_receipt_checks_hash_without_network(monkeypatch,tmp_path):
    site='PILOT-SOUTH-LHONAK';oid=site+'__CONSTRUCTED'
    folder=tmp_path/'data/raw/expansion_2023/optical'/oid;folder.mkdir(parents=True)
    target=tmp_path/'fixture.npz';target.write_bytes(b'explicit constructed fixture')
    receipt={'observation_id':oid,'site_id':site,'product_id':'CONSTRUCTED.SAFE',
        'values_file':'fixture.npz','sha256':x.sha(target),'acquired_at':'2023-01-01T00:00:00Z',
        'grid':{},'qa_clear_count':0,'valid_per_channel':[0]*5}
    x.write(folder/'receipt.json',receipt)
    monkeypatch.setattr(x,'fetch',lambda *a,**k:pytest.fail('resume must not request provider bytes'))
    assert x.optical_observation(tmp_path,site,{'id':'CONSTRUCTED'},None,0)['sha256']==x.sha(target)
    target.write_bytes(b'altered')
    with pytest.raises(ValueError,match='receipt/hash mismatch'):
        x.optical_observation(tmp_path,site,{'id':'CONSTRUCTED'},None,0)
