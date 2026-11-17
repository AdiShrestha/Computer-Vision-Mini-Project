#!/usr/bin/env python3
"""Bind native XML request receipts into a pre-analysis SR-1 manifest epoch."""
from __future__ import annotations
import argparse, hashlib, json, shutil, re
from datetime import datetime, timezone
from pathlib import Path
from defusedxml import ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
def one(root,name):
    found={n.text.strip() for n in root.iter() if n.tag.rsplit('}',1)[-1]==name and n.text and n.text.strip()}
    if len(found)!=1:raise ValueError(f'expected one native XML {name}, found {len(found)}')
    return next(iter(found))
def same_instant(a,b):
    def parse(value):
        result=datetime.fromisoformat(str(value).replace('Z','+00:00'))
        if result.tzinfo is None:raise ValueError('timestamp has no timezone')
        return result.astimezone(timezone.utc)
    return parse(a)==parse(b)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--attempt',required=True);ap.add_argument('--epoch',default='002')
    ap.add_argument('--rebind-from-preserved',action='store_true',
        help='rebuild from the immutable pre-enrichment manifest after a failed receipt-binding epoch')
    a=ap.parse_args()
    attempt=ROOT/'data/raw/sr1_multiyear'/a.attempt;manifest_path=ROOT/'data/derived/sr1_multiyear'/a.attempt/'observations.sr1.json'
    manifest=json.loads(manifest_path.read_text());ledger=json.loads((attempt/'transfer_ledger.json').read_text())
    preserved=manifest_path.with_name('observations.sr1.pre_native_receipt_enrichment.json')
    if preserved.exists():
        if sha(preserved)!=sha(manifest_path):
            if not a.rebind_from_preserved:
                raise SystemExit(f'refusing to overwrite preserved original manifest {preserved}')
            manifest=json.loads(preserved.read_text())
    else:shutil.copy2(manifest_path,preserved)
    by_url={}
    for request in ledger['requests']:
        by_url.setdefault(request['url'],[]).append(request)
    enriched=0;retained_2023=0
    for row in manifest['observations']:
        if row.get('provider')=='retained verified 2023 C1 ROI':
            parent=row['parent_manifest_observation']
            parent_receipt_path=ROOT/parent['receipt_file']
            if sha(parent_receipt_path)!=row['parent_receipt_sha256']:
                raise ValueError(f'2023 parent receipt SHA mismatch for {row["item_id"]}')
            original=json.loads(parent_receipt_path.read_text());native=original['native']
            metadata_receipts=native['receipts'];metadata_roots={}
            for role in ('product_metadata','granule_metadata'):
                r=metadata_receipts[role];xml_path=ROOT/r['path']
                if (sha(xml_path)!=r['sha256'] or r.get('http_status')!=200
                        or r.get('received_bytes')!=xml_path.stat().st_size):
                    raise ValueError(f'2023 parent native XML receipt invalid for {row["item_id"]}/{role}')
                metadata_roots[role]=ET.parse(str(xml_path)).getroot()
            product=metadata_roots['product_metadata']
            quant=float(one(product,'BOA_QUANTIFICATION_VALUE'))
            offsets={node.attrib.get('band_id'):float(node.text) for node in product.iter()
                     if node.tag.rsplit('}',1)[-1]=='BOA_ADD_OFFSET' and node.text}
            bands={}
            for node in product.iter():
                if node.tag.rsplit('}',1)[-1]!='Spectral_Information':continue
                physical=node.attrib.get('physicalBand','').upper();match=re.fullmatch(r'B0?(\d{1,2})',physical)
                if match:bands[node.attrib.get('bandId')]=f'B{int(match.group(1)):02d}'
            offset_tags={};radiometry={}
            for band in ('B03','B08','B11'):
                bid=next((key for key,value in bands.items() if value==band),None)
                if bid is None or bid not in offsets:raise ValueError(f'2023 native offset tag missing for {row["item_id"]}/{band}')
                expected=native['radiometry'][band]
                if abs(expected['scale']-1.0/quant)>1e-12 or abs(expected['offset']-offsets[bid]/quant)>1e-12:
                    raise ValueError(f'2023 native radiometry does not match exact XML tags for {row["item_id"]}/{band}')
                offset_tags[band]={'band_id':bid,'boa_add_offset_tag':offsets[bid]}
                radiometry[band]={'quantification':quant,'offset_tag':offsets[bid],
                    'scale':1.0/quant,'offset':offsets[bid]/quant,
                    'offset_status':'EXPLICIT_NATIVE_XML_TAG'}
            payload=json.loads((ROOT/row['receipt_file']).read_text())
            payload.update({'processing_baseline':native['baseline'],'boa_radiometry':radiometry,
                'parent_native_radiometry':native['radiometry'],
                'native_boa_quantification_value':quant,'native_boa_offset_tags':offset_tags,
                'native_sensing_time':native['native_sensing_time'],
                'native_product_start_time':one(product,'PRODUCT_START_TIME'),
                'native_product_stop_time':one(product,'PRODUCT_STOP_TIME'),
                'metadata_receipts':metadata_receipts})
            old_receipt=ROOT/row['receipt_file']
            receipt_v2=old_receipt.with_name(f'{old_receipt.stem}.native-bound-{a.epoch}.json');write(receipt_v2,payload)
            row.update({k:payload[k] for k in ('processing_baseline','boa_radiometry','parent_native_radiometry','native_boa_quantification_value',
                'native_boa_offset_tags','native_sensing_time','native_product_start_time','native_product_stop_time','metadata_receipts')})
            row['receipt_file']=receipt_v2.relative_to(ROOT).as_posix();row['receipt_sha256']=sha(receipt_v2)
            retained_2023+=1;continue
        if row.get('provider')!='Earth Search C1 L2A':continue
        scene=attempt/'provider'/row['site_id']/row['item_id']
        item=json.loads((scene/'item.json').read_text())
        metadata_receipts={}
        native={}
        for role in ('product_metadata','granule_metadata'):
            asset=item['assets'][role];url=asset['href'];xml_path=scene/f'{role}.xml'
            actual=sha(xml_path)
            matches=[r for r in by_url.get(url,[]) if r.get('sha256')==actual and r.get('body_status')=='COMPLETE_RESPONSE']
            if not matches:raise ValueError(f'no completed transport receipt matches native XML {row["item_id"]}/{role}')
            receipt=matches[-1]
            if receipt.get('http_status')!=200 or receipt.get('received_bytes')!=xml_path.stat().st_size:
                raise ValueError(f'native XML transport status/length mismatch for {row["item_id"]}/{role}')
            metadata_receipts[role]={**receipt,'path':xml_path.relative_to(ROOT).as_posix(),'sha256':actual,
                                     'asset_checksum':asset.get('file:checksum')}
            native[role]=ET.parse(str(xml_path)).getroot()
        start=one(native['product_metadata'],'PRODUCT_START_TIME')
        stop=one(native['product_metadata'],'PRODUCT_STOP_TIME')
        if not same_instant(row.get('native_sensing_time'),one(native['granule_metadata'],'SENSING_TIME')):
            raise ValueError(f'native sensing timestamp mismatch for {row["item_id"]}')
        old_receipt=ROOT/row['receipt_file'];payload=json.loads(old_receipt.read_text())
        payload['metadata_receipts']=metadata_receipts
        payload['native_product_start_time']=start;payload['native_product_stop_time']=stop
        receipt_v2=old_receipt.with_name(f'{old_receipt.stem}.native-bound-{a.epoch}.json')
        write(receipt_v2,payload)
        row['receipt_file']=receipt_v2.relative_to(ROOT).as_posix()
        row['receipt_sha256']=sha(receipt_v2)
        row['native_boa_quantification_value']=next(iter(row['boa_radiometry'].values()))['quantification']
        row['native_boa_offset_tags']={band:{'boa_add_offset_tag':values['offset_tag']}
                                       for band,values in row['boa_radiometry'].items()}
        payload['native_boa_quantification_value']=row['native_boa_quantification_value']
        payload['native_boa_offset_tags']=row['native_boa_offset_tags']
        write(receipt_v2,payload);row['receipt_sha256']=sha(receipt_v2)
        row['metadata_receipts']=metadata_receipts
        row['native_product_start_time']=start;row['native_product_stop_time']=stop
        enriched+=1
    manifest['native_metadata_receipt_enrichment']={
        'status':'ALL_PROVIDER_NATIVE_XML_RECEIPTS_BOUND',
        'provider_rows_enriched':enriched,'retained_2023_rows_verified':retained_2023,
        'pre_enrichment_manifest_sha256':sha(preserved),
        'transfer_ledger_sha256':sha(attempt/'transfer_ledger.json'),
        'method':'Each retained XML file hash and byte count joins a complete HTTP 200 ledger receipt matched by the exact STAC asset URL; original receipts and manifest are retained.'
    }
    write(manifest_path,manifest)
    print(json.dumps({'status':manifest['native_metadata_receipt_enrichment']['status'],
                      'provider_rows_enriched':enriched,'retained_2023_rows_verified':retained_2023,'manifest_sha256':sha(manifest_path),
                      'pre_enrichment_manifest':preserved.relative_to(ROOT).as_posix()},indent=2))
    return 0
if __name__=='__main__':raise SystemExit(main())
