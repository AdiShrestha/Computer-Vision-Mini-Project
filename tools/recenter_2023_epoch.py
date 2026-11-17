"""Build a new, sealed epoch from verified retained sources and bounded new ranges.

Old expansion directories are read-only inputs. No lake outlines or QA values
are inferred from the coordinate reference. Grid-definition TIFFs are geometry,
not observations. Provider failures remain in the new transfer ledger.
"""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np
import rasterio
from pyproj import Transformer
from sentinel_gl.ingestion import expansion2023 as x
from sentinel_gl.panel_v2 import load_panel_v2

ROOT=Path(__file__).resolve().parents[1]
EPOCH='expansion_2023_recentered'
RAW=ROOT/'data/raw'/EPOCH
DERIVED=ROOT/'data/derived'/EPOCH
OLD=ROOT/'data/derived/expansion_2023/observations.v2.json'
SOUTH='PILOT-SOUTH-LHONAK';KHANG='PILOT-KHANGCHUNG-TSHO'

def load(path):return json.loads(Path(path).read_text())
def grid_definitions(rows):
    grids={}
    for site in [SOUTH,KHANG]:
        grid=copy.deepcopy(next(r['grid'] for r in rows if r['site_id']==site))
        if site==KHANG:
            grid['transform']=[20.0,0.0,673540.0,0.0,-20.0,3102360.0]
        target=RAW/'grids'/site/'grid_definition.tif'
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            with rasterio.open(target,'w',driver='GTiff',height=499,width=499,count=1,
                               dtype='uint8',crs=grid['crs'],transform=rasterio.Affine(*grid['transform']),
                               compress='deflate',nodata=0) as raster:
                raster.write(np.zeros((499,499),dtype=np.uint8),1)
                raster.update_tags(origin='GRID_DEFINITION_ONLY_NO_OBSERVATIONS',
                                   purpose='Geometry reference; values are not QA or measured water masks')
        grid['grid_origin_reference']=target.relative_to(ROOT).as_posix()
        grid['grid_reference_sha256']=x.sha(target);grids[site]=grid
    return grids

def geometry_check(rows,grids):
    grid=grids[KHANG];a=rasterio.Affine(*grid['transform'])
    east,north=Transformer.from_crs('EPSG:4326',grid['crs'],always_xy=True).transform(88.815,27.988333)
    col,row=(~a)*(east,north);cx,cy=a*(249.5,249.5)
    checks=[]
    for record in rows:
        if record['site_id']!=KHANG or record['sensor']!='SENTINEL_2_OPTICAL':continue
        item=load((ROOT/record['receipt_file']).parent/'item.json')
        for role in ['green','nir','swir16','scl']:
            asset=item['assets'][role];native=rasterio.Affine(*asset['proj:transform'][:6])
            h,w=asset['proj:shape'];left,bottom,right,top=rasterio.transform.array_bounds(499,499,a)
            al,ab,ar,at=rasterio.transform.array_bounds(h,w,native)
            inside=al<=left<=right<=ar and ab<=bottom<=top<=at
            c,r=(~native)*(left,top)
            aligned=abs(c-round(c))<1e-8 and abs(r-round(r))<1e-8
            if not inside or not aligned:raise ValueError('Corrected grid outside or unaligned with retained optical tile')
            checks.append({'product_id':record['product_id'],'asset':role,'full_grid_inside':inside,'native_aligned':aligned})
    result={'origin':'GEOMETRY_CALCULATION_NOT_LAKE_OUTLINE','reference_lon_lat':[88.815,27.988333],
        'reference_projected':[east,north],'reference_grid_column_row':[col,row],
        'reference_inside_grid':0<=col<499 and 0<=row<499,'grid_center_projected':[cx,cy],
        'grid_center_distance_from_reference_m':float(np.hypot(cx-east,cy-north)),
        'grid':grid,'optical_coverage_checks':checks,
        'lake_outline_coverage':'UNMEASURED_NO_AUTHENTICATED_POLYGON',
        'note':'PI-specified exact affine retained. Its pixel-area center differs slightly from the rounded NRSC coordinate; no exact centering or full-lake coverage claim.'}
    x.write(DERIVED/'geometry_check.json',result)
    return result

def reuse_south(record,grid):
    folder=RAW/'optical'/record['observation_id'];folder.mkdir(parents=True,exist_ok=True)
    receipt_file=folder/'receipt.json'
    if receipt_file.exists():return x.optical_row(ROOT,receipt_file)
    old_receipt_path=ROOT/record['receipt_file'];receipt=load(old_receipt_path)
    old_values=ROOT/record['values_file']
    if x.sha(old_values)!=record['sha256'] or receipt['sha256']!=record['sha256']:
        raise ValueError('Retained South Lhonak source binding changed')
    with np.load(old_values,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    arrays['origin']=np.asarray('observational')
    target=DERIVED/'rois'/(record['observation_id']+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(target)
    np.savez_compressed(target,**arrays)
    with np.load(target,allow_pickle=False) as z:
        for name,array in arrays.items():
            if not np.array_equal(z[name],array,equal_nan=(array.dtype.kind=='f')):
                raise ValueError('Source-preserving NPZ copy differs')
    receipt.update({'origin':'observational','grid':grid,'values_file':target.relative_to(ROOT).as_posix(),
        'sha256':x.sha(target),'reused_observation':{'values_file':record['values_file'],'sha256':record['sha256'],
        'receipt_file':record['receipt_file'],'receipt_sha256':x.sha(old_receipt_path),
        'operation':'Source arrays unchanged; new explicit origin metadata and sealed receipt only'}})
    for role,binding in receipt['native']['receipts'].items():
        binding['path']=(old_receipt_path.parent/(role+'.xml')).relative_to(ROOT).as_posix()
    item=old_receipt_path.parent/'item.json'
    receipt['stac_item']={'path':item.relative_to(ROOT).as_posix(),'sha256':x.sha(item)}
    x.write(receipt_file,receipt)
    return x.optical_row(ROOT,receipt_file)

def stage_sar(rows,grids):
    old_receipt=load(ROOT/next(r['receipt_file'] for r in rows if r['sensor']=='SENTINEL_1_IW_GRD'))
    paths={pol:{role:ROOT/b['path'] for role,b in roles.items()} for pol,roles in old_receipt['source_roles'].items()}
    for roles in old_receipt['source_roles'].values():
        for b in roles.values():
            if x.sha(ROOT/b['path'])!=b['sha256']:raise ValueError('Retained SAR role hash mismatch')
    archive=old_receipt['archive']
    if x.sha(ROOT/archive['path'])!=archive['sha256']:raise ValueError('Retained SAR archive hash mismatch')
    row=load(ROOT/'data/raw/wp03/cdse_stage_20261014_restart002/fresh_product_metadata.json')
    folder=RAW/'sar_retained_20230901';folder.mkdir(parents=True,exist_ok=True)
    if all((folder/(site+'_receipt.json')).exists() for site in grids):
        result=[]
        for site in grids:
            rp=folder/(site+'_receipt.json');r=load(rp)
            result.append({'observation_id':r['observation_id'],'site_id':site,'sensor':'SENTINEL_1_IW_GRD',
                'product_id':r['product_id'],'acquired_at':r['acquired_at'],'values_file':r['values_file'],
                'sha256':r['sha256'],'grid':r['grid'],'channels':x.SAR,'receipt_file':rp.relative_to(ROOT).as_posix(),
                'receipt_sha256':x.sha(rp),'origin':'observational'})
        return result
    coverage=[]
    for site,grid in grids.items():
        meta=x.parse_native_sentinel1_product(paths['VV']['annotation'].read_bytes(),
                 paths['VV']['calibration'].read_bytes(),paths['VV']['noise'].read_bytes())
        r,c,window=x._native_mapping(meta,grid)
        covered=np.isfinite(r)&np.isfinite(c)&(r+window.row_off>=0)&(r+window.row_off<=meta['rows']-1)&(c+window.col_off>=0)&(c+window.col_off<=meta['columns']-1)
        coverage.append({'site_id':site,'grid_pixel_count':int(covered.size),'inside_native_tiepoint_support':int(covered.sum())})
        if not covered.all():raise ValueError('PI grid not fully inside retained SAR tiepoint support')
    x.write(DERIVED/'sar_geometry_check.json',{'network_calls':0,'coverage':coverage,'terrain_correction':'NOT_APPLIED'})
    return x.calibrate_sar_inputs(ROOT,row,paths,archive,folder,grids=grids,derived_epoch=EPOCH)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['optical','sar','seal'],required=True);args=parser.parse_args()
    RAW.mkdir(parents=True,exist_ok=True);DERIVED.mkdir(parents=True,exist_ok=True)
    rows=load(OLD)['observations'];grids=grid_definitions(rows);geometry_check(rows,grids)
    if args.stage=='optical':
        budget=x.Budget(RAW/'transfer_ledger.json',cap=750000000)
        for record in rows:
            if record['sensor']!='SENTINEL_2_OPTICAL':continue
            if record['site_id']==SOUTH:result=reuse_south(record,grids[SOUTH])
            else:
                item=load((ROOT/record['receipt_file']).parent/'item.json')
                result=x.optical_observation(ROOT,KHANG,item,budget,0,grid=grids[KHANG],raw_epoch=EPOCH,derived_epoch=EPOCH)
            print(json.dumps({'event':'OPTICAL_READY','observation_id':result['observation_id'],'received_bytes':budget.received}),flush=True)
    elif args.stage=='sar':
        print(json.dumps({'event':'RETAINED_SAR_READY','observations':len(stage_sar(rows,grids))}),flush=True)
    else:
        result=[x.optical_row(ROOT,p) for p in sorted((RAW/'optical').glob('*/receipt.json'))]
        result+=stage_sar(rows,grids)
        if len(result)!=len(rows):raise ValueError('New epoch incomplete; preserve partial attempts, do not fill missing sources')
        manifest=DERIVED/'observations.v2.json'
        if manifest.exists():raise FileExistsError('Sealed manifest already exists')
        result.sort(key=lambda r:(r['site_id'],r['acquired_at'],r['sensor']))
        x.write(manifest,{'schema_version':2,'origin':'observational','study':'RETROSPECTIVE_2023_TWO_DEVELOPMENTAL_ROIS',
            'previous_epoch':{'path':OLD.relative_to(ROOT).as_posix(),'sha256':x.sha(OLD)},
            'geometry_reference':'NRSC/ISRO Khangchung Tso GLOF Risk Assessment, section 3.1',
            'observations':result})
        panel=load_panel_v2(ROOT,manifest,mode='observational')
        x.write(DERIVED/'READY.json',{'status':'SEALED_SOURCE_BINDINGS_VERIFIED','manifest':manifest.relative_to(ROOT).as_posix(),
             'sha256':panel.manifest_sha256,'observations':len(panel.observations),'origin':panel.document['origin']})
        print(json.dumps(load(DERIVED/'READY.json')),flush=True)
if __name__=='__main__':main()
