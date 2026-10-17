"""Bounded 2023 source-derived ROI acquisition. No observation synthesis."""
from __future__ import annotations
import hashlib,io,json,math,os,re,threading,time,zipfile,fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone,timedelta
from pathlib import Path
from urllib.parse import urlencode,urlsplit
import numpy as np
import requests
import rasterio
from rasterio.windows import Window,from_bounds
from pyproj import Transformer
from scipy.interpolate import LinearNDInterpolator
from defusedxml import ElementTree as ET
from .sentinel1_calibration import (parse_native_sentinel1_product,calibrate_native_noise_window,
    bilinear_resample_with_valid_weights,strict_power_to_db)
from .cdse_single_granule import footprint_contains
from . import asf_grd

HOSTS={'catalogue.dataspace.copernicus.eu','download.dataspace.copernicus.eu',
       'earth-search.aws.element84.com','e84-earth-search-sentinel-data.s3.us-west-2.amazonaws.com'}
SITES={'PILOT-SOUTH-LHONAK':(88.204,27.915),'PILOT-KHANGCHUNG-TSHO':(88.752,27.985)}
OPTICAL=['B03','B08','B11','NDWI','MNDWI']
SAR=['SIGMA0_VV_RAW_LINEAR','SIGMA0_VH_RAW_LINEAR']
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def now():return datetime.now(timezone.utc).isoformat()
def check_url(url):
    p=urlsplit(url)
    if p.scheme!='https' or p.hostname not in HOSTS or p.username or p.password or p.port not in {None,443}:
        raise ValueError('URL outside the authorized exact-host HTTPS allowlist')
    return url
class Budget:
    def __init__(self,path,cap=8262017611):
        self.path=Path(path);self.cap=cap;self.lock=threading.Lock()
        old=json.loads(self.path.read_text()) if self.path.exists() else {'received_bytes':0,'requests':[]}
        self.received=old['received_bytes'];self.rows=old['requests']
    def add(self,row):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.lock, self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            current=json.loads(self.path.read_text()) if self.path.exists() else {'received_bytes':0,'requests':[]}
            n=row['received_bytes']
            self.received=current['received_bytes']+n;self.rows=current['requests']+[row]
            write(self.path,{'schema_version':2,'cap_bytes':self.cap,'received_bytes':self.received,
                'remaining_bytes':self.cap-self.received,'requests':self.rows})
            fcntl.flock(lock,fcntl.LOCK_UN)
            if self.received>self.cap:raise ValueError('Actual received payload exceeds authorized byte cap; receipt retained')
    @property
    def remaining(self):return self.cap-self.received
def fetch(url,budget,*,method='GET',payload=None,headers=None,limit=12000000,session=None):
    check_url(url);start=now();r=(session or requests).request(method,url,json=payload,headers=headers,
        stream=True,timeout=(20,90),allow_redirects=False)
    blocks=[];n=0;h=hashlib.sha256();failure=None
    try:
        for block in r.iter_content(256*1024):
            n+=len(block);h.update(block);blocks.append(block)
            if n>min(limit,budget.remaining):
                raise ValueError('Bounded response exceeded declared cap')
    except Exception as exc:
        failure=exc
    finally:
        receipt={'url':url,'method':method,'http_status':r.status_code,'received_bytes':n,
            'sha256':h.hexdigest(),'started_at':start,'finished_at':now(),
            'content_range':r.headers.get('Content-Range'),'content_length':r.headers.get('Content-Length'),
            'etag':r.headers.get('ETag'),'credential_logged':False,
            'body_status':'PARTIAL_ABORTED' if failure else 'COMPLETE_RESPONSE',
            'error':None if failure is None else type(failure).__name__+': '+str(failure)}
        r.close();budget.add(receipt)
    if failure:raise failure
    data=b''.join(blocks)
    if r.status_code>=400:raise requests.HTTPError(f'HTTP {r.status_code}: '+data[:200].decode(errors='replace'))
    if 300<=r.status_code<400:raise ValueError('Redirect refused; no unverified endpoint contacted')
    return data,receipt

class RangeStore:
    """Exact hashed HTTP ranges, cached only in RAM; never fetches an entire COG."""
    def __init__(self,url,size,budget):
        self.url=check_url(url);self.size=int(size);self.budget=budget;self.cache={};self.receipts=[]
        self.session=requests.Session();self.block=65536
    def read(self,start,n):
        if start>=self.size:return b''
        end=min(self.size,start+n);bases=list(range((start//self.block)*self.block,end,self.block))
        missing=[base for base in bases if base not in self.cache]
        while missing:
            base=missing[0];last=base
            for candidate in missing[1:]:
                if candidate!=last+self.block or candidate-base>=4*1024*1024:break
                last=candidate
            stop=min(last+self.block,self.size)-1
            data,receipt=fetch(self.url,self.budget,headers={'Range':f'bytes={base}-{stop}',
                'Accept-Encoding':'identity'},limit=stop-base+2,session=self.session)
            expected=f'bytes {base}-{stop}/{self.size}'
            if receipt['http_status']!=206 or receipt['content_range']!=expected or len(data)!=stop-base+1:
                raise ValueError('COG range status/content-range/length mismatch')
            for offset in range(base,last+self.block,self.block):
                self.cache[offset]=data[offset-base:offset-base+self.block]
            self.receipts.append(receipt)
            missing=[x for x in missing if x>last]
        return b''.join(self.cache[base][max(start,base)-base:min(end,base+self.block)-base] for base in bases)

class RangeFile(io.RawIOBase):
    def __init__(self,store):super().__init__();self.store=store;self.pos=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.store.size+offset
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        if n<0:n=self.store.size-self.pos
        out=self.store.read(self.pos,n);self.pos+=len(out);return out
    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)
def grid_for(root,site):
    path=Path(root)/'configs/roi_grids.json'
    grid=json.loads(path.read_text())[site]
    if grid['crs']!='EPSG:32645' or grid['shape']!=[499,499] or len(grid['transform'])!=6:
        raise ValueError('Unexpected fixed scientific ROI grid')
    return {**grid,'grid_origin_reference':'configs/roi_grids.json','grid_reference_sha256':sha(path)}
def local_xml(root,name):
    values=[n.text.strip() for n in root.iter() if n.tag.rsplit('}',1)[-1]==name and n.text and n.text.strip()]
    if len(set(values))!=1:raise ValueError('Ambiguous/missing native XML '+name)
    return values[0]
def optical_metadata(item,folder,budget):
    paths={};receipts={}
    for role in ['product_metadata','granule_metadata']:
        asset=item['assets'][role];data,receipt=fetch(asset['href'],budget,limit=3000000)
        expected=asset.get('file:checksum','')
        if expected.startswith('1220') and hashlib.sha256(data).hexdigest()!=expected[4:]:
            raise ValueError('Native XML SHA-256 differs from STAC multihash')
        p=folder/(role+'.xml');p.write_bytes(data);paths[role]=p;receipts[role]=receipt
    product=ET.parse(paths['product_metadata']).getroot();granule=ET.parse(paths['granule_metadata']).getroot()
    baseline=local_xml(product,'PROCESSING_BASELINE')
    if baseline not in {'05.09','05.10','05.11'} or baseline!=item['properties']['s2:processing_baseline']:
        raise ValueError('Unreviewed or mismatched native processing baseline')
    if local_xml(product,'PRODUCT_URI')!=item['properties']['s2:product_uri']:
        raise ValueError('Native product/STAC identity mismatch')
    quant=float(local_xml(product,'BOA_QUANTIFICATION_VALUE'));ids={};offsets={}
    if not math.isfinite(quant) or quant<=0:raise ValueError('Invalid BOA quantification')
    for n in product.iter():
        tag=n.tag.rsplit('}',1)[-1]
        if tag=='Spectral_Information':ids[n.attrib.get('physicalBand')]=n.attrib.get('bandId')
        if tag=='BOA_ADD_OFFSET':offsets[n.attrib.get('band_id')]=float(n.text)
    radiometry={}
    for band in ['B03','B08','B11']:
        bid=ids.get(band,ids.get(band.replace('B0','B')))
        if bid not in offsets:raise ValueError('Native BOA offset missing for '+band)
        radiometry[band]={'scale':1/quant,'offset':offsets[bid]/quant}
    return {'baseline':baseline,'radiometry':radiometry,'receipts':receipts,
        'native_sensing_time':local_xml(granule,'SENSING_TIME'),'product_uri':local_xml(product,'PRODUCT_URI')}
def read_cog(asset,band,grid,budget,ordinal):
    store=RangeStore(asset['href'],asset['file:size'],budget)
    filename=f'sentinel_{ordinal}_{band}.tif'
    def opener(path,mode='rb'):
        if Path(path).name!=filename:raise FileNotFoundError(path)
        return RangeFile(store)
    affine=rasterio.Affine(*grid['transform']);h,w=grid['shape']
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_NUM_THREADS='1'):
        with rasterio.open(filename,opener=opener) as src:
            if str(src.crs)!=grid['crs']:raise ValueError('COG CRS differs from target grid')
            ratio=affine.a/src.transform.a
            if abs(ratio-round(ratio))>1e-8 or round(ratio) not in {1,2}:
                raise ValueError('Unsupported source/target pixel scale')
            ratio=int(round(ratio));col=(affine.c-src.transform.c)/src.transform.a;row=(affine.f-src.transform.f)/src.transform.e
            if abs(col-round(col))>1e-6 or abs(row-round(row))>1e-6:
                raise ValueError('Target grid is not aligned to source pixels')
            window=Window(round(col),round(row),w*ratio,h*ratio)
            if window.col_off<0 or window.row_off<0 or window.col_off+window.width>src.width or window.row_off+window.height>src.height:
                raise ValueError('Target ROI extends beyond source tile bounds')
            raw=src.read(1,window=window);valid=(raw!=src.nodata) & np.isfinite(raw)
            if ratio==2:
                shaped=raw.reshape(h,2,w,2);vm=valid.reshape(h,2,w,2)
                raw=shaped.astype(np.float64).mean(axis=(1,3));valid=vm.all(axis=(1,3))
            meta={'source_shape':[src.height,src.width],'source_transform':list(src.transform)[:6],
                'window':[window.col_off,window.row_off,window.width,window.height],
                'source_scale':float(src.scales[0]),'source_offset':float(src.offsets[0]),
                'received_range_count':len(store.receipts),'received_bytes':sum(x['received_bytes'] for x in store.receipts),
                'range_receipts':store.receipts,'full_cog_sha256':'NOT_COMPUTED_NOT_FULLY_DOWNLOADED',
                'provider_full_cog_multihash':asset.get('file:checksum')}
    store.session.close()
    return raw,valid,meta
def select_optical(features,count=24):
    # Fixed equal-duration bins; lowest scene-level cloud metadata, then time/id. No anomaly scores.
    start=datetime(2023,1,1,tzinfo=timezone.utc);end=datetime(2023,11,16,tzinfo=timezone.utc)
    bins=[[] for _ in range(count)]
    for item in features:
        t=datetime.fromisoformat(item['properties']['datetime'].replace('Z','+00:00'))
        index=min(count-1,int((t-start).total_seconds()/(end-start).total_seconds()*count))
        bins[index].append(item)
    selected=[]
    for candidates in bins:
        if candidates:selected.append(min(candidates,key=lambda x:(x['properties'].get('eo:cloud_cover',100),
            x['properties']['datetime'],x['id'])))
    return sorted(selected,key=lambda x:x['properties']['datetime'])
def discover_optical(root,budget):
    result={}
    for site,(lon,lat) in SITES.items():
        folder=Path(root)/'data/raw/expansion_2023/catalogue'/site;folder.mkdir(parents=True,exist_ok=True)
        body={'collections':['sentinel-2-c1-l2a'],'intersects':{'type':'Point','coordinates':[lon,lat]},
            'datetime':'2023-01-01T00:00:00Z/2023-11-15T23:59:59Z','limit':100}
        features=[]
        for page in range(5):
            data,receipt=fetch('https://earth-search.aws.element84.com/v1/search',budget,method='POST',payload=body)
            (folder/f'page_{page:02d}.json').write_bytes(data);write(folder/f'page_{page:02d}.receipt.json',receipt)
            obj=json.loads(data);features.extend(obj.get('features',[]))
            nxt=next((x for x in obj.get('links',[]) if x.get('rel')=='next'),None)
            if not nxt:break
            check_url(nxt['href']);body=nxt['body']
        unique={x['id']:x for x in features}
        grid=grid_for(root,site);target=rasterio.Affine(*grid['transform'])
        h,w=grid['shape'];left,bottom,right,top=rasterio.transform.array_bounds(h,w,target)
        eligible=[]
        for candidate in unique.values():
            asset=candidate['assets']['scl'];t=rasterio.Affine(*asset['proj:transform'][:6])
            height,width=asset['proj:shape'];al,ab,ar,at=rasterio.transform.array_bounds(height,width,t)
            if candidate['properties'].get('proj:epsg')==32645 and al<=left and ab<=bottom and ar>=right and at>=top:
                eligible.append(candidate)
        result[site]=select_optical(eligible)
        write(folder/'selection.json',{'catalogued_items':len(unique),'full_ROI_coverage_items':len(eligible),'selection_policy':'24 equal-duration bins; lowest STAC scene cloud cover, then timestamp and ID; no score-based selection',
            'selected_ids':[x['id'] for x in result[site]],'selected_times':[x['properties']['datetime'] for x in result[site]]})
    return result
def optical_observation(root,site,item,budget,index):
    root=Path(root);oid=site+'__'+item['id'];folder=root/'data/raw/expansion_2023/optical'/oid
    if folder.exists():
        receipt_file=folder/'receipt.json'
        if receipt_file.exists():
            receipt=json.loads(receipt_file.read_text());target=root/receipt['values_file']
            if receipt['observation_id']!=oid or sha(target)!=receipt['sha256']:
                raise ValueError('Retained optical observation receipt/hash mismatch')
            return optical_row(root,receipt_file)
        attempt=2
        while (folder/f'attempt_{attempt:03d}').exists():attempt+=1
        folder=folder/f'attempt_{attempt:03d}'
    folder.mkdir(parents=True,exist_ok=False);write(folder/'item.json',item)
    grid=grid_for(root,site);native=optical_metadata(item,folder,budget);arrays={};receipts={}
    for role,band in [('green','B03'),('nir','B08'),('swir16','B11'),('scl','SCL')]:
        asset=item['assets'][role];raw,valid,receipt=read_cog(asset,band,grid,budget,index)
        if band!='SCL':
            declared=asset['raster:bands'][0];expected=native['radiometry'][band]
            if abs(float(declared['scale'])-expected['scale'])>1e-12 or abs(float(declared['offset'])-expected['offset'])>1e-12:
                raise ValueError('STAC/native BOA radiometry mismatch')
            if (receipt['source_scale'],receipt['source_offset'])!=(1.0,0.0):
                if abs(receipt['source_scale']-expected['scale'])>1e-12 or abs(receipt['source_offset']-expected['offset'])>1e-12:
                    raise ValueError('COG/native BOA tag mismatch')
            raw=raw.astype(np.float64)*expected['scale']+expected['offset']
        arrays[band]=(raw,valid);receipts[band]=receipt
    scl,scl_valid=arrays['SCL'];clear=scl_valid & np.isin(scl,[4,5,6]);values=[];valids=[]
    for band in ['B03','B08','B11']:
        a,v=arrays[band];v=v & clear & np.isfinite(a);values.append(np.where(v,a,np.nan));valids.append(v)
    for first,second in [('B03','B08'),('B03','B11')]:
        second_index=1 if second=='B08' else 2
        a=np.asarray(values[0],dtype=np.float32).astype(np.float64);b=np.asarray(values[second_index],dtype=np.float32).astype(np.float64)
        va=valids[0];vb=valids[second_index];den=a+b
        v=va & vb & clear & np.isfinite(den) & (den!=0)
        out=np.full(a.shape,np.nan);np.divide(a-b,den,out=out,where=v)
        values.append(out);valids.append(v)
    values=np.asarray(values,dtype=np.float32);valids=np.asarray(valids,dtype=bool);values[~valids]=np.nan
    target=root/'data/derived/expansion_2023/rois'/(oid+('' if folder.name==oid else '__'+folder.name)+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,values=values,observed_valid=valids,scl_classes=scl.astype(np.uint8),qa_clear=clear,
        candidate_water=valids[3]&valids[4]&(values[3]>0)&(values[4]>0))
    receipt={'observation_id':oid,'site_id':site,'product_id':item['properties']['s2:product_uri'],
        'acquired_at':item['properties']['datetime'],'native':native,'assets':receipts,'grid':grid,
        'values_file':target.relative_to(root).as_posix(),'sha256':sha(target),'qa_clear_count':int(clear.sum()),
        'valid_per_channel':[int(x.sum()) for x in valids],
        'historical_operational_availability':'UNVERIFIED; acquisition-time retrospective analysis only',
        'lake_polygon_validation':'NOT_ESTABLISHED_QUERY_CENTROID_ROI'}
    rp=folder/'receipt.json';write(rp,receipt)
    return optical_row(root,rp)
def optical_row(root,receipt_file):
    receipt=json.loads(Path(receipt_file).read_text())
    return {'observation_id':receipt['observation_id'],'site_id':receipt['site_id'],'sensor':'SENTINEL_2_OPTICAL',
        'acquired_at':receipt['acquired_at'],'product_id':receipt['product_id'],
        'values_file':receipt['values_file'],'sha256':receipt['sha256'],'channels':OPTICAL,
        'grid':receipt['grid'],'receipt_file':Path(receipt_file).relative_to(root).as_posix(),
        'qa_clear_count':receipt['qa_clear_count'],'valid_per_channel':receipt['valid_per_channel']}

def save_observations(root,observations):
    path=Path(root)/'data/derived/expansion_2023/observations.v2.json';path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=json.loads(path.read_text())['observations'] if path.exists() else []
        combined={r['observation_id']:r for r in old}
        combined.update({r['observation_id']:r for r in observations})
        ordered=sorted(combined.values(),key=lambda x:(x['site_id'],x['acquired_at'],x['sensor']))
        write(path,{'schema_version':2,'study':'RETROSPECTIVE_2023_TWO_DEVELOPMENTAL_ROIS','observations':ordered,
            'geometry':'20m EPSG32645 query-centroid grids; SAR ellipsoid tiepoints, no DEM terrain correction',
            'control_status':'PI_DESIGNATED_DEVELOPMENTAL_COMPARATOR; no independently ascertained non-event label',
            'historical_availability':'UNVERIFIED','network_authority':'EXACT_FOUR_PROVIDER_HOSTS'})
        fcntl.flock(lock,fcntl.LOCK_UN)

def run_optical(root,budget,workers=3):
    root=Path(root);selected=discover_optical(root,budget);observations=[];failures=[]
    path=root/'data/derived/expansion_2023/observations.v2.json'
    if path.exists():observations=json.loads(path.read_text())['observations']
    done={x['observation_id'] for x in observations};jobs=[]
    for site,items in selected.items():
        for item in items:
            if site+'__'+item['id'] not in done:jobs.append((site,item,len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(optical_observation,root,s,i,budget,n):(s,i['id']) for s,i,n in jobs}
        for future in as_completed(futures):
            site,itemid=futures[future]
            try:
                observation=future.result();observations.append(observation);save_observations(root,observations)
                print(json.dumps({'event':'OPTICAL_STAGED','site':site,'item':itemid,'clear':observation['qa_clear_count'],'observations':len(observations),'received_bytes':budget.received}),flush=True)
            except Exception as exc:
                failure={'site':site,'item':itemid,'error':type(exc).__name__+': '+str(exc)}
                failures.append(failure);write(root/'data/raw/expansion_2023/optical_failures.json',failures)
                print(json.dumps({'event':'OPTICAL_FAILED',**failure}),flush=True)
                if len(failures)>=2:
                    for pending in futures:pending.cancel()
                    raise RuntimeError('Two unexpected optical failures; successes preserved')
    return observations

def _native_mapping(metadata,grid):
    affine=rasterio.Affine(*grid['transform']);h,w=grid['shape'];rr,cc=np.indices((h,w),dtype=np.float64)
    x=affine.c+affine.a*(cc+.5)+affine.b*(rr+.5);y=affine.f+affine.d*(cc+.5)+affine.e*(rr+.5)
    lon,lat=Transformer.from_crs(grid['crs'],'EPSG:4326',always_xy=True).transform(x,y)
    points=metadata['geolocation_grid_points']
    ll=np.asarray([[p['longitude'],p['latitude']] for p in points])
    ij=np.asarray([[p['line'],p['pixel']] for p in points])
    mapped=LinearNDInterpolator(ll,ij,fill_value=np.nan)(np.column_stack([lon.ravel(),lat.ravel()])).reshape(h,w,2)
    rows,samples=mapped[...,0],mapped[...,1]
    inside=np.isfinite(rows)&np.isfinite(samples)&(rows>=0)&(rows<=metadata['rows']-1)&(samples>=0)&(samples<=metadata['columns']-1)
    if not inside.any():raise ValueError('Target grid outside native geolocation support')
    rmin=max(0,int(np.floor(rows[inside].min()))-2);cmin=max(0,int(np.floor(samples[inside].min()))-2)
    rmax=min(metadata['rows'],int(np.ceil(rows[inside].max()))+3);cmax=min(metadata['columns'],int(np.ceil(samples[inside].max()))+3)
    return rows-rmin,samples-cmin,Window(cmin,rmin,cmax-cmin,rmax-rmin)
def _utc_native(value):
    stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
    if stamp.tzinfo is None:stamp=stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat().replace('+00:00','Z')
def calibrate_sar_inputs(root,row,paths_by_pol,source_record,folder):
    root=Path(root);folder=Path(folder);metadata_by_pol={};source_hashes={}
    for pol,paths in paths_by_pol.items():
        metadata_by_pol[pol]=parse_native_sentinel1_product(paths['annotation'].read_bytes(),
            paths['calibration'].read_bytes(),paths['noise'].read_bytes())
        source_hashes[pol]={role:{'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':sha(path)}
            for role,path in paths.items()}
        native=metadata_by_pol[pol]
        start=_utc_native(native['start_time']);cat=_utc_native(row['ContentDate']['Start'])
        if abs((datetime.fromisoformat(start.replace('Z','+00:00'))-datetime.fromisoformat(cat.replace('Z','+00:00'))).total_seconds())>.001:
            raise ValueError('Native/catalogue UTC interval mismatch')
        if native['polarization']!=pol:raise ValueError('Native polarization mismatch')
    observations=[]
    for site in SITES:
        grid=grid_for(root,site);raw_values=[];raw_valid=[];signed_values=[];signed_valid=[];nesz_values=[];details={}
        for pol in ['VV','VH']:
            metadata=metadata_by_pol[pol];rows,samples,window=_native_mapping(metadata,grid)
            with rasterio.open(paths_by_pol[pol]['measurement']) as src:
                dn=src.read(1,window=window);source_valid=src.read_masks(1,window=window)>0
            raw,rv,nesz,signed,sv=calibrate_native_noise_window(dn,metadata['calibration_vectors'],
                metadata['noise_range_vectors'],metadata['noise_azimuth_blocks'],
                line_offset=int(window.row_off),sample_offset=int(window.col_off),
                thermal_noise_correction_applied=metadata['thermal_noise_correction_applied'])
            rv &= source_valid;sv &= source_valid
            target_raw,coverage,valid=bilinear_resample_with_valid_weights(raw,rv,rows,samples)
            target_signed,signed_coverage,svalid=bilinear_resample_with_valid_weights(signed,sv,rows,samples)
            target_nesz,_,_=bilinear_resample_with_valid_weights(nesz,sv,rows,samples)
            raw_values.append(target_raw);raw_valid.append(valid);signed_values.append(target_signed);signed_valid.append(svalid);nesz_values.append(target_nesz)
            details[pol]={'native_window':[int(window.col_off),int(window.row_off),int(window.width),int(window.height)],
                'native_full_shape':[metadata['rows'],metadata['columns']],'raw_valid':int(valid.sum()),
                'signed_valid':int(svalid.sum()),'nonpositive_signed_count':int((svalid&(target_signed<=0)).sum()),
                'thermal_noise_correction_applied_in_source':metadata['thermal_noise_correction_applied']}
        values=np.asarray(raw_values,dtype=np.float32);valid=np.asarray(raw_valid,dtype=bool)
        signed=np.asarray(signed_values,dtype=np.float32);svalid=np.asarray(signed_valid,dtype=bool)
        values[~valid]=np.nan;signed[~svalid]=np.nan
        db,dbv=strict_power_to_db(values,valid);sdb,sdbv=strict_power_to_db(signed,svalid)
        oid=site+'__'+row['Name'].removesuffix('.SAFE');target=root/'data/derived/expansion_2023/rois'/(oid+'.npz')
        target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,values=values,observed_valid=valid,signed_values=signed,signed_valid=svalid,
            raw_db=db,raw_db_valid=dbv,signed_db=sdb,signed_db_valid=sdbv,nesz_values=np.asarray(nesz_values,dtype=np.float32))
        acquired=_utc_native(metadata_by_pol['VV']['start_time'])
        receipt={'observation_id':oid,'site_id':site,'product_id':row['Name'],'acquired_at':acquired,
            'end_at':_utc_native(metadata_by_pol['VV']['stop_time']),'native_windows':details,'grid':grid,
            'source_roles':source_hashes,'archive':source_record,'catalogue_content_date':row['ContentDate'],
            'timezone_binding':'SAFE native timestamps interpreted as UTC per ESA product-specification UTC convention; exact product/time joined to explicit UTC CDSE ContentDate within provider precision',
            'geometry':'ANNOTATION_ELLIPSOID_TIEPOINT_INVERSE_NO_DEM_TERRAIN_CORRECTION',
            'historical_operational_availability':'UNVERIFIED; retrospective acquisition-time analysis',
            'primary_model_radiometry':'RAW_SIGMA0_LINEAR; signed and NESZ auxiliary outputs retained separately',
            'values_file':target.relative_to(root).as_posix(),'sha256':sha(target)}
        rp=folder/(site+'_receipt.json');write(rp,receipt)
        observations.append({'observation_id':oid,'site_id':site,'sensor':'SENTINEL_1_IW_GRD','acquired_at':acquired,
            'product_id':row['Name'],'values_file':receipt['values_file'],'sha256':receipt['sha256'],
            'channels':SAR,'grid':grid,'receipt_file':rp.relative_to(root).as_posix(),
            'valid_per_channel':[int(x.sum()) for x in valid]})
    return observations
def reuse_retained_sar(root):
    root=Path(root);config=root/'configs/sar_inputs.local.json'
    spec=json.loads(config.read_text())
    row=json.loads((root/spec['catalogue_item']).read_text())
    paths={pol:{role:root/path for role,path in members.items()} for pol,members in spec['paths_by_pol'].items()}
    source=spec['archive']
    archive=root/source['path']
    if archive.stat().st_size!=source['bytes'] or sha(archive)!=source['sha256']:
        raise ValueError('Local SAR archive size or SHA-256 differs from supplied binding')
    folder=root/'data/raw/expansion_2023/sar_retained_20230901';folder.mkdir(parents=True,exist_ok=True)
    rows=calibrate_sar_inputs(root,row,paths,source,folder);save_observations(root,rows)
    return rows
def select_sar(root):
    rows=json.loads((Path(root)/'data/raw/expansion_2023/sar_catalogue/response.json').read_text())['value']
    eligible=[]
    for row in rows:
        attrs={x['Name']:x.get('Value') for x in row.get('Attributes',[])}
        if (row.get('Online') and '_COG.' not in row['Name'] and attrs.get('relativeOrbitNumber')==12 and
            attrs.get('orbitDirection')=='ASCENDING' and all(footprint_contains(row['GeoFootprint'],p) for p in SITES.values())):
            eligible.append(row)
    targets=['2023-10-07','2023-05-04','2023-05-28','2023-10-19']
    selected=[]
    for target in targets:
        t=datetime.fromisoformat(target).replace(tzinfo=timezone.utc)
        candidates=[r for r in eligible if (r['ContentDate']['Start'][:7]==target[:7])]
        if not candidates:continue
        pick=min(candidates,key=lambda r:abs((datetime.fromisoformat(r['ContentDate']['Start'].replace('Z','+00:00'))-t).total_seconds()))
        if pick['Id'] not in {r['Id'] for r in selected}:selected.append(pick)
    return selected
def stage_sar_product(root,row,budget):
    root=Path(root);folder=root/'data/raw/expansion_2023/sar'/row['Id'];folder.mkdir(parents=True,exist_ok=False)
    write(folder/'catalogue_item.json',row)
    token=os.environ.get('CDSE_ACCESS_TOKEN')
    if not token:return [],{'status':'TOKEN_ABSENT','product':row['Name']}
    url='https://download.dataspace.copernicus.eu/odata/v1/Products('+row['Id']+')/$value';check_url(url)
    if row['ContentLength']+15000000>budget.remaining:return [],{'status':'TRANSFER_BUDGET_RESERVATION_UNAVAILABLE','product':row['Name']}
    archive=folder/'product.zip';h=hashlib.sha256();md5=hashlib.md5();received=0;started=now()
    with requests.get(url,headers={'Authorization':'Bearer '+token,'Accept-Encoding':'identity'},
            stream=True,timeout=(20,120),allow_redirects=False) as response:
        if response.status_code==401:
            error=response.content[:10000];receipt={'url':url,'method':'GET','http_status':401,'received_bytes':len(error),
                'sha256':hashlib.sha256(error).hexdigest(),'message':error.decode(errors='replace'),'credential_logged':False}
            budget.add(receipt);write(folder/'download_receipt.json',receipt)
            return [],{'status':'TOKEN_EXPIRED_OR_UNAUTHORIZED','product':row['Name'],'http_status':401}
        if response.status_code!=200:raise ValueError('SAR download HTTP '+str(response.status_code))
        expected=int(response.headers['Content-Length']) if response.headers.get('Content-Length') else None
        cap=min(row['ContentLength']+10485760,budget.remaining-10000000)
        with archive.open('xb') as out:
            for block in response.iter_content(1024*1024):
                received+=len(block)
                if received>cap:raise ValueError('SAR stream exceeded reserved byte cap')
                out.write(block);h.update(block);md5.update(block)
        if expected is not None and received!=expected:raise ValueError('SAR stream length mismatch')
    receipt={'url':url,'method':'GET','http_status':200,'received_bytes':received,'sha256':h.hexdigest(),
        'local_md5':md5.hexdigest(),'catalogue_checksums':row.get('Checksum'),'catalogue_checksum_scope':'Native-content/ZIP representation equivalence not assumed',
        'started_at':started,'finished_at':now(),'eof':True,'response_content_length':expected,'credential_logged':False}
    budget.add(receipt);write(folder/'download_receipt.json',receipt)
    paths_by_pol={};members={}
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP member CRC failed: '+bad)
        receipt['all_member_crc']='PASS';receipt['zip_members']=len(z.infolist())
        for pol in ['VV','VH']:
            roles=asf_grd.inspect_safe_roles(archive,row['Name'].removesuffix('.SAFE'),pol,origin='provider_download')['roles']
            paths={}
            for role in ['annotation','calibration','noise','measurement']:
                member=roles[role];suffix='.tiff' if role=='measurement' else '.xml';target=folder/(pol+'_'+role+suffix)
                with z.open(member) as src,target.open('xb') as dst:
                    digest=hashlib.sha256()
                    while block:=src.read(1024*1024):dst.write(block);digest.update(block)
                paths[role]=target;members[member]={'sha256':digest.hexdigest(),'bytes':target.stat().st_size}
            paths_by_pol[pol]=paths
        source={'path':archive.relative_to(root).as_posix(),'sha256':h.hexdigest(),'md5':md5.hexdigest(),
            'bytes':received,'crc':'PASS','member_hashes':members,'retention':'TEMPORARY_ARCHIVE_DELETED_AFTER_VERIFIED_ROI_CREATION'}
        rows=calibrate_sar_inputs(root,row,paths_by_pol,source,folder)
    save_observations(root,rows)
    # Peak archive + two original TIFFs is approximately 3.5 GB; no native full-strip output is produced.
    deleted=[]
    for paths in paths_by_pol.values():
        p=paths['measurement'];deleted.append({'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)});p.unlink()
    deleted.append({'path':archive.relative_to(root).as_posix(),'bytes':received,'sha256':h.hexdigest()});archive.unlink()
    write(folder/'temporary_retirement.json',{'authority':'DELETE_RECOMPUTABLE_TEMPORARY_INPUTS_AFTER_ROI_HASHING','deleted':deleted,'retained_ROI_observations':[r['observation_id'] for r in rows]})
    write(folder/'download_receipt.json',receipt)
    return rows,{'status':'SAR_ROIS_CALIBRATED','product':row['Name'],'observations':len(rows)}
def run_sar(root,budget):
    existing=Path(root)/'data/derived/expansion_2023/observations.v2.json'
    current=json.loads(existing.read_text())['observations'] if existing.exists() else []
    if not any(x['sensor']=='SENTINEL_1_IW_GRD' and x['acquired_at'].startswith('2023-09-01') for x in current):
        reuse_retained_sar(root);print(json.dumps({'event':'RETAINED_SAR_WINDOW_CALIBRATED'}),flush=True)
    selected=select_sar(root);write(Path(root)/'data/raw/expansion_2023/sar_catalogue/selection.json',{'policy':'Same ascending relative orbit12, exact coverage of both query centroids; prioritise first post-breach date, two fitting-period May dates, second post-breach date; no score selection','products':selected})
    for row in selected:
        current=json.loads(existing.read_text())['observations']
        if any(x['product_id']==row['Name'] for x in current):continue
        rows,status=stage_sar_product(root,row,budget);print(json.dumps(status),flush=True)
        if status['status']!='SAR_ROIS_CALIBRATED':
            write(Path(root)/'data/raw/expansion_2023/sar_stop.json',status);break
