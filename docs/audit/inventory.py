import ast, csv, hashlib, json, os, pathlib, re, collections
import numpy as np

PAT = re.compile(r'random|dummy|fallback|hardcod|mock|fabricat|nan_to_num|windows\[:5\]|labels\[-6|\[\-6:|0\.75 \+|climatology_c|def |class |assert |except |TODO|REPLACE_ME', re.I)

def pairs(items):
    obj = {}
    for k, v in items:
        if k in obj: raise ValueError('duplicate key '+k)
        obj[k] = v
    return obj

def bad(x): raise ValueError('nonfinite JSON constant '+x)

def inspect(root):
    rows=[]
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d != '.git')
        for name in sorted(files):
            p=pathlib.Path(directory)/name; rel=p.relative_to(root).as_posix()
            r={'path':rel,'size_bytes':p.lstat().st_size}
            if p.is_symlink():
                r['inspection']='symlink_not_followed'; rows.append(r);continue
            raw=p.read_bytes();r['sha256']=hashlib.sha256(raw).hexdigest()
            if p.suffix=='.npz':
                r['inspection']='numeric_array_inspection_no_pickle';r['arrays']={}
                try:
                    with np.load(p,allow_pickle=False) as z:
                        for key in z.files:
                            try:
                                a=z[key]; stats={'shape':list(a.shape),'dtype':str(a.dtype)}
                                if np.issubdtype(a.dtype,np.number):
                                    finite=np.isfinite(a); valid=a[finite]
                                    stats.update(finite_fraction=float(finite.mean()),nonfinite_count=int((~finite).sum()))
                                    if valid.size: stats.update(min=float(valid.min()),max=float(valid.max()),mean=float(valid.mean()))
                                elif a.dtype.kind in 'US' and a.size: stats.update(first=str(a.flat[0]),last=str(a.flat[-1]))
                                r['arrays'][key]=stats
                            except ValueError as e:r['arrays'][key]={'not_loaded':str(e)}
                except Exception as e:r['error']=str(e)
            elif p.suffix in {'.pt','.pyc','.png','.jpg','.pdf','.zip','.nc','.gz'} or b'\0' in raw:
                r['inspection']='binary_hashed_not_executed'
            else:
                try: txt=raw.decode('utf-8');r['inspection']='full_text_read';r['line_count']=len(txt.splitlines())
                except UnicodeDecodeError:r['inspection']='binary_hashed_not_executed';rows.append(r);continue
                if p.suffix=='.py':
                    try:
                        tree=ast.parse(txt);r['ast_valid']=True
                        r['definitions']=[{'name':n.name,'line':n.lineno,'end_line':n.end_lineno} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
                    except SyntaxError as e:r['ast_valid']=False;r['parse_error']=str(e)
                    r['review_signals']=[{'line':i,'text':line[:260]} for i,line in enumerate(txt.splitlines(),1) if PAT.search(line)]
                elif p.suffix=='.json':
                    try:
                        j=json.loads(txt,object_pairs_hook=pairs,parse_constant=bad)
                        r['strict_json_valid']=True;r['top_keys']=list(j) if isinstance(j,dict) else None
                    except Exception as e:r['strict_json_valid']=False;r['parse_error']=str(e)
                elif p.suffix=='.csv':
                    try:
                        cr=list(csv.DictReader(txt.splitlines()));r['csv_rows']=len(cr);r['columns']=list(cr[0]) if cr else []
                        r['duplicate_dates']=len(cr)-len({v.get('date') for v in cr}) if cr and 'date' in cr[0] else None
                    except Exception as e:r['parse_error']=str(e)
                elif p.suffix=='.md':
                    r['headings']=[line for line in txt.splitlines() if line.startswith('#')]
            rows.append(r)
    return rows


if __name__ == "__main__":
    import argparse
    a=argparse.ArgumentParser();a.add_argument("root",type=pathlib.Path);a.add_argument("output",type=pathlib.Path);args=a.parse_args()
    rows=inspect(args.root.resolve());counts=collections.Counter(r["inspection"] for r in rows)
    result={"scope":"all regular files except Git internals; symlinks not followed; no pickle execution", "file_count":len(rows),"total_bytes":sum(r["size_bytes"] for r in rows),"inspection_counts":dict(counts),"files":rows}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
