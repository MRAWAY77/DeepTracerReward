#!/usr/bin/env python3
"""Dependency-free DeepTraceReward dataset explorer."""
from __future__ import annotations
import argparse, errno, json, math, mimetypes, re
from collections import Counter, defaultdict
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from inference.explanation_metrics import evaluate_explanations
from inference.evaluate_predictions import bbox_distance, iou, normalize_bbox

APP_ROOT = Path(__file__).resolve().parent
ROOT = APP_ROOT
STATIC = APP_ROOT / "static"


def json_safe(value):
    """Replace non-finite floats recursively so API JSON stays standards-compliant."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def create_server(host, preferred_port, attempts):
    """Bind to the requested port or the next available one."""
    if not 0 <= preferred_port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    if attempts < 1:
        raise SystemExit("--port-attempts must be at least 1")
    ports = [0] if preferred_port == 0 else range(preferred_port, min(65536, preferred_port + attempts))
    last_error = None
    for port in ports:
        try:
            server = ThreadingHTTPServer((host, port), Handler)
            return server, int(server.server_address[1])
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            last_error = exc
    final_port = min(65535, preferred_port + attempts - 1)
    raise SystemExit(f"No available port in {preferred_port}-{final_port} on {host}: {last_error}")

def paper_category(tags):
    text = " ".join(tags or []).lower(); out = []
    rules = [("Object Merging", r"merg|overlap"), ("Object Splitting", r"splitt"),
             ("Object Disappearance", r"disappear"), ("Redundant Object", r"appearance|redundant|extra object"),
             ("Sudden Blurring", r"blurr"), ("Background Distortion", r"background distortion"),
             ("Unexpected Move", r"unexpected move|still things"),
             ("Object Trajectory", r"weird motion|movement paths|trajectory|unsmooth motion"),
             ("Object Distortion", r"shape distortion|distortions \(object|deformation")]
    for name, pattern in rules:
        if re.search(pattern, text): out.append(name)
    return out

class Dataset:
    def __init__(self, path):
        if not path.is_file():
            raise SystemExit(
                f"Missing {path}. Download RewardData as documented in README.md."
            )
        raw = json.loads(path.read_text(encoding="utf-8")); self.rows=[]; self.by_video=defaultdict(list)
        for i, row in enumerate(raw):
            r=dict(row); r["_index"]=i; r["paper_categories"]=paper_category(r.get("faketrace_check_applies"))
            self.rows.append(r); self.by_video[r["video_id"]].append(r)
        self.videos=[]
        for annotations in self.by_video.values():
            base=dict(annotations[0]); base["annotation_count"]=len(annotations) if base["video_type"]=="fake_video" else 0
            base["has_explanation"]=any(bool((a.get("faketrace_explanations") or "").strip()) for a in annotations)
            base["paper_categories"]=sorted({c for a in annotations for c in a["paper_categories"]})
            base.pop("faketrace_bbox", None); self.videos.append(base)
        # The release JSON stores every fake annotation before every real row.
        # Interleave unique samples so the default gallery visibly represents
        # both halves of the balanced dataset from page one.
        fake=[v for v in self.videos if v["video_type"]=="fake_video"]
        real=[v for v in self.videos if v["video_type"]=="real_video"]
        self.videos=[v for pair in zip(fake,real) for v in pair]
    @staticmethod
    def public(row): return {k:v for k,v in row.items() if not k.startswith("_")}
    def query(self, p, allowed_ids=None):
        items=self.videos; q=p.get("q",[""])[0].strip().lower()
        if allowed_ids is not None: items=[x for x in items if x["video_id"] in allowed_ids]
        for key, field in [("type","video_type"),("source","video_source"),("split","split")]:
            val=p.get(key,["all"])[0]
            if val!="all": items=[x for x in items if x[field]==val]
        cat=p.get("category",["all"])[0]; exp=p.get("explanation",["all"])[0]
        if cat!="all": items=[x for x in items if cat in x["paper_categories"]]
        if exp=="yes": items=[x for x in items if x["has_explanation"]]
        if exp=="no": items=[x for x in items if not x["has_explanation"]]
        if q: items=[x for x in items if q in " ".join(map(str,[x.get("video_id",""),x.get("video_prompt",""),x.get("video_source","")])).lower()]
        total=len(items); breakdown=Counter(x["video_type"] for x in items)
        page=max(1,int(p.get("page",["1"])[0])); size=min(100,max(1,int(p.get("size",["24"])[0])))
        return items[(page-1)*size:page*size], total, {"fake":breakdown["fake_video"],"real":breakdown["real_video"]}
    def stats(self):
        fake=[r for r in self.rows if r["video_type"]=="fake_video"]; fv=[v for v in self.videos if v["video_type"]=="fake_video"]
        counts=lambda field,seq: dict(Counter(x.get(field,"Unknown") for x in seq).most_common())
        cats=Counter(c for r in fake for c in r["paper_categories"]); tagged=sum(bool(r.get("faketrace_check_applies")) for r in fake)
        explained=sum(bool((r.get("faketrace_explanations") or "").strip()) for r in fake); durations=[float(v.get("video_duration") or 0) for v in self.videos]
        ratios=[min(1,float(r.get("faketrace_duration") or 0)/float(r.get("video_duration") or 1)) for r in fake]
        bins=[("≤5s",0,5),("5–6s",5,6),("6–10s",6,10),("10–20s",10,20),(">20s",20,10**9)]
        return {"summary":{"rows":len(self.rows),"videos":len(self.videos),"fake_videos":len(fv),"real_videos":len(self.videos)-len(fv),"trace_annotations":len(fake),"explained_pct":round(100*explained/len(fake),1),"categorized_pct":round(100*tagged/len(fake),1),"avg_video_duration":round(sum(durations)/len(durations),2),"avg_trace_coverage_pct":round(100*sum(ratios)/len(ratios),1)},
                "source_annotations":counts("video_source",fake),"source_videos":counts("video_source",self.videos),"splits_videos":counts("split",self.videos),"categories":dict(cats.most_common()),
                "duration_bins":{name:sum((lo<d<=hi) if lo else d<=hi for d in durations) for name,lo,hi in bins},"annotations_per_fake_video":dict(sorted(Counter(v["annotation_count"] for v in fv).items())),
                "release_notes":{"uncategorized_annotations":len(fake)-tagged,"category_mapping":"Raw Labelbox tags mapped heuristically to the paper's nine-category taxonomy; categories are multi-label."}}

DATA=None
class Predictions:
    def __init__(self): self.signature=None; self.models={}; self.by_key={}
    def refresh(self):
        files=sorted((ROOT/"predictions").glob("*.jsonl")) if (ROOT/"predictions").exists() else []
        sig=tuple((str(p),p.stat().st_mtime_ns,p.stat().st_size) for p in files)
        if sig==self.signature: return
        self.signature=sig; self.models={}; self.by_key={}
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip(): continue
                try: row=json.loads(line)
                except json.JSONDecodeError: continue
                model=row.get("model_id",path.stem); vid=row.get("video_id")
                if not vid: continue
                self.models[model]={"model_id":model,"task":row.get("task","unknown"),"source_file":path.name}
                self.by_key[(model,vid)]=row
    def options(self): self.refresh(); return sorted(self.models.values(),key=lambda x:x["model_id"])
    @staticmethod
    def response_kind(row):
        if not row or row.get("error"): return "missing"
        response=str(row.get("explanation") or row.get("raw_output") or "").strip()
        if not response:return "missing"
        normalized=re.sub(r"[.!\s]+$","",response).strip().upper()
        typical=bool(re.fullmatch(r"(?:REAL|FAKE|TRUE|FALSE|REAL\s+OR\s+FAKE|THE ANSWER IS:?\s*(?:REAL|FAKE))",normalized))
        return "typical" if typical else "atypical"
    def get(self,model,vid):
        self.refresh(); row=self.by_key.get((model,vid))
        return dict(row,response_kind=self.response_kind(row)) if row else None
    def video_ids(self,model,kind):
        self.refresh(); return {vid for (m,vid),row in self.by_key.items() if m==model and self.response_kind(row)==kind}
    def performance(self,model):
        self.refresh(); rows=[p for (m,_),p in self.by_key.items() if m==model]; task=Counter(str(p.get("task") or "unknown") for p in rows).most_common(1)[0][0] if rows else "unknown"; labels=Counter(); confusion=Counter(); sources=defaultdict(lambda:[0,0]); latencies=[]; explanation_rows=0; substantive=0; errors=0; ious=[]; bdists=[]; tdists=[]; valid_boxes=0; valid_times=0; fake_scores=[]
        trivial=re.compile(r"^(true|false|real|fake|real or fake|the answer is:?\s*(real|fake)|\[?fake,?\s*0(?:\.0)?\]?)\.?$",re.I)
        for p in rows:
            gt_rows=DATA.by_video.get(p.get("video_id"),[])
            if not gt_rows: continue
            truth="FAKE" if gt_rows[0]["video_type"]=="fake_video" else "REAL"; pred=str(p.get("predicted_label") or "INVALID").upper(); pred=pred if pred in {"REAL","FAKE"} else "INVALID"
            labels[pred]+=1; confusion[(truth,pred)]+=1; source=gt_rows[0]["video_source"]; sources[source][1]+=1; sources[source][0]+=pred==truth; errors+=bool(p.get("error"))
            try:
                latency=float(p.get("latency_seconds"));
                if latency>=0: latencies.append(latency)
            except (TypeError,ValueError): pass
            explanation=str(p.get("explanation") or "").strip(); explanation_rows+=bool(explanation)
            substantive+=bool(explanation and len(explanation.split())>=4 and not trivial.match(explanation))
            try:
                score=float(p.get("fake_probability"));
                if math.isfinite(score):fake_scores.append(score)
            except (TypeError,ValueError):pass
            if task=="localization" and truth=="FAKE":
                width=float(gt_rows[0]["video_width"]);height=float(gt_rows[0]["video_height"]);pred_box=normalize_bbox(p.get("bbox"),width,height)
                try:start=float(p.get("start_time")) if p.get("start_time") is not None else None;start=start if start is not None and math.isfinite(start) else None
                except (TypeError,ValueError):start=None
                valid_boxes+=pred_box is not None;valid_times+=start is not None
                for gt in gt_rows:
                    truth_box=normalize_bbox(gt.get("faketrace_bbox"),width,height);ious.append(iou(pred_box,truth_box));bdists.append(bbox_distance(pred_box,truth_box,width,height));truth_time=gt["faketrace_first_frame"]/gt["video_frame_rate"];tdists.append(min(1,abs(start-truth_time)/gt["video_duration"]) if start is not None else 1)
        n=len(rows); fake_n=sum(v for (t,_),v in confusion.items() if t=="FAKE"); real_n=sum(v for (t,_),v in confusion.items() if t=="REAL"); correct=sum(v for (t,p),v in confusion.items() if t==p)
        test_gt=defaultdict(list)
        for vid,gt_rows in DATA.by_video.items():
            for gt in gt_rows:
                if gt.get("split")=="test":test_gt[vid].append(gt)
        explanation=evaluate_explanations({p.get("video_id"):p for p in rows},test_gt) if task!="localization" else None
        localization={"valid_box_fake_videos":valid_boxes,"valid_time_fake_videos":valid_times,"bbox_iou":round(100*sum(ious)/len(ious),3) if ious else None,"bbox_distance":round(100*sum(bdists)/len(bdists),3) if bdists else None,"time_distance":round(100*sum(tdists)/len(tdists),3) if tdists else None,"mean_fake_score":round(sum(fake_scores)/len(fake_scores),4) if fake_scores else None,"trace_comparisons":len(ious)} if task=="localization" else None
        return {"model_id":model,"task":task,"summary":{"videos":n,"accuracy":round(100*correct/n,3) if n else None,"fake_accuracy":round(100*confusion[("FAKE","FAKE")]/fake_n,3) if fake_n else None,"real_accuracy":round(100*confusion[("REAL","REAL")]/real_n,3) if real_n else None,"runtime_errors":errors,"response_text_coverage":round(100*explanation_rows/n,1) if n else 0,"substantive_explanation_coverage":round(100*substantive/n,1) if n else 0},"predicted_labels":dict(labels),"confusion":{f"{t} → {p}":v for (t,p),v in confusion.items()},"per_source":{k:{"correct":v[0],"total":v[1],"accuracy":round(100*v[0]/v[1],2)} for k,v in sources.items()},"latencies":latencies,"explanation_metrics":explanation,"localization_metrics":localization}
PREDICTIONS=Predictions()
class Handler(SimpleHTTPRequestHandler):
    def send_json(self,value,status=200):
        body=json.dumps(json_safe(value),ensure_ascii=False,allow_nan=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(body))); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        p=urlparse(self.path)
        if p.path=="/api/videos":
            params=parse_qs(p.query); model=params.get("model",[""])[0]; kind=params.get("response",["all"])[0]
            allowed=PREDICTIONS.video_ids(model,kind) if model and kind in {"typical","atypical","missing"} else None
            items,total,breakdown=DATA.query(params,allowed); return self.send_json({"items":[DATA.public(x) for x in items],"total":total,"breakdown":breakdown})
        if p.path.startswith("/api/video/"):
            vid=unquote(p.path.removeprefix("/api/video/")); rows=DATA.by_video.get(vid)
            return self.send_json({"video":DATA.public(rows[0]),"annotations":[DATA.public(x) for x in rows]}) if rows else self.send_json({"error":"not found"},404)
        if p.path=="/api/stats": return self.send_json(DATA.stats())
        if p.path=="/api/options": return self.send_json({"sources":sorted({x["video_source"] for x in DATA.videos}),"splits":sorted({x["split"] for x in DATA.videos}),"categories":sorted({c for x in DATA.videos for c in x["paper_categories"]})})
        if p.path=="/api/prediction-models": return self.send_json({"models":PREDICTIONS.options()})
        if p.path=="/api/model-performance":
            model=parse_qs(p.query).get("model",[""])[0]; return self.send_json(PREDICTIONS.performance(model))
        if p.path.startswith("/api/prediction/"):
            vid=unquote(p.path.removeprefix("/api/prediction/")); model=parse_qs(p.query).get("model",[""])[0]
            return self.send_json({"prediction":PREDICTIONS.get(model,vid)})
        if p.path.startswith("/media/"):
            target=(ROOT/Path(unquote(p.path.removeprefix("/media/")))).resolve()
            if ROOT not in target.parents or not target.is_file(): return self.send_error(404)
            return self.send_media(target)
        self.path="/index.html" if p.path=="/" else p.path; return super().do_GET()
    def translate_path(self,path):
        clean=urlparse(path).path.lstrip("/")
        base=ROOT if clean=="readme.html" or clean.startswith(("videos/","real_videos/","reports/")) else STATIC
        target=(base/clean).resolve()
        if target!=base and base not in target.parents:
            return str(STATIC/"__not_found__")
        return str(target)
    def send_media(self, target):
        size=target.stat().st_size; start,end=0,size-1; partial=False
        header=self.headers.get("Range","")
        match=re.match(r"bytes=(\d*)-(\d*)",header)
        if match:
            partial=True
            if match.group(1): start=int(match.group(1))
            if match.group(2): end=min(int(match.group(2)),size-1)
            if not match.group(1) and match.group(2): start=max(0,size-int(match.group(2)))
            if start>end or start>=size: return self.send_error(416)
        length=end-start+1; self.send_response(206 if partial else 200)
        self.send_header("Content-Type",mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Accept-Ranges","bytes"); self.send_header("Content-Length",str(length))
        if partial: self.send_header("Content-Range",f"bytes {start}-{end}/{size}")
        self.end_headers()
        with target.open("rb") as f:
            f.seek(start); remaining=length
            while remaining:
                chunk=f.read(min(1024*1024,remaining))
                if not chunk: break
                try: self.wfile.write(chunk)
                except (BrokenPipeError,ConnectionResetError): break
                remaining-=len(chunk)
    def log_message(self,fmt,*args): print(f"[{self.log_date_time_string()}] {fmt%args}")
def main():
    global ROOT, DATA, PREDICTIONS
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(APP_ROOT), help="RewardData root (default: app.py directory)")
    ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--port",type=int,default=8000,help="preferred port; use 0 for any free port")
    ap.add_argument("--port-attempts",type=int,default=100,help="consecutive ports to try (default: 100)")
    a=ap.parse_args()
    ROOT=Path(a.root).expanduser().resolve()
    DATA=Dataset(ROOT/"all_data.json")
    PREDICTIONS=Predictions()
    server,port=create_server(a.host,a.port,a.port_attempts)
    print(f"Loaded {len(DATA.videos):,} videos and {len(DATA.rows):,} rows")
    if port!=a.port and a.port!=0: print(f"Port {a.port} is occupied; using {port} instead")
    print(f"Open http://{a.host}:{port}")
    try:
        with server: server.serve_forever()
    except KeyboardInterrupt:
        print("\nExplorer stopped")
if __name__=="__main__": main()
