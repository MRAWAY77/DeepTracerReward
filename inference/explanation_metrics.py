"""Lightweight reference-based explanation metrics shared by CLI and visualizer."""
from __future__ import annotations
import re
from collections import defaultdict

STOPWORDS=set("a an and are as at be because been but by can could did do does for from had has have he her here him his i if in into is it its may more most no not of on or our she so than that the their them there these they this to was were what when where which who will with would you your video clip scene appears appear shows show showing seems seem looks look due during around".split())
CATEGORY_RULES={
 "Object Merging":r"merg|fuse|blend|overlap|intersect",
 "Object Splitting":r"split|divid|duplicat.*part",
 "Object Disappearance":r"disappear|vanish|missing|fade out",
 "Redundant Object":r"extra (?:object|limb|finger|person)|redundant|additional object|sudden(?:ly)? appear",
 "Sudden Blurring":r"blur|smear|fuzz",
 "Background Distortion":r"background.*(?:distort|warp|morph|flicker)|(?:distort|warp).*background",
 "Unexpected Move":r"unexpected (?:move|motion)|moves? (?:unnaturally|unexpectedly)|still (?:thing|object).*mov",
 "Object Trajectory":r"trajector|motion path|movement path|jerk|unsmooth|inconsistent motion|temporal inconsist",
 "Object Distortion":r"distort|deform|warp|morph|malform|anatom|shape chang|unnatural shape",
}

def tokens(text): return re.findall(r"[a-z0-9]+",str(text or "").lower())
def content_tokens(text): return [x for x in tokens(text) if x not in STOPWORDS and len(x)>2]
def prf(pred,ref):
    p,r=set(pred),set(ref); hit=len(p&r)
    precision=hit/len(p) if p else 0.0; recall=hit/len(r) if r else 0.0
    return precision,recall,2*precision*recall/(precision+recall) if precision+recall else 0.0
def rouge_l(pred,ref):
    a,b=tokens(pred),tokens(ref)
    if not a or not b:return 0.0
    row=[0]*(len(b)+1)
    for x in a:
        previous=0
        for j,y in enumerate(b,1):
            old=row[j];row[j]=previous+1 if x==y else max(row[j],row[j-1]);previous=old
    lcs=row[-1];precision=lcs/len(a);recall=lcs/len(b)
    return 2*precision*recall/(precision+recall) if precision+recall else 0.0
def categories(text): return {name for name,pattern in CATEGORY_RULES.items() if re.search(pattern,str(text or "").lower())}
def pair_scores(pred,ref):
    _,_,token_f1=prf(tokens(pred),tokens(ref));_,_,content_f1=prf(content_tokens(pred),content_tokens(ref));_,_,category_f1=prf(categories(pred),categories(ref))
    return {"rouge_l_f1":rouge_l(pred,ref),"token_f1":token_f1,"content_f1":content_f1,"artifact_category_f1":category_f1}

def evaluate_explanations(predictions,gt_by_video):
    """Return 0–100 aggregates. Missing predictions score zero, not silently vanish."""
    fake_rows=[r for rows in gt_by_video.values() for r in rows if r.get("video_type")=="fake_video"]
    fake_videos={r["video_id"] for r in fake_rows}; refs=[r for r in fake_rows if str(r.get("faketrace_explanations") or "").strip()]
    evaluable_videos={r["video_id"] for r in refs}; by_video=defaultdict(list); response_videos=set(); pairs=[]
    for r in refs:
        p=predictions.get(r["video_id"],{}); response=str(p.get("explanation") or "").strip()
        if response:response_videos.add(r["video_id"])
        score=pair_scores(response,r["faketrace_explanations"]);pairs.append(score);by_video[r["video_id"]].append(score)
    keys=("rouge_l_f1","token_f1","content_f1","artifact_category_f1")
    avg=lambda xs,k:sum(x[k] for x in xs)/len(xs) if xs else 0.0
    annotation={k:avg(pairs,k) for k in keys}
    video_mean={k:sum(avg(v,k) for v in by_video.values())/len(by_video) if by_video else 0.0 for k in keys}
    video_best={k:sum(max(x[k] for x in v) for v in by_video.values())/len(by_video) if by_video else 0.0 for k in keys}
    scale=lambda d:{k:round(100*v,2) for k,v in d.items()}
    return {"reference_coverage":{"fake_trace_annotations":len(fake_rows),"written_reference_annotations":len(refs),"fake_videos":len(fake_videos),"evaluable_fake_videos":len(evaluable_videos),"prediction_response_videos":len(response_videos)},"annotation_macro":scale(annotation),"video_all_trace_mean":scale(video_mean),"video_oracle_best_trace":scale(video_best),"metric_note":"Automatic overlap metrics are diagnostics, not a substitute for the paper's 0/0.5/1 GPT-4.1 or human correctness rubric. Oracle-best asks whether any one reference trace was matched; all-trace mean rewards coverage of every documented trace."}
