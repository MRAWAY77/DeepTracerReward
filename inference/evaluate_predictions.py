#!/usr/bin/env python3
"""Evaluate prediction JSONL with strict, paper-aligned deterministic metrics."""
import argparse, json, math, re, statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VALID_LABELS={"REAL","FAKE"}

def normalize_bbox(value,width,height):
    """Return clipped left/top/width/height or None for malformed boxes."""
    try:
        if isinstance(value,(list,tuple)) and len(value)==4:
            x0,y0,x1,y1=map(float,value)  # prompt specifies [x0,y0,x1,y1]
        elif isinstance(value,dict) and all(k in value for k in ("left","top","width","height")):
            x0,y0=float(value["left"]),float(value["top"]); x1=x0+float(value["width"]); y1=y0+float(value["height"])
        elif isinstance(value,dict) and all(k in value for k in ("x0","y0","x1","y1")):
            x0,y0,x1,y1=map(float,(value["x0"],value["y0"],value["x1"],value["y1"]))
        else:return None
        if not all(math.isfinite(x) for x in (x0,y0,x1,y1)):return None
        x0=max(0,min(float(width),x0));x1=max(0,min(float(width),x1));y0=max(0,min(float(height),y0));y1=max(0,min(float(height),y1))
        if x1<=x0 or y1<=y0:return None
        return {"left":x0,"top":y0,"width":x1-x0,"height":y1-y0}
    except (TypeError,ValueError,KeyError):return None

def iou(a,b):
    if not a or not b:return 0.0
    ax1,ay1=a["left"],a["top"];ax2,ay2=ax1+a["width"],ay1+a["height"];bx1,by1=b["left"],b["top"];bx2,by2=bx1+b["width"],by1+b["height"]
    inter=max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1));union=a["width"]*a["height"]+b["width"]*b["height"]-inter
    return inter/union if union>0 else 0.0

def bbox_distance(a,b,width,height):
    if not a or not b:return 1.0
    acx=(a["left"]+a["width"]/2)/width;acy=(a["top"]+a["height"]/2)/height;bcx=(b["left"]+b["width"]/2)/width;bcy=(b["top"]+b["height"]/2)/height
    return min(1.0,math.hypot(acx-bcx,acy-bcy)/math.sqrt(2))

def pct(num,den):return round(100*num/den,3) if den else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument("prediction_file");a=ap.parse_args()
    data=json.loads((ROOT/"all_data.json").read_text());gt=defaultdict(list)
    for row in data:gt[row["video_id"]].append(row)
    raw_preds=[json.loads(x) for x in Path(a.prediction_file).read_text().splitlines() if x.strip()]
    recovered=0
    for p in raw_preds:
        raw=str(p.get("raw_output","")).strip().upper().rstrip(".! ")
        if p.get("error") and raw in {"TRUE","FALSE"}:
            p["predicted_label"]="FAKE" if raw=="TRUE" else "REAL";p.pop("error",None);recovered+=1
    predictions={p.get("video_id"):p for p in raw_preds if p.get("video_id") in gt}
    duplicate_rows=len(raw_preds)-len({p.get("video_id") for p in raw_preds})
    correct=fake_correct=real_correct=0;fake_n=real_n=0;errors=invalid_labels=0;labels=Counter();ious=[];bdists=[];tdists=[];valid_boxes=valid_times=0;trace_count=0
    for vid,p in predictions.items():
        rows=gt[vid];truth="FAKE" if rows[0]["video_type"]=="fake_video" else "REAL";label=str(p.get("predicted_label","")).strip().upper();labels[label or "<missing>"]+=1
        errors+=bool(p.get("error"));invalid_labels+=label not in VALID_LABELS;correct+=label==truth
        if truth=="FAKE":fake_n+=1;fake_correct+=label==truth
        else:real_n+=1;real_correct+=label==truth
        if truth!="FAKE":continue
        width=float(rows[0]["video_width"]);height=float(rows[0]["video_height"]);is_localizer=p.get("task")=="localization";pred_box=normalize_bbox(p.get("bbox"),width,height) if label=="FAKE" or is_localizer else None
        start=p.get("start_time") if label=="FAKE" or is_localizer else None
        try:start=float(start) if start is not None and math.isfinite(float(start)) else None
        except (TypeError,ValueError):start=None
        valid_boxes+=pred_box is not None;valid_times+=start is not None
        for row in rows:  # paper test set contains 440 fake trace annotations
            trace_count+=1;truth_box=normalize_bbox(row.get("faketrace_bbox"),width,height);ious.append(iou(pred_box,truth_box));bdists.append(bbox_distance(pred_box,truth_box,width,height))
            truth_time=row["faketrace_first_frame"]/row["video_frame_rate"]
            tdists.append(min(1.0,abs(start-truth_time)/row["video_duration"]) if start is not None else 1.0)
    n=len(predictions)
    trivial=re.compile(r"^(?:true|false|real|fake|real or fake|the answer is:?\s*(?:real|fake))\.?$",re.I)
    response_by_label=Counter();substantive_by_label=Counter();word_counts=[];warning_types=Counter()
    for p in predictions.values():
        label=str(p.get("predicted_label") or "INVALID").upper();label=label if label in VALID_LABELS else "INVALID"
        explanation=str(p.get("explanation") or "").strip()
        if explanation:
            response_by_label[label]+=1;words=len(explanation.split());word_counts.append(words)
            if words>=4 and not trivial.fullmatch(explanation):substantive_by_label[label]+=1
        for warning in str(p.get("format_warning") or "").split(";"):
            if warning.strip():warning_types[warning.strip()]+=1
    out={
      "coverage":{"jsonl_rows":len(raw_preds),"unique_matched_videos":n,"duplicate_rows":duplicate_rows,"fake_videos":fake_n,"real_videos":real_n,"fake_trace_annotations":trace_count},
      "output_quality":{"runtime_error_rows":errors,"invalid_label_rows":invalid_labels,"recovered_boolean_rows":recovered,"valid_bbox_fake_videos":valid_boxes,"valid_time_fake_videos":valid_times,"predicted_labels":dict(labels),"response_text_rows":sum(response_by_label.values()),"response_text_by_predicted_label":dict(response_by_label),"substantive_looking_response_rows":sum(substantive_by_label.values()),"substantive_looking_by_predicted_label":dict(substantive_by_label),"mean_explanation_words":round(statistics.mean(word_counts),2) if word_counts else None,"median_explanation_words":round(statistics.median(word_counts),2) if word_counts else None,"format_warning_rows":sum(bool(p.get("format_warning")) for p in predictions.values()),"format_warning_types":dict(warning_types)},
      "paper_scale_0_100":{"accuracy":pct(correct,n),"fake_accuracy":pct(fake_correct,fake_n),"real_accuracy":pct(real_correct,real_n),"bbox_iou":round(100*sum(ious)/len(ious),3) if ious else None,"bbox_distance":round(100*sum(bdists)/len(bdists),3) if bdists else None,"time_distance":round(100*sum(tdists)/len(tdists),3) if tdists else None},
      "scoring_policy":"Malformed/missing labels are incorrect. Missing or invalid boxes/times receive worst-case IoU=0 and normalized distance=100. Track-B localization rows score produced boxes/times independently of the image-level REAL/FAKE decision; Track-C rows require FAKE. List boxes are interpreted as [x0,y0,x1,y1].",
      "note":"Response-text and substantive-looking counts measure availability/format only, not factual correctness. Explanation score is omitted because reproducing the paper requires its GPT-4.1 judge or a human audit."
    }
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
