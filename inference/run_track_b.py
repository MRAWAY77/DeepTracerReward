#!/usr/bin/env python3
"""Resume-safe frame extraction and Track-B adapter orchestration."""
import argparse,fcntl,json,subprocess,sys,tempfile,time
from pathlib import Path
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
MODELS = {
 "trufor": {"checkpoint": "third_party/track_b/TruFor/TruFor_train_test/weights/trufor.pth.tar"},
 "iml-vit": {"checkpoint": "third_party/track_b/IML-ViT/checkpoints/iml-vit_checkpoint.pth"},
}
def frame(path,t,out):subprocess.run(["ffmpeg","-loglevel","error","-ss",f"{t:.5f}","-i",str(path),"-frames:v","1","-q:v","2","-y",str(out)],check=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--model",choices=MODELS,required=True);ap.add_argument("--split",default="test");ap.add_argument("--frames",type=int,default=2,help="frames per video; 2 is the validated 6 GB default");ap.add_argument("--threshold",type=float,default=.5);ap.add_argument("--limit",type=int);ap.add_argument("--output");ap.add_argument("--checkpoint");ap.add_argument("--resume",action="store_true");ap.add_argument("--retry-errors",action="store_true",help="with --resume, remove failed rows and recompute them");a=ap.parse_args()
 import torch
 if not torch.cuda.is_available():raise SystemExit("CUDA is unavailable. Check `nvidia-smi` and reboot/reload the NVIDIA driver before Track-B inference; CPU fallback is intentionally disabled.")
 lock=open("/tmp/deeptrace-track-b-gpu.lock","w")
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:raise SystemExit("Another Track-B run owns the GPU lock. Finish or stop it before starting this model.")
 spec=MODELS[a.model];ckpt=Path(a.checkpoint or ROOT/spec["checkpoint"])
 if not ckpt.is_file():raise SystemExit(f"Missing official checkpoint: {ckpt}\nSee inference/TRACK_B.md; do not start the full run before a pilot.")
 data=json.loads((ROOT/"all_data.json").read_text());unique={}
 for r in data:
  if r["split"]==a.split:unique.setdefault(r["video_id"],r)
 vals=list(unique.values());fake=[x for x in vals if x["video_type"]=="fake_video"];real=[x for x in vals if x["video_type"]=="real_video"];samples=[x for z in zip(fake,real) for x in z]
 if a.limit is not None:samples=samples[:a.limit]
 output=Path(a.output or ROOT/"predictions"/f"{a.model}-track-b-{a.split}.jsonl");done=set()
 if a.resume and output.exists():
  existing=[json.loads(x) for x in output.read_text().splitlines() if x.strip()]
  if a.retry_errors:
   existing=[x for x in existing if not x.get("error")];output.write_text("".join(json.dumps(x)+"\n" for x in existing),encoding="utf-8")
  done={x["video_id"] for x in existing}
 output.parent.mkdir(exist_ok=True)
 with output.open("a" if a.resume else "w",encoding="utf-8") as dst:
  for n,s in tqdm(enumerate(samples,1),total=len(samples),desc=f"{a.model} Track B",unit="video",dynamic_ncols=True):
   if s["video_id"] in done:continue
   started=time.perf_counter();base={"video_id":s["video_id"],"model_id":a.model,"task":"localization","sampled_frames":a.frames,"threshold":a.threshold,"checkpoint":ckpt.name}
   try:
    with tempfile.TemporaryDirectory() as td:
     td=Path(td);duration=float(s["video_duration"]);times=[duration*(i+.5)/a.frames for i in range(a.frames)];paths=[]
     for i,t in enumerate(times):p=td/f"frame_{i:03d}.jpg";frame(ROOT/s["video_path"],t,p);paths.append(p)
     proc=subprocess.run([sys.executable,str(ROOT/"inference"/"track_b_worker.py"),"--model",a.model,"--checkpoint",str(ckpt),"--threshold",str(a.threshold),"--width",str(int(float(s["video_width"]))),"--height",str(int(float(s["video_height"]))),*map(str,paths)],capture_output=True,text=True,check=True)
     result=json.loads(proc.stdout.strip().splitlines()[-1]);idx=int(result.pop("best_frame_index"));base.update(result);base["start_time"]=times[idx];base["end_time"]=times[idx];base["sample_times"]=times
   except subprocess.CalledProcessError as e:base["error"]=f"worker failed ({e.returncode}): {(e.stderr or e.stdout or str(e)).strip()[-1200:]}"
   except Exception as e:base["error"]=f"{type(e).__name__}: {e}"
   base["latency_seconds"]=round(time.perf_counter()-started,3);dst.write(json.dumps(base)+"\n");dst.flush();tqdm.write(f"[{n}/{len(samples)}] {s['video_id']} {base.get('predicted_label',base.get('error'))}")
 print(f"Saved {output}")
if __name__=="__main__":main()
