#!/usr/bin/env python3
"""Run compact Track-C video VLMs over DeepTraceReward into visualizer JSONL."""
import argparse,json,math,os,re,subprocess,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROMPT_VERSION="deeptrace-v2-no-semantic-example"
MODELS={
 "smolvlm2-500m":{"repo":"HuggingFaceTB/SmolVLM2-500M-Video-Instruct","adapter":"pipeline","status":"negative-control"},
 "qwen2.5-vl-3b":{"repo":"Qwen/Qwen2.5-VL-3B-Instruct","adapter":"qwen","status":"recommended"},
 "videollama3-2b":{"repo":"DAMO-NLP-SG/VideoLLaMA3-2B","adapter":"native-video","status":"recommended"},
 "marlin-2b":{"repo":"NemoStation/Marlin-2B","adapter":"marlin","status":"recommended-gated"},
}
PROMPT="""Inspect the complete video for visible evidence of AI generation.
Classify it as FAKE only when you identify a concrete visible anomaly; otherwise classify it as REAL.
Check object distortion, sudden blur, unnatural trajectories, extra objects, merging, splitting,
background deformation, disappearance, and unexpected movement.

Return exactly one JSON object. `label` must be exactly `REAL` or `FAKE`.
For FAKE, explain the specific object and anomaly, provide its first visible time in seconds,
its end time, and one bounding box in original-video pixels. For REAL, explain briefly why no
specific anomaly is visible and use null for time and bbox.
Use exactly these six keys: `label`, `fake_probability`, `start_time`, `end_time`,
`bbox`, and `explanation`. The `bbox` value must be null or an object containing only
`left`, `top`, `width`, and `height`. Do not repeat or describe these instructions in
the answer. Do not invent an anomaly that is not visibly present."""

def local_source(repo):
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(repo,local_files_only=True)
    except Exception:return repo

def extract_frames(path,count,duration,tmp):
    pattern=str(tmp/"frame_%03d.jpg");rate=count/max(float(duration),.001)
    subprocess.run(["ffmpeg","-loglevel","error","-i",str(path),"-vf",f"fps={rate:.8f}","-frames:v",str(count),"-q:v","3",pattern],check=True)
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in sorted(tmp.glob("frame_*.jpg"))]

def constrain_pixels(images,max_pixels):
    """Resize in place to cap visual tokens while preserving aspect ratio."""
    from PIL import Image
    resized=[]
    for image in images:
        pixels=image.width*image.height
        if pixels>max_pixels:
            scale=math.sqrt(max_pixels/pixels);size=(max(28,int(image.width*scale)//28*28),max(28,int(image.height*scale)//28*28))
            image=image.resize(size,Image.Resampling.LANCZOS)
        resized.append(image)
    return resized

def parse_output(text):
    cleaned=re.sub(r"^\s*<think>.*?</think>\s*","",text,flags=re.S)
    match=re.search(r"\{.*\}",cleaned,re.S)
    if match:
        try:out=json.loads(match.group())
        except json.JSONDecodeError as e:return {"raw_output":text,"error":f"invalid JSON: {e}"}
        label=str(out.get("label","")).strip().upper()
        probability=out.get("fake_probability");start=out.get("start_time");end=out.get("end_time");bbox=out.get("bbox")
        if isinstance(bbox,dict) and not any(v is not None for v in bbox.values()):bbox=None
        warnings=[]
        if label not in {"REAL","FAKE"}:warnings.append("label is not exactly REAL or FAKE")
        if label=="REAL" and any(v is not None for v in (start,end,bbox)):warnings.append("REAL prediction contains fake localization")
        if label=="FAKE" and any(v is None for v in (start,end,bbox)):warnings.append("FAKE prediction is missing required localization")
        explanation=str(out.get("explanation") or "")
        if label=="FAKE" and re.search(r"\bno (?:visible |specific |significant )?(?:anomal(?:y|ies)|distortions?|artifacts?)\b",explanation,re.I):warnings.append("FAKE explanation claims no anomaly")
        try:
            p=float(probability)
            if (label=="REAL" and p>.5) or (label=="FAKE" and p<.5):warnings.append("fake_probability contradicts label")
        except (TypeError,ValueError):pass
        return {"predicted_label":label,"fake_probability":probability,"start_time":start,"end_time":end,"bbox":bbox,"explanation":explanation,"raw_output":text,"format_warning":"; ".join(warnings) or None}
    upper=cleaned.upper().strip();label="FAKE" if re.search(r"\bFAKE\b|AI[- ]GENERATED|^TRUE[.!]?$",upper) else "REAL" if re.search(r"\bREAL\b|^FALSE[.!]?$",upper) else ""
    if label:return {"predicted_label":label,"fake_probability":None,"start_time":None,"end_time":None,"bbox":None,"explanation":cleaned.strip(),"raw_output":text,"format_warning":"plain text response instead of JSON"}
    return {"raw_output":text,"error":"no valid JSON or recognizable REAL/FAKE label"}

def load_pipeline(spec,args,torch):
    from transformers import pipeline
    pipe=pipeline("image-text-to-text",model=local_source(spec["repo"]),device_map="auto",dtype=torch.float16)
    def infer(sample,images,path):
        content=[{"type":"image","image":im} for im in images]+[{"type":"text","text":sample_prompt(sample)}]
        result=pipe([{"role":"user","content":content}],generate_kwargs={"max_new_tokens":args.max_new_tokens,"do_sample":False})[0].get("generated_text","")
        return result[-1].get("content","") if isinstance(result,list) else str(result)
    return infer

def load_qwen(spec,args,torch):
    from transformers import AutoProcessor,BitsAndBytesConfig,Qwen2_5_VLForConditionalGeneration
    source=local_source(spec["repo"]);kwargs={"device_map":"auto"}
    if args.quantization=="4bit":kwargs.update(dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    else:kwargs["dtype"]=torch.bfloat16
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(source,**kwargs)
    processor=AutoProcessor.from_pretrained(source,min_pixels=4*28*28,max_pixels=args.max_pixels)
    def infer(sample,images,path):
        content=[]
        for i,image in enumerate(images):
            t=(i+.5)*float(sample["video_duration"])/len(images);content.extend([{"type":"text","text":f"Frame sampled at approximately {t:.2f} seconds:"},{"type":"image","image":image}])
        content.append({"type":"text","text":sample_prompt(sample)});messages=[{"role":"user","content":content}]
        chat=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);inputs=processor(text=[chat],images=images,padding=True,return_tensors="pt").to(model.device)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
        trimmed=out[:,inputs.input_ids.shape[1]:];return processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    return infer

def load_native_video(spec,args,torch):
    from transformers import AutoModelForCausalLM,AutoProcessor,BitsAndBytesConfig
    # VideoLLaMA3's remote processor imports VideoInput from the Transformers
    # 4.x location, while Transformers 5.x (required by Marlin) moved it to
    # video_utils. Provide the removed alias without modifying site-packages.
    if spec["repo"].startswith("DAMO-NLP-SG/VideoLLaMA3"):
        import transformers.image_utils as image_utils
        from transformers.video_utils import VideoInput
        if not hasattr(image_utils,"VideoInput"):image_utils.VideoInput=VideoInput
    # Keep the repository ID for remote-code models. A weights-complete local
    # snapshot can still be missing Python modules referenced by the processor;
    # Transformers will reuse cached weights and fetch only missing code files.
    source=spec["repo"]
    model_kwargs={"trust_remote_code":True}
    if args.quantization=="4bit":
        model_kwargs["device_map"]="auto"
        model_kwargs.update(dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    else:model_kwargs.update(dtype=torch.bfloat16,device_map="auto")
    model=AutoModelForCausalLM.from_pretrained(source,**model_kwargs)
    if spec["repo"].startswith("DAMO-NLP-SG/VideoLLaMA3"):
        # ProcessorMixin 5.x added a positional processor_dict argument. The
        # checkpoint's 4.x remote class overrides the old signature, so wrap
        # it locally instead of downgrading the shared environment.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import sys
        from typing import TypedDict
        processor_cls=get_class_from_dynamic_module("processing_videollama3.Videollama3Qwen2Processor",source,trust_remote_code=True)
        kwargs_cls=sys.modules[processor_cls.__module__].Videollama3Qwen2ProcessorKwargs
        if "common_kwargs" not in kwargs_cls.__annotations__:
            class EmptyCommonKwargs(TypedDict,total=False):pass
            kwargs_cls.__annotations__["common_kwargs"]=EmptyCommonKwargs
        original=processor_cls._get_arguments_from_pretrained.__func__
        def compatible_arguments(cls,pretrained_model_name_or_path,processor_dict=None,**kwargs):
            return original(cls,pretrained_model_name_or_path,**kwargs)
        processor_cls._get_arguments_from_pretrained=classmethod(compatible_arguments)
        processor=processor_cls.from_pretrained(source,trust_remote_code=True)
    else:
        processor=AutoProcessor.from_pretrained(source,trust_remote_code=True)
    def infer(sample,images,path):
        fps=args.frames/max(float(sample["video_duration"]),.001)
        if spec["repo"].startswith("DAMO-NLP-SG/VideoLLaMA3"):
            # Pre-sample before tokenization. The checkpoint's legacy native
            # video reader can ignore the intended frame cap under newer deps.
            video_content={"type":"video","video":images,"num_frames":len(images),"timestamps":[(i+.5)/fps for i in range(len(images))]}
            conversation=[{"role":"system","content":"You are a precise video-forensics analyst."},{"role":"user","content":[video_content,{"type":"text","text":sample_prompt(sample)}]}]
            inputs=processor(conversation=conversation,images=[("video",images)],return_tensors="pt")
        else:
            conversation=[{"role":"system","content":"You are a precise video-forensics analyst."},{"role":"user","content":[{"type":"video","video":{"video_path":str(path),"fps":fps,"max_frames":args.frames}},{"type":"text","text":sample_prompt(sample)}]}]
            inputs=processor(conversation=conversation,return_tensors="pt")
        inputs={k:v.to(model.device) if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
        if "pixel_values" in inputs:inputs["pixel_values"]=inputs["pixel_values"].to(torch.float16 if args.quantization=="4bit" else torch.bfloat16)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
        text=processor.batch_decode(out,skip_special_tokens=True)[0].strip();return text
    return infer

def load_marlin(spec,args,torch):
    from transformers import AutoModelForCausalLM,AutoProcessor,BitsAndBytesConfig
    os.environ["FPS_MAX_FRAMES"]=str(args.frames);os.environ["FPS_MIN_FRAMES"]=str(min(4,args.frames));os.environ["FPS"]="1.0"
    source=spec["repo"];model_kwargs={"trust_remote_code":True,"device_map":{"":"cuda"}}
    if args.quantization=="4bit":model_kwargs.update(dtype=torch.float16,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=True))
    else:model_kwargs["dtype"]=torch.bfloat16
    model=AutoModelForCausalLM.from_pretrained(source,**model_kwargs);processor=AutoProcessor.from_pretrained(source,trust_remote_code=True)
    def infer(sample,images,path):
        messages=[{"role":"user","content":[{"type":"video","video":str(path)},{"type":"text","text":sample_prompt(sample)}]}]
        inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt",return_dict=True).to(model.device)
        with torch.inference_mode():out=model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
        out=out[:,inputs["input_ids"].shape[1]:];return processor.batch_decode(out,skip_special_tokens=True)[0]
    return infer

def sample_prompt(s):return PROMPT+f"\nVideo dimensions: {int(s['video_width'])}x{int(s['video_height'])}. Duration: {float(s['video_duration']):.3f} seconds."

def build_adapter(spec,args,torch):
    return {"pipeline":load_pipeline,"qwen":load_qwen,"native-video":load_native_video,"marlin":load_marlin}[spec["adapter"]](spec,args,torch)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--model",choices=MODELS,required=True);ap.add_argument("--split",default="test");ap.add_argument("--limit",type=int);ap.add_argument("--frames",type=int,default=8);ap.add_argument("--max-new-tokens",type=int,default=320);ap.add_argument("--max-pixels",type=int,default=256*28*28,help="maximum pixels per sampled frame (default: 200704)");ap.add_argument("--quantization",choices=["4bit","bf16"],default="4bit");ap.add_argument("--output");ap.add_argument("--resume",action="store_true");args=ap.parse_args()
    import torch
    from tqdm.auto import tqdm
    spec=MODELS[args.model];data=json.loads((ROOT/"all_data.json").read_text());unique={}
    for row in data:
        if row["split"]==args.split:unique.setdefault(row["video_id"],row)
    values=list(unique.values());fake=[x for x in values if x["video_type"]=="fake_video"];real=[x for x in values if x["video_type"]=="real_video"];samples=[x for pair in zip(fake,real) for x in pair]
    if args.limit is not None:samples=samples[:args.limit]
    output=Path(args.output or ROOT/"predictions"/f"{args.model}-{args.split}.jsonl");output.parent.mkdir(exist_ok=True);done=set()
    if args.resume and output.exists():done={json.loads(x)["video_id"] for x in output.read_text().splitlines() if x.strip()}
    print(f"Loading {args.model} ({spec['repo']}) with {spec['adapter']} adapter");infer=build_adapter(spec,args,torch)
    with output.open("a" if args.resume else "w",encoding="utf-8") as f:
        progress=tqdm(enumerate(samples,1),total=len(samples),desc=f"{args.model} {args.split}",unit="video",dynamic_ncols=True)
        for n,s in progress:
            if s["video_id"] in done:continue
            actual_quantization=args.quantization if spec["adapter"] in {"qwen","native-video","marlin"} else "fp16"
            base={"video_id":s["video_id"],"model_id":args.model,"model_repo":spec["repo"],"task":"reasoning","prompt_version":PROMPT_VERSION,"sampled_frames":args.frames,"adapter":spec["adapter"],"quantization":actual_quantization};started=time.perf_counter()
            try:
                with tempfile.TemporaryDirectory() as td:
                    images=extract_frames(ROOT/s["video_path"],args.frames,s["video_duration"],Path(td));pixel_budget=args.max_pixels
                    if args.model=="videollama3-2b":pixel_budget=min(pixel_budget,max(28*28,int(4*100352/max(1,args.frames))))
                    images=constrain_pixels(images,pixel_budget);base["max_pixels_per_frame"]=pixel_budget;base["input_frame_size"]=[images[0].width,images[0].height] if images else None;base.update(parse_output(infer(s,images,ROOT/s["video_path"])))
            except Exception as e:base["error"]=f"{type(e).__name__}: {e}"
            base["latency_seconds"]=round(time.perf_counter()-started,3);f.write(json.dumps(base,ensure_ascii=False)+"\n");f.flush()
            result=base.get("predicted_label",base.get("error"));progress.set_postfix_str(str(result)[:48]);tqdm.write(f"[{n}/{len(samples)}] {s['video_id']} {result}")
    print(f"Saved {output}")
if __name__=="__main__":main()
