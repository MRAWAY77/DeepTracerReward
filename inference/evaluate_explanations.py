#!/usr/bin/env python3
"""Compare model explanations with DeepTraceReward written test references."""
import argparse,json
from collections import defaultdict
from pathlib import Path
from explanation_metrics import evaluate_explanations

ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("prediction_file");ap.add_argument("--output",help="optional summary JSON path");args=ap.parse_args()
    gt=defaultdict(list)
    for row in json.loads((ROOT/"all_data.json").read_text(encoding="utf-8")):
        if row.get("split")=="test":gt[row["video_id"]].append(row)
    predictions={}
    for line in Path(args.prediction_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row=json.loads(line);predictions[row.get("video_id")]=row
    result=evaluate_explanations(predictions,gt);text=json.dumps(result,indent=2);print(text)
    if args.output:Path(args.output).write_text(text+"\n",encoding="utf-8")
if __name__=="__main__":main()
