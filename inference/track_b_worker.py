#!/usr/bin/env python3
"""Isolated model worker for the supported Track-B localizers.

The parent runner launches this file once per video. Keeping model imports in a
short-lived subprocess prevents memory retained by one localizer from affecting
the next sample or another model.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def summarize_maps(maps, threshold, width, height):
    """Convert anomaly maps into a label and strongest-frame bounding box."""
    scores = [float(np.asarray(item).mean()) for item in maps]
    best_index = int(np.argmax(scores))
    anomaly_map = np.asarray(maps[best_index], dtype=np.float32)
    mask = (anomaly_map >= threshold).astype("uint8")
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    bbox = None
    if component_count > 1:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, box_width, box_height, _ = stats[component]
        scale_x = width / anomaly_map.shape[1]
        scale_y = height / anomaly_map.shape[0]
        bbox = {
            "left": round(x * scale_x, 2),
            "top": round(y * scale_y, 2),
            "width": round(box_width * scale_x, 2),
            "height": round(box_height * scale_y, 2),
        }

    return {
        "predicted_label": "FAKE" if scores[best_index] >= threshold else "REAL",
        "fake_probability": scores[best_index],
        "bbox": bbox,
        "best_frame_index": best_index,
        "frame_scores": scores,
    }


def run_trufor(args):
    repo = ROOT / "third_party/track_b/TruFor/TruFor_train_test"
    sys.path[:0] = [str(repo), str(repo.parent)]
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-deeptrace")
    os.chdir(repo)

    from dataset.dataset_test import TestDataset
    from lib.config import config, update_config
    from lib.utils import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_args = argparse.Namespace(
        experiment="trufor_ph3",
        opts=["TEST.MODEL_FILE", str(args.checkpoint)],
    )
    update_config(config, config_args)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = get_model(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    maps = []
    loader = torch.utils.data.DataLoader(TestDataset(list_img=args.images), batch_size=1)
    with torch.inference_mode():
        for rgb, _ in loader:
            rgb = rgb.to(device)
            prediction, confidence, detection, noiseprint = model(rgb)
            maps.append(torch.softmax(prediction.squeeze(0), 0)[1].float().cpu().numpy())
            del rgb, prediction, confidence, detection, noiseprint
            if device == "cuda":
                torch.cuda.empty_cache()
    return maps


def run_iml_vit(args):
    from PIL import Image
    from torchvision.transforms import functional as transforms

    repo = ROOT / "third_party/track_b/IML-ViT"
    sys.path.insert(0, str(repo))
    from iml_vit_model import iml_vit_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = iml_vit_model()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.to(device).eval()

    maps = []
    with torch.inference_mode():
        for image_path in args.images:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((1024, 1024))
            tensor = transforms.to_tensor(image)
            tensor = transforms.normalize(tensor, [.485, .456, .406], [.229, .224, .225])
            height, width = tensor.shape[-2:]
            tensor = transforms.pad(tensor, [0, 0, 1024 - width, 1024 - height])
            tensor = tensor.unsqueeze(0).to(device)
            dummy = torch.zeros((1, 1, 1024, 1024), device=device)
            loss, prediction, edge = model(tensor, dummy, dummy)
            maps.append(prediction[0, 0, :height, :width].float().cpu().numpy())
            del tensor, dummy, loss, prediction, edge
            if device == "cuda":
                torch.cuda.empty_cache()
    return maps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("trufor", "iml-vit"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.images = [str(Path(path).resolve()) for path in args.images]

    maps = run_trufor(args) if args.model == "trufor" else run_iml_vit(args)
    print(json.dumps(summarize_maps(
        maps, args.threshold, args.width, args.height
    )))


if __name__ == "__main__":
    main()
