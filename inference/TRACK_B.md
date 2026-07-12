# Track B — spatial forensic localization

Track B samples frames uniformly across each video, runs an official image-manipulation
localizer on every frame, selects the strongest heatmap, converts its largest
connected component to an original-pixel bounding box, and writes resume-safe
JSONL understood by the existing evaluator and visualizer.

The parent runner uses one consolidated `track_b_worker.py` adapter and launches
it as an isolated subprocess per video. This keeps model-specific imports and GPU
memory out of the orchestration process while sharing one output schema.

## Final shortlist for the 6 GB laptop

1. **TruFor** — recommended first; broad RGB + Noiseprint++ anomaly localizer.
2. **IML-ViT** — ViT-B manipulation-localization baseline; run one-video pilot
   because its 1024-square input is close to the memory limit.
DeCLIP has been removed from this benchmark: its official release pins Python
3.10/PyTorch 2.2.2 and cannot share the required `deeptracereward` Python 3.14
stack. MVSS-Net is also excluded because its official environment is Python 3.6,
CUDA 10.1 and Apex. The final Track-B comparison therefore contains two valid,
reproducible pretrained localizers rather than padding the table with incompatible
or unvalidated results.

These models localize still-image manipulation fingerprints, not human-perceived
video-motion traces. Their results are out-of-domain transfer baselines.

The validated 6 GB default is two frames per video. Four or eight consecutive
full-resolution passes can terminate a worker at peak memory on this laptop.
Keep `--frames 2` identical for both active models and report sparse temporal
sampling as a benchmark limitation.

The runner refuses CPU fallback. If `nvidia-smi` cannot communicate with the
driver after a CUDA memory failure or laptop suspend, reboot/reload the NVIDIA
driver before rerunning; otherwise a full CPU run is impractically slow.

An exclusive GPU lock prevents TruFor and IML-ViT from running concurrently.
After an older concurrent run, retry failed rows sequentially with both
`--resume --retry-errors`; the runner removes only rows containing `error` before
recomputing them.

## Checkpoints

- TruFor: download the official `TruFor_weights.zip` from
  <https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip>. Its
  documented MD5 is `7bee48f3476c75616c3c5721ab256ff8`. Put
  `trufor.pth.tar` at
  `third_party/track_b/TruFor/TruFor_train_test/weights/trufor.pth.tar`.
- IML-ViT: download an official checkpoint from the Google Drive linked in
  `third_party/track_b/IML-ViT/README.md` and put it at
  `third_party/track_b/IML-ViT/checkpoints/iml-vit_checkpoint.pth`.
Install the two active adapters' small dependencies only after Track C is idle:

```bash
conda activate deeptracereward
python -m pip install albumentations==1.3.0 scikit-learn timm fvcore yacs
```

## Required pilot sequence

```bash
python inference/run_track_b.py --model trufor --split test --limit 2 \
  --output predictions/trufor-track-b-pilot2.jsonl
python inference/run_track_b.py --model iml-vit --split test --limit 2 \
  --output predictions/iml-vit-track-b-pilot2.jsonl
```

Inspect and evaluate each pilot before proceeding. Then run the complete test:

```bash
python inference/run_track_b.py --model trufor --split test \
  --output predictions/trufor-track-b-test.jsonl --resume
python inference/run_track_b.py --model iml-vit --split test \
  --output predictions/iml-vit-track-b-test.jsonl --resume
```

If the evaluation reports runtime errors:

```bash
python inference/run_track_b.py --model trufor --split test \
  --output predictions/trufor-track-b-test.jsonl --resume --retry-errors
python inference/run_track_b.py --model iml-vit --split test \
  --output predictions/iml-vit-track-b-test.jsonl --resume --retry-errors
```

Evaluate with the existing strict trace metric:

```bash
python inference/evaluate_predictions.py predictions/trufor-track-b-test.jsonl
python inference/evaluate_predictions.py predictions/iml-vit-track-b-test.jsonl
```

For Track B, spatial and temporal localization are scored whenever the localizer
produces them, independently of its image-level REAL/FAKE threshold. Classification
accuracy still uses the image-level decision. Track C retains FAKE-only localization.

The default threshold is 0.5. Do not tune it on the test set. If threshold tuning
is desired, choose it once on the 664-video validation split and record it.

## Completed test results

| Model | Rows | Errors | Accuracy ↑ | Fake acc. ↑ | Real acc. ↑ | Box IoU ↑ | Time distance ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| TruFor | 664 | 0 | 47.590 | 1.205 | 93.976 | 13.962 | 38.497 |
| IML-ViT | 664 | 0 | 45.181 | 0.301 | 90.060 | 11.437 | 40.914 |

The near-zero fake accuracy alongside high real accuracy means both localizers
mostly call videos REAL. Overall accuracy alone would hide this failure. Their
non-zero box IoU still provides a useful spatial-transfer signal, but neither is
a movement-aware DeepTraceReward model.
