# Lightweight model recommendations

This shortlist targets an RTX 1000 Ada laptop GPU with 6 GB VRAM. The goal is to
measure how far small, pretrained transfer baselines are from the paper's larger,
task-trained systems—not to claim a like-for-like reproduction.

## Track A — video-level classification

Track A remains proposed work. Recommended DeepfakeBench checkpoints are:

1. **MesoInception** for an extremely small classical baseline.
2. **Xception** for the recognizable FaceForensics++ baseline.
3. **EfficientNet-B4** for a stronger compact CNN.
4. **F3Net or SPSL** for frequency-aware transfer.

These are face/manipulation detectors, while DeepTraceReward includes whole-body
and motion failures. Sample frames consistently and report mean, maximum, and
top-k video aggregation because short anomalies can disappear under a plain mean.

## Track B — spatial forensic localization

The completed shortlist is:

1. **TruFor** — RGB plus Noiseprint++ integrity localization.
2. **IML-ViT** — ViT-B image-manipulation localization.

Both official checkpoints ran successfully through the shared two-frame protocol.

| Model | Accuracy ↑ | Fake acc. ↑ | Real acc. ↑ | Box IoU ↑ | Time distance ↓ |
|---|---:|---:|---:|---:|---:|
| TruFor | 47.590 | 1.205 | 93.976 | 13.962 | 38.497 |
| IML-ViT | 45.181 | 0.301 | 90.060 | 11.437 | 40.914 |

These models localize still-image manipulation fingerprints rather than
human-perceived motion traces. Treat the result as out-of-domain transfer. DeCLIP
and MVSS-Net were removed: their official pinned environments conflict with this
Python 3.14 stack and were not replaced with unverified ports. Full setup and
commands are in [`inference/TRACK_B.md`](inference/TRACK_B.md).

## Track C — video reasoning and explanation

| Model | Role | Accuracy ↑ | Fake acc. ↑ | Real acc. ↑ | Box IoU ↑ |
|---|---|---:|---:|---:|---:|
| Qwen2.5-VL-3B, NF4 | strongest compact control | 66.566 | 56.928 | 76.205 | 13.808 |
| VideoLLaMA3-2B, NF4 | same-family scaling control | 47.892 | 10.542 | 85.241 | 0.000 |
| SmolVLM2-500M | ultra-small negative control | 18.373 | 15.361 | 21.386 | 0.000 |

**Qwen2.5-VL-3B is the recommended compact baseline.** It is the only tested
small model with reasonably balanced fake/real transfer, and its video interface
supports timestamps, boxes, and structured output. It is still not deepfake-tuned.

**VideoLLaMA3-2B is useful as a scaling control** because the paper uses the 7B
family. It produces sentence-level responses, but its strong REAL bias and weak
grounding show that fluent output is not forensic correctness.

**SmolVLM2-500M is retained only as a negative control.** Its responses were
mostly labels, booleans, or copied/generic text, with no valid grounding.

**Marlin-2B is not a valid final result.** Although conceptually relevant for
temporal grounding, 651 of 664 rows failed at runtime in this environment. Keep
the failed JSONL for auditability; exclude it from performance comparisons.

FakeVLM-7B is conceptually relevant, but its separate Python 3.10 LLaVA stack and
frame/contact-sheet protocol make it unsuitable for this shared 6 GB video
benchmark. Penguin-VL-2B likewise depends on incompatible loading internals.

## Fair-comparison protocol

- Evaluate the official test split: 332 fake and 332 real unique videos.
- Split and deduplicate by `video_id`, never annotation row.
- Report overall, fake, and real accuracy together to expose majority/REAL bias.
- Score missing boxes and times under the documented strict policy.
- Report quantization, sampled frames, threshold, failure rows, and latency.
- Keep different prompts and sampling protocols in separate files.
- Treat automatic explanation similarity as diagnostic. Direct comparison with
  the paper requires its judge rubric or a documented human audit.
