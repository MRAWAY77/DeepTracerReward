---
name: profile-media-dataset
description: RewardData specialization of the shared profile-media-dataset workflow.
---

# Profile Media Dataset — DeepTraceReward specialization

Read `/home/alvinwong/Desktop/src/Codex_Skills/profile-media-dataset/SKILL.md`
first. This file contains only verified local rules and overrides.

## Dataset roots and exclusions

- Ground-truth metadata: `all_data.json`
- Generated-video media: `videos/*.mp4`
- Real-video media: `real_videos/*.mp4`
- Exclude `.git/`, `third_party/`, `assets/`, `predictions/`, model caches,
  presentations, and generated `reports/` from source-media counts.
- Treat media as read-only. Audit scripts may only write beneath `reports/`.

## Join and identity rules

- `video_id` is the unique sample identity and `video_path` is the media join key.
- `all_data.json` has 7,652 rows but 6,636 unique videos because a generated
  video can have multiple expert fake-trace annotations.
- Never split, deduplicate, or evaluate by annotation row. Use `video_id`.
- Expected unique labeled videos: 3,318 generated + 3,318 real.
- Expected split totals by unique video: 5,308 `train`, 664 `val`, 664 `test`.
- Expected expert trace rows: 4,334.
- Retain physical MP4s missing from `all_data.json` as `unmatched_media`; do not
  invent labels or silently exclude them. The 2026-07-14 audit found two such
  files under `videos/`.

## Semantic caveats

- `video_type=fake_video` means AI-generated; `real_video` is the real control.
- Fake traces are human-perceived movement/artifact annotations, not objective
  forensic proof and not pixel-perfect segmentation masks.
- Paper categories are heuristic, multi-label mappings from released Labelbox
  tags. Keep the mapping local and label it normalized/exploratory.
- Blank explanations and category tags are valid missing annotations.

## Project-owned artifacts

- Audit command: `python tools/dataset_audit.py .`
- Per-file manifest: `reports/dataset_manifest.jsonl`
- Aggregate summary: `reports/dataset_summary.json`
- Technical audit report: `reports/dataset_audit.html`
- Explorer launcher: `python app.py`
- Explorer UI: `static/index.html`
- Curated guide: `readme.html` (the audit must not overwrite it)

## Explorer compatibility contract

Preserve the existing UI and API:

- Explore, Dataset analytics, Model performance, and About tabs;
- all `/api/videos`, `/api/video/<id>`, `/api/stats`, `/api/options`,
  `/api/prediction-models`, `/api/model-performance`, `/api/prediction/<id>`,
  and `/media/<path>` behavior;
- real/fake interleaving, filters, response filters, expert/model overlays,
  SAME/DIFFERENT review, explanation metrics, and Track-B metrics;
- `python app.py` launches the same visualizer on port 8000 when available.

## Regression checks

- Current local manifest: 6,638 physical MP4s, 6,636 annotation-matched, two
  unmatched, and no annotation-referenced media missing.
- API default page interleaves fake and real videos.
- `/media/` supports HTTP byte ranges and rejects paths outside the dataset root.
- Completed prediction files remain discoverable without rebuilding the audit.
