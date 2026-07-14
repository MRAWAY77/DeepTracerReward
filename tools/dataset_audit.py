#!/usr/bin/env python3
"""Generate RewardData per-video audit artifacts without modifying media."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


EXPECTED = {
    "annotation_rows": 7652,
    "labeled_videos": 6636,
    "fake_videos": 3318,
    "real_videos": 3318,
    "trace_annotations": 4334,
    "splits": {"train": 5308, "val": 664, "test": 664},
}


def number(value, kind=float):
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_path"]].append(row)
    result = {}
    for path, annotations in grouped.items():
        base = annotations[0]
        result[path] = {
            "video_id": base["video_id"],
            "video_type": base["video_type"],
            "source": base.get("video_source"),
            "split": base.get("split"),
            "prompt": base.get("video_prompt"),
            "mime_type": base.get("video_mime_type"),
            "annotation_count": len(annotations) if base["video_type"] == "fake_video" else 0,
            "written_explanation_count": sum(bool(str(row.get("faketrace_explanations") or "").strip()) for row in annotations),
            "raw_tagged_annotation_count": sum(bool(row.get("faketrace_check_applies")) for row in annotations),
            "width": number(base.get("video_width"), int),
            "height": number(base.get("video_height"), int),
            "fps": number(base.get("video_frame_rate")),
            "frame_count": number(base.get("video_frame_count"), int),
            "duration_seconds": number(base.get("video_duration")),
        }
    return result


def inventory(root, labels):
    files = sorted(path for folder in ("videos", "real_videos") for path in (root / folder).glob("*") if path.is_file())
    present = {path.relative_to(root).as_posix() for path in files}
    records = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        metadata = labels.get(relative)
        stat = path.stat()
        record = {
            "path": relative,
            "filename": path.name,
            "extension": path.suffix.lower(),
            "media_type": "video",
            "size_bytes": stat.st_size,
            "modified_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            "status": "ok",
            "integrity": "annotation_matched" if metadata else "unmatched_media",
            "collection": metadata["video_type"] if metadata else path.parent.name,
            "source": metadata["source"] if metadata else "unknown",
            "split": metadata["split"] if metadata else "unassigned",
            "probe_status": "not_requested",
        }
        if metadata:
            record.update(metadata)
        records.append(record)
    for relative, metadata in sorted(labels.items()):
        if relative not in present:
            records.append({
                "path": relative,
                "filename": Path(relative).name,
                "extension": Path(relative).suffix.lower(),
                "media_type": "video",
                "size_bytes": 0,
                "status": "error",
                "integrity": "missing_media",
                "probe_status": "not_available",
                "probe_error": "Referenced by all_data.json but file is missing",
                "collection": metadata["video_type"],
                **metadata,
            })
    return records


def probe(path):
    command = [
        "ffprobe", "-v", "error", "-show_entries",
        "format=format_name,duration,bit_rate:stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
        "-of", "json", str(path),
    ]
    try:
        payload = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
        streams = payload.get("streams", [])
        video = next((item for item in streams if item.get("codec_type") == "video"), {})
        audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
        rate = str(video.get("r_frame_rate") or "")
        if "/" in rate:
            numerator, denominator = rate.split("/", 1)
            fps = float(numerator) / float(denominator) if float(denominator) else None
        else:
            fps = number(rate)
        fmt = payload.get("format", {})
        return {
            "probe_status": "ok",
            "probed_format": fmt.get("format_name"),
            "probed_codec": video.get("codec_name"),
            "probed_width": number(video.get("width"), int),
            "probed_height": number(video.get("height"), int),
            "probed_fps": round(fps, 6) if fps is not None else None,
            "probed_duration_seconds": number(fmt.get("duration")),
            "probed_bit_rate": number(fmt.get("bit_rate"), int),
            "probed_has_audio": bool(audio),
            "probed_audio_codec": audio.get("codec_name"),
            "probed_audio_sample_rate_hz": number(audio.get("sample_rate"), int),
            "probed_audio_channels": number(audio.get("channels"), int),
            "probe_error": None,
        }
    except Exception as exc:
        return {"probe_status": "error", "probe_error": f"{type(exc).__name__}: {exc}"}


def select_probe_paths(records, mode, sample_per_group):
    valid = [record for record in records if record["status"] == "ok"]
    if mode == "all":
        return {record["path"] for record in valid}
    if mode == "none":
        return set()
    groups = defaultdict(list)
    for record in valid:
        groups[(record["collection"], record["source"])].append(record["path"])
    return {path for paths in groups.values() for path in paths[:sample_per_group]}


def run_probes(root, records, paths, workers):
    selected = {record["path"]: record for record in records if record["path"] in paths}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe, root / path): path for path in selected}
        for future in as_completed(futures):
            selected[futures[future]].update(future.result())


def counts(values):
    return dict(Counter(str(value) for value in values).most_common())


def summarize(root, rows, records, mode):
    labeled = [record for record in records if record["integrity"] == "annotation_matched"]
    traces = [row for row in rows if row["video_type"] == "fake_video"]
    split_counts = Counter(record["split"] for record in labeled)
    totals = {
        "physical_media_files": sum(record["status"] == "ok" for record in records),
        "physical_bytes": sum(record["size_bytes"] for record in records),
        "annotation_rows": len(rows),
        "labeled_videos": len(labeled),
        "fake_videos": sum(record.get("video_type") == "fake_video" for record in labeled),
        "real_videos": sum(record.get("video_type") == "real_video" for record in labeled),
        "trace_annotations": len(traces),
        "written_trace_explanations": sum(bool(str(row.get("faketrace_explanations") or "").strip()) for row in traces),
        "unmatched_media": sum(record["integrity"] == "unmatched_media" for record in records),
        "missing_media": sum(record["integrity"] == "missing_media" for record in records),
        "probed_files": sum(record["probe_status"] == "ok" for record in records),
        "probe_errors": sum(record["probe_status"] == "error" for record in records),
    }
    checks = {
        key: totals[key] == EXPECTED[key]
        for key in ("annotation_rows", "labeled_videos", "fake_videos", "real_videos", "trace_annotations")
    }
    checks.update({"split_totals": dict(split_counts) == EXPECTED["splits"], "no_missing_annotation_media": totals["missing_media"] == 0})
    probed = [record for record in records if record["probe_status"] == "ok"]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": ".",
        "probe_mode": mode,
        "totals": totals,
        "regression_checks": checks,
        "distributions": {
            "integrity": counts(record["integrity"] for record in records),
            "collections": counts(record["collection"] for record in records),
            "splits_labeled_videos": dict(split_counts),
            "sources_labeled_videos": counts(record["source"] for record in labeled),
            "resolutions_declared": counts(f"{record.get('width')}×{record.get('height')}" for record in labeled),
            "video_codecs_probed": counts(record.get("probed_codec") or "unknown" for record in probed),
            "audio_presence_probed": counts(record.get("probed_has_audio") for record in probed),
        },
        "unmatched_paths": [record["path"] for record in records if record["integrity"] == "unmatched_media"],
        "missing_paths": [record["path"] for record in records if record["integrity"] == "missing_media"],
        "notes": [
            "Labels and splits come from all_data.json, not directory names alone.",
            "Multiple fake trace rows are joined by video_id before video-level counts.",
            "Codec/audio distributions cover probed files only unless --probe all is used.",
            "Source videos are never hashed or modified by this audit.",
        ],
    }


def html_table(mapping):
    rows = "".join(f"<tr><td>{html.escape(str(key))}</td><td>{value:,}</td></tr>" for key, value in mapping.items())
    return f"<table><tr><th>Value</th><th>Count</th></tr>{rows}</table>"


def report(summary):
    totals = summary["totals"]
    cards = "".join(f'<div class="card"><span>{label}</span><b>{totals[key]:,}</b></div>' for key, label in (
        ("physical_media_files", "Physical videos"), ("labeled_videos", "Labeled videos"),
        ("trace_annotations", "Expert traces"), ("unmatched_media", "Unmatched media"),
        ("missing_media", "Missing media"), ("probe_errors", "Probe errors")))
    checks = "".join(f'<li class="{"ok" if passed else "bad"}">{"PASS" if passed else "FAIL"}: {html.escape(name.replace("_", " "))}</li>' for name, passed in summary["regression_checks"].items())
    unmatched = "".join(f"<li><code>{html.escape(path)}</code></li>" for path in summary["unmatched_paths"]) or "<li>None</li>"
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in summary["notes"])
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RewardData technical audit</title><style>body{{max-width:1100px;margin:36px auto;padding:0 20px;background:#0d1117;color:#dce3ed;font:15px/1.55 system-ui}}a{{color:#6ee7b7}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}.card,section{{background:#151b25;border:1px solid #303848;border-radius:10px;padding:16px}}.card span{{color:#9aa7b8}}.card b{{display:block;font-size:25px}}section{{margin-top:16px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #303848;padding:8px;text-align:left}}.ok{{color:#6ee7b7}}.bad{{color:#fb7185}}code{{overflow-wrap:anywhere}}</style></head><body><h1>RewardData technical audit</h1><p>Generated {html.escape(summary['generated_at_utc'])}; probe mode <code>{html.escape(summary['probe_mode'])}</code>.</p><p><a href="dataset_manifest.jsonl">Per-file JSONL manifest</a> · <a href="dataset_summary.json">Aggregate JSON</a> · <a href="../readme.html">Curated explorer guide</a></p><div class="grid">{cards}</div><section><h2>Regression checks</h2><ul>{checks}</ul></section><section><h2>Integrity</h2>{html_table(summary['distributions']['integrity'])}<h3>Unmatched physical files</h3><ul>{unmatched}</ul></section><section><h2>Labeled videos by split</h2>{html_table(summary['distributions']['splits_labeled_videos'])}</section><section><h2>Video codecs (probed files)</h2>{html_table(summary['distributions']['video_codecs_probed'])}</section><section><h2>Method caveats</h2><ul>{notes}</ul></section></body></html>'''


def write(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--probe", choices=("none", "sample", "all"), default="sample")
    parser.add_argument("--sample-per-group", type=int, default=2)
    parser.add_argument("--probe-workers", type=int, default=8)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    annotation_path = root / "all_data.json"
    if not annotation_path.is_file():
        raise SystemExit(f"Missing annotation file: {annotation_path}")
    if args.probe != "none" and not shutil.which("ffprobe"):
        raise SystemExit("ffprobe is required for --probe sample/all")
    rows = json.loads(annotation_path.read_text(encoding="utf-8"))
    records = inventory(root, aggregate(rows))
    paths = select_probe_paths(records, args.probe, max(1, args.sample_per_group))
    run_probes(root, records, paths, max(1, args.probe_workers))
    summary = summarize(root, rows, records, args.probe)
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write(output / "dataset_manifest.jsonl", "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records))
    write(output / "dataset_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write(output / "dataset_audit.html", report(summary))
    print(json.dumps({"output": str(output), **summary["totals"], "checks_passed": all(summary["regression_checks"].values())}, indent=2))


if __name__ == "__main__":
    main()
