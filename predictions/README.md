# Prediction files

Track B uses the final two-model shortlist `trufor` and `iml-vit`. Their JSONL
rows use `task: "localization"`, contain per-frame anomaly scores, and store the
strongest sampled-frame box in original-video pixels. DeCLIP is intentionally
excluded because its official environment is incompatible with `deeptracereward`.

## Explanation metrics

Run `python inference/evaluate_explanations.py <prediction.jsonl>` after inference.
The visualizer computes the same lightweight metrics. Missing explanations score
zero. Read annotation macro, all-trace video mean, and oracle-best trace together.

Put one JSON object per line in `predictions/<run-name>.jsonl`. The visualizer
discovers files automatically; no restart is required. Completed full-test files
are small enough to version; `*pilot*.jsonl` is ignored so a pilot cannot be
mistaken for the 664-video benchmark.

Required fields are `video_id`, `model_id`, and `task`. Supported fields include:
`predicted_label`, `fake_probability`, `explanation`, `bbox` (`left`, `top`,
`width`, `height`), `start_time`, `end_time`, `latency_seconds`, `raw_output`,
`error`, `format_warning`, `prompt_version`, `adapter`, `quantization`, `sampled_frames`,
`max_pixels_per_frame`, and `input_frame_size`.

Example:

```json
{"video_id":"abc","model_id":"qwen2.5-vl-3b","task":"reasoning","adapter":"qwen","quantization":"4bit","sampled_frames":8,"predicted_label":"FAKE","fake_probability":0.73,"explanation":"The hand deforms while moving.","bbox":{"left":100,"top":80,"width":220,"height":190},"start_time":1.2,"end_time":2.7,"latency_seconds":3.1}
```

Track-C model IDs currently recognized by `inference/run_vlm.py` are
`qwen2.5-vl-3b`, `videollama3-2b`, `marlin-2b`, and the retained
negative control `smolvlm2-500m`. Use separate `*-pilot2.jsonl` and
`*-pilot32.jsonl` files so pilot runs cannot be mistaken for full test results.

The completed Marlin run had 651/664 runtime failures. It remains readable for
audit purposes but is not a valid performance comparison.

`penguin-vl-2b` is intentionally not recognized. Its official custom checkpoint
uses Transformers 4-era loading hooks that are incompatible with the Transformers
5 environment required by Marlin. FakeVLM is also separate because it requires a
Python 3.10 LLaVA/lmms-finetune image workflow.

## Output validity

The runner preserves contradictory model outputs and records them in
`format_warning`. For example, a `REAL` label with a high `fake_probability` or a
non-null fake location is not silently corrected. This makes instruction-following
failures reviewable in the visualizer.

Runs with different `prompt_version` values must remain in different files. The
current prompt is `deeptrace-v2-no-semantic-example`; it removes the concrete
artifact example that contaminated the first VideoLLaMA3 pilot through copying.

The evaluator uses strict scoring:

- invalid or missing labels count as incorrect;
- missing/invalid boxes receive IoU 0 and normalized distance 100;
- missing/invalid times receive normalized distance 100;
- list boxes are interpreted as `[x0, y0, x1, y1]`;
- explanation correctness is not fabricated without the paper's judge or a human audit.

It also reports `response_text_rows`, response coverage by predicted label,
substantive-looking response counts, explanation word-length statistics, and
format-warning types. These describe response availability and compliance only;
they are not explanation-accuracy metrics.

Evaluate a run with:

```bash
python inference/evaluate_predictions.py predictions/<run-name>.jsonl
```
