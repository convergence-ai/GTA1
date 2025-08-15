# Grounding Evals (Transformers inference)

This directory contains evaluations for grounding tasks using plain `transformers` with Qwen-VL style checkpoints.

Supported:

- ScreenSpot (v2 and Pro)
- OSWorld-G

## Prerequisites
- A local Qwen-compatible checkpoint (works with `transformers` `AutoProcessor` and `Qwen2_5_VLForConditionalGeneration`)
- Python environment with project dependencies installed

Optional:
- YAML config files to override model path, dtype, attention implementation, system prompt, and generation args. See `eval/transformers_inference/configs/`.

Available example configs:
- `eval/transformers_inference/configs/jedi.yaml`
- `eval/transformers_inference/configs/gta1.yaml`

---

## OSWorld-G

Evaluate a Qwen-VL style checkpoint on the OSWorld-G benchmark JSON and images using plain `transformers`. Saves annotated images and reports accuracy.

### Default dataset location
- Default root: `/home/laura_convergence_ai/OSWorld-G/benchmark` (must contain `images/` and `OSWorld-G.json`)

### Single-GPU
```bash
python -m eval.transformers_inference.run_osworldg \
  --osworldg-root /home/laura_convergence_ai/OSWorld-G/benchmark \
  --model-path /path/to/your/qwen_checkpoint/
```

With a YAML config (overrides prompts/generation and optionally model path):
```bash
python -m eval.transformers_inference.run_osworldg \
  --osworldg-root /home/laura_convergence_ai/OSWorld-G/benchmark \
  --config eval/transformers_inference/configs/jedi.yaml
```

### Multi-GPU (single node)
```bash
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
torchrun --standalone --nproc-per-node=${NUM_GPUS} -m \
  eval.transformers_inference.run_osworldg \
  --osworldg-root /home/laura_convergence_ai/OSWorld-G/benchmark
```

Notes:
- Outputs are saved to `results_overlays_OSWorld-G/<model_name>/{correct,incorrect}/...` mirroring the dataset file structure.
- The script binds to `LOCAL_RANK` when launched with `torchrun` and aggregates accuracy across ranks.
- Generation is deterministic (`do_sample=False`) for stable evaluation.

---

## ScreenSpot

Evaluate a Qwen-VL style checkpoint on ScreenSpot v2 or Pro using plain `transformers`. Saves annotated images and reports accuracy.

### YAML config (optional)
Same YAML format as OSWorld-G; see `eval/transformers_inference/configs/`.
```bash
python -m eval.transformers_inference.run_screenspot \
  --dataset-version pro \
  --config eval/transformers_inference/configs/gta1.yaml
```

### Single-GPU
```bash
python -m eval.transformers_inference.run_screenspot \
  --dataset-version pro \
  --model-path /path/to/your/qwen_checkpoint/
```

### Multi-GPU (single node)
```bash
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
torchrun --standalone --nproc-per-node=${NUM_GPUS} -m \
  eval.transformers_inference.run_screenspot \
  --dataset-version pro
```

Notes:
- Dataset sources: `HongxinLi/ScreenSpot_v2` for v2 and `HongxinLi/ScreenSpot-Pro` for Pro.
- Outputs are saved to `results_overlays_<ScreenSpot_v2|ScreenSpot-Pro>/<model_name>/{correct,incorrect}/...`.
- The script aggregates accuracy across ranks when launched with `torchrun`.
- Generation is deterministic (`do_sample=False`). 