import torch
from datasets import load_dataset

from eval.transformers_inference.visualisation_utils import draw_coordinates_on_image

from tqdm import tqdm
from io import BytesIO
import os
import argparse
import logging
import numpy as np
from typing import Any, Sequence
import pandas as pd

from eval.transformers_inference.common import (
    DEFAULT_SYSTEM_PROMPT,
    GENERATION_DEFAULTS,
    load_yaml_config,
    select_device,
    build_model_and_processor,
    construct_messages,
    get_resized_size,
    predict_coordinates,
    compute_shard_indices,
    init_distributed,
    aggregate_counts,
    destroy_distributed_if_initialized,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# Defaults (can be moved to CLI later if needed)
DEFAULT_DATASET_NAME = "lmms-lab/ScreenSpot-Pro"
MAX_NEW_TOKENS = 100
LOG_INTERVAL = 20


def get_original_coords(normalized_coords: Sequence[float], image_width: int, image_height: int) -> np.ndarray:
    original_coords = np.array([
        normalized_coords[0] * image_width,
        normalized_coords[1] * image_height,
        (normalized_coords[2] - normalized_coords[0]) * image_width,
            (normalized_coords[3] - normalized_coords[1]) * image_height
    ])
    original_coords = original_coords.astype(int)
    return original_coords


def main(dataset_name: str, model_path: str, screenspot_df: pd.DataFrame, rank: int = 0, world_size: int = 1, config: dict[str, Any] | None = None) -> tuple[int, int]:
    dataset_name = dataset_name.split("/")[-1]
    model_name = os.path.basename(os.path.normpath(model_path)) if "checkpoint" not in model_path else model_path.split("/")[-3]
    res_dir = os.path.abspath(f"results_overlays_{dataset_name}/{model_name}")
    os.makedirs(res_dir, exist_ok=True)

    # Device, model, processor
    device = select_device(local_rank=None, rank=rank)
    model, processor = build_model_and_processor(model_path, device, config=config)

    gen_cfg = (config or {}).get("generation", {}) or {}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": gen_cfg.get("max_new_tokens", MAX_NEW_TOKENS or GENERATION_DEFAULTS["max_new_tokens"]),
        "do_sample": gen_cfg.get("do_sample", GENERATION_DEFAULTS["do_sample"]),
        "temperature": gen_cfg.get("temperature", GENERATION_DEFAULTS["temperature"]),
    }
    system_prompt = config.get("prompts", {}).get("system") or DEFAULT_SYSTEM_PROMPT

    correct = 0
    total = 0
    for i, row in tqdm(screenspot_df.iterrows(), total=len(screenspot_df), desc=f"rank {rank}"):
        img_bytes = BytesIO(row['image']['bytes'])
        _, resized_width, resized_height, scale_x, scale_y, img = get_resized_size(img_bytes, processor)
        instruction = row['instruction']
        messages = construct_messages(instruction, img, resized_height, resized_width, system_prompt)
        pred_x_scaled, pred_y_scaled, output_text, _ = predict_coordinates(
            model=model, processor=processor, messages=messages, scale_x=scale_x, scale_y=scale_y, generation_kwargs=generation_kwargs
        )
        screenspot_df.loc[i, 'pred_x_scaled'] = pred_x_scaled
        screenspot_df.loc[i, 'pred_y_scaled'] = pred_y_scaled
        gt_coords = row['bbox']
        gt_coords_original = get_original_coords(gt_coords, img.width, img.height)
        correct_ans = pred_x_scaled >= gt_coords_original[0] and pred_x_scaled <= gt_coords_original[0] + gt_coords_original[2] and pred_y_scaled >= gt_coords_original[1] and pred_y_scaled <= gt_coords_original[1] + gt_coords_original[3]
        screenspot_df.loc[i, 'correct'] = correct_ans
        total += 1
        correct += correct_ans

        x, y, w, h = gt_coords_original
        bbox = [x, y, x + w, y + h]

        correct_dir = "correct" if correct_ans else "incorrect"
        os.makedirs(os.path.join(res_dir, correct_dir), exist_ok=True)
        im_path = row['file_name']
        img_path = os.path.join(res_dir, correct_dir, im_path)

        draw_coordinates_on_image(
            img,
            output_path=img_path,
            predicted_coords=[pred_x_scaled, pred_y_scaled, pred_x_scaled, pred_y_scaled],
            output_text=output_text,
            ground_truth_coords=bbox,
            instruction=instruction,
        )
        print(f"[rank {rank}] saved: {img_path}") if os.path.exists(img_path) else print(f"[rank {rank}] FAILED: {img_path}")

        if total % LOG_INTERVAL == 0:
            print(f"[rank {rank}] {correct}/{total} or {correct/total:.4f}")

    return int(correct), int(total)


if __name__ == "__main__":

    # Distributed setup via torchrun
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-version",
        choices=["v2", "pro"],
        default="pro",
        help="ScreenSpot dataset version to evaluate: 'v2' (HongxinLi/ScreenSpot_v2) or 'pro' (lmms-lab/ScreenSpot-Pro)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Model path or repo id to evaluate (overrides config)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/evals/grounding/transformers_inference/configs/jedi.yaml",
        help="Path to YAML config overriding model path, prompts, and generation args",
    )
    args, _ = parser.parse_known_args()

    # YAML config
    cfg: dict[str, Any] = load_yaml_config(args.config) if args.config else {}

    # Override system prompt if provided
    prompt_cfg = (cfg.get("prompts", {}) or {}).get("system") if cfg else None
    if prompt_cfg:
        SYSTEM_PROMPT = str(prompt_cfg).strip()

    # Resolve model path precedence
    cfg_model_path = (cfg.get("model", {}) or {}).get("path") if cfg else None
    effective_model_path = args.model_path or cfg_model_path

    # Resolve dataset from CLI
    dataset_name = (
        "HongxinLi/ScreenSpot_v2" if args.dataset_version == "v2" else "HongxinLi/ScreenSpot-Pro"
    )
    screenspot = load_dataset(dataset_name)

    full_df = screenspot['test'].to_pandas()

    # Shard dataframe across ranks
    start_idx, end_idx = compute_shard_indices(len(full_df), world_size, rank)
    shard_df = full_df.iloc[start_idx:end_idx].copy()

    if rank == 0:
        print(f"Dataset: {dataset_name}; WORLD_SIZE={world_size}; processing {len(full_df)} rows -> shard size per rank ≈ {end_idx - start_idx}")
    print(f"[rank {rank}/{world_size}] handling rows [{start_idx}, {end_idx}) -> {len(shard_df)} items")

    # Initialize process group for cross-rank aggregation if needed
    dist_backend = init_distributed(world_size, rank)

    # Run main loop on this shard
    shard_correct, shard_total = main(
        dataset_name, effective_model_path, shard_df, rank=rank, world_size=world_size, config=cfg
    )
    shard_acc = (shard_correct / shard_total) if shard_total > 0 else 0.0
    print(f"[rank {rank}] Shard accuracy: {shard_acc:.6f} ({shard_correct}/{shard_total})")

    # Aggregate global accuracy across ranks
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    global_correct, global_total = aggregate_counts(shard_correct, shard_total, world_size, dist_backend, device)
    if world_size > 1:
        if rank == 0:
            global_acc = (global_correct / global_total) if global_total > 0 else 0.0
            print(f"Global accuracy: {global_acc:.6f} ({global_correct}/{global_total})")
        destroy_distributed_if_initialized()
    else:
        print(f"Global accuracy: {shard_acc:.6f} ({shard_correct}/{shard_total})") 