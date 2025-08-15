import os
import argparse
import logging
from typing import Any, Sequence

import torch
from tqdm import tqdm

from eval.transformers_inference.visualisation_utils import draw_coordinates_on_image
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

# Defaults (can be overridden by CLI or YAML config)
DEFAULT_OSWORLDG_ROOT = "/home/laura_convergence_ai/OSWorld-G/benchmark"
MAX_NEW_TOKENS = 100
LOG_INTERVAL = 20

# Geometry helpers for OSWorld-G

def is_point_in_rectangle(point: tuple[float, float], rect: tuple[float, float, float, float]) -> bool:
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def is_point_in_polygon(point: tuple[float, float], polygon: Sequence[float]) -> bool:
    x, y = point
    n = len(polygon) // 2
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]

        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def is_correct(gt_coords: Sequence[float], pred_x_scaled: float, pred_y_scaled: float, box_type: str) -> bool:
    match box_type:
        case "bbox":
            gt_x, gt_y, gt_w, gt_h = gt_coords[0], gt_coords[1], gt_coords[2], gt_coords[3]
            return is_point_in_rectangle((pred_x_scaled, pred_y_scaled), (gt_x, gt_y, gt_x + gt_w, gt_y + gt_h))
        case "polygon":
            return is_point_in_polygon((pred_x_scaled, pred_y_scaled), gt_coords)
        case "refusal":
            return float(pred_x_scaled) == 0.0 and float(pred_y_scaled) == 0.0
        case _:
            return False


def polygon_to_bbox_coords(polygon: Sequence[float]) -> list[int]:
    if not polygon:
        return []
    xs = polygon[0::2]
    ys = polygon[1::2]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    return [int(x1), int(y1), int(x2), int(y2)]


def main(
    osworldg_root: str,
    model_path: str,
    df,  # pandas.DataFrame
    rank: int = 0,
    world_size: int = 1,
    local_rank: int | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[int, int]:
    dataset_name = "OSWorld-G"
    model_name = os.path.basename(os.path.normpath(model_path)) if "checkpoint" not in model_path else model_path.split("/")[-3]
    res_dir = os.path.abspath(f"results_overlays_{dataset_name}/{model_name}")
    os.makedirs(res_dir, exist_ok=True)

    # Device, model, processor
    device = select_device(local_rank, rank)
    model, processor = build_model_and_processor(model_path, device, config=config)

    # Generation kwargs
    gen_cfg = config.get("generation", {}) or {}
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": gen_cfg.get("max_new_tokens", MAX_NEW_TOKENS or GENERATION_DEFAULTS["max_new_tokens"]),
        "do_sample": gen_cfg.get("do_sample", GENERATION_DEFAULTS["do_sample"]),
        "temperature": gen_cfg.get("temperature", GENERATION_DEFAULTS["temperature"]),
    }
    system_prompt = config.get("prompts", {}).get("system") or DEFAULT_SYSTEM_PROMPT
    correct = 0
    total = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"rank {rank}"):
        img_path = os.path.join(osworldg_root, "images", row["image_path"])  # str
        _, resized_width, resized_height, scale_x, scale_y, img = get_resized_size(img_path, processor)
        instruction = row["instruction"]
        messages = construct_messages(instruction, img, resized_height, resized_width, system_prompt)
        pred_x_scaled, pred_y_scaled, output_text, _ = predict_coordinates(
            model=model, processor=processor, messages=messages, scale_x=scale_x, scale_y=scale_y, generation_kwargs=generation_kwargs
        )

        gt_coords = row["box_coordinates"]
        box_type = row["box_type"]
        correct_ans = is_correct(gt_coords, pred_x_scaled, pred_y_scaled, box_type)
        total += 1
        correct += int(correct_ans)

        # Build overlays
        if box_type == "bbox" and isinstance(gt_coords, (list, tuple)) and len(gt_coords) >= 4:
            gx, gy, gw, gh = gt_coords[:4]
            gt_vis = [int(gx), int(gy), int(gx + gw), int(gy + gh)]
        elif box_type == "polygon" and isinstance(gt_coords, (list, tuple)) and len(gt_coords) >= 6:
            gt_vis = polygon_to_bbox_coords(gt_coords)
        else:
            gt_vis = None

        correct_dir = "correct" if correct_ans else "incorrect"
        os.makedirs(os.path.join(res_dir, correct_dir), exist_ok=True)
        out_path = os.path.join(res_dir, correct_dir, row["image_path"])  # mirror dataset path

        draw_coordinates_on_image(
            img,
            output_path=out_path,
            predicted_coords=[pred_x_scaled, pred_y_scaled, pred_x_scaled, pred_y_scaled],
            output_text=output_text,
            ground_truth_coords=gt_vis,
            instruction=instruction,
        )
        print(f"[rank {rank}] saved: {out_path}") if os.path.exists(out_path) else print(f"[rank {rank}] FAILED: {out_path}")

        if total % LOG_INTERVAL == 0:
            print(f"[rank {rank}] {correct}/{total} or {correct/total:.4f}")

    return int(correct), int(total)


if __name__ == "__main__":
    import pandas as pd
    import json

    # Distributed setup via torchrun-like env
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))

    # CLI args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--osworldg-root",
        type=str,
        default=DEFAULT_OSWORLDG_ROOT,
        help="Path to OSWorld-G benchmark root containing images/ and OSWorld-G.json",
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
    parser.add_argument(
        "--refined-dataset",
        action="store_true",
        help="Use refined dataset",
        default=False,
    )
    args, _ = parser.parse_known_args()

    # Load config and override prompt
    cfg: dict[str, Any] = load_yaml_config(args.config) if args.config else {}
    prompt_cfg = (cfg.get("prompts", {}) or {}).get("system") if cfg else None
    if prompt_cfg:
        SYSTEM_PROMPT = str(prompt_cfg).strip()

    # Resolve model path precedence
    cfg_model_path = (cfg.get("model", {}) or {}).get("path") if cfg else None
    effective_model_path = args.model_path or cfg_model_path

    # Load dataset json
    if args.refined_dataset:
        json_path = os.path.join(args.osworldg_root, "OSWorld-G_refined.json")
    else:
        json_path = os.path.join(args.osworldg_root, "OSWorld-G.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    full_df = pd.DataFrame(data)

    # Shard dataframe across ranks
    start_idx, end_idx = compute_shard_indices(len(full_df), world_size, rank)
    shard_df = full_df.iloc[start_idx:end_idx].copy()

    if rank == 0:
        print(f"Dataset: OSWorld-G; WORLD_SIZE={world_size}; processing {len(full_df)} rows -> shard size per rank ≈ {end_idx - start_idx}")
    print(f"[rank {rank}/{world_size}] handling rows [{start_idx}, {end_idx}) -> {len(shard_df)} items")

    # Initialize process group for cross-rank aggregation if needed
    dist_backend = init_distributed(world_size, rank)

    # Run main loop on this shard
    shard_correct, shard_total = main(
        osworldg_root=args.osworldg_root,
        model_path=effective_model_path,
        df=shard_df,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        config=cfg,
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