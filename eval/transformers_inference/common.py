from __future__ import annotations

import re
from io import BytesIO
from typing import Any, Union

import torch
from PIL import Image

from qwen_vl_utils import process_vision_info, smart_resize
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

import yaml
import torch.distributed as dist

DEFAULT_SYSTEM_PROMPT = (
    """
You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)
"""
).strip()

GENERATION_DEFAULTS: dict[str, Any] = {
    "do_sample": False,
    "temperature": 0.0,
    "max_new_tokens": 256,
}


def load_yaml_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def select_device(local_rank: int | None, rank: int) -> torch.device:
    if torch.cuda.is_available():
        device_index = local_rank if (local_rank is not None) else rank
        torch.cuda.set_device(device_index)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        return torch.device(f"cuda:{device_index}")
    return torch.device("cpu")


def build_model_and_processor(
    model_path: str,
    device: torch.device,
    config: dict[str, Any] | None = None,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    cfg = config or {}

    attn_kwargs: dict[str, Any] = {}
    attn_impl = ((cfg.get("model", {}) or {}).get("attn_implementation"))
    if device.type == "cuda":
        attn_kwargs["attn_implementation"] = attn_impl or "flash_attention_2"

    dtype_cfg = ((cfg.get("model", {}) or {}).get("dtype") or "auto").lower()
    if device.type == "cuda":
        if dtype_cfg in ("auto", "bf16", "bfloat16"):
            torch_dtype = torch.bfloat16
        elif dtype_cfg in ("fp16", "float16"):
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    else:
        torch_dtype = torch.float32

    model: Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
        **attn_kwargs,
    )
    model.to(device)
    model.eval()

    processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
    processor: AutoProcessor = AutoProcessor.from_pretrained(
        model_path,
        **processor_kwargs,
    )

    return model, processor


def construct_messages(
    instruction: str,
    image: Image.Image,
    resized_height: int,
    resized_width: int,
    system_prompt: str,
) -> list[dict[str, Any]]:
    system_message = system_prompt.format(width=resized_width, height=resized_height)
    return [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def get_resized_size(
    img_path_or_bytes: Union[str, BytesIO],
    processor: AutoProcessor,
) -> tuple[Image.Image, int, int, float, float, Image.Image]:
    image = Image.open(img_path_or_bytes)
    width, height = image.width, image.height

    resized_height, resized_width = smart_resize(
        image.height,
        image.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    scale_x, scale_y = width / resized_width, height / resized_height

    resized_image = image.resize((resized_width, resized_height))
    return resized_image, resized_width, resized_height, scale_x, scale_y, image


def predict_coordinates(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    messages: list[dict[str, Any]],
    scale_x: float,
    scale_y: float,
    generation_kwargs: dict[str, Any] | None = None,
) -> tuple[float, float, str, str]:
    generation_kwargs = generation_kwargs or {}
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs, pad_token_id=processor.tokenizer.pad_token_id)

    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    match = re.search(r"(-?\d*\.?\d+).*?(-?\d*\.?\d+)", output_text)
    if match:
        pred_x, pred_y = match.groups()
        pred_x_scaled = scale_x * float(pred_x)
        pred_y_scaled = scale_y * float(pred_y)
        return pred_x_scaled, pred_y_scaled, output_text, text
    else:
        return 0.0, 0.0, output_text, text


def compute_shard_indices(total_rows: int, world_size: int, rank: int) -> tuple[int, int]:
    if world_size <= 0:
        return 0, total_rows
    import math
    rows_per_rank = math.ceil(total_rows / world_size)
    start_idx = rank * rows_per_rank
    end_idx = min(total_rows, start_idx + rows_per_rank)
    return start_idx, end_idx


def init_distributed(world_size: int, rank: int) -> str:
    """Initialize torch.distributed process group if needed; return backend used."""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://", rank=rank, world_size=world_size)
    return backend


def aggregate_counts(
    shard_correct: int,
    shard_total: int,
    world_size: int,
    backend: str,
    device: torch.device,
) -> tuple[int, int]:
    """All-reduce (correct,total) across ranks if world_size>1 and return global counts."""
    if world_size > 1:
        counts_tensor = torch.tensor([shard_correct, shard_total], dtype=torch.long, device=(device if backend == "nccl" else torch.device("cpu")))
        if backend == "gloo" and counts_tensor.is_cuda:
            counts_tensor = counts_tensor.cpu()
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        return int(counts_tensor[0].item()), int(counts_tensor[1].item())
    return shard_correct, shard_total


def destroy_distributed_if_initialized():
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group() 