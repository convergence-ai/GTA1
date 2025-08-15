#!/bin/bash

# Set a run name (edit as needed)
RUN_NAME="gta1-jedi-0.0001beta-gspo-enforced-json"

# Paths
DATASET_PATH="/home/laura_convergence_ai/datasets/GTA1_grounding_data/inp.json"
IMAGE_ROOT="$HOME/datasets/GTA1_grounding_data/image"

export CUDA_LAUNCH_BLOCKING=1     # forces sync so the stack-trace is accurate
export TORCH_USE_CUDA_DSA=1       # device-side assertions
# Launch training

torchrun \
    --nproc_per_node 8 \
    src/grpo_grounding.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir grounding/$RUN_NAME \
    --model_name_or_path "xlangai/Jedi-7B-1080p"  \
    --dataset_name "$DATASET_PATH" \
    --image_root "$IMAGE_ROOT" \
    --max_prompt_length 1024 \
    --max_completion_length 128 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --freeze_vision_modules true \
    --reward_funcs accuracy \
    --beta 0.0001 \
    --dataloader_num_workers 2 \
    --max_pixels $((4096 * 2160)) \
    --gradient_accumulation_steps 16 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name output/$RUN_NAME \
    --save_steps 50 \
    --save_total_limit 40 \
    --save_only_model false \
    --resume_from_checkpoint true \
    --importance_sampling_level "sequence"