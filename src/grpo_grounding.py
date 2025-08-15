# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
import pathlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from trainer import Qwen2VLGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math
from qwen_vl_utils import smart_resize
from transformers import AutoProcessor
from liger_kernel.transformers import apply_liger_kernel_to_qwen2_5_vl

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False

# SYSTEM_PROMPT = '''
# You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

# Output the coordinate pair exactly:
# (x,y)
# '''
SYSTEM_PROMPT = '''
    You are a helpful assistant.
    # Tools
    You may call one or more functions to assist with the user query.
    You are provided with function signatures within <tools></tools> XML tags:
    <tools>
    {{"type": "function", "function": {{"name": "computer_use", "description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\n* The screen's resolution is {width}x{height}.\n.* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {{"properties": {{"action": {{"description": "The action to perform. The available actions are:\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n* `type`: Type a string of text on the keyboard.\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button.\n* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n* `right_click`: Click the right mouse button.\n* `middle_click`: Click the middle mouse button.\n* `double_click`: Double-click the left mouse button.\n* `scroll`: Performs a scroll of the mouse scroll wheel.\n* `wait`: Wait specified seconds for the change to happen.\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", "double_click", "scroll", "wait", "terminate"], "type": "string"}}, "keys": {{"description": "Required only by `action=key`.", "type": "array"}}, "text": {{"description": "Required only by `action=type`.", "type": "string"}}, "coordinate": {{"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move`, `action=left_click_drag`, `action=left_click`, `action=right_click`, `action=double_click`.", "type": "array"}}, "pixels": {{"description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.", "type": "number"}}, "time": {{"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"}}, "status": {{"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}}}, "required": ["action"], "type": "object"}}}}}}
    </tools>
    For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
    <tool_call>
    {{"name": <function-name>, "arguments": <args-json-object>}}
    </tool_call>
'''
SYSTEM_PROMPT = SYSTEM_PROMPT.strip()

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, processing_class:None):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        cur_data_dict = []
        if data_path.endswith(".json"):
            with open(data_path, "r") as json_file:
                self.list_data_dict = json.load(json_file)
                print(f"Before filtering: {len(self.list_data_dict)}")
                self.list_data_dict = self._filter_non_existint_images()
                print(f"After filtering: {len(self.list_data_dict)}")
        else:
            self.list_data_dict = []
            with open(data_path, "r") as f:
                for line in f:
                    self.list_data_dict.append(json.loads(line))
        self.processor=processing_class

    def _filter_non_existint_images(self):
        image_root = self.script_args.image_root
        filtered_list = []
        for data_dict in self.list_data_dict:
            image_path = os.path.join(image_root, data_dict['image'])
            if os.path.exists(image_path) and "Wendy" not in data_dict['image']:
                filtered_list.append(data_dict)
        return filtered_list
        
    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        def make_conversation_image(example, height,width):
            instruction = example['conversations'][0]['value']
            instruction = instruction.replace("<image>","")
            return {
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT.format(height=height,width=width)}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": instruction}
                        ],
                    },
                ],
            }
        example = self.list_data_dict[i]
        image_root = self.script_args.image_root   
        image_path = os.path.join(image_root, example['image'])
        image = Image.open(image_path).convert("RGB")
        image_height, image_width =  image.height, image.width
        resized_height, resized_width  = smart_resize(
                    image.height,
                    image.width,
                    factor=self.processor.image_processor.patch_size * self.processor.image_processor.merge_size,
                    min_pixels=self.processor.image_processor.min_pixels,
                    max_pixels=self.processor.image_processor.max_pixels,
                    )
        image = image.resize((resized_width, resized_height))
        box = example['bbox']
        image = [image]
        

        solution = [box,resized_height/1000,resized_width/1000]

        return {
            'image': image,
            'problem': example['conversations'][0]['value'],
            'solution': solution,
            'prompt': make_conversation_image(example, resized_height,resized_width)['prompt']
        }


def parse_coordinates_strict(raw_string, scale_x: float = 1.0, scale_y: float = 1.0):
    # Extract the most recent <tool_call> ... </tool_call> block
    try:
        tool_calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", raw_string, flags=re.DOTALL)
        if not tool_calls:
            return -1, -1

        # Check the first tool call in the string
        allowed_actions = {"mouse_move", "left_click", "right_click", "double_click", "left_click_drag"}
        # allowed_actions = {"left_click"}
        for block in tool_calls:
            payload = block.strip()
            # Strip optional code fences if present
            if payload.startswith("```") and payload.endswith("```"):
                payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload.strip())

            try:
                obj = json.loads(payload)
            except Exception:
                continue

            if not isinstance(obj, dict):
                continue

            name = obj.get("name")
            arguments = obj.get("arguments")
            if name != "computer_use" or not isinstance(arguments, dict):
                continue

            action = arguments.get("action")
            if action not in allowed_actions:
                continue

            coordinate = arguments.get("coordinate")
            if not (isinstance(coordinate, (list, tuple)) and len(coordinate) == 2):
                continue

            x_raw, y_raw = coordinate

            def to_float(value):
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, str):
                    val = value.strip()
                    # Remove any stray non-numeric characters
                    val = re.sub(r"[^0-9eE+\-\.]", "", val)
                    return float(val) if val else None
                return None

            x = to_float(x_raw)
            y = to_float(y_raw)
            if x is None or y is None or math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                continue

            return scale_x * x, scale_y * y

    except Exception:
        pass

    return -1, -1


def parse_coordinates(raw_string, scale_x: float = 1.0, scale_y: float = 1.0):
    match = re.search(r"(-?\d*\.?\d+).*?(-?\d*\.?\d+)", raw_string)
    if match:
        pred_x, pred_y = match.groups()
        pred_x_scaled = scale_x * float(pred_x)
        pred_y_scaled = scale_y * float(pred_y)
        return pred_x_scaled, pred_y_scaled
    else:
        return -1,-1


def click_reward(completions, solution, **kwargs):
    def isin(x,y, sol):
        boxs,ratio_h,ratio_w=sol[:3]
        if not isinstance(boxs[0], list):
            boxs = [boxs]
        for box in boxs:
            x0,y0,x1,y1 = box
            x0 = int(x0*ratio_w)
            y0 = int(y0*ratio_h)
            x1 = int(x1*ratio_w)
            y1 = int(y1*ratio_h)
            if x<=x1 and x>=x0:
                if y<=y1 and y>=y0:
                    return True
        return False
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        reward = 0.0
        try:
            pred_x, pred_y = parse_coordinates_strict(content)
            if isin(pred_x, pred_y, sol):
                # print(f"Reward: 1.0, correct prediction: {pred_x},{pred_y}")
                reward = 1.0
            # else:
                # print(f"Reward: 0.0, incorrect prediction: {pred_x},{pred_y}")
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards

def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>\(\d+,\s*\d+\)</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content.strip(), re.DOTALL) for content in completion_contents]
    reward=[1.0 if match else 0.0 for match in matches]
    return reward

reward_funcs_registry = {
    "accuracy": click_reward,
    "format": format_reward,
}


def main(script_args, training_args, model_args):

    print(script_args)
    print(training_args)
    print(model_args)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    processing_class = AutoProcessor.from_pretrained(model_args.model_name_or_path,  max_pixels=script_args.max_pixels, min_pixels=script_args.min_pixels)
    # Load the dataset
    dataset = LazySupervisedDataset(script_args.dataset_name, script_args, processing_class)
    trainer_cls = Qwen2VLGRPOTrainer

    apply_liger_kernel_to_qwen2_5_vl(fused_linear_cross_entropy=False)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )


    # Train and push the model to the Hub
    latest_checkpoint = None
    checkpoints = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if checkpoints:
        # Sort checkpoints by step number and get the latest one
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split("-")[1]))
        print(f"Found checkpoint: {latest_checkpoint}")
        
        # Verify that trainer_state.json exists in the checkpoint
        trainer_state_path = latest_checkpoint / "trainer_state.json"
        if trainer_state_path.exists():
            print(f"Resuming training from {latest_checkpoint}")
            trainer.train(resume_from_checkpoint=str(latest_checkpoint))
        else:
            print(f"Warning: No trainer_state.json found in {latest_checkpoint}, starting from scratch")
            trainer.train()
    else:
        print("No checkpoints found, starting training from scratch")
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
