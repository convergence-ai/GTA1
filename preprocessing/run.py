from PIL import Image
import json
import os
import tqdm
import torch
import numpy as np
from ultralytics import YOLO
from util.utils import (
    get_som_labeled_img,
    check_ocr_box,
    get_caption_model_processor
)
from PIL import Image
import fire

class VLMParser:
    def __init__(
        self,
        som_model_path: str,
        caption_model_name: str,
        caption_model_path: str,
        device: str = "cuda",
    ):

        self.device = device
        self.som_model = YOLO(som_model_path)
        self.som_model.to(self.device)
        self.caption_processor = get_caption_model_processor(
            model_name=caption_model_name,
            model_name_or_path=caption_model_path,
            device=device
        )

    def process(
        self,
        image_path: str,
        box_threshold: float = 0.05,
        ocr_args: dict = None,
        paddle: bool = True,
        draw_bbox_config: dict = None,
    ):
        """
        Runs OCR + SOM detection + captioning on the image at image_path,
        then rescales all bboxes in parsed_content_list from [0–1] ratios
        back to pixel coords. Returns the list of items.
        """
        if ocr_args is None:
            ocr_args = {"paragraph": False, "text_threshold": 0.9}

        img = Image.open(image_path)
        width, height = img.width, img.height

        (ocr_text, ocr_bboxes), _ = check_ocr_box(
            image_path,
            display_img=False,
            output_bb_format="xyxy",
            goal_filtering=None,
            easyocr_args=ocr_args,
            use_paddleocr=paddle
        )

        dino_img_b64, label_coords, parsed_content_list = get_som_labeled_img(
            image_path,
            model=self.som_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bboxes,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_processor,
            ocr_text=ocr_text,
            use_local_semantics=True,
            iou_threshold=0.7,
            scale_img=False,
            batch_size=128
        )

        scaled_results = []
        for entry in parsed_content_list:
            x0, y0, x1, y1 = entry["bbox"]
            scaled_bbox = [
                int(x0 * width),
                int(y0 * height),
                int(x1 * width),
                int(y1 * height),
            ]
            e = entry.copy()
            e["bbox"] = scaled_bbox
            scaled_results.append(e)
        return scaled_results

parser = VLMParser(
    som_model_path="OmniParser-v2.0/icon_detect/model.pt",
    caption_model_name="florence2",
    caption_model_path="OmniParser-v2.0/icon_caption_florence/",
    device="cuda"
)

def main(data, output):
    processed = set()
    if os.path.exists(output):
        with open(output, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                for path, _ in result.items():
                    processed.add(path)

    dataset = [i for i in data if i not in processed]
    with open(output, 'a', encoding='utf-8') as output_file:
        for image_path in tqdm.tqdm(dataset):
            with torch.inference_mode():
                try:
                    parsed = parser.process(
                        image_path,
                        box_threshold=0.05,
                        draw_bbox_config={
                            "text_scale": 0.8,
                            "text_thickness": 2,
                            "text_padding": 3,
                            "thickness": 3,
                        }
                    )
                    item = {image_path: parsed}
                except:
                    item = {image_path: None}
                output_file.write(json.dumps(item) + '\n')
                output_file.flush()
    
def setup(idx=0,total_split=2, data_path="inp.json"):
    with open(data_path,"r") as f:
        data = json.load(f)
        data = [i["image"] for i in data]
    
    data = sorted(list(set(data)))
    data = np.array_split(data,total_split)[idx]
    os.makedirs(f"log/",exist_ok=True)
    output = f"log/{idx}.json"
    main(data, output)
    
if __name__ == "__main__":
    fire.Fire(setup)
    
