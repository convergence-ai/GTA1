from typing import Optional
import os
import logging
from PIL import Image, ImageFont, ImageDraw

# Logger used by callers
global_logger = logging.getLogger("global_logger")

# Panel/layout defaults (kept identical in effect)
_BASE_WIDTH = 1200
_BASE_HEIGHT = 800
_BASE_TITLE_FONT_SIZE = 16
_BASE_TEXT_FONT_SIZE = 14
_BASE_PADDING = 20
_BASE_LINE_HEIGHT = 25
_BASE_THICK_LINE_WIDTH = 3
_BASE_THIN_LINE_WIDTH = 1
_BASE_RADIUS = 5
_BASE_SMALL_RADIUS = 4
_BASE_LABEL_OFFSET = 25
_BASE_SMALL_LABEL_OFFSET = 20
_BASE_LARGE_LABEL_OFFSET = 35


def _load_fonts(title_font_size: int, text_font_size: int):
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            title_font_size,
        )
        text_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            text_font_size,
        )
        return title_font, text_font
    except Exception:
        try:
            return ImageFont.load_default(), ImageFont.load_default()
        except Exception:
            return None, None


def draw_coordinates_on_image(
    original_image: Image.Image,
    predicted_coords: list[int],
    output_text: str,
    ground_truth_coords: Optional[list[int]] = None,
    output_path: Optional[str] = None,
    instruction: str = "",
) -> str:
    """Draw predicted and ground truth coordinates on image and save.

    Returns path to the saved image with overlays.
    """
    try:
        original_width, original_height = original_image.size

        # Scales
        width_scale = original_width / _BASE_WIDTH
        height_scale = original_height / _BASE_HEIGHT
        scale_factor = min(width_scale, height_scale, 2.0)

        # Font sizes
        title_font_size = max(
            12, min(24, int(_BASE_TITLE_FONT_SIZE * scale_factor * 0.8))
        )
        text_font_size = max(
            10, min(20, int(_BASE_TEXT_FONT_SIZE * scale_factor * 0.8))
        )
        title_font, text_font = _load_fonts(title_font_size, text_font_size)

        # Dimensions
        padding = max(10, int(_BASE_PADDING * scale_factor * 0.7))
        line_height = max(15, int(_BASE_LINE_HEIGHT * scale_factor * 0.7))
        max_width = original_width - 2 * padding

        # Helpers
        def format_model_response(response_text: str) -> str:
            if not response_text or response_text == "None":
                return "Model response: None"
            cleaned_text = str(response_text).strip()
            if any(c in cleaned_text for c in ['{', '}', '[', ']', '":']):
                try:
                    import re as _re
                    action_match = _re.search(r'"action":\s*"([^"]*)"', cleaned_text)
                    coord_match = _re.search(r'"coordinate":\s*\[([^\]]*)\]', cleaned_text)
                    if action_match and coord_match:
                        return f"Model response: {action_match.group(1)} at [{coord_match.group(1)}]"
                except Exception:
                    pass
                max_len = 80
            else:
                max_len = 150
            return (
                f"Model response: {cleaned_text[:max_len]}..."
                if len(cleaned_text) > max_len
                else f"Model response: {cleaned_text}"
            )

        def format_coords(coords: list[int]) -> str:
            if not coords:
                return "None"
            try:
                if len(coords) >= 4:
                    a, b, c, d = coords[:4]
                    rounded = [round(float(x), 1) for x in [a, b, c, d]]
                    return f"[{rounded[0]}, {rounded[1]}, {rounded[2]}, {rounded[3]}]"
                rounded = [round(float(x), 1) for x in coords]
                return str(rounded)
            except Exception:
                return str(coords)

        def wrap_text(text: str, width_px: int, font_obj: ImageFont.ImageFont):
            if not font_obj:
                chars_per_line = max(30, width_px // max(6, int(8 * scale_factor * 0.8)))
                words = text.split(" ")
                lines, current, cur_len = [], [], 0
                for word in words:
                    if len(word) > int(chars_per_line * 0.7):
                        if current:
                            lines.append(" ".join(current))
                            current, cur_len = [], 0
                        while len(word) > chars_per_line:
                            lines.append(word[:chars_per_line])
                            word = word[chars_per_line:]
                        if word:
                            current, cur_len = [word], len(word)
                    elif cur_len + len(word) + (1 if current else 0) <= chars_per_line:
                        current.append(word)
                        cur_len += len(word) + (1 if len(current) > 1 else 0)
                    else:
                        if current:
                            lines.append(" ".join(current))
                        current, cur_len = [word], len(word)
                if current:
                    lines.append(" ".join(current))
                return lines
            # Font-aware wrap
            words = text.split(" ")
            lines, current = [], []
            for word in words:
                test_line = " ".join(current + [word])
                try:
                    bbox = font_obj.getbbox(test_line)
                    text_width = bbox[2] - bbox[0]
                except Exception:
                    text_width = len(test_line) * 8
                if text_width <= width_px:
                    current.append(word)
                else:
                    if current:
                        lines.append(" ".join(current))
                        current = []
                    try:
                        word_bbox = font_obj.getbbox(word)
                        word_width = word_bbox[2] - word_bbox[0]
                    except Exception:
                        word_width = len(word) * 8
                    if word_width > width_px:
                        temp = ""
                        for ch in word:
                            t2 = temp + ch
                            try:
                                tb = font_obj.getbbox(t2)
                                w2 = tb[2] - tb[0]
                            except Exception:
                                w2 = len(t2) * 8
                            if w2 <= width_px:
                                temp = t2
                            else:
                                if temp:
                                    lines.append(temp)
                                temp = ch
                        if temp:
                            current = [temp]
                    else:
                        current = [word]
            if current:
                lines.append(" ".join(current))
            return lines

        instruction_text = f"Instruction: {instruction}" if instruction else "Instruction: [No instruction provided]"
        pred_text = f"Rescaled coordinates: {format_coords(predicted_coords)}"
        gt_text = f"Ground Truth: {format_coords(ground_truth_coords)}"
        output_text_fmt = format_model_response(output_text)

        instruction_lines = wrap_text(instruction_text, max_width, text_font)
        pred_lines = wrap_text(pred_text, max_width, title_font)
        gt_lines = wrap_text(gt_text, max_width, title_font)
        output_lines = wrap_text(output_text_fmt, max_width, title_font)

        extra_spacing = max(5, int(8 * scale_factor * 0.6))
        panel_height = (
            padding * 2
            + line_height * len(instruction_lines)
            + extra_spacing
            + line_height * len(pred_lines)
            + line_height * len(gt_lines)
            + extra_spacing
            + line_height * len(output_lines)
        )

        new_image = Image.new("RGB", (original_width, original_height + panel_height), "white")

        draw = ImageDraw.Draw(new_image)
        border_width = max(1, int(2 * scale_factor))
        draw.rectangle(
            [0, 0, original_width, panel_height],
            fill="lightgrey",
            outline="darkgrey",
            width=border_width,
        )

        y_offset = padding
        for line in instruction_lines:
            draw.text((padding, y_offset), line, fill="black", font=text_font or None)
            y_offset += line_height
        y_offset += extra_spacing

        for line in pred_lines:
            draw.text((padding, y_offset), line, fill="darkred", font=title_font or None)
            y_offset += line_height
        for line in gt_lines:
            draw.text((padding, y_offset), line, fill="darkgreen", font=title_font or None)
            y_offset += line_height
        y_offset += extra_spacing
        for line in output_lines:
            draw.text((padding, y_offset), line, fill="black", font=title_font or None)
            y_offset += line_height

        new_image.paste(original_image, (0, panel_height))
        draw = ImageDraw.Draw(new_image)

        thick_line_width = max(2, min(4, int(_BASE_THICK_LINE_WIDTH * scale_factor * 0.8)))
        thin_line_width = max(1, int(_BASE_THIN_LINE_WIDTH * scale_factor * 0.8))
        radius = max(3, min(8, int(_BASE_RADIUS * scale_factor * 0.8)))
        small_radius = max(3, min(6, int(_BASE_SMALL_RADIUS * scale_factor * 0.8)))
        label_offset = max(15, int(_BASE_LABEL_OFFSET * scale_factor * 0.7))
        small_label_offset = max(12, int(_BASE_SMALL_LABEL_OFFSET * scale_factor * 0.7))
        large_label_offset = max(25, int(_BASE_LARGE_LABEL_OFFSET * scale_factor * 0.7))

        if ground_truth_coords and len(ground_truth_coords) >= 4:
            x1, y1, x2, y2 = ground_truth_coords[:4]
            if x2 < x1 or y2 < y1:
                w, h = x2, y2
                x2, y2 = x1 + w, y1 + h
            y1 += panel_height
            y2 += panel_height
            if x1 == x2 and y1 == y2:
                draw.ellipse([x1 - radius, y1 - radius, x1 + radius, y1 + radius], outline="lime", width=thick_line_width)
                draw.ellipse([x1 - radius - 1, y1 - radius - 1, x1 + radius + 1, y1 + radius + 1], outline="white", width=thin_line_width)
            else:
                draw.rectangle([x1, y1, x2, y2], outline="lime", width=thick_line_width)
                draw.rectangle([x1 - 1, y1 - 1, x2 + 1, y2 + 1], outline="white", width=thin_line_width)
            label_text = "Ground Truth"
            label_y = y1 - label_offset
            if title_font:
                text_padding = max(1, int(2 * scale_factor))
                bbox = draw.textbbox((x1, label_y), label_text, font=title_font)
                draw.rectangle(
                    [bbox[0] - text_padding, bbox[1] - text_padding, bbox[2] + text_padding, bbox[3] + text_padding],
                    fill="white",
                    outline="black",
                )
                draw.text((x1, label_y), label_text, fill="darkgreen", font=title_font)
            else:
                draw.text((x1, label_y), label_text, fill="darkgreen")

        if predicted_coords and len(predicted_coords) >= 4:
            px1, py1, px2, py2 = predicted_coords[:4]
            py1 += panel_height
            py2 += panel_height
            if px1 == px2 and py1 == py2:
                draw.ellipse([px1 - small_radius, py1 - small_radius, px1 + small_radius, py1 + small_radius], outline="red", width=max(2, int(3 * scale_factor)))
                draw.ellipse([px1 - small_radius - 1, py1 - small_radius - 1, px1 + small_radius + 1, py1 + small_radius + 1], outline="white", width=max(1, int(1 * scale_factor)))
            else:
                pred_line_width = max(2, min(3, int(2.5 * scale_factor * 0.8)))
                draw.rectangle([px1, py1, px2, py2], outline="red", width=pred_line_width)
                draw.rectangle([px1 - 1, py1 - 1, px2 + 1, py2 + 1], outline="white", width=max(1, int(0.8 * scale_factor * 0.8)))
            label_text = "Predicted"
            label_y = py1 - (large_label_offset if ground_truth_coords and len(ground_truth_coords) >= 4 else small_label_offset)
            if title_font:
                text_padding = max(1, int(2 * scale_factor))
                bbox = draw.textbbox((px1, label_y), label_text, font=title_font)
                draw.rectangle(
                    [bbox[0] - text_padding, bbox[1] - text_padding, bbox[2] + text_padding, bbox[3] + text_padding],
                    fill="white",
                    outline="black",
                )
                draw.text((px1, label_y), label_text, fill="darkred", font=title_font)
            else:
                draw.text((px1, label_y), label_text, fill="darkred")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        new_image.save(output_path)
        global_logger.info(f"Saved image with overlays to: {output_path}")
        return output_path

    except Exception as e:
        global_logger.error(f"Error drawing coordinates on image: {e}")
        return output_path