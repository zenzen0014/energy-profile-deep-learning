"""Isometric architecture diagrams based on VisualTorch's traced tensor shapes."""

import copy
from math import log2
from pathlib import Path

from matplotlib import font_manager
from PIL import Image, ImageColor, ImageDraw, ImageFont
from torch import nn
import torch


COLORS = {
    "Input": "#8BA8B8",
    "Conv1d": "#EFC879",
    "Conv2d": "#EFC879",
    "BatchNorm1d": "#8CC8DF",
    "BatchNorm2d": "#8CC8DF",
    "ReLU": "#8BD684",
    "MaxPool1d": "#EC9675",
    "MaxPool2d": "#EC9675",
    "Flatten": "#B7B2C8",
    "Linear": "#C39ACC",
    "Dropout": "#DCC77D",
}


def _mix(color, target, amount):
    source = ImageColor.getrgb(color)
    target = ImageColor.getrgb(target)
    return tuple(round(a * (1 - amount) + b * amount) for a, b in zip(source, target))


def _groups(model, input_shape):
    """Use VisualTorch's traced output shapes, then group adjacent operations."""
    try:
        from visualtorch.backend import extract_architecture
    except ImportError as exc:
        raise ImportError("Install VisualTorch with: pip install visualtorch==1.4.1") from exc

    with torch.random.fork_rng(devices=[]):
        architecture = extract_architecture(
            copy.deepcopy(model).cpu().eval(), (1, *input_shape)
        )
    layers = [
        layer for column in architecture.columns for layer in column
        if type(layer.module).__name__ not in {"Input", "Output"}
    ]
    groups = [{"name": "Input", "types": ["Input"], "shape": (1, *input_shape)}]
    conv_number = 0
    index = 0
    while index < len(layers):
        layer = layers[index]
        kind = type(layer.module).__name__
        operations = [kind]
        if isinstance(layer.module, (nn.Conv1d, nn.Conv2d)):
            conv_number += 1
            name = f"Conv block {conv_number}"
            following = (nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU)
        elif isinstance(layer.module, nn.Linear):
            name = "Dense + dropout" if any(
                isinstance(other.module, nn.Dropout) for other in layers[index + 1:index + 3]
            ) else "Output"
            following = (nn.ReLU, nn.Dropout)
        else:
            name = kind
            following = ()
        while index + 1 < len(layers) and isinstance(layers[index + 1].module, following):
            index += 1
            operations.append(type(layers[index].module).__name__)
        groups.append({"name": name, "types": operations, "shape": layer.output_shape})
        index += 1
    return groups


def _shape_label(shape):
    if len(shape) == 4:
        return f"{shape[1]} ch × {shape[2]} × {shape[3]}"
    if len(shape) == 3:
        return f"{shape[1]} ch × {shape[2]} steps"
    return f"{shape[-1]} features" if shape[-1] != 3 else "3 class logits"


def _box_size(shape):
    if len(shape) == 4:
        return round(42 + 1.6 * shape[3]), round(76 + 5.7 * shape[2])
    if len(shape) == 3:
        channels, steps = shape[1:]
        return round(42 + 6 * log2(channels + 1)), round(76 + 5.7 * steps)
    return round(44 + 5 * log2(shape[-1] + 1)), 94


def _depth(shape):
    return round(18 + 4 * log2(shape[1] + 1)) if len(shape) == 4 else 20


def _font(size, bold=False):
    properties = font_manager.FontProperties(
        family="DejaVu Sans", weight="bold" if bold else "normal"
    )
    return ImageFont.truetype(font_manager.findfont(properties), size)


def _center(draw, x, y, value, font, fill="#24313C"):
    bounds = draw.textbbox((0, 0), value, font=font)
    draw.text((x - (bounds[2] - bounds[0]) / 2, y), value, font=font, fill=fill)


def _prism(draw, x, center_y, width, height, color, dx=20):
    top = round(center_y - height / 2)
    bottom = top + height
    dy = 17
    edge = "#52616C"
    draw.polygon(
        [(x, top), (x + dx, top - dy), (x + width + dx, top - dy), (x + width, top)],
        fill=_mix(color, "#FFFFFF", 0.30), outline=edge,
    )
    draw.polygon(
        [(x + width, top), (x + width + dx, top - dy),
         (x + width + dx, bottom - dy), (x + width, bottom)],
        fill=_mix(color, "#52616C", 0.16), outline=edge,
    )
    draw.rectangle((x, top, x + width, bottom), fill=color, outline=edge, width=2)


def plot_architecture(model, input_shape, save_path):
    """Draw schematic volumes labeled with exact traced tensor sizes."""
    if len(input_shape) not in (2, 3):
        raise ValueError("Expected (channels, steps) or (channels, height, width)")
    is_2d = len(input_shape) == 3
    groups = _groups(model, input_shape)
    image = Image.new("RGB", (2300, 625), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(31, bold=True)
    label_font = _font(19, bold=True)
    detail_font = _font(17)
    muted = "#60717B"

    if is_2d:
        title = "2D CNN architecture · 24×24 load-profile image"
        subtitle = "Feature-map volume: channels × height × width"
        legend = ("Conv2d", "BatchNorm2d", "ReLU", "MaxPool2d", "Linear", "Dropout")
    else:
        title = "1D CNN architecture · 24-hour load profile"
        subtitle = "Feature-map volume: channels × time steps"
        legend = ("Conv1d", "BatchNorm1d", "ReLU", "MaxPool1d", "Linear", "Dropout")
    draw.text((70, 38), title, font=title_font, fill="#24313C")
    draw.text((72, 91), subtitle, font=detail_font, fill=muted)
    legend_x = 745
    for kind in legend:
        draw.rectangle((legend_x, 96, legend_x + 18, 114), fill=COLORS[kind],
                       outline="#52616C")
        draw.text((legend_x + 25, 92), kind, font=detail_font, fill="#24313C")
        legend_x += 58 + draw.textlength(kind, font=detail_font)

    positions = [130 + round(i * 2030 / max(1, len(groups) - 1)) for i in range(len(groups))]
    center_y = 325
    for index in range(len(groups) - 1):
        if is_2d:
            group = groups[index]
            width, _ = _box_size(group["shape"])
            start = positions[index] + (len(group["types"]) - 1) * 35
            start += width + _depth(group["shape"]) + 8
        else:
            start = positions[index] + (151 if len(groups[index]["types"]) > 1 else 102)
        end = positions[index + 1] - 12
        if end > start:
            draw.line((start, center_y, end, center_y), fill="#8796A0", width=2)
            draw.polygon([(end, center_y), (end - 10, center_y - 5),
                          (end - 10, center_y + 5)], fill="#8796A0")

    for group, x in zip(groups, positions):
        shape = group["shape"]
        width, height = _box_size(shape)
        operations = group["types"]
        for offset, kind in enumerate(operations):
            _prism(draw, x + offset * 35, center_y, width, height,
                   COLORS[kind], dx=_depth(shape))
        group_center = x + (len(operations) - 1) * 17.5 + width / 2
        _center(draw, group_center, 163, group["name"], label_font)
        _center(draw, group_center, 485, _shape_label(shape), detail_font, muted)

    draw.line((70, 551, 2230, 551), fill="#D6DDE1", width=2)
    draw.text((72, 570), "Volumes are schematic; labels show exact traced tensor sizes.",
              font=detail_font, fill=muted)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
    return image
