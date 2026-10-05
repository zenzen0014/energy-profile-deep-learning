"""Graphviz diagrams with tensor shapes traced directly from PyTorch models."""

import copy
from html import escape
from pathlib import Path

from graphviz import Digraph
from PIL import Image
from torch import nn
import torch


COLORS = {
    "Input": "#0EA5A8",
    "Convolution": "#356AE6",
    "Classifier": "#8756D9",
    "Output": "#E96B84",
}

def _groups(model, input_shape):
    """Trace tensor sizes and combine implementation layers into readable stages."""
    try:
        from visualtorch.backend import extract_architecture
        with torch.random.fork_rng(devices=[]):
            architecture = extract_architecture(
                copy.deepcopy(model).cpu().eval(), (1, *input_shape)
            )
        layers = [
            (layer.module, layer.output_shape)
            for column in architecture.columns for layer in column
            if type(layer.module).__name__ not in {"Input", "Output"}
        ]
    except ImportError:
        # VisualTorch is optional: hooks make the diagram portable to a plain
        # PyTorch teaching environment while retaining actual output dimensions.
        traced_model = copy.deepcopy(model).cpu().eval()
        layers = []

        def capture(module, _inputs, output):
            if isinstance(output, torch.Tensor):
                layers.append((module, tuple(output.shape)))

        hooks = [
            module.register_forward_hook(capture)
            for module in traced_model.modules()
            if module is not traced_model and not any(module.children())
        ]
        try:
            with torch.no_grad():
                traced_model(torch.zeros((1, *input_shape)))
        finally:
            for hook in hooks:
                hook.remove()
    groups = [{
        "name": "Input",
        "shape": (1, *input_shape),
        "color": "Input",
        "operations": "Load-profile tensor",
    }]
    conv_number = 0
    index = 0
    while index < len(layers):
        layer, _ = layers[index]
        kind = type(layer).__name__
        if isinstance(layer, (nn.Conv1d, nn.Conv2d)):
            conv_number += 1
            name = f"Conv block {conv_number}"
            following = (nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU, nn.MaxPool1d, nn.MaxPool2d)
            color = "Convolution"
        elif isinstance(layer, nn.Flatten):
            index += 1
            continue
        elif isinstance(layer, nn.Linear):
            has_dropout = any(isinstance(other, nn.Dropout) for other, _ in layers[index + 1:index + 3])
            name = "Classifier" if has_dropout else "Prediction"
            following = (nn.ReLU, nn.Dropout)
            color = "Classifier" if has_dropout else "Output"
        else:
            index += 1
            continue
        operations = [kind]
        while index + 1 < len(layers) and isinstance(layers[index + 1][0], following):
            index += 1
            operations.append(type(layers[index][0]).__name__)
        groups.append({
            "name": name,
            "shape": layers[index][1],
            "color": color,
            "operations": "  ·  ".join(operations),
        })
        index += 1
    return groups


def _shape_label(shape):
    if len(shape) == 4:
        return f"{shape[1]} ch × {shape[2]} × {shape[3]}"
    if len(shape) == 3:
        return f"{shape[1]} ch × {shape[2]} steps"
    return f"{shape[-1]} features" if shape[-1] != 3 else "3 class logits"


def plot_architecture(model, input_shape, save_path):
    """Render a clean, presentation-ready CNN diagram using Graphviz."""
    if len(input_shape) not in (2, 3):
        raise ValueError("Expected (channels, steps) or (channels, height, width)")
    is_2d = len(input_shape) == 3
    groups = _groups(model, input_shape)
    if is_2d:
        title = "2D CNN architecture · 24×24 load-profile image"
        subtitle = "Feature-map volume: channels × height × width"
    else:
        title = "1D CNN architecture · 24-hour load profile"
        subtitle = "Feature-map volume: channels × time steps"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    graph = Digraph("cnn_architecture", format="png")
    graph.attr(
        rankdir="LR", bgcolor="#F7F9FC", pad="0.45", nodesep="0.45", ranksep="0.90",
        fontname="Helvetica", labelloc="t", labeljust="l",
        label=f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="20"><B>{escape(title)}</B></FONT></TD></TR>
<TR><TD HEIGHT="8"></TD></TR>
<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="14" COLOR="#65758B">{escape(subtitle)}</FONT></TD></TR>
</TABLE>>''',
    )
    graph.attr("node", shape="plain", fontname="Helvetica")
    graph.attr("edge", color="#A9B8CC", penwidth="2", arrowsize="0.75")

    for index, group in enumerate(groups):
        color = COLORS[group["color"]]
        operation = group["operations"].replace("BatchNorm2d", "Batch norm").replace("BatchNorm1d", "Batch norm")
        operation = operation.replace("MaxPool2d", "Max pool").replace("MaxPool1d", "Max pool")
        operation = operation.replace("Conv2d", "Conv").replace("Conv1d", "Conv").replace("Linear", "Dense")
        label = f'''<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="12" BGCOLOR="white">
<TR><TD BGCOLOR="{color}"><FONT COLOR="white"><B>{escape(group["name"])}</B></FONT></TD></TR>
<TR><TD><FONT POINT-SIZE="18" COLOR="#172033"><B>{escape(_shape_label(group["shape"]))}</B></FONT></TD></TR>
<TR><TD><FONT POINT-SIZE="11" COLOR="#65758B">{escape(operation)}</FONT></TD></TR>
</TABLE>>'''
        graph.node(f"stage_{index}", label=label, margin="0", color="#DDE5EE")
        if index:
            graph.edge(f"stage_{index - 1}", f"stage_{index}")

    rendered_path = Path(graph.render(filename=str(save_path.with_suffix("")), cleanup=True))
    if rendered_path != save_path:
        rendered_path.replace(save_path)
    with Image.open(save_path) as rendered:
        return rendered.convert("RGB").copy()
