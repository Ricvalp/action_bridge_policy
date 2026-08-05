"""Interactive 3D diagnostics for JAX RLBench action chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _numpy(value):
    return np.asarray(value)


def _rgb_strings(rgb: np.ndarray):
    values = np.asarray(rgb)
    if values.dtype != np.uint8:
        values = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    return [f"rgb({r},{g},{b})" for r, g, b in values]


def prediction_chunk_figure(
    batch: Dict[str, np.ndarray],
    output: Dict[str, np.ndarray],
    *,
    num_examples: int = 4,
) -> go.Figure:
    """Show point cloud, expert/predicted paths, and learned attractors."""

    batch = {key: _numpy(value) for key, value in batch.items()}
    output = {key: _numpy(value) for key, value in output.items()}
    count = min(int(num_examples), int(batch["future_actions"].shape[0]))
    figure = make_subplots(
        rows=1,
        cols=count,
        specs=[[{"type": "scene"}] * count],
        subplot_titles=[f"Example {index}" for index in range(count)],
    )
    center = np.asarray((0.0, 0.0, 1.25), dtype=np.float32)
    scale = np.asarray((1.0, 1.0, 1.25), dtype=np.float32)
    for index in range(count):
        cloud = batch["point_cloud_hist"][index, -1]
        valid = batch["point_valid_hist"][index, -1].astype(bool)
        rgb = batch.get("rgb_hist")
        colors = (
            _rgb_strings(rgb[index, -1, valid])
            if rgb is not None
            else "#94a3b8"
        )
        figure.add_trace(
            go.Scatter3d(
                x=cloud[valid, 0],
                y=cloud[valid, 1],
                z=cloud[valid, 2],
                mode="markers",
                marker={"size": 1.8, "color": colors, "opacity": 0.55},
                name="point cloud",
                legendgroup="cloud",
                showlegend=index == 0,
            ),
            row=1,
            col=index + 1,
        )
        paths = [
            (batch["future_actions"][index, :, :3], "expert", "#16a34a", None),
            (output["actions"][index, :, :3], "free rollout", "#7c3aed", None),
        ]
        if "teacher_position" in output:
            paths.append(
                (
                    output["teacher_position"][index] * scale + center,
                    "teacher-forced",
                    "#2563eb",
                    "dash",
                )
            )
        if "attractor" in output:
            paths.append(
                (
                    output["attractor"][index] * scale + center,
                    "attractor m(k)",
                    "#dc2626",
                    "dot",
                )
            )
        for path, name, color, dash in paths:
            figure.add_trace(
                go.Scatter3d(
                    x=path[:, 0],
                    y=path[:, 1],
                    z=path[:, 2],
                    mode="lines+markers",
                    line={"color": color, "width": 6, "dash": dash},
                    marker={"color": color, "size": 3},
                    name=name,
                    legendgroup=name,
                    showlegend=index == 0,
                ),
                row=1,
                col=index + 1,
            )
    figure.update_layout(
        title="RLBench predicted chunks and contact attractors",
        height=560,
        width=max(650, 540 * count),
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
    )
    for index in range(1, count + 1):
        scene_name = "scene" if index == 1 else f"scene{index}"
        figure.layout[scene_name].update(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            aspectmode="data",
        )
    return figure


def write_prediction_chunk_html(figure: go.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path

