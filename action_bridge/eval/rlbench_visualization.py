"""Interactive 3D visualizations for cached RLBench episodes and training batches."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from phi_rlbench.data.actions import decode_action_chunk
from phi_rlbench.data.cache import RLBenchCacheStore
from plotly.subplots import make_subplots

_AXIS_COLORS = ("#ef4444", "#22c55e", "#3b82f6")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _rgb_strings(rgb: np.ndarray) -> List[str]:
    values = np.asarray(rgb)
    if values.dtype != np.uint8:
        if values.size and float(np.nanmax(values)) <= 1.0 + 1e-6:
            values = values * 255.0
        values = np.clip(values, 0, 255).astype(np.uint8)
    return [
        f"rgb({int(red)},{int(green)},{int(blue)})"
        for red, green, blue in values.reshape(-1, 3)
    ]


def _quaternion_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _orientation_traces(
    state: np.ndarray,
    *,
    axis_length: float,
    showlegend: bool,
    scene: Optional[str] = None,
) -> List[go.Scatter3d]:
    state = np.asarray(state, dtype=np.float32)
    origin = state[:3]
    rotation = _quaternion_rotation_matrix(state[3:7])
    traces = []
    for axis, (name, color) in enumerate(zip(("gripper x", "gripper y", "gripper z"), _AXIS_COLORS)):
        endpoint = origin + float(axis_length) * rotation[:, axis]
        kwargs: Dict[str, Any] = {}
        if scene is not None:
            kwargs["scene"] = scene
        traces.append(
            go.Scatter3d(
                x=[origin[0], endpoint[0]],
                y=[origin[1], endpoint[1]],
                z=[origin[2], endpoint[2]],
                mode="lines",
                line={"color": color, "width": 6},
                name=name,
                legendgroup=f"orientation-{axis}",
                showlegend=showlegend,
                hoverinfo="skip",
                **kwargs,
            )
        )
    return traces


def _point_trace(
    xyz: np.ndarray,
    rgb: np.ndarray,
    valid: np.ndarray,
    mask_id: Optional[np.ndarray],
    *,
    name: str = "RGB point cloud",
    marker_size: float = 2.5,
    showlegend: bool = True,
    scene: Optional[str] = None,
) -> go.Scatter3d:
    valid = np.asarray(valid, dtype=np.bool_).reshape(-1)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)[valid]
    rgb = np.asarray(rgb).reshape(-1, 3)[valid]
    customdata = None
    hovertemplate = "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>"
    if mask_id is not None:
        customdata = np.asarray(mask_id).reshape(-1)[valid, None]
        hovertemplate = (
            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
            "<br>mask=%{customdata[0]}<extra></extra>"
        )
    kwargs: Dict[str, Any] = {}
    if scene is not None:
        kwargs["scene"] = scene
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        marker={"size": marker_size, "color": _rgb_strings(rgb), "opacity": 0.9},
        customdata=customdata,
        hovertemplate=hovertemplate,
        name=name,
        legendgroup="point-cloud",
        showlegend=showlegend,
        **kwargs,
    )


def _path_trace(
    positions: np.ndarray,
    *,
    name: str,
    color: str,
    width: float,
    showlegend: bool = True,
    dash: Optional[str] = None,
    scene: Optional[str] = None,
) -> go.Scatter3d:
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    kwargs: Dict[str, Any] = {}
    if scene is not None:
        kwargs["scene"] = scene
    line: Dict[str, Any] = {"color": color, "width": width}
    if dash is not None:
        line["dash"] = dash
    return go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="lines+markers",
        line=line,
        marker={"size": 3, "color": color},
        name=name,
        legendgroup=name,
        showlegend=showlegend,
        hovertemplate="x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        **kwargs,
    )


def _marker_trace(
    position: np.ndarray,
    *,
    name: str,
    color: str,
    symbol: str = "circle",
    size: float = 7,
    showlegend: bool = True,
    scene: Optional[str] = None,
) -> go.Scatter3d:
    position = np.asarray(position, dtype=np.float32).reshape(3)
    kwargs: Dict[str, Any] = {}
    if scene is not None:
        kwargs["scene"] = scene
    return go.Scatter3d(
        x=[position[0]],
        y=[position[1]],
        z=[position[2]],
        mode="markers",
        marker={"size": size, "color": color, "symbol": symbol, "line": {"width": 1, "color": "#111827"}},
        name=name,
        legendgroup=name,
        showlegend=showlegend,
        hovertemplate=f"{name}<br>x=%{{x:.3f}}<br>y=%{{y:.3f}}<br>z=%{{z:.3f}}<extra></extra>",
        **kwargs,
    )


def _axis_ranges(point_sets: Sequence[np.ndarray], padding_fraction: float = 0.06) -> Dict[str, List[float]]:
    finite_sets = []
    for points in point_sets:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            finite_sets.append(points)
    if not finite_sets:
        return {"x": [-1.0, 1.0], "y": [-1.0, 1.0], "z": [0.0, 2.0]}
    merged = np.concatenate(finite_sets, axis=0)
    low = merged.min(axis=0)
    high = merged.max(axis=0)
    extent = np.maximum(high - low, 0.1)
    padding = float(padding_fraction) * extent
    return {
        "x": [float(low[0] - padding[0]), float(high[0] + padding[0])],
        "y": [float(low[1] - padding[1]), float(high[1] + padding[1])],
        "z": [float(low[2] - padding[2]), float(high[2] + padding[2])],
    }


def _scene_settings(ranges: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    return {
        "xaxis": {"title": "x (m)", "range": list(ranges["x"]), "backgroundcolor": "#f8fafc"},
        "yaxis": {"title": "y (m)", "range": list(ranges["y"]), "backgroundcolor": "#f8fafc"},
        "zaxis": {"title": "z (m)", "range": list(ranges["z"]), "backgroundcolor": "#f8fafc"},
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.05}},
    }


def episode_animation_figure(
    store: RLBenchCacheStore,
    variation_index: int,
    episode_id: int,
    *,
    frame_stride: int = 2,
    max_frames: int = 80,
    chunk_horizon: int = 16,
    point_count: Optional[int] = None,
    orientation_axis_length: float = 0.06,
) -> go.Figure:
    """Build an animated RGB point-cloud episode with the BC target horizon."""

    length = store.episode_length(variation_index, episode_id)
    stride = max(1, int(frame_stride))
    if int(max_frames) > 0:
        stride = max(stride, int(ceil(length / int(max_frames))))
    time_indices = np.arange(0, length, stride, dtype=np.int64)
    if time_indices[-1] != length - 1:
        time_indices = np.concatenate([time_indices, [length - 1]])

    fields = set(store.available_fields(variation_index, episode_id))
    requested = ["xyz", "valid", "state"]
    if "rgb" in fields:
        requested.append("rgb")
    if "mask_id" in fields:
        requested.append("mask_id")
    frames_data = store.load_episode_slices(
        variation_index,
        episode_id,
        time_indices,
        fields=requested,
    )
    full_state = store.load_episode_slices(
        variation_index,
        episode_id,
        np.arange(length),
        fields=("state",),
    )["state"].astype(np.float32)

    cached_points = int(frames_data["xyz"].shape[1])
    display_points = cached_points if point_count is None else min(int(point_count), cached_points)
    point_indices = np.linspace(0, cached_points - 1, display_points).round().astype(np.int64)
    xyz = frames_data["xyz"][:, point_indices].astype(np.float32)
    valid = frames_data["valid"][:, point_indices].astype(np.bool_)
    rgb = frames_data.get(
        "rgb",
        np.full((*xyz.shape[:-1], 3), [70, 130, 220], dtype=np.uint8),
    )[:, point_indices]
    mask_id = frames_data.get("mask_id")
    if mask_id is not None:
        mask_id = mask_id[:, point_indices]

    key = store.keys[int(variation_index)]
    task_title = key.task.replace("_", " ")
    ranges = _axis_ranges([xyz.reshape(-1, 3), full_state[:, :3]])

    def dynamic_traces(frame_number: int) -> List[go.Scatter3d]:
        time_index = int(time_indices[frame_number])
        current = full_state[time_index]
        future_end = min(length, time_index + 1 + int(chunk_horizon))
        future = full_state[time_index + 1 : future_end, :3]
        if not len(future):
            future = current[None, :3]
        open_state = float(current[7]) >= 0.5
        traces: List[go.Scatter3d] = [
            _point_trace(
                xyz[frame_number],
                rgb[frame_number],
                valid[frame_number],
                None if mask_id is None else mask_id[frame_number],
            ),
            _path_trace(
                full_state[: time_index + 1, :3],
                name="executed so far",
                color="#7c3aed",
                width=7,
            ),
            _path_trace(
                future,
                name=f"next {chunk_horizon} targets",
                color="#f59e0b",
                width=7,
            ),
            _marker_trace(
                current[:3],
                name="current gripper",
                color="#22c55e" if open_state else "#ef4444",
                size=9,
            ),
        ]
        traces.extend(
            _orientation_traces(
                current,
                axis_length=orientation_axis_length,
                showlegend=True,
            )
        )
        return traces

    initial_dynamic = dynamic_traces(0)
    figure = go.Figure(
        data=[
            initial_dynamic[0],
            _path_trace(
                full_state[:, :3],
                name="full expert trajectory",
                color="#64748b",
                width=3,
                dash="dot",
            ),
            *initial_dynamic[1:],
            _marker_trace(full_state[0, :3], name="episode start", color="#0ea5e9", symbol="diamond"),
            _marker_trace(full_state[-1, :3], name="episode end", color="#111827", symbol="diamond"),
        ]
    )
    dynamic_trace_indices = [0, 2, 3, 4, 5, 6, 7]
    animation_frames = []
    for frame_number, time_index in enumerate(time_indices):
        state = full_state[int(time_index)]
        animation_frames.append(
            go.Frame(
                name=str(int(time_index)),
                data=dynamic_traces(frame_number),
                traces=dynamic_trace_indices,
                layout=go.Layout(
                    title={
                        "text": (
                            f"{task_title} · variation {key.variation} · episode {episode_id}"
                            f"<br><sup>frame {int(time_index)}/{length - 1} · "
                            f"gripper {'open' if state[7] >= 0.5 else 'closed'} · "
                            f"orange = action chunk beginning at t+1</sup>"
                        )
                    }
                ),
            )
        )
    figure.frames = animation_frames
    slider_steps = [
        {
            "label": str(int(time_index)),
            "method": "animate",
            "args": [
                [str(int(time_index))],
                {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
            ],
        }
        for time_index in time_indices
    ]
    figure.update_layout(
        title={
            "text": (
                f"{task_title} · variation {key.variation} · episode {episode_id}"
                f"<br><sup>frame 0/{length - 1} · orange = action chunk beginning at t+1</sup>"
            )
        },
        template="plotly_white",
        scene=_scene_settings(ranges),
        height=760,
        margin={"l": 10, "r": 10, "t": 95, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 0.0,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 140, "redraw": True},
                                "transition": {"duration": 0},
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "frame "},
                "pad": {"t": 35},
                "steps": slider_steps,
            }
        ],
    )
    return figure


def training_batch_figure(
    batch: Mapping[str, Any],
    sample_metadata: Sequence[Mapping[str, Any]],
    *,
    action_representation: str,
    columns: int = 3,
) -> go.Figure:
    """Visualize a collated training batch in world coordinates."""

    point_cloud = _as_numpy(batch["point_cloud_hist"]).astype(np.float32)
    point_valid = _as_numpy(batch["point_valid_hist"]).astype(np.bool_)
    obs_hist = _as_numpy(batch["obs_hist"]).astype(np.float32)
    future_actions = _as_numpy(batch["future_actions"]).astype(np.float32)
    future_mask = _as_numpy(batch["future_action_mask"]).astype(np.bool_)
    rgb = _as_numpy(batch["rgb_hist"]) if "rgb_hist" in batch else None
    mask_id = _as_numpy(batch["mask_id_hist"]) if "mask_id_hist" in batch else None
    batch_size = int(point_cloud.shape[0])
    columns = min(max(1, int(columns)), batch_size)
    rows = int(ceil(batch_size / columns))
    subplot_titles = []
    for metadata in sample_metadata:
        subplot_titles.append(
            f"{metadata['task']} · v{metadata['variation']} · e{metadata['episode_id']} · t={metadata['time_index']}"
        )
    subplot_titles.extend([""] * (rows * columns - len(subplot_titles)))
    figure = make_subplots(
        rows=rows,
        cols=columns,
        specs=[[{"type": "scene"} for _ in range(columns)] for _ in range(rows)],
        subplot_titles=subplot_titles,
        horizontal_spacing=0.025,
        vertical_spacing=0.08,
    )

    all_points = []
    for sample_index in range(batch_size):
        row = sample_index // columns + 1
        column = sample_index % columns + 1
        scene_name = "scene" if sample_index == 0 else f"scene{sample_index + 1}"
        latest_xyz = point_cloud[sample_index, -1]
        latest_valid = point_valid[sample_index, -1]
        latest_rgb = (
            rgb[sample_index, -1]
            if rgb is not None
            else np.full((latest_xyz.shape[0], 3), [70, 130, 220], dtype=np.uint8)
        )
        latest_mask = None if mask_id is None else mask_id[sample_index, -1]
        absolute_future = decode_action_chunk(
            future_actions[sample_index],
            observation_state=obs_hist[sample_index],
            representation=action_representation,
        )
        absolute_future = absolute_future[future_mask[sample_index]]
        showlegend = sample_index == 0
        traces: List[go.Scatter3d] = [
            _point_trace(
                latest_xyz,
                latest_rgb,
                latest_valid,
                latest_mask,
                marker_size=2.2,
                showlegend=showlegend,
                scene=scene_name,
            ),
            _path_trace(
                obs_hist[sample_index, :, :3],
                name="observation history",
                color="#7c3aed",
                width=7,
                showlegend=showlegend,
                scene=scene_name,
            ),
            _path_trace(
                np.concatenate(
                    [obs_hist[sample_index, -1:, :3], absolute_future[:, :3]],
                    axis=0,
                ),
                name="target action chunk",
                color="#f59e0b",
                width=7,
                showlegend=showlegend,
                scene=scene_name,
            ),
            _marker_trace(
                obs_hist[sample_index, -1, :3],
                name="current gripper",
                color="#22c55e" if obs_hist[sample_index, -1, 7] >= 0.5 else "#ef4444",
                size=8,
                showlegend=showlegend,
                scene=scene_name,
            ),
        ]
        traces.extend(
            _orientation_traces(
                obs_hist[sample_index, -1],
                axis_length=0.05,
                showlegend=showlegend,
                scene=scene_name,
            )
        )
        for trace in traces:
            figure.add_trace(trace, row=row, col=column)
        all_points.extend([latest_xyz[latest_valid], obs_hist[sample_index, :, :3], absolute_future[:, :3]])

    ranges = _axis_ranges(all_points)
    figure.update_scenes(**_scene_settings(ranges))
    figure.update_layout(
        title={
            "text": (
                f"RLBench training batch · {action_representation}"
                f"<br><sup>point_cloud_hist={tuple(point_cloud.shape)} · "
                f"obs_hist={tuple(obs_hist.shape)} · "
                f"future_actions={tuple(future_actions.shape)} · "
                "orange targets are decoded to world coordinates</sup>"
            )
        },
        template="plotly_white",
        width=520 * columns,
        height=455 * rows + 120,
        margin={"l": 10, "r": 10, "t": 115, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    return figure


def write_figure_html(
    figure: go.Figure,
    path: Path | str,
    *,
    embed_plotly: bool = False,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        path,
        include_plotlyjs=True if embed_plotly else "cdn",
        full_html=True,
        auto_play=False,
    )
    return path
