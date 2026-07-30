"""Build flexible dense RLBench caches from generated demonstration episodes."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import zlib

import h5py
import numpy as np
from tqdm import tqdm

from action_bridge.data.rlbench_cache import CACHE_SCHEMA_NAME, CACHE_SCHEMA_VERSION


DEFAULT_WORKSPACE_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.5))
DEFAULT_LOW_DIM_FIELDS = (
    "gripper_pose",
    "gripper_open",
    "joint_positions",
    "joint_velocities",
    "joint_forces",
    "gripper_joint_positions",
    "gripper_touch_forces",
    "task_low_dim_state",
)
MASK_NAMES_TO_IGNORE = (
    "Floor",
    "Wall1",
    "Wall2",
    "Wall3",
    "Wall4",
    "Roof",
    "workspace",
    "diningTable_visible",
)
MASK_NAME_SUBSTRINGS_TO_IGNORE = (
    "floor",
    "wall",
    "roof",
    "workspace",
    "table",
    "panda_link",
)


def _numbered_directories(parent: Path, prefix: str) -> List[Path]:
    paths = []
    for path in parent.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        try:
            number = int(path.name.removeprefix(prefix))
        except ValueError:
            continue
        paths.append((number, path))
    return [path for _, path in sorted(paths)]


def _numbered_files(parent: Path, suffix: str) -> List[Path]:
    paths = []
    for path in parent.glob(f"*{suffix}"):
        try:
            number = int(path.stem)
        except ValueError:
            continue
        paths.append((number, path))
    return [path for _, path in sorted(paths)]


def _load_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not unpickle {path}. Raw RLBench observations retain their Python "
            "class, so install the same RLBench/PyRep package used to generate them. "
            "The resulting HDF5 cache does not require RLBench to load."
        ) from exc


def _load_label_map(variation_dir: Path) -> Dict[int, str]:
    path = variation_dir / "mask_to_label.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return {int(key): str(value) for key, value in json.load(handle).items()}


def _ignored_mask_ids(label_map: Mapping[int, str]) -> Tuple[int, ...]:
    ignored = set()
    for mask_id, name in label_map.items():
        lower = str(name).lower()
        if name in MASK_NAMES_TO_IGNORE or any(
            token in lower for token in MASK_NAME_SUBSTRINGS_TO_IGNORE
        ):
            ignored.add(int(mask_id))
    return tuple(sorted(ignored))


def _workspace_mask(
    points: np.ndarray,
    bounds: Optional[Sequence[Sequence[float]]],
) -> np.ndarray:
    if bounds is None:
        return np.ones(points.shape[0], dtype=np.bool_)
    if len(bounds) != 3 or any(len(axis) != 2 for axis in bounds):
        raise ValueError("workspace_bounds must be ((xmin,xmax),(ymin,ymax),(zmin,zmax)).")
    return np.logical_and.reduce(
        [
            points[:, axis] >= float(bounds[axis][0])
            for axis in range(3)
        ]
        + [
            points[:, axis] <= float(bounds[axis][1])
            for axis in range(3)
        ]
    )


def _normalize_colors(colors: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors)
    if np.issubdtype(colors.dtype, np.floating):
        finite_max = float(np.nanmax(colors)) if colors.size else 0.0
        if finite_max <= 1.0 + 1e-6:
            colors = colors * 255.0
    return np.clip(colors, 0, 255).astype(np.uint8)


def _sample_points(
    points: np.ndarray,
    colors: np.ndarray,
    masks: np.ndarray,
    *,
    num_points: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    if points.shape[0] == 0:
        return {
            "xyz": np.zeros((num_points, 3), dtype=np.float32),
            "rgb": np.zeros((num_points, 3), dtype=np.uint8),
            "mask_id": np.zeros((num_points,), dtype=np.int32),
            "valid": np.zeros((num_points,), dtype=np.bool_),
        }
    indices = rng.choice(
        points.shape[0],
        size=int(num_points),
        replace=points.shape[0] < int(num_points),
    )
    return {
        "xyz": points[indices].astype(np.float32, copy=False),
        "rgb": colors[indices].astype(np.uint8, copy=False),
        "mask_id": masks[indices].astype(np.int32, copy=False),
        "valid": np.ones((int(num_points),), dtype=np.bool_),
    }


def _load_point_frame(
    path: Path,
    *,
    num_points: int,
    rng: np.random.Generator,
    ignored_ids: Sequence[int],
    workspace_bounds: Optional[Sequence[Sequence[float]]],
) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        missing = [key for key in ("points", "colors", "masks") if key not in data]
        if missing:
            raise KeyError(f"{path} is missing point-cloud arrays: {missing}")
        points = np.asarray(data["points"], dtype=np.float32).reshape(-1, 3)
        colors = _normalize_colors(np.asarray(data["colors"]).reshape(-1, 3))
        masks = np.asarray(data["masks"], dtype=np.int32).reshape(-1)
    if not (len(points) == len(colors) == len(masks)):
        raise ValueError(
            f"Point/color/mask lengths differ in {path}: "
            f"{len(points)}, {len(colors)}, {len(masks)}."
        )
    keep = np.isfinite(points).all(axis=1)
    if ignored_ids:
        keep &= ~np.isin(masks, np.asarray(ignored_ids, dtype=np.int32))
    keep &= _workspace_mask(points, workspace_bounds)
    return _sample_points(
        points[keep],
        colors[keep],
        masks[keep],
        num_points=num_points,
        rng=rng,
    )


def _numeric_observation_field(observation: Any, field: str) -> Optional[np.ndarray]:
    value = getattr(observation, field, None)
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        return None
    if array.size == 0:
        return None
    if field == "gripper_open":
        array = array.reshape(1)
    return array.astype(np.float32).reshape(-1)


def _extract_low_dim_fields(
    observations: Sequence[Any],
    fields: Sequence[str],
) -> Dict[str, np.ndarray]:
    output: Dict[str, np.ndarray] = {}
    for field in fields:
        values = [_numeric_observation_field(observation, field) for observation in observations]
        present = [value for value in values if value is not None]
        if not present:
            continue
        if len(present) != len(values):
            raise ValueError(f"Low-dimensional field {field!r} is missing in part of an episode.")
        shapes = {value.shape for value in present}
        if len(shapes) != 1:
            raise ValueError(f"Low-dimensional field {field!r} changes shape: {sorted(shapes)}.")
        output[field] = np.stack(present, axis=0).astype(np.float32)

    if "gripper_pose" not in output or "gripper_open" not in output:
        raise ValueError("RLBench observations must contain gripper_pose and gripper_open.")
    state = np.concatenate([output["gripper_pose"], output["gripper_open"]], axis=-1)
    output["state"] = state.astype(np.float32)
    output["action"] = state.astype(np.float32, copy=True)
    return output


def _episode_seed(seed: int, task: str, variation: int, episode_id: int) -> np.random.SeedSequence:
    task_hash = zlib.crc32(task.encode("utf-8"))
    return np.random.SeedSequence([int(seed), int(task_hash), int(variation), int(episode_id)])


def _episode_arrays(
    episode_dir: Path,
    *,
    task: str,
    variation: int,
    episode_id: int,
    seed: int,
    num_points: int,
    ignored_ids: Sequence[int],
    workspace_bounds: Optional[Sequence[Sequence[float]]],
    include_rgb: bool,
    include_mask_id: bool,
    low_dim_fields: Sequence[str],
    strict_lengths: bool,
) -> Dict[str, np.ndarray]:
    low_dim_path = episode_dir / "low_dim_obs.pkl"
    if not low_dim_path.is_file():
        raise FileNotFoundError(f"Missing RLBench low-dimensional observations: {low_dim_path}")
    observations = list(_load_pickle(low_dim_path))
    point_files = _numbered_files(episode_dir / "merged_point_cloud", ".npz")
    if strict_lengths and len(observations) != len(point_files):
        raise ValueError(
            f"Frame count mismatch in {episode_dir}: {len(observations)} low-dimensional "
            f"observations versus {len(point_files)} point clouds."
        )
    length = min(len(observations), len(point_files))
    if length < 1:
        raise ValueError(f"RLBench episode is empty: {episode_dir}")
    observations = observations[:length]
    point_files = point_files[:length]
    low_dim = _extract_low_dim_fields(observations, low_dim_fields)
    rng = np.random.default_rng(_episode_seed(seed, task, variation, episode_id))
    point_frames = [
        _load_point_frame(
            path,
            num_points=num_points,
            rng=rng,
            ignored_ids=ignored_ids,
            workspace_bounds=workspace_bounds,
        )
        for path in point_files
    ]
    output = {
        name: np.stack([frame[name] for frame in point_frames], axis=0)
        for name in ("xyz", "valid", "rgb", "mask_id")
    }
    if not include_rgb:
        output.pop("rgb")
    if not include_mask_id:
        output.pop("mask_id")
    output.update(low_dim)
    output["frame_index"] = np.asarray([int(path.stem) for path in point_files], dtype=np.int32)
    return output


def _compression_kwargs(compression: str) -> Dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4}
    if compression == "lzf":
        return {"compression": "lzf"}
    raise ValueError("compression must be gzip, lzf, or none.")


def _write_dataset(group: h5py.Group, name: str, array: np.ndarray, compression: str) -> None:
    kwargs = _compression_kwargs(compression)
    array = np.asarray(array)
    if name == "xyz":
        data = array.astype(np.float16)
        chunks = (1, data.shape[1], 3)
    elif name == "valid":
        data = array.astype(np.bool_)
        chunks = (1, data.shape[1])
    elif name == "rgb":
        data = array.astype(np.uint8)
        chunks = (1, data.shape[1], 3)
    elif name == "mask_id":
        data = array.astype(np.int32)
        chunks = (1, data.shape[1])
    else:
        data = array
        if np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        chunks = (min(max(1, data.shape[0]), 64),) + data.shape[1:]
    group.create_dataset(name, data=data, chunks=chunks, **kwargs)


def _variation_descriptions(variation_dir: Path) -> List[str]:
    for filename in ("variation_descriptions.pkl", "variation_descriptions.json"):
        path = variation_dir / filename
        if not path.is_file():
            continue
        if path.suffix == ".pkl":
            values = _load_pickle(path)
        else:
            with path.open("r", encoding="utf-8") as handle:
                values = json.load(handle)
        return [str(value) for value in values]
    return []


def _write_variation(
    path: Path,
    *,
    task: str,
    variation: int,
    variation_dir: Path,
    episodes: Sequence[Tuple[int, Dict[str, np.ndarray], Path]],
    num_points: int,
    ignored_ids: Sequence[int],
    label_map: Mapping[int, str],
    workspace_bounds: Optional[Sequence[Sequence[float]]],
    low_dim_fields: Sequence[str],
    compression: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema_name"] = CACHE_SCHEMA_NAME
            handle.attrs["schema_version"] = CACHE_SCHEMA_VERSION
            handle.attrs["task"] = task
            handle.attrs["variation"] = int(variation)
            handle.attrs["N"] = int(num_points)
            handle.attrs["action_semantics"] = "absolute_gripper_pose_plus_open"
            handle.attrs["workspace_bounds"] = json.dumps(workspace_bounds)
            handle.attrs["ignored_mask_ids"] = np.asarray(ignored_ids, dtype=np.int64)
            handle.attrs["mask_to_label"] = json.dumps(
                {str(key): value for key, value in label_map.items()}
            )
            handle.attrs["low_dim_fields"] = json.dumps(list(low_dim_fields))
            handle.attrs["variation_descriptions"] = json.dumps(
                _variation_descriptions(variation_dir)
            )
            episode_ids = np.asarray([episode_id for episode_id, _, _ in episodes], dtype=np.int64)
            handle.create_dataset("episode_ids", data=episode_ids, dtype="i8")
            root = handle.create_group("episodes")
            for episode_id, arrays, source_dir in episodes:
                group = root.create_group(str(int(episode_id)))
                group.attrs["episode_id"] = int(episode_id)
                group.attrs["T"] = int(arrays["action"].shape[0])
                group.attrs["N"] = int(num_points)
                group.attrs["source_episode"] = str(source_dir)
                for name, array in arrays.items():
                    _write_dataset(group, name, array, compression)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_cache_manifest(cache_root: Path | str) -> Path:
    """Scan a cache root and write a human-readable manifest."""

    root = Path(cache_root)
    variations: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/variation*.h5")):
        with h5py.File(path, "r") as handle:
            episode_ids = np.asarray(handle["episode_ids"][:], dtype=np.int64)
            lengths = [
                int(handle["episodes"][str(int(episode_id))].attrs["T"])
                for episode_id in episode_ids
            ]
            fields: List[str] = []
            if episode_ids.size:
                group = handle["episodes"][str(int(episode_ids[0]))]
                fields = sorted(name for name, value in group.items() if isinstance(value, h5py.Dataset))
            variations.append(
                {
                    "task": str(handle.attrs["task"]),
                    "variation": int(handle.attrs["variation"]),
                    "path": str(path.relative_to(root)),
                    "num_episodes": int(len(episode_ids)),
                    "num_frames": int(sum(lengths)),
                    "min_episode_length": int(min(lengths)) if lengths else 0,
                    "max_episode_length": int(max(lengths)) if lengths else 0,
                    "num_points": int(handle.attrs.get("N", 0)),
                    "fields": fields,
                }
            )
    manifest = {
        "schema_name": CACHE_SCHEMA_NAME,
        "schema_version": CACHE_SCHEMA_VERSION,
        "num_tasks": len({item["task"] for item in variations}),
        "num_variations": len(variations),
        "num_episodes": sum(item["num_episodes"] for item in variations),
        "num_frames": sum(item["num_frames"] for item in variations),
        "variations": variations,
    }
    output_path = root / "manifest.json"
    temporary = output_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=False)
    temporary.replace(output_path)
    return output_path


def convert_rlbench_dataset(
    raw_root: Path | str,
    cache_root: Path | str,
    *,
    tasks: Sequence[str] = (),
    start_variation: int = 0,
    num_variations: int = -1,
    num_points: int = 1024,
    compression: str = "gzip",
    seed: int = 0,
    include_rgb: bool = True,
    include_mask_id: bool = True,
    ignore_background: bool = True,
    workspace_bounds: Optional[Sequence[Sequence[float]]] = DEFAULT_WORKSPACE_BOUNDS,
    low_dim_fields: Sequence[str] = DEFAULT_LOW_DIM_FIELDS,
    strict_lengths: bool = True,
    max_episodes_per_variation: Optional[int] = None,
    overwrite: bool = False,
) -> Path:
    """Convert raw RLBench episodes into per-variation HDF5 files."""

    raw_root = Path(raw_root)
    cache_root = Path(cache_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw RLBench root not found: {raw_root}")
    if int(num_points) < 1:
        raise ValueError("num_points must be positive.")
    task_dirs = [raw_root / task for task in tasks] if tasks else sorted(
        path for path in raw_root.iterdir() if path.is_dir()
    )
    missing = [str(path) for path in task_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Raw RLBench task directories not found: {missing}")
    cache_root.mkdir(parents=True, exist_ok=True)

    for task_dir in task_dirs:
        task = task_dir.name
        variation_dirs = [
            path
            for path in _numbered_directories(task_dir, "variation")
            if int(path.name.removeprefix("variation")) >= int(start_variation)
        ]
        if int(num_variations) >= 0:
            variation_dirs = variation_dirs[: int(num_variations)]
        if not variation_dirs:
            raise RuntimeError(f"No selected variations found in {task_dir}.")
        output_task_dir = cache_root / task
        output_task_dir.mkdir(parents=True, exist_ok=True)

        for variation_dir in variation_dirs:
            variation = int(variation_dir.name.removeprefix("variation"))
            output_path = output_task_dir / f"variation{variation}.h5"
            if output_path.exists() and not overwrite:
                print(f"Skipping existing cache: {output_path}", flush=True)
                continue
            label_map = _load_label_map(variation_dir)
            ignored_ids = _ignored_mask_ids(label_map) if ignore_background else ()
            episode_dirs = _numbered_directories(variation_dir / "episodes", "episode")
            if max_episodes_per_variation is not None:
                episode_dirs = episode_dirs[: int(max_episodes_per_variation)]
            if not episode_dirs:
                raise RuntimeError(f"No RLBench episodes found in {variation_dir}.")
            episodes = []
            description = f"{task}/variation{variation}"
            for episode_dir in tqdm(episode_dirs, desc=description, unit="episode"):
                episode_id = int(episode_dir.name.removeprefix("episode"))
                arrays = _episode_arrays(
                    episode_dir,
                    task=task,
                    variation=variation,
                    episode_id=episode_id,
                    seed=seed,
                    num_points=int(num_points),
                    ignored_ids=ignored_ids,
                    workspace_bounds=workspace_bounds,
                    include_rgb=include_rgb,
                    include_mask_id=include_mask_id,
                    low_dim_fields=low_dim_fields,
                    strict_lengths=strict_lengths,
                )
                episodes.append((episode_id, arrays, episode_dir))
            _write_variation(
                output_path,
                task=task,
                variation=variation,
                variation_dir=variation_dir,
                episodes=episodes,
                num_points=int(num_points),
                ignored_ids=ignored_ids,
                label_map=label_map,
                workspace_bounds=workspace_bounds,
                low_dim_fields=low_dim_fields,
                compression=compression,
            )
            print(f"Wrote {output_path}", flush=True)
    return write_cache_manifest(cache_root)
