"""Geometric helpers for Push-T reference processes."""

from __future__ import annotations

import math
from typing import Tuple

import torch


def default_t_polygon_local(scale: float = 30.0) -> torch.Tensor:
    """Return the outer CCW boundary of the Push-T T in local coordinates.

    The shape matches gym-pusht's two rectangles: a 4*scale by scale cap and a
    scale by 3*scale stem. Coordinates follow the same screen/world convention
    used by the environment, with positive y downward.
    """

    length = 4.0
    half_cap = length * scale / 2.0
    half_stem = scale / 2.0
    return torch.tensor(
        [
            [-half_cap, 0.0],
            [half_cap, 0.0],
            [half_cap, scale],
            [half_stem, scale],
            [half_stem, length * scale],
            [-half_stem, length * scale],
            [-half_stem, scale],
            [-half_cap, scale],
        ],
        dtype=torch.float32,
    )


def sample_polygon_boundary_local(poly: torch.Tensor, n_per_edge: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample boundary points and outward normals for a CCW polygon."""

    points = []
    normals = []
    count = int(poly.shape[0])
    for i in range(count):
        a = poly[i]
        b = poly[(i + 1) % count]
        edge = b - a
        tangent = edge / edge.norm().clamp_min(1e-8)
        normal = torch.stack([tangent[1], -tangent[0]])
        for j in range(int(n_per_edge)):
            t = (float(j) + 0.5) / float(n_per_edge)
            points.append((1.0 - t) * a + t * b)
            normals.append(normal)
    return torch.stack(points, dim=0), torch.stack(normals, dim=0)


def wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def safe_unit_torch(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norm = torch.linalg.norm(x, dim=dim, keepdim=True)
    return x / norm.clamp_min(eps)


def sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def rotmat_batch(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    return torch.stack(
        [
            torch.stack([c, -s], dim=-1),
            torch.stack([s, c], dim=-1),
        ],
        dim=-2,
    )


def transform_boundary_batch(
    boundary_local: torch.Tensor,
    normals_local: torch.Tensor,
    block_pos: torch.Tensor,
    block_theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotation = rotmat_batch(block_theta)
    boundary = torch.einsum("bij,nj->bni", rotation, boundary_local) + block_pos[:, None, :]
    normals = torch.einsum("bij,nj->bni", rotation, normals_local)
    return boundary, normals


def target_pose_tensor(device: torch.device, dtype: torch.dtype, target_pose) -> torch.Tensor:
    if target_pose is None:
        target_pose = [256.0, 256.0, math.pi / 4.0]
    return torch.as_tensor(target_pose, device=device, dtype=dtype)
