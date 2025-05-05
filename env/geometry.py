"""
Geometric utilities for map processing and zone interpolation.
"""

from pathlib import Path
from typing import List

import numpy as np
from scipy.interpolate import make_interp_spline


def line_plane_intersection(plane_normal: np.ndarray, plane_point: np.ndarray,
                            ray_direction: np.ndarray, ray_point: np.ndarray,
                            epsilon=1e-6) -> np.ndarray:
    """
    Compute intersection point between a line and a plane.
    """
    ndotu = plane_normal.dot(ray_direction)
    if abs(ndotu) < epsilon:
        raise ValueError("No intersection or line is parallel to the plane.")
    w = ray_point - plane_point
    si = -plane_normal.dot(w) / ndotu
    return ray_point + si * ray_direction


def compute_fraction_in_zone(current_zone: np.ndarray, next_zone: np.ndarray,
                              pos: np.ndarray, next_pos: np.ndarray) -> float:
    """
    Fraction of motion from pos to next_pos that lies within the current zone.
    """
    plane_normal = next_zone - current_zone
    numerator = -plane_normal.dot(pos - (next_zone + current_zone) / 2)
    denominator = plane_normal.dot(next_pos - pos)
    if denominator == 0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def resample_centerline(path: List[np.ndarray], spacing: float, save_path: Path) -> np.ndarray:
    """
    Resample 3D path with uniform spacing and save as .npy file.
    """
    interp = make_interp_spline(x=range(len(path)), y=path, k=1)
    densified = interp(np.arange(0, len(path) - 1 + 1e-6, 0.01))

    a = np.array(densified)
    distances = np.linalg.norm(a[1:] - a[:-1], axis=1)
    cumulative = np.pad(distances.cumsum(), (1, 0))
    num_zones = round(cumulative[-1] / spacing - 0.5) + 0.5
    segment_length = cumulative[-1] / num_zones

    idx_next = np.unique(cumulative // segment_length, return_index=True)[1][1:]
    idx_prev = idx_next - 1
    w1 = 1 - (cumulative[idx_prev] % segment_length) / segment_length
    w2 = (cumulative[idx_next] % segment_length) / segment_length

    zone_centers = a[idx_prev] + (a[idx_next] - a[idx_prev]) * (w1 / (1e-4 + w1 + w2)).reshape((-1, 1))
    zone_centers = np.vstack((a[0][None, :], zone_centers, (2 * a[-1] - zone_centers[-1])[None, :]))
    np.save(save_path / "map.npy", zone_centers.round(4))
    return zone_centers
