"""
Loads and processes zone centers for TrackMania map simulation.
"""

from pathlib import Path
import numpy as np
from config import config


def load_zone_centers(path: Path) -> np.ndarray:
    """
    Load map zone centers from .npy file and pad with extra zones before/after the track.
    """
    zone_centers = np.load(str(path))

    before = zone_centers[0] + (zone_centers[0] - zone_centers[1]) * np.arange(
        config.N_ZONES_BEFORE, 0, -1
    )[:, None]
    after = zone_centers[-1] + (zone_centers[-1] - zone_centers[-2]) * np.arange(
        1, config.N_ZONES_AFTER + 1
    )[:, None]

    full_zone_centers = np.vstack([before, zone_centers, after])
    full_zone_centers[5:-5] = 0.5 * (full_zone_centers[:-10] + full_zone_centers[10:])  # smoothing

    return full_zone_centers


def precalculate_zone_info(zone_centers: np.ndarray):
    """
    Precompute zone transition data: midpoints, distances, and unit direction vectors.
    """
    transitions = 0.5 * (zone_centers[1:] + zone_centers[:-1])
    deltas = transitions[1:] - transitions[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    cumulative_lengths = np.hstack(([0], lengths.cumsum()))
    directions = deltas / lengths[:, None]

    return transitions, lengths, cumulative_lengths, directions
