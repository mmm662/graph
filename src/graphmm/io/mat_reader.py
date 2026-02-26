from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.io import loadmat

@dataclass
class FloorMap:
    coord_xy: np.ndarray
    E: np.ndarray
    bump: Optional[np.ndarray]
    row: int
    col: int

def read_mat(mat_file_path: str, ppm: float) -> FloorMap:
    data = loadmat(mat_file_path)
    if "E" not in data or "v" not in data:
        raise KeyError(f"MAT missing required keys 'E' and 'v': {mat_file_path}")

    E = np.unique(data["E"], axis=0)
    v = data["v"]
    bump = data.get("bump", None)

    # default size (edit if your map differs)
    w = 1850
    h = 1100
    row, col = h, w

    # add bump column (zeros)
    new_col = np.zeros((E.shape[0], 1), dtype=E.dtype)
    E = np.hstack((E, new_col))

    coord = (v - 1) * ppm
    coord = coord[:, [1, 0]]  # to (x,y)
    return FloorMap(coord_xy=coord.astype(np.float32), E=E, bump=bump, row=row, col=col)
