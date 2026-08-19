from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_demo_grid(config: dict[str, Any]) -> pd.DataFrame:
    """Construit une petite grille UTM structurée, masquée par une forme urbaine.

    Cette grille sert uniquement aux tests de bout en bout. Le CRS, la résolution
    nominale et les identifiants suivent les conventions de la future grille réelle.
    L'espacement est élargi pour couvrir la ville avec peu de cellules de démonstration.
    """
    rows = int(config["grid"]["demo_rows"])
    cols = int(config["grid"]["demo_cols"])
    lon0, lat0 = config["study_area"]["centroid_wgs84"]

    x_offsets = np.linspace(-7200.0, 7200.0, cols, dtype=np.float64)
    y_offsets = np.linspace(-6200.0, 6200.0, rows, dtype=np.float64)
    xx, yy = np.meshgrid(x_offsets, y_offsets)

    # Limite urbaine simplifiée avec une façade atlantique et une encoche au nord-est.
    ellipse = (xx / 7300.0) ** 2 + (yy / 6500.0) ** 2 <= 1.0
    ocean_cut = xx > -7000.0 + 0.08 * yy
    estuary_cut = ~((xx > 3500.0) & (yy > 3500.0) & ((xx + yy) > 9000.0))
    mask = ellipse & ocean_cut & estuary_cut

    x_center = 700_000.0
    y_center = 3_761_000.0
    x_m = x_center + xx[mask]
    y_m = y_center + yy[mask]

    meters_per_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    meters_per_lat = 110_540.0
    lon = lon0 + (x_m - x_center) / meters_per_lon
    lat = lat0 + (y_m - y_center) / meters_per_lat

    grid = pd.DataFrame(
        {
            "pixel_id": [f"P{i:06d}" for i in range(mask.sum())],
            "grid_row": np.where(mask)[0].astype(np.int32),
            "grid_col": np.where(mask)[1].astype(np.int32),
            "x_m": x_m.astype(np.float32),
            "y_m": y_m.astype(np.float32),
            "lon": lon.astype(np.float64),
            "lat": lat.astype(np.float64),
        }
    )
    if grid["pixel_id"].duplicated().any():
        raise AssertionError("Les identifiants de pixels doivent être uniques.")
    return grid


def normalized_coordinates(grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = grid["x_m"].to_numpy(dtype=np.float64)
    y = grid["y_m"].to_numpy(dtype=np.float64)
    x_norm = 2.0 * (x - x.min()) / (x.max() - x.min()) - 1.0
    y_norm = 2.0 * (y - y.min()) / (y.max() - y.min()) - 1.0
    return x_norm, y_norm
