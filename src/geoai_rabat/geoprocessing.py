from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .io_utils import write_json


@dataclass(frozen=True)
class ReferenceGrid:
    crs: str
    resolution_m: float
    width: int
    height: int
    left: float
    bottom: float
    right: float
    top: float

    @property
    def transform_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.resolution_m,
            0.0,
            self.left,
            0.0,
            -self.resolution_m,
            self.top,
        )


def reference_grid_from_boundary(
    boundary_path: Path,
    crs: str = "EPSG:32629",
    resolution_m: float = 30.0,
) -> ReferenceGrid:
    """Construit une grille dont l'origine est alignée sur un multiple de 30 m."""
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("Installez les dépendances géospatiales avec `pip install -e .[geo]`.") from exc

    boundary = gpd.read_file(boundary_path).to_crs(crs)
    left, bottom, right, top = boundary.total_bounds
    left = math.floor(left / resolution_m) * resolution_m
    bottom = math.floor(bottom / resolution_m) * resolution_m
    right = math.ceil(right / resolution_m) * resolution_m
    top = math.ceil(top / resolution_m) * resolution_m
    width = int(round((right - left) / resolution_m))
    height = int(round((top - bottom) / resolution_m))
    return ReferenceGrid(crs, resolution_m, width, height, left, bottom, right, top)


def reproject_raster_to_grid(
    source_path: Path,
    destination_path: Path,
    grid: ReferenceGrid,
    *,
    resampling: Literal["nearest", "bilinear", "average"] = "bilinear",
    destination_dtype: str = "float32",
    nodata: float = np.nan,
) -> Path:
    """Reprojette une bande sur la grille commune et conserve ses métadonnées."""
    try:
        import rasterio
        from rasterio.transform import Affine
        from rasterio.warp import Resampling, reproject
    except ImportError as exc:
        raise RuntimeError("Installez les dépendances géospatiales avec `pip install -e .[geo]`.") from exc

    methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
    }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination = np.full((grid.height, grid.width), nodata, dtype=destination_dtype)
    with rasterio.open(source_path) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=Affine(*grid.transform_tuple),
            dst_crs=grid.crs,
            dst_nodata=nodata,
            resampling=methods[resampling],
        )
        profile = source.profile.copy()
    profile.update(
        {
            "driver": "GTiff",
            "height": grid.height,
            "width": grid.width,
            "count": 1,
            "crs": grid.crs,
            "transform": Affine(*grid.transform_tuple),
            "dtype": destination_dtype,
            "nodata": nodata,
            "compress": "deflate",
            "tiled": True,
        }
    )
    with rasterio.open(destination_path, "w", **profile) as target:
        target.write(destination, 1)
    return destination_path


def rasterize_presence(
    vector_path: Path,
    grid: ReferenceGrid,
    *,
    all_touched: bool = True,
) -> np.ndarray:
    """Rasterise la présence de bâtiments, routes, parcs ou eau sur la grille."""
    try:
        import geopandas as gpd
        from rasterio.features import rasterize
        from rasterio.transform import Affine
    except ImportError as exc:
        raise RuntimeError("Installez les dépendances géospatiales avec `pip install -e .[geo]`.") from exc

    features = gpd.read_file(vector_path).to_crs(grid.crs)
    shapes = ((geometry, 1) for geometry in features.geometry if geometry is not None)
    return rasterize(
        shapes,
        out_shape=(grid.height, grid.width),
        transform=Affine(*grid.transform_tuple),
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )


def distance_to_presence(presence: np.ndarray, resolution_m: float = 30.0) -> np.ndarray:
    """Distance euclidienne métrique au pixel positif le plus proche."""
    from scipy.ndimage import distance_transform_edt

    array = np.asarray(presence)
    if array.ndim != 2:
        raise ValueError("La couche de présence doit être bidimensionnelle.")
    if not np.any(array > 0):
        raise ValueError("La couche ne contient aucun objet positif.")
    return distance_transform_edt(array == 0, sampling=resolution_m).astype(np.float32)


def save_reference_grid(path: Path, grid: ReferenceGrid, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = asdict(grid)
    payload["transform"] = list(grid.transform_tuple)
    if extra:
        payload.update(extra)
    write_json(path, payload)
