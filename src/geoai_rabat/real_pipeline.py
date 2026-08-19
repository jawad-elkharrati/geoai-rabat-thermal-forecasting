from __future__ import annotations

import json
import math
import pickle
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
import requests
from PIL import Image, ImageDraw
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, rasterize
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from shapely.geometry import LineString, Polygon, box, mapping, shape
from shapely.ops import transform as shapely_transform

from .config import load_config
from .io_utils import ensure_directories, file_record, write_json
from .lite_models import (
    GradientBoostingLiteRegressor,
    LinearRegressorLite,
    RandomForestLiteRegressor,
)
from .modeling import split_dates
from .quality import landsat_clear_mask, scale_landsat_surface_temperature
from .table_io import read_table, write_table


STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
REAL_FEATURES = [
    "elevation_m",
    "slope_deg",
    "ndvi",
    "ndbi",
    "building_density",
    "road_density",
    "impervious_fraction",
    "distance_to_water_m",
    "distance_to_green_m",
    "air_temperature_c",
    "relative_humidity_pct",
    "wind_speed_ms",
    "shortwave_radiation_wm2",
    "day_of_year_sin",
    "day_of_year_cos",
]


def load_real_config(path: str | Path = "configs/rabat_real_pilot.json") -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _catalog() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _grid(config: dict[str, Any]) -> dict[str, Any]:
    bbox = config["bbox_wgs84"]
    transformer = Transformer.from_crs("EPSG:4326", config["crs"], always_xy=True)
    corners = [
        transformer.transform(bbox[0], bbox[1]),
        transformer.transform(bbox[0], bbox[3]),
        transformer.transform(bbox[2], bbox[1]),
        transformer.transform(bbox[2], bbox[3]),
    ]
    resolution = float(config["resolution_m"])
    left = math.floor(min(point[0] for point in corners) / resolution) * resolution
    right = math.ceil(max(point[0] for point in corners) / resolution) * resolution
    bottom = math.floor(min(point[1] for point in corners) / resolution) * resolution
    top = math.ceil(max(point[1] for point in corners) / resolution) * resolution
    width = int(round((right - left) / resolution))
    height = int(round((top - bottom) / resolution))
    return {
        "crs": config["crs"],
        "resolution_m": resolution,
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
        "width": width,
        "height": height,
        "transform": Affine(resolution, 0, left, 0, -resolution, top),
    }


def _boundary_mask(config: dict[str, Any], grid: dict[str, Any]) -> np.ndarray:
    with Path(config["boundary_path"]).open(encoding="utf-8") as stream:
        boundary_geojson = json.load(stream)
    geometry = shape(boundary_geojson["features"][0]["geometry"])
    transformer = Transformer.from_crs("EPSG:4326", config["crs"], always_xy=True)
    projected = shapely_transform(transformer.transform, geometry)
    return geometry_mask(
        [mapping(projected)],
        out_shape=(grid["height"], grid["width"]),
        transform=grid["transform"],
        invert=True,
    )


def _read_item_asset(
    item: Any,
    asset_name: str,
    grid: dict[str, Any],
    resampling: Resampling,
    *,
    dtype: str = "float32",
    nodata: float | int = np.nan,
) -> np.ndarray:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            feature = {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [grid["left"], grid["bottom"]],
                            [grid["right"], grid["bottom"]],
                            [grid["right"], grid["top"]],
                            [grid["left"], grid["top"]],
                            [grid["left"], grid["bottom"]],
                        ]
                    ],
                },
            }
            method = "nearest" if resampling == Resampling.nearest else "bilinear"
            url = (
                "https://planetarycomputer.microsoft.com/api/data/v1/item/crop/"
                f"{grid['width']}x{grid['height']}.tif"
            )
            response = requests.post(
                url,
                params={
                    "collection": item.collection_id,
                    "item": item.id,
                    "assets": asset_name,
                    "coord_crs": grid["crs"],
                    "dst_crs": grid["crs"],
                    "resampling": method,
                    "reproject": method,
                    "return_mask": "false",
                    "unscale": "false",
                },
                json=feature,
                timeout=180,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Data API {response.status_code} pour {item.id}/{asset_name}: "
                    f"{response.text[:1000]}"
                )
            with MemoryFile(response.content) as memory:
                with memory.open() as source:
                    data = source.read(1, masked=True).filled(nodata)
            if data.shape != (grid["height"], grid["width"]):
                raise ValueError(f"Forme inattendue pour {item.id}/{asset_name}: {data.shape}")
            return data.astype(dtype)
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(3 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _search_items(catalog: pystac_client.Client, **kwargs: Any) -> list[Any]:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return list(catalog.search(**kwargs).items())
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(3 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _write_raster(
    path: Path,
    array: np.ndarray,
    grid: dict[str, Any],
    *,
    nodata: float | int,
    dtype: str,
    descriptions: list[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = array[np.newaxis, ...] if array.ndim == 2 else array
    profile = {
        "driver": "GTiff",
        "height": grid["height"],
        "width": grid["width"],
        "count": data.shape[0],
        "crs": grid["crs"],
        "transform": grid["transform"],
        "dtype": dtype,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
    }
    with rasterio.open(path, "w", **profile) as target:
        target.write(data.astype(dtype))
        if descriptions:
            target.descriptions = tuple(descriptions)
    return path


def _select_landsat_items(config: dict[str, Any], catalog: pystac_client.Client) -> list[Any]:
    settings = config["landsat"]
    aoi = box(*config["bbox_wgs84"])
    items = _search_items(
        catalog,
        collections=[settings["collection"]],
        bbox=config["bbox_wgs84"],
        datetime=settings["datetime"],
        query={"eo:cloud_cover": {"lt": settings["max_cloud_percent"]}},
    )
    by_date: dict[str, list[tuple[float, float, Any]]] = defaultdict(list)
    for item in items:
        if not {"lwir11", "qa_pixel"}.issubset(item.assets):
            continue
        date = item.datetime.date()
        if date.month not in settings["months"]:
            continue
        coverage = shape(item.geometry).intersection(aoi).area / aoi.area
        if coverage < 0.98:
            continue
        cloud = float(item.properties.get("eo:cloud_cover", 100))
        by_date[date.isoformat()].append((coverage, cloud, item))

    best_dates: list[Any] = []
    for candidates in by_date.values():
        candidates.sort(key=lambda value: (-value[0], value[1]))
        best_dates.append(candidates[0][2])

    by_year: dict[int, list[Any]] = defaultdict(list)
    for item in best_dates:
        by_year[item.datetime.year].append(item)
    selected: list[Any] = []
    for year in sorted(by_year):
        choices = sorted(
            by_year[year],
            key=lambda item: float(item.properties.get("eo:cloud_cover", 100)),
        )
        for item in choices:
            if all(abs((item.datetime.date() - other.datetime.date()).days) >= 24 for other in selected if other.datetime.year == year):
                selected.append(item)
            if sum(other.datetime.year == year for other in selected) >= 2:
                break
    selected = sorted(selected, key=lambda item: item.datetime)[: int(settings["max_dates"])]
    if len(selected) < 6:
        raise RuntimeError(f"Seulement {len(selected)} dates Landsat exploitables.")
    return selected


def _select_sentinel_item(config: dict[str, Any], catalog: pystac_client.Client) -> Any:
    settings = config["sentinel2"]
    aoi = box(*config["bbox_wgs84"])
    items = _search_items(
        catalog,
        collections=[settings["collection"]],
        bbox=config["bbox_wgs84"],
        datetime=settings["datetime"],
        query={"eo:cloud_cover": {"lt": settings["max_cloud_percent"]}},
    )
    candidates = []
    for item in items:
        coverage = shape(item.geometry).intersection(aoi).area / aoi.area
        if coverage >= 0.99 and {"B02", "B03", "B04", "B08", "B11", "SCL"}.issubset(item.assets):
            candidates.append((float(item.properties.get("eo:cloud_cover", 100)), item))
    if not candidates:
        raise RuntimeError("Aucune scène Sentinel-2 complète et peu nuageuse.")
    return min(candidates, key=lambda value: value[0])[1]


def _chamfer_distance(target: np.ndarray, resolution: float) -> np.ndarray:
    if not np.any(target):
        return np.full(target.shape, 10_000.0, dtype=np.float32)
    distance = np.where(target, 0.0, np.inf)
    diagonal = math.sqrt(2.0)
    height, width = target.shape
    for row in range(height):
        for col in range(width):
            if row > 0:
                distance[row, col] = min(distance[row, col], distance[row - 1, col] + 1)
            if col > 0:
                distance[row, col] = min(distance[row, col], distance[row, col - 1] + 1)
            if row > 0 and col > 0:
                distance[row, col] = min(distance[row, col], distance[row - 1, col - 1] + diagonal)
    for row in range(height - 1, -1, -1):
        for col in range(width - 1, -1, -1):
            if row + 1 < height:
                distance[row, col] = min(distance[row, col], distance[row + 1, col] + 1)
            if col + 1 < width:
                distance[row, col] = min(distance[row, col], distance[row, col + 1] + 1)
            if row + 1 < height and col + 1 < width:
                distance[row, col] = min(distance[row, col], distance[row + 1, col + 1] + diagonal)
    return (distance * resolution).astype(np.float32)


def _stretch_rgb(red: np.ndarray, green: np.ndarray, blue: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = []
    for band in [red, green, blue]:
        values = band[valid]
        low, high = np.nanpercentile(values, [2, 98])
        stretched = np.clip((band - low) / max(high - low, 1e-6), 0, 1)
        output.append((stretched * 255).astype(np.uint8))
    rgb = np.dstack(output)
    rgb[~valid] = 245
    return rgb


def _save_rgb(path: Path, rgb: np.ndarray, title: str) -> None:
    scale = 3
    image = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale))
    canvas = Image.new("RGB", (image.width, image.height + 70), "white")
    canvas.paste(image, (0, 70))
    ImageDraw.Draw(canvas).text((20, 22), title, fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_heatmap(path: Path, values: np.ndarray, valid: np.ndarray, title: str) -> None:
    low, high = np.nanpercentile(values[valid], [2, 98])
    normalized = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    red = np.clip(1.7 * normalized, 0, 1)
    green = np.clip(1.8 - 2.0 * np.abs(normalized - 0.5), 0, 1)
    blue = np.clip(1.7 * (1 - normalized), 0, 1)
    rgb = (np.dstack([red, green, blue]) * 255).astype(np.uint8)
    rgb[~valid] = 245
    scale = 3
    image = Image.fromarray(rgb).resize((rgb.shape[1] * scale, rgb.shape[0] * scale))
    canvas = Image.new("RGB", (image.width, image.height + 90), "white")
    canvas.paste(image, (0, 70))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 20), f"{title} | echelle {low:.1f} - {high:.1f} deg C", fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_osm_map(
    path: Path,
    building_density: np.ndarray,
    road_density: np.ndarray,
    boundary: np.ndarray,
) -> None:
    height, width = building_density.shape
    rgb = np.full((height, width, 3), 248, dtype=np.uint8)
    buildings = np.nan_to_num(building_density, nan=0.0)
    roads = np.nan_to_num(road_density, nan=0.0)
    rgb[..., 0] = np.clip(248 - roads * 170, 60, 248).astype(np.uint8)
    rgb[..., 1] = np.clip(248 - roads * 170 - buildings * 90, 70, 248).astype(np.uint8)
    rgb[..., 2] = np.clip(248 - roads * 170 - buildings * 190, 45, 248).astype(np.uint8)
    rgb[~boundary] = 255
    scale = 3
    image = Image.fromarray(rgb).resize((width * scale, height * scale))
    canvas = Image.new("RGB", (image.width, image.height + 70), "white")
    canvas.paste(image, (0, 70))
    ImageDraw.Draw(canvas).text(
        (20, 22),
        "OpenStreetMap reel - densites batiments et routes",
        fill="#111827",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _fetch_weather(config: dict[str, Any], dates: list[str], output_path: Path) -> pd.DataFrame:
    lon = (config["bbox_wgs84"][0] + config["bbox_wgs84"][2]) / 2
    lat = (config["bbox_wgs84"][1] + config["bbox_wgs84"][3]) / 2
    common_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": min(dates),
        "end_date": max(dates),
        "wind_speed_unit": "ms",
        "timezone": config["weather"]["timezone"],
    }

    def request_model(model: str, variables: str) -> tuple[dict[str, Any], str]:
        params = {**common_params, "hourly": variables, "models": model}
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params,
            timeout=180,
        )
        response.raise_for_status()
        return response.json(), response.url

    era5_land, era5_land_url = request_model(
        config["weather"].get("model", "era5_land"),
        "temperature_2m,relative_humidity_2m",
    )
    era5, era5_url = request_model("era5", "wind_speed_10m,shortwave_radiation")
    weather = pd.DataFrame(era5_land["hourly"]).merge(
        pd.DataFrame(era5["hourly"]),
        on="time",
        how="inner",
        validate="one_to_one",
    )
    payload = {
        "hourly": weather.to_dict(orient="list"),
        "hourly_units": {
            **era5_land.get("hourly_units", {}),
            **era5.get("hourly_units", {}),
        },
        "_provenance": {
            "primary_dataset": "Copernicus ERA5-Land",
            "supplementary_dataset": "Copernicus ERA5",
            "variable_mapping": {
                "temperature_2m": "ERA5-Land",
                "relative_humidity_2m": "ERA5-Land",
                "wind_speed_10m": "ERA5",
                "shortwave_radiation": "ERA5",
            },
            "delivery_service": "Open-Meteo Historical Weather API",
            "era5_land_url": era5_land_url,
            "era5_url": era5_url,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_json(output_path, payload)
    weather["_provenance_dataset"] = "ERA5-Land + ERA5"
    weather["time"] = pd.to_datetime(weather["time"])
    hour = int(config["weather"]["overpass_hour_local"])
    wanted = weather.loc[
        weather["time"].dt.strftime("%Y-%m-%d").isin(dates) & (weather["time"].dt.hour == hour)
    ].copy()
    if wanted[
        [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "shortwave_radiation",
        ]
    ].isna().any().any():
        raise RuntimeError("Variables ERA5-Land/ERA5 manquantes aux dates Landsat.")
    wanted["date"] = wanted["time"].dt.strftime("%Y-%m-%d")
    return wanted.set_index("date")


def _fetch_osm_urban(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    if output_path.exists():
        with output_path.open(encoding="utf-8") as stream:
            cached = json.load(stream)
        if cached.get("elements"):
            return cached
    west, south, east, north = config["bbox_wgs84"]
    middle_lon = (west + east) / 2
    middle_lat = (south + north) / 2
    tiles = [
        (south, west, middle_lat, middle_lon),
        (south, middle_lon, middle_lat, east),
        (middle_lat, west, north, middle_lon),
        (middle_lat, middle_lon, north, east),
    ]
    endpoints = [
        "https://overpass.private.coffee/api/interpreter",
        "https://overpass-api.de/api/interpreter",
        "https://gall.openstreetmap.de/api/interpreter",
        "https://lambert.openstreetmap.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    last_error: Exception | None = None
    elements: dict[tuple[str, int], dict[str, Any]] = {}
    requests_used = []
    for tile_south, tile_west, tile_north, tile_east in tiles:
        query = f"""
        [out:json][timeout:90];
        (
          way["building"]({tile_south},{tile_west},{tile_north},{tile_east});
          way["highway"]({tile_south},{tile_west},{tile_north},{tile_east});
        );
        out geom;
        """
        tile_complete = False
        for endpoint in endpoints:
            try:
                response = requests.post(
                    endpoint,
                    data={"data": query},
                    headers={"User-Agent": "GeoAI-Rabat-student-project/1.0"},
                    timeout=120,
                )
                response.raise_for_status()
                tile_payload = response.json()
                for element in tile_payload.get("elements", []):
                    elements[(str(element.get("type")), int(element["id"]))] = element
                requests_used.append({"endpoint": endpoint, "query": query.strip()})
                tile_complete = True
                break
            except (requests.RequestException, ValueError, KeyError) as error:
                last_error = error
        if not tile_complete:
            raise RuntimeError(
                f"Impossible de télécharger une tuile OSM après {len(endpoints)} miroirs : "
                f"{last_error}"
            )
    if not elements:
        raise RuntimeError("La requête OSM n'a retourné aucun bâtiment ni aucune route.")
    payload = {
        "version": 0.6,
        "generator": "GeoAI Rabat tiled Overpass acquisition",
        "elements": list(elements.values()),
        "_provenance": {
            "source": "OpenStreetMap contributors",
            "license": "ODbL 1.0",
            "requests": requests_used,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    }
    write_json(output_path, payload)
    return payload


def _osm_density_layers(
    payload: dict[str, Any],
    grid: dict[str, Any],
    boundary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    transformer = Transformer.from_crs("EPSG:4326", grid["crs"], always_xy=True)
    buildings = []
    roads = []
    for element in payload.get("elements", []):
        coordinates = [
            transformer.transform(float(point["lon"]), float(point["lat"]))
            for point in element.get("geometry", [])
            if "lon" in point and "lat" in point
        ]
        tags = element.get("tags", {})
        if "building" in tags and len(coordinates) >= 4:
            if coordinates[0] != coordinates[-1]:
                coordinates.append(coordinates[0])
            geometry = Polygon(coordinates)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if not geometry.is_empty:
                buildings.append(geometry)
        if "highway" in tags and len(coordinates) >= 2:
            geometry = LineString(coordinates)
            if not geometry.is_empty:
                roads.append(geometry)

    scale = 3
    fine_shape = (grid["height"] * scale, grid["width"] * scale)
    fine_transform = Affine(
        grid["resolution_m"] / scale,
        0,
        grid["left"],
        0,
        -grid["resolution_m"] / scale,
        grid["top"],
    )

    def density(geometries: list[Any]) -> np.ndarray:
        if not geometries:
            return np.zeros((grid["height"], grid["width"]), dtype=np.float32)
        fine = rasterize(
            [(mapping(geometry), 1) for geometry in geometries],
            out_shape=fine_shape,
            transform=fine_transform,
            fill=0,
            all_touched=True,
            dtype="uint8",
        )
        coarse = fine.reshape(grid["height"], scale, grid["width"], scale).mean(axis=(1, 3))
        return np.where(boundary, coarse, np.nan).astype(np.float32)

    return (
        density(buildings),
        density(roads),
        {"building_ways": len(buildings), "road_ways": len(roads), "supersampling_m": 10},
    )


def acquire_real_data(config: dict[str, Any]) -> dict[str, Any]:
    paths = {name: Path(value) for name, value in config["paths"].items() if name != "report_pdf"}
    ensure_directories(*paths.values())
    grid = _grid(config)
    boundary = _boundary_mask(config, grid)
    catalog = _catalog()

    sentinel = _select_sentinel_item(config, catalog)
    sentinel_bands = {
        name: _read_item_asset(
            sentinel,
            name,
            grid,
            Resampling.nearest if name == "SCL" else Resampling.bilinear,
            dtype="float32",
            nodata=np.nan,
        )
        for name in ["B02", "B03", "B04", "B08", "B11", "SCL"]
    }
    scl = sentinel_bands["SCL"].astype(np.int16)
    sentinel_valid = boundary & np.isfinite(sentinel_bands["B04"]) & ~np.isin(scl, [0, 1, 3, 8, 9, 10, 11])
    for name in ["B02", "B03", "B04", "B08", "B11"]:
        sentinel_bands[name] = sentinel_bands[name] * 0.0001
    red = sentinel_bands["B04"]
    green = sentinel_bands["B03"]
    blue = sentinel_bands["B02"]
    nir = sentinel_bands["B08"]
    swir = sentinel_bands["B11"]
    ndvi = np.where(sentinel_valid, (nir - red) / np.where(np.abs(nir + red) > 1e-6, nir + red, np.nan), np.nan)
    ndbi = np.where(sentinel_valid, (swir - nir) / np.where(np.abs(swir + nir) > 1e-6, swir + nir, np.nan), np.nan)
    mndwi = np.where(sentinel_valid, (green - swir) / np.where(np.abs(green + swir) > 1e-6, green + swir, np.nan), np.nan)
    impervious = np.clip((ndbi + 0.15) * 1.8 * (1.0 - np.clip(ndvi, 0, 1)), 0, 1)
    distance_water = _chamfer_distance((mndwi > 0.12) & sentinel_valid, grid["resolution_m"])
    distance_green = _chamfer_distance((ndvi > 0.45) & sentinel_valid, grid["resolution_m"])
    osm_path = paths["raw"] / "osm_urban_buildings_roads.json"
    osm_payload = _fetch_osm_urban(config, osm_path)
    building_density, road_density, osm_summary = _osm_density_layers(
        osm_payload, grid, boundary
    )
    _save_osm_map(
        paths["figures"] / "osm_densites_urbaines.png",
        building_density,
        road_density,
        boundary,
    )

    dem_items = _search_items(
        catalog,
        collections=[config["dem"]["collection"]],
        bbox=config["bbox_wgs84"],
    )
    elevation = np.full((grid["height"], grid["width"]), np.nan, dtype=np.float32)
    for item in dem_items:
        tile = _read_item_asset(
            item,
            "data",
            grid,
            Resampling.bilinear,
            dtype="float32",
            nodata=np.nan,
        )
        fill = np.isnan(elevation) & np.isfinite(tile)
        elevation[fill] = tile[fill]
    filled_elevation = np.where(np.isfinite(elevation), elevation, np.nanmedian(elevation))
    gradient_y, gradient_x = np.gradient(filled_elevation, grid["resolution_m"])
    slope = np.degrees(np.arctan(np.sqrt(gradient_x**2 + gradient_y**2))).astype(np.float32)

    static_stack = np.stack(
        [
            elevation,
            slope,
            ndvi,
            ndbi,
            building_density,
            road_density,
            impervious,
            distance_water,
            distance_green,
        ]
    ).astype(np.float32)
    static_path = _write_raster(
        paths["interim"] / "static_real_features.tif",
        static_stack,
        grid,
        nodata=np.nan,
        dtype="float32",
        descriptions=[
            "elevation_m",
            "slope_deg",
            "ndvi",
            "ndbi",
            "building_density",
            "road_density",
            "impervious_fraction",
            "distance_to_water_m",
            "distance_to_green_m",
        ],
    )
    rgb_path = paths["figures"] / "sentinel2_rgb_zone_pilote.png"
    _save_rgb(rgb_path, _stretch_rgb(red, green, blue, sentinel_valid), f"Sentinel-2 reel - {sentinel.datetime.date()}")

    landsat_items = _select_landsat_items(config, catalog)
    dates = [item.datetime.date().isoformat() for item in landsat_items]
    weather_path = paths["raw"] / "era5_land_hourly_landsat_dates.json"
    weather = _fetch_weather(config, dates, weather_path)
    rows, cols = np.indices((grid["height"], grid["width"]))
    x_coords = grid["left"] + (cols + 0.5) * grid["resolution_m"]
    y_coords = grid["top"] - (rows + 0.5) * grid["resolution_m"]
    inverse = Transformer.from_crs(config["crs"], "EPSG:4326", always_xy=True)
    lon, lat = inverse.transform(x_coords, y_coords)

    frames: list[pd.DataFrame] = []
    landsat_records = []
    first_heatmap: Path | None = None
    static_valid = boundary & np.all(np.isfinite(static_stack), axis=0)
    for item in landsat_items:
        date = item.datetime.date().isoformat()
        raw_st = _read_item_asset(
            item,
            "lwir11",
            grid,
            Resampling.nearest,
            dtype="float32",
            nodata=0,
        )
        qa = _read_item_asset(
            item,
            "qa_pixel",
            grid,
            Resampling.nearest,
            dtype="uint16",
            nodata=65535,
        )
        lst = scale_landsat_surface_temperature(raw_st)
        valid = static_valid & landsat_clear_mask(qa.astype(np.uint16)) & np.isfinite(lst) & (lst >= 10) & (lst <= 70)
        if int(valid.sum()) < 500:
            continue
        lst_path = _write_raster(
            paths["interim"] / "landsat" / f"{date}_lst_c.tif",
            np.where(valid, lst, np.nan),
            grid,
            nodata=np.nan,
            dtype="float32",
            descriptions=["lst_c"],
        )
        _write_raster(
            paths["interim"] / "landsat" / f"{date}_valid_mask.tif",
            valid.astype(np.uint8),
            grid,
            nodata=0,
            dtype="uint8",
            descriptions=["valid_pixel"],
        )
        if first_heatmap is None:
            first_heatmap = paths["figures"] / "landsat_lst_reelle.png"
            _save_heatmap(first_heatmap, lst, valid, f"Landsat LST reelle - {date}")
        weather_row = weather.loc[date]
        day = pd.Timestamp(date).dayofyear
        frame = pd.DataFrame(
            {
                "pixel_id": [f"R{row:04d}C{col:04d}" for row, col in zip(rows[valid], cols[valid])],
                "date": pd.Timestamp(date),
                "x_m": x_coords[valid],
                "y_m": y_coords[valid],
                "lon": lon[valid],
                "lat": lat[valid],
                "elevation_m": elevation[valid],
                "slope_deg": slope[valid],
                "ndvi": ndvi[valid],
                "ndbi": ndbi[valid],
                "building_density": building_density[valid],
                "road_density": road_density[valid],
                "impervious_fraction": impervious[valid],
                "distance_to_water_m": distance_water[valid],
                "distance_to_green_m": distance_green[valid],
                "air_temperature_c": float(weather_row["temperature_2m"]),
                "relative_humidity_pct": float(weather_row["relative_humidity_2m"]),
                "wind_speed_ms": float(weather_row["wind_speed_10m"]),
                "shortwave_radiation_wm2": float(weather_row["shortwave_radiation"]),
                "day_of_year_sin": math.sin(2 * math.pi * day / 365.25),
                "day_of_year_cos": math.cos(2 * math.pi * day / 365.25),
                "lst_c": lst[valid],
            }
        )
        frames.append(frame)
        landsat_records.append(
            {
                "id": item.id,
                "date": date,
                "scene_cloud_percent": item.properties.get("eo:cloud_cover"),
                "valid_pixels": int(valid.sum()),
                "lst_path": lst_path.as_posix(),
                "assets": {
                    "lwir11": item.assets["lwir11"].href.split("?")[0],
                    "qa_pixel": item.assets["qa_pixel"].href.split("?")[0],
                },
            }
        )
    if len(frames) < 6:
        raise RuntimeError(f"Seulement {len(frames)} dates ont produit assez de pixels valides.")
    dataset = pd.concat(frames, ignore_index=True)
    dataset_path = write_table(dataset, paths["processed"] / "pixel_date_real.parquet")
    dictionary = pd.DataFrame(
        [
            ("pixel_id", "-", "identifier", "30 m reference grid"),
            ("date", "date", "group", "Landsat acquisition"),
            ("x_m", "m", "location", "EPSG:32629 grid"),
            ("y_m", "m", "location", "EPSG:32629 grid"),
            ("lon", "degree", "location", "WGS84 grid"),
            ("lat", "degree", "location", "WGS84 grid"),
            ("elevation_m", "m", "feature", "Copernicus DEM GLO-30"),
            ("slope_deg", "degree", "feature", "Copernicus DEM derived"),
            ("ndvi", "-1..1", "feature", "Sentinel-2 B08/B04"),
            ("ndbi", "-1..1", "feature", "Sentinel-2 B11/B08"),
            ("building_density", "0..1", "feature", "OpenStreetMap buildings at 10 m"),
            ("road_density", "0..1", "feature", "OpenStreetMap highways at 10 m"),
            (
                "impervious_fraction",
                "0..1",
                "feature",
                "Sentinel-2 spectral proxy (not cadastral density)",
            ),
            ("distance_to_water_m", "m", "feature", "Sentinel-2 MNDWI derived"),
            ("distance_to_green_m", "m", "feature", "Sentinel-2 NDVI derived"),
            ("air_temperature_c", "deg C", "feature", "Copernicus ERA5-Land"),
            ("relative_humidity_pct", "%", "feature", "Copernicus ERA5-Land"),
            ("wind_speed_ms", "m/s", "feature", "Copernicus ERA5"),
            ("shortwave_radiation_wm2", "W/m2", "feature", "Copernicus ERA5"),
            ("day_of_year_sin", "-", "feature", "acquisition date"),
            ("day_of_year_cos", "-", "feature", "acquisition date"),
            ("lst_c", "deg C", "target", "Landsat C2 L2 ST_B10 + QA_PIXEL"),
        ],
        columns=["variable", "unit", "role", "source_or_definition"],
    )
    dictionary.to_csv(paths["processed"] / "data_dictionary_real.csv", index=False)
    audit = {
        "status": "ok",
        "rows": int(len(dataset)),
        "dates": int(dataset["date"].nunique()),
        "unique_pixels": int(dataset["pixel_id"].nunique()),
        "duplicate_pixel_date": int(dataset.duplicated(["pixel_id", "date"]).sum()),
        "missing_percent": (dataset.isna().mean() * 100).round(5).to_dict(),
        "target_summary_c": {
            "min": float(dataset["lst_c"].min()),
            "mean": float(dataset["lst_c"].mean()),
            "max": float(dataset["lst_c"].max()),
            "std": float(dataset["lst_c"].std()),
        },
        "rows_by_date": {
            pd.Timestamp(key).date().isoformat(): int(value)
            for key, value in dataset.groupby("date").size().items()
        },
    }
    write_json(paths["processed"] / "quality_audit_real.json", audit)
    write_json(
        paths["processed"] / "reference_grid_real.json",
        {
            **{key: value for key, value in grid.items() if key != "transform"},
            "transform": list(grid["transform"])[:6],
            "boundary_mask_pixels": int(boundary.sum()),
        },
    )

    provenance = {
        "kind": "real_observation_pilot",
        "scientific_validation": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bbox_wgs84": config["bbox_wgs84"],
        "crs": config["crs"],
        "resolution_m": config["resolution_m"],
        "grid": {key: value for key, value in grid.items() if key != "transform"},
        "sentinel2": {
            "id": sentinel.id,
            "date": sentinel.datetime.date().isoformat(),
            "cloud_percent": sentinel.properties.get("eo:cloud_cover"),
        },
        "dem_items": [item.id for item in dem_items],
        "osm": {
            **osm_summary,
            "path": osm_path.as_posix(),
            "source": "OpenStreetMap",
            "license": "ODbL 1.0",
        },
        "landsat": landsat_records,
        "weather_path": weather_path.as_posix(),
        "weather": {
            "primary_dataset": "Copernicus ERA5-Land",
            "supplementary_dataset": "Copernicus ERA5",
            "delivery_service": "Open-Meteo Historical Weather API",
            "model": config["weather"].get("model", "era5_land"),
        },
        "dataset": file_record(dataset_path, "real_sources"),
        "static_raster": file_record(static_path, "real_sources"),
        "rows": int(len(dataset)),
        "dates": int(dataset["date"].nunique()),
        "unique_pixels": int(dataset["pixel_id"].nunique()),
    }
    write_json(paths["processed"] / "real_dataset_manifest.json", provenance)
    return provenance


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = np.asarray(y_true) - np.asarray(y_pred)
    denominator = np.sum((np.asarray(y_true) - np.mean(y_true)) ** 2)
    return {
        "mae_c": float(np.mean(np.abs(residual))),
        "rmse_c": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / denominator) if denominator else 0.0,
    }


def _save_metrics_chart(metrics: pd.DataFrame, path: Path) -> None:
    ordered = metrics.sort_values("mae_c")
    width, height = 1100, 150 + len(ordered) * 85
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 25), "Modeles entraines sur observations reelles - validation par dates", fill="#111827")
    maximum = float(ordered["mae_c"].max())
    for index, row in enumerate(ordered.itertuples(index=False)):
        y = 90 + index * 85
        draw.text((35, y), row.model, fill="#111827")
        bar_width = int(650 * row.mae_c / max(maximum, 0.01))
        draw.rectangle((300, y - 2, 300 + bar_width, y + 24), fill="#2563eb" if not row.is_baseline else "#94a3b8")
        draw.text((320 + bar_width, y), f"MAE {row.mae_c:.2f} C | R2 {row.r2:.2f}", fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


class XGBoostNativeRegressor:
    def __init__(
        self,
        *,
        n_estimators: int = 220,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        min_child_weight: float = 8.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.9,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> "XGBoostNativeRegressor":
        import xgboost as xgb

        matrix = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=np.float32)
        self.n_features_in_ = matrix.shape[1]
        self.booster_ = xgb.train(
            {
                "objective": "reg:squarederror",
                "eta": self.learning_rate,
                "max_depth": self.max_depth,
                "min_child_weight": self.min_child_weight,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "lambda": 1.0,
                "tree_method": "hist",
                "seed": self.random_state,
                "nthread": -1,
            },
            xgb.DMatrix(matrix, label=target),
            num_boost_round=self.n_estimators,
        )
        scores = self.booster_.get_score(importance_type="gain")
        importance = np.array(
            [float(scores.get(f"f{index}", 0.0)) for index in range(self.n_features_in_)],
            dtype=float,
        )
        total = importance.sum()
        self.feature_importances_ = importance / total if total else importance
        return self

    def predict(self, X: Any) -> np.ndarray:
        import xgboost as xgb

        return self.booster_.predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)))


def train_real_models(config: dict[str, Any]) -> dict[str, Any]:
    paths = {name: Path(value) for name, value in config["paths"].items() if name != "report_pdf"}
    data = read_table(paths["processed"] / "pixel_date_real.parquet")
    data["date"] = pd.to_datetime(data["date"])
    split = split_dates(data["date"], validation_fraction=0.2, test_dates_reserved=2)
    date_text = data["date"].dt.strftime("%Y-%m-%d")
    train = data[date_text.isin(split["train"])]
    validation = data[date_text.isin(split["validation"])]
    development = data[date_text.isin(split["train"] + split["validation"])]
    test = data[date_text.isin(split["test_reserved"])]

    X_train = train[REAL_FEATURES]
    y_train = train["lst_c"]
    X_validation = validation[REAL_FEATURES]
    y_validation = validation["lst_c"]
    project_config = load_config("configs/rabat.json")
    rf = project_config["models"]["random_forest"]
    hgb = project_config["models"]["hist_gradient_boosting"]
    model_factories = {
        "linear_regression": lambda: LinearRegressorLite(),
        "random_forest": lambda: RandomForestLiteRegressor(
            n_estimators=int(rf["n_estimators"]),
            max_depth=int(rf["max_depth"]),
            min_samples_leaf=int(rf["min_samples_leaf"]),
            random_state=42,
        ),
        "gradient_boosting": lambda: GradientBoostingLiteRegressor(
            n_estimators=int(hgb["max_iter"]),
            learning_rate=float(hgb["learning_rate"]),
            random_state=42,
        ),
        "xgboost": lambda: XGBoostNativeRegressor(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
        ),
    }
    models = {name: factory() for name, factory in model_factories.items()}
    global_mean = float(y_train.mean())
    monthly_means = train.assign(month=train["date"].dt.month).groupby("month")["lst_c"].mean()
    seasonal_prediction = (
        validation["date"].dt.month.map(monthly_means).fillna(global_mean).to_numpy()
    )
    results = [
        {
            "model": "global_mean_baseline",
            **_metrics(y_validation, np.full(len(validation), global_mean)),
            "is_baseline": True,
        },
        {
            "model": "monthly_seasonal_baseline",
            **_metrics(y_validation, seasonal_prediction),
            "is_baseline": True,
        },
    ]
    fitted = {}
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        results.append({"model": name, **_metrics(y_validation, prediction), "is_baseline": False})
        fitted[name] = model
        predictions[name] = prediction
    metrics = pd.DataFrame(results).sort_values("mae_c")
    selected_name = str(metrics.loc[~metrics["is_baseline"]].iloc[0]["model"])
    final_model = model_factories[selected_name]()
    final_model.fit(development[REAL_FEATURES], development["lst_c"])

    ensure_directories(paths["artifacts"], paths["figures"])
    with (paths["artifacts"] / "model_real_week5.pkl").open("wb") as stream:
        pickle.dump(final_model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metrics.to_csv(paths["artifacts"] / "metrics_real_validation.csv", index=False)
    write_json(paths["artifacts"] / "splits_real.json", split)
    validation_output = validation[["pixel_id", "date", "x_m", "y_m", "lon", "lat", "lst_c"]].copy()
    validation_output["pred_global_mean_baseline"] = global_mean
    validation_output["pred_monthly_seasonal_baseline"] = seasonal_prediction
    for name, values in predictions.items():
        validation_output[f"pred_{name}"] = values
    write_table(validation_output, paths["artifacts"] / "validation_predictions_real.parquet")
    chart = paths["figures"] / "metrics_modeles_reels.png"
    _save_metrics_chart(metrics, chart)
    selected_row = metrics.loc[metrics["model"] == selected_name].iloc[0]
    if hasattr(final_model, "feature_importances_"):
        importance_values = np.asarray(final_model.feature_importances_, dtype=float)
        importance_method = "tree_gain"
    elif hasattr(final_model, "coefficients_"):
        importance_values = np.abs(np.asarray(final_model.coefficients_[1:], dtype=float))
        importance_method = "absolute_standardized_coefficient"
    else:
        rng = np.random.default_rng(42)
        sample_size = min(20_000, len(validation))
        sample_indices = rng.choice(len(validation), size=sample_size, replace=False)
        sample_X = validation[REAL_FEATURES].iloc[sample_indices].to_numpy(copy=True)
        sample_y = y_validation.iloc[sample_indices].to_numpy()
        base_mae = _metrics(
            sample_y,
            fitted[selected_name].predict(sample_X),
        )["mae_c"]
        permutation_scores = []
        for feature_index in range(len(REAL_FEATURES)):
            shuffled = sample_X.copy()
            rng.shuffle(shuffled[:, feature_index])
            permuted_mae = _metrics(
                sample_y,
                fitted[selected_name].predict(shuffled),
            )["mae_c"]
            permutation_scores.append(max(0.0, permuted_mae - base_mae))
        importance_values = np.asarray(permutation_scores, dtype=float)
        importance_method = "permutation_mae_on_validation_sample"
    importance = pd.DataFrame(
        {"feature": REAL_FEATURES, "importance": importance_values}
    ).sort_values("importance", ascending=False)
    importance["method"] = importance_method
    importance.to_csv(paths["artifacts"] / "feature_importance_real.csv", index=False)
    metadata = {
        "dataset_kind": "real_observation_pilot",
        "selected_model": selected_name,
        "selected_metrics": {
            "mae_c": float(selected_row["mae_c"]),
            "rmse_c": float(selected_row["rmse_c"]),
            "r2": float(selected_row["r2"]),
        },
        "best_overall_model": str(metrics.iloc[0]["model"]),
        "best_baseline_model": str(metrics.loc[metrics["is_baseline"]].iloc[0]["model"]),
        "candidate_gain_vs_best_baseline_percent": float(
            100
            * (
                float(metrics.loc[metrics["is_baseline"]].iloc[0]["mae_c"])
                - float(selected_row["mae_c"])
            )
            / float(metrics.loc[metrics["is_baseline"]].iloc[0]["mae_c"])
        ),
        "features": REAL_FEATURES,
        "models_evaluated": metrics["model"].tolist(),
        "baseline_monthly_means_c": {
            str(int(month)): float(value) for month, value in monthly_means.items()
        },
        "feature_importance_method": importance_method,
        "rows": {
            "train": len(train),
            "validation": len(validation),
            "test_reserved": len(test),
        },
        "dates": split,
        "test_scored": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(paths["artifacts"] / "model_real_metadata.json", metadata)
    model_card = (
        "# Model card - semaine 5 réelle\n\n"
        f"- Meilleur modèle candidat : **{selected_name}**\n"
        f"- Meilleur résultat global : **{str(metrics.iloc[0]['model'])}**\n"
        f"- MAE validation : **{float(selected_row['mae_c']):.3f} °C**\n"
        f"- RMSE validation : **{float(selected_row['rmse_c']):.3f} °C**\n"
        f"- R² validation : **{float(selected_row['r2']):.3f}**\n"
        "- Validation : dates complètes, sans mélange de pixels entre partitions.\n"
        "- Test final : deux dates 2025 réservées et non scorées.\n"
        "- Données : observations Landsat réelles, ERA5-Land et variables urbaines OSM.\n"
        f"- Gain du candidat contre la meilleure baseline : "
        f"**{metadata['candidate_gain_vs_best_baseline_percent']:.2f} %**.\n"
        "- Limite : entraînement réalisé sur la zone pilote, pas encore toute la commune.\n"
    )
    (paths["artifacts"] / "MODEL_CARD_REAL.md").write_text(model_card, encoding="utf-8")
    return metadata


def run_real_pipeline(config_path: str | Path = "configs/rabat_real_pilot.json") -> dict[str, Any]:
    config = load_real_config(config_path)
    acquisition = acquire_real_data(config)
    training = train_real_models(config)
    result = {"acquisition": acquisition, "training": training}
    write_json(Path(config["paths"]["artifacts"]) / "real_pipeline_result.json", result)
    return result


def verify_real_pipeline(config_path: str | Path = "configs/rabat_real_pilot.json") -> dict[str, Any]:
    config = load_real_config(config_path)
    paths = {name: Path(value) for name, value in config["paths"].items() if name != "report_pdf"}
    expected = [
        paths["processed"] / "pixel_date_real.parquet",
        paths["processed"] / "real_dataset_manifest.json",
        paths["processed"] / "data_dictionary_real.csv",
        paths["processed"] / "quality_audit_real.json",
        paths["processed"] / "reference_grid_real.json",
        paths["interim"] / "static_real_features.tif",
        paths["artifacts"] / "metrics_real_validation.csv",
        paths["artifacts"] / "model_real_week5.pkl",
        paths["artifacts"] / "model_real_metadata.json",
        paths["artifacts"] / "feature_importance_real.csv",
        paths["artifacts"] / "MODEL_CARD_REAL.md",
    ]
    missing = [path.as_posix() for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Livrables réels manquants : {missing}")
    data = read_table(paths["processed"] / "pixel_date_real.parquet")
    if data["date"].nunique() < 6:
        raise AssertionError("Moins de six dates Landsat réelles.")
    if data.duplicated(["pixel_id", "date"]).any():
        raise AssertionError("Clés pixel-date réelles dupliquées.")
    if data["lst_c"].isna().any() or not data["lst_c"].between(10, 70).all():
        raise AssertionError("Cible LST réelle invalide.")
    missing_features = sorted(set(REAL_FEATURES) - set(data.columns))
    if missing_features:
        raise AssertionError(f"Variables de semaine 4 manquantes : {missing_features}")

    rasters = [paths["interim"] / "static_real_features.tif"] + sorted(
        (paths["interim"] / "landsat").glob("*_lst_c.tif")
    )
    signatures = set()
    for path in rasters:
        with rasterio.open(path) as source:
            signatures.add(
                (
                    source.crs.to_string(),
                    source.width,
                    source.height,
                    tuple(round(value, 6) for value in source.transform)[:6],
                )
            )
    if len(signatures) != 1:
        raise AssertionError(f"Rasters non alignés : {signatures}")
    with (paths["artifacts"] / "model_real_metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    evaluated = set(metadata.get("models_evaluated", []))
    required_models = {
        "global_mean_baseline",
        "monthly_seasonal_baseline",
        "linear_regression",
        "random_forest",
        "gradient_boosting",
        "xgboost",
    }
    if not required_models.issubset(evaluated):
        raise AssertionError(
            f"Modèles de semaine 5 manquants : {sorted(required_models - evaluated)}"
        )
    if metadata.get("test_scored") is not False:
        raise AssertionError("Le test réel réservé a été ouvert avant la semaine 6.")
    return {
        "status": "ok",
        "rows": int(len(data)),
        "dates": int(data["date"].nunique()),
        "unique_pixels": int(data["pixel_id"].nunique()),
        "aligned_rasters": len(rasters),
        "selected_model": metadata["selected_model"],
        "test_scored": metadata["test_scored"],
    }
