from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .grid import build_demo_grid, normalized_coordinates
from .io_utils import ensure_directories, file_record, write_json
from .schema import DATA_DICTIONARY, FEATURE_COLUMNS, TARGET_COLUMN
from .table_io import write_table


def _gaussian(x: np.ndarray, y: np.ndarray, cx: float, cy: float, scale: float) -> np.ndarray:
    return np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2.0 * scale**2))


def build_static_features(grid: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Crée des couches statiques spatialement cohérentes pour la démonstration."""
    rng = np.random.default_rng(seed)
    x, y = normalized_coordinates(grid)

    park_1 = _gaussian(x, y, -0.15, 0.05, 0.20)
    park_2 = _gaussian(x, y, 0.45, -0.45, 0.16)
    park_3 = _gaussian(x, y, -0.50, 0.48, 0.13)
    green_signal = np.maximum.reduce([park_1, park_2, park_3])

    centre = _gaussian(x, y, 0.05, 0.00, 0.48)
    industrial = _gaussian(x, y, 0.58, 0.25, 0.24)
    building = np.clip(0.08 + 0.68 * centre + 0.30 * industrial - 0.55 * green_signal, 0, 1)
    building += rng.normal(0, 0.035, size=len(grid))
    building = np.clip(building, 0, 1)

    road_pattern = 0.22 + 0.35 * centre + 0.10 * np.sin(8 * x) ** 2 + 0.08 * np.cos(7 * y) ** 2
    road_density = np.clip(road_pattern - 0.22 * green_signal, 0, 1)
    impervious = np.clip(0.70 * building + 0.35 * road_density - 0.12 * green_signal, 0, 1)

    # Atlantique à l'ouest et Bouregreg au nord-est.
    coastal_distance = (x - x.min()) * 7_000.0
    river_distance = np.sqrt(((x - 0.72) * 6_000.0) ** 2 + ((y - 0.78) * 5_000.0) ** 2)
    distance_water = np.minimum(coastal_distance, river_distance)

    park_centres = [(-0.15, 0.05), (0.45, -0.45), (-0.50, 0.48)]
    park_distances = [
        np.sqrt(((x - cx) * 7_000.0) ** 2 + ((y - cy) * 6_000.0) ** 2)
        for cx, cy in park_centres
    ]
    distance_park = np.minimum.reduce(park_distances)

    elevation = 12.0 + 62.0 * (x + 1.0) / 2.0 + 18.0 * _gaussian(x, y, 0.25, -0.25, 0.32)
    elevation += 3.0 * np.sin(3.5 * y) + rng.normal(0, 1.2, size=len(grid))
    slope = np.clip(0.9 + 5.5 * np.abs(np.cos(2.3 * x) * np.sin(2.0 * y)), 0, 15)

    ndvi = np.clip(0.12 + 0.68 * green_signal - 0.20 * impervious + rng.normal(0, 0.025, len(grid)), -0.2, 0.9)
    ndbi = np.clip(-0.08 + 0.62 * impervious - 0.25 * green_signal + rng.normal(0, 0.02, len(grid)), -0.5, 0.8)

    features = grid.copy()
    features["elevation_m"] = elevation.astype(np.float32)
    features["slope_deg"] = slope.astype(np.float32)
    features["ndvi_static"] = ndvi.astype(np.float32)
    features["ndbi_static"] = ndbi.astype(np.float32)
    features["building_density"] = building.astype(np.float32)
    features["road_density"] = road_density.astype(np.float32)
    features["impervious_fraction"] = impervious.astype(np.float32)
    features["distance_to_water_m"] = distance_water.astype(np.float32)
    features["distance_to_park_m"] = distance_park.astype(np.float32)
    return features


def build_demo_weather(dates: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)
    date_values = pd.to_datetime(dates)
    day_of_year = date_values.dayofyear.to_numpy()
    summer_peak = np.cos(2 * np.pi * (day_of_year - 205) / 365.25)
    year_trend = (date_values.year.to_numpy() - date_values.year.min()) * 0.11

    air = 23.5 + 5.4 * summer_peak + year_trend + rng.normal(0, 1.4, len(dates))
    humidity = 67.0 - 10.0 * summer_peak - 0.8 * (air - air.mean()) + rng.normal(0, 3.0, len(dates))
    wind = np.clip(3.2 + rng.normal(0, 0.9, len(dates)), 0.8, 7.5)
    radiation = 690.0 + 125.0 * summer_peak + rng.normal(0, 45.0, len(dates))

    weather = pd.DataFrame(
        {
            "date": date_values,
            "air_temperature_c": air.astype(np.float32),
            "relative_humidity_pct": np.clip(humidity, 30, 90).astype(np.float32),
            "wind_speed_ms": wind.astype(np.float32),
            "shortwave_radiation_wm2": np.clip(radiation, 350, 900).astype(np.float32),
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / 365.25).astype(np.float32),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / 365.25).astype(np.float32),
        }
    )
    return weather


def build_demo_pixel_date(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = int(config["project"]["random_seed"])
    rng = np.random.default_rng(seed + 200)
    grid = build_demo_grid(config)
    static = build_static_features(grid, seed)
    weather = build_demo_weather(config["history"]["demo_dates"], seed)

    x_norm, y_norm = normalized_coordinates(static)
    records: list[pd.DataFrame] = []
    for date_index, weather_row in weather.iterrows():
        n_pixels = len(static)
        cloud_target = float(np.clip(rng.beta(2.2, 8.0), 0.04, 0.34))
        cloud_field = (
            rng.random(n_pixels)
            + 0.20 * np.sin(3.0 * x_norm + date_index)
            + 0.15 * np.cos(4.0 * y_norm - date_index / 2)
        )
        threshold = np.quantile(cloud_field, cloud_target)
        valid = cloud_field > threshold
        scene_cloud_fraction = 1.0 - valid.mean()

        dynamic_ndvi = np.clip(
            static["ndvi_static"].to_numpy()
            + 0.025 * weather_row["day_of_year_cos"]
            + rng.normal(0, 0.018, n_pixels),
            -0.2,
            0.9,
        )
        dynamic_ndbi = np.clip(
            static["ndbi_static"].to_numpy() + rng.normal(0, 0.014, n_pixels),
            -0.5,
            0.8,
        )

        lst = (
            weather_row["air_temperature_c"]
            + 7.2
            + 6.7 * static["building_density"].to_numpy()
            + 4.2 * static["impervious_fraction"].to_numpy()
            + 2.8 * dynamic_ndbi
            - 5.6 * dynamic_ndvi
            - 0.00042 * static["distance_to_water_m"].to_numpy()
            - 0.00025 * static["distance_to_park_m"].to_numpy()
            + 0.0105 * (weather_row["shortwave_radiation_wm2"] - 600.0)
            - 0.038 * (weather_row["relative_humidity_pct"] - 55.0)
            - 0.33 * weather_row["wind_speed_ms"]
            + 0.55 * x_norm
            - 0.25 * y_norm
            + rng.normal(0, 0.85, n_pixels)
        )
        lst = np.clip(lst, 12.0, 68.0)

        frame = static.loc[
            valid,
            [
                "pixel_id",
                "x_m",
                "y_m",
                "lon",
                "lat",
                "elevation_m",
                "slope_deg",
                "building_density",
                "road_density",
                "impervious_fraction",
                "distance_to_water_m",
                "distance_to_park_m",
            ],
        ].copy()
        frame["date"] = weather_row["date"]
        frame["ndvi"] = dynamic_ndvi[valid].astype(np.float32)
        frame["ndbi"] = dynamic_ndbi[valid].astype(np.float32)
        for column in [
            "air_temperature_c",
            "relative_humidity_pct",
            "wind_speed_ms",
            "shortwave_radiation_wm2",
            "day_of_year_sin",
            "day_of_year_cos",
        ]:
            frame[column] = np.float32(weather_row[column])
        frame["landsat_cloud_fraction"] = np.float32(scene_cloud_fraction)
        frame[TARGET_COLUMN] = lst[valid].astype(np.float32)
        records.append(frame)

    dataset = pd.concat(records, ignore_index=True)
    dataset = dataset[
        [
            "pixel_id",
            "date",
            "x_m",
            "y_m",
            "lon",
            "lat",
            *FEATURE_COLUMNS,
            "landsat_cloud_fraction",
            TARGET_COLUMN,
        ]
    ]

    # Lacunes réalistes de Sentinel-2 : elles seront imputées dans le pipeline ML.
    for column in ["ndvi", "ndbi"]:
        missing = rng.random(len(dataset)) < 0.008
        dataset.loc[missing, column] = np.nan

    return dataset, static, weather


def _quality_audit(dataset: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    duplicate_keys = int(dataset.duplicated(["pixel_id", "date"]).sum())
    missing_percent = (dataset.isna().mean() * 100).round(4).to_dict()
    target = dataset[TARGET_COLUMN]
    by_date = (
        dataset.groupby("date")
        .agg(
            rows=("pixel_id", "size"),
            pixels=("pixel_id", "nunique"),
            lst_mean_c=(TARGET_COLUMN, "mean"),
            lst_std_c=(TARGET_COLUMN, "std"),
            cloud_fraction=("landsat_cloud_fraction", "first"),
        )
        .reset_index()
    )

    issues: list[str] = []
    if duplicate_keys:
        issues.append(f"{duplicate_keys} clés pixel-date dupliquées")
    if target.isna().any():
        issues.append("cible manquante")
    q = config["quality"]
    if not target.between(q["lst_valid_min_c"], q["lst_valid_max_c"]).all():
        issues.append("cible hors plage configurée")
    excessive_missing = {
        name: value
        for name, value in missing_percent.items()
        if name in FEATURE_COLUMNS and value > q["max_feature_missing_percent"]
    }
    if excessive_missing:
        issues.append(f"features trop incomplètes : {excessive_missing}")

    return {
        "status": "ok" if not issues else "failed",
        "issues": issues,
        "rows": int(len(dataset)),
        "unique_pixels": int(dataset["pixel_id"].nunique()),
        "dates": int(dataset["date"].nunique()),
        "date_min": dataset["date"].min().date().isoformat(),
        "date_max": dataset["date"].max().date().isoformat(),
        "duplicate_pixel_date_keys": duplicate_keys,
        "missing_percent": missing_percent,
        "target_summary_c": {
            "min": float(target.min()),
            "mean": float(target.mean()),
            "max": float(target.max()),
            "std": float(target.std()),
        },
        "by_date": by_date.to_dict(orient="records"),
    }


def build_and_save_demo_dataset(config: dict[str, Any]) -> dict[str, Any]:
    processed = Path(config["paths"]["processed"])
    ensure_directories(processed)
    dataset, static, weather = build_demo_pixel_date(config)

    dataset_path = processed / "pixel_date_demo.parquet"
    sample_csv_path = processed / "pixel_date_demo_sample.csv"
    grid_path = processed / "static_grid_demo.parquet"
    weather_path = processed / "weather_demo.csv"
    dictionary_path = processed / "data_dictionary.csv"
    audit_path = processed / "quality_audit.json"

    dataset_path = write_table(dataset, dataset_path)
    dataset.head(1000).to_csv(sample_csv_path, index=False)
    grid_path = write_table(static, grid_path)
    weather.to_csv(weather_path, index=False)
    pd.DataFrame(
        DATA_DICTIONARY,
        columns=["variable", "dtype", "unit", "role", "source", "description"],
    ).to_csv(dictionary_path, index=False)

    audit = _quality_audit(dataset, config)
    write_json(audit_path, audit)
    if audit["status"] != "ok":
        raise ValueError(f"Audit qualité en échec : {audit['issues']}")

    files = [
        file_record(dataset_path, "demo_generator"),
        file_record(sample_csv_path, "demo_generator"),
        file_record(grid_path, "demo_generator"),
        file_record(weather_path, "demo_generator"),
        file_record(dictionary_path, "project_schema"),
        file_record(audit_path, "quality_audit"),
    ]
    manifest = {
        "dataset_kind": "synthetic_geospatial_demo",
        "scientific_validation": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": int(config["project"]["random_seed"]),
        "crs": config["grid"]["crs"],
        "nominal_resolution_m": config["grid"]["resolution_m"],
        "rows": len(dataset),
        "files": files,
    }
    write_json(processed / "dataset_manifest.json", manifest)
    return {"audit": audit, "manifest": manifest}
