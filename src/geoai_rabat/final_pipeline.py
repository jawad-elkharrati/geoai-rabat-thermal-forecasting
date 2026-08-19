from __future__ import annotations

import json
import math
import pickle
import re
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image, ImageDraw, ImageFont

from .io_utils import ensure_directories, file_record, write_json
from .real_pipeline import REAL_FEATURES, _grid, _write_raster, load_real_config, verify_real_pipeline
from .table_io import read_table, write_table


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    residual = truth - prediction
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "mae_c": float(np.mean(np.abs(residual))),
        "rmse_c": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1 - np.sum(residual**2) / denominator) if denominator else 0.0,
        "bias_c": float(np.mean(prediction - truth)),
    }


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    values = config["paths"]
    return {
        "processed": Path(values["processed"]),
        "week5": Path(values["artifacts"]),
        "raw": Path(values["raw"]),
        "final": Path(values["final_artifacts"]),
        "figures": Path(values["final_figures"]),
        "rasters": Path(values["prediction_rasters"]),
    }


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _thermal_rgb(values: np.ndarray, low: float, high: float) -> np.ndarray:
    normalized = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    red = np.clip(1.8 * normalized, 0, 1)
    green = np.clip(1.75 - 2.1 * np.abs(normalized - 0.5), 0, 1)
    blue = np.clip(1.8 * (1 - normalized), 0, 1)
    return (np.nan_to_num(np.dstack([red, green, blue]), nan=1.0) * 255).astype(np.uint8)


def _residual_rgb(values: np.ndarray, limit: float) -> np.ndarray:
    normalized = np.clip(values / max(limit, 1e-6), -1, 1)
    rgb = np.ones((*values.shape, 3), dtype=float)
    positive = normalized >= 0
    rgb[..., 0] = np.where(positive, 1.0, 1.0 + normalized)
    rgb[..., 1] = 1.0 - np.abs(normalized) * 0.82
    rgb[..., 2] = np.where(positive, 1.0 - normalized, 1.0)
    return (np.nan_to_num(np.clip(rgb, 0, 1), nan=1.0) * 255).astype(np.uint8)


def _save_map(
    path: Path,
    values: np.ndarray,
    valid: np.ndarray,
    title: str,
    low: float,
    high: float,
    *,
    residual: bool = False,
) -> None:
    rgb = _residual_rgb(values, max(abs(low), abs(high))) if residual else _thermal_rgb(values, low, high)
    rgb[~valid] = 245
    scale = 3
    map_image = Image.fromarray(rgb).resize((values.shape[1] * scale, values.shape[0] * scale))
    canvas = Image.new("RGB", (map_image.width + 120, map_image.height + 120), "white")
    canvas.paste(map_image, (20, 75))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 18), title, font=_font(22, bold=True), fill="#12304A")
    draw.text((20, 48), f"Echelle commune : {low:.1f} a {high:.1f} deg C", font=_font(14), fill="#475569")
    bar_x = map_image.width + 48
    bar_top = 95
    bar_height = map_image.height - 40
    for row in range(bar_height):
        value = high - (high - low) * row / max(bar_height - 1, 1)
        color = _residual_rgb(np.array([[value]]), max(abs(low), abs(high)))[0, 0] if residual else _thermal_rgb(np.array([[value]]), low, high)[0, 0]
        draw.line((bar_x, bar_top + row, bar_x + 24, bar_top + row), fill=tuple(color.tolist()))
    draw.text((bar_x + 32, bar_top - 8), f"{high:.1f}", font=_font(12), fill="#111827")
    draw.text((bar_x + 32, bar_top + bar_height - 8), f"{low:.1f}", font=_font(12), fill="#111827")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _array_from_rows(frame: pd.DataFrame, column: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    positions = frame["pixel_id"].str.extract(r"R(\d+)C(\d+)").astype(int)
    array = np.full(shape, np.nan, dtype=np.float32)
    array[positions[0].to_numpy(), positions[1].to_numpy()] = frame[column].to_numpy(dtype=np.float32)
    return array, np.isfinite(array)


def evaluate_final_test(config_path: str | Path = "configs/rabat_real_pilot.json") -> dict[str, Any]:
    config = load_real_config(config_path)
    paths = _paths(config)
    ensure_directories(paths["final"], paths["figures"], paths["rasters"])
    marker_path = paths["final"] / "test_evaluation_marker.json"
    metrics_path = paths["final"] / "test_metrics_final.json"
    if marker_path.exists() and metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        write_final_model_card(config_path)
        return result

    metadata = json.loads((paths["week5"] / "model_real_metadata.json").read_text(encoding="utf-8"))
    data = read_table(paths["processed"] / "pixel_date_real.parquet")
    data["date"] = pd.to_datetime(data["date"])
    test_dates = metadata["dates"]["test_reserved"]
    date_text = data["date"].dt.strftime("%Y-%m-%d")
    development = data[~date_text.isin(test_dates)].copy()
    test = data[date_text.isin(test_dates)].copy()
    if set(test["date"].dt.strftime("%Y-%m-%d").unique()) != set(test_dates):
        raise AssertionError("Les deux dates de test réservées ne sont pas disponibles.")

    with (paths["week5"] / "model_real_week5.pkl").open("rb") as stream:
        model = pickle.load(stream)
    test["pred_random_forest_c"] = model.predict(test[REAL_FEATURES])
    global_mean = float(development["lst_c"].mean())
    monthly_means = development.assign(month=development["date"].dt.month).groupby("month")["lst_c"].mean()
    test["pred_seasonal_baseline_c"] = test["date"].dt.month.map(monthly_means).fillna(global_mean)
    test["residual_random_forest_c"] = test["lst_c"] - test["pred_random_forest_c"]

    overall = {
        "random_forest": _metrics(test["lst_c"].to_numpy(), test["pred_random_forest_c"].to_numpy()),
        "monthly_seasonal_baseline": _metrics(test["lst_c"].to_numpy(), test["pred_seasonal_baseline_c"].to_numpy()),
    }
    by_date: dict[str, Any] = {}
    grid = _grid(config)
    residual_arrays = []
    for date, group in test.groupby(test["date"].dt.strftime("%Y-%m-%d")):
        by_date[date] = {
            "rows": int(len(group)),
            "random_forest": _metrics(group["lst_c"].to_numpy(), group["pred_random_forest_c"].to_numpy()),
            "monthly_seasonal_baseline": _metrics(group["lst_c"].to_numpy(), group["pred_seasonal_baseline_c"].to_numpy()),
        }
        residual_array, valid = _array_from_rows(group, "residual_random_forest_c", (grid["height"], grid["width"]))
        residual_arrays.append(residual_array[valid])
        _write_raster(
            paths["rasters"] / f"residual_test_{date}.tif",
            residual_array,
            grid,
            nodata=np.nan,
            dtype="float32",
            descriptions=["observed_minus_predicted_c"],
        )
    residual_limit = float(np.nanpercentile(np.abs(np.concatenate(residual_arrays)), 98))
    for date, group in test.groupby(test["date"].dt.strftime("%Y-%m-%d")):
        residual_array, valid = _array_from_rows(group, "residual_random_forest_c", (grid["height"], grid["width"]))
        _save_map(
            paths["figures"] / f"residual_test_{date}.png",
            residual_array,
            valid,
            f"Residus test final - {date}",
            -residual_limit,
            residual_limit,
            residual=True,
        )

    predictions_path = write_table(test, paths["final"] / "test_predictions_final.parquet")
    result = {
        "status": "ok",
        "test_opened_once": True,
        "test_opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_dates": test_dates,
        "rows": int(len(test)),
        "frozen_model": "random_forest",
        "overall": overall,
        "by_date": by_date,
        "best_on_test": min(overall, key=lambda name: overall[name]["mae_c"]),
        "prediction_file": file_record(predictions_path, "week6_final_test"),
    }
    write_json(metrics_path, result)
    write_json(
        marker_path,
        {
            "opened_at_utc": result["test_opened_at_utc"],
            "dates": test_dates,
            "model_file": file_record(paths["week5"] / "model_real_week5.pkl", "week5_frozen_model"),
            "policy": "Evaluation unique, aucun réentraînement après ouverture du test.",
        },
    )
    write_final_model_card(config_path)
    return result


def write_final_model_card(
    config_path: str | Path = "configs/rabat_real_pilot.json",
) -> Path:
    config = load_real_config(config_path)
    paths = _paths(config)
    metrics = json.loads(
        (paths["final"] / "test_metrics_final.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (paths["week5"] / "model_real_metadata.json").read_text(encoding="utf-8")
    )
    rf = metrics["overall"]["random_forest"]
    baseline = metrics["overall"]["monthly_seasonal_baseline"]
    dates = ", ".join(metrics["test_dates"])
    content = f"""# Model card finale - GeoAI Rabat

## Usage

- Cible : température diurne de surface terrestre (LST), en °C.
- Périmètre : zone pilote de Rabat, grille de 30 m en EPSG:32629.
- Usage prévu : preuve de concept, comparaison spatiale et cartes J+1/J+2.
- Usage interdit : température de l'air, ressenti humain ou alerte sanitaire officielle.

## Modèle

- Modèle figé : `{metadata['selected_model']}`.
- Variables : {len(metadata['features'])} variables explicatives.
- Sélection : dates complètes de validation, sans utiliser le test final.
- Test final ouvert une seule fois : {dates}.

## Résultats sur {metrics['rows']:,} observations de test

| Méthode | MAE (°C) | RMSE (°C) | R² | Biais (°C) |
|---|---:|---:|---:|---:|
| Random Forest figée | {rf['mae_c']:.3f} | {rf['rmse_c']:.3f} | {rf['r2']:.3f} | {rf['bias_c']:+.3f} |
| Baseline saisonnière | {baseline['mae_c']:.3f} | {baseline['rmse_c']:.3f} | {baseline['r2']:.3f} | {baseline['bias_c']:+.3f} |

La Random Forest bat la baseline sur le test final en MAE et RMSE. Son R² reste
modeste et son biais est positif : le modèle doit être étendu et surveillé avant
toute mise en production.

## Limites et surveillance

- zone pilote et seulement 14 dates Landsat ;
- variables Sentinel-2 et urbaines considérées statiques pour J+1/J+2 ;
- météo future extraite au point central de la zone ;
- vérifier la dérive, le biais et la qualité des entrées à chaque nouvelle campagne ;
- conserver le modèle, la réponse météo, la date de génération et les cartes.
"""
    output = paths["final"] / "MODEL_CARD_FINAL.md"
    output.write_text(content, encoding="utf-8")
    return output


def _fetch_forecast(config: dict[str, Any], output_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    west, south, east, north = config["bbox_wgs84"]
    params = {
        "latitude": (south + north) / 2,
        "longitude": (west + east) / 2,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
        "wind_speed_unit": "ms",
        "timezone": config["weather"]["timezone"],
        "forecast_days": 3,
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    payload["_provenance"] = {
        "url": response.url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "Open-Meteo Forecast API",
    }
    write_json(output_path, payload)
    weather = pd.DataFrame(payload["hourly"])
    weather["time"] = pd.to_datetime(weather["time"])
    local_now = datetime.now(ZoneInfo(config["weather"]["timezone"]))
    target_dates = [(local_now.date() + timedelta(days=offset)).isoformat() for offset in (1, 2)]
    hour = int(config["weather"]["overpass_hour_local"])
    wanted = weather[
        weather["time"].dt.strftime("%Y-%m-%d").isin(target_dates) & (weather["time"].dt.hour == hour)
    ].copy()
    wanted["date"] = wanted["time"].dt.strftime("%Y-%m-%d")
    if set(wanted["date"]) != set(target_dates):
        raise RuntimeError(f"Prévisions J+1/J+2 incomplètes : {target_dates}")
    return wanted.set_index("date"), payload["_provenance"]


def generate_two_day_forecast(config_path: str | Path = "configs/rabat_real_pilot.json") -> dict[str, Any]:
    config = load_real_config(config_path)
    paths = _paths(config)
    ensure_directories(paths["final"], paths["figures"], paths["rasters"], paths["raw"])
    weather, provenance = _fetch_forecast(config, paths["raw"] / "open_meteo_forecast_j1_j2.json")
    grid = _grid(config)
    with rasterio.open(Path(config["paths"]["interim"]) / "static_real_features.tif") as source:
        static = source.read().astype(np.float32)
        descriptions = list(source.descriptions)
    if descriptions != REAL_FEATURES[: len(descriptions)]:
        raise AssertionError(f"Ordre des variables statiques inattendu : {descriptions}")
    valid = np.all(np.isfinite(static), axis=0)
    with (paths["week5"] / "model_real_week5.pkl").open("rb") as stream:
        model = pickle.load(stream)

    predictions: dict[str, np.ndarray] = {}
    weather_records = {}
    for date, row in weather.iterrows():
        day = pd.Timestamp(date).dayofyear
        features = [band[valid] for band in static]
        features.extend(
            [
                np.full(valid.sum(), float(row["temperature_2m"]), dtype=np.float32),
                np.full(valid.sum(), float(row["relative_humidity_2m"]), dtype=np.float32),
                np.full(valid.sum(), float(row["wind_speed_10m"]), dtype=np.float32),
                np.full(valid.sum(), float(row["shortwave_radiation"]), dtype=np.float32),
                np.full(valid.sum(), math.sin(2 * math.pi * day / 365.25), dtype=np.float32),
                np.full(valid.sum(), math.cos(2 * math.pi * day / 365.25), dtype=np.float32),
            ]
        )
        matrix = np.column_stack(features)
        if matrix.shape[1] != len(REAL_FEATURES):
            raise AssertionError(f"Matrice de prévision invalide : {matrix.shape}")
        prediction = np.full(valid.shape, np.nan, dtype=np.float32)
        prediction[valid] = model.predict(matrix).astype(np.float32)
        predictions[date] = prediction
        weather_records[date] = {
            "air_temperature_c": float(row["temperature_2m"]),
            "relative_humidity_pct": float(row["relative_humidity_2m"]),
            "wind_speed_ms": float(row["wind_speed_10m"]),
            "shortwave_radiation_wm2": float(row["shortwave_radiation"]),
            "reference_hour_local": int(config["weather"]["overpass_hour_local"]),
        }
        _write_raster(
            paths["rasters"] / f"lst_forecast_{date}.tif",
            prediction,
            grid,
            nodata=np.nan,
            dtype="float32",
            descriptions=["predicted_lst_c"],
        )

    combined = np.concatenate([array[np.isfinite(array)] for array in predictions.values()])
    low, high = [float(value) for value in np.nanpercentile(combined, [2, 98])]
    hotspot_frames = []
    for date, prediction in predictions.items():
        _save_map(
            paths["figures"] / f"lst_forecast_{date}.png",
            prediction,
            np.isfinite(prediction),
            f"Temperature de surface prevue - {date}",
            low,
            high,
        )
        rows, cols = np.where(np.isfinite(prediction))
        values = prediction[rows, cols]
        order = np.argsort(values)[-100:][::-1]
        x = grid["left"] + (cols[order] + 0.5) * grid["resolution_m"]
        y = grid["top"] - (rows[order] + 0.5) * grid["resolution_m"]
        hotspot_frames.append(
            pd.DataFrame(
                {
                    "date": date,
                    "rank": np.arange(1, len(order) + 1),
                    "pixel_id": [f"R{r:04d}C{c:04d}" for r, c in zip(rows[order], cols[order])],
                    "x_m": x,
                    "y_m": y,
                    "predicted_lst_c": values[order],
                }
            )
        )
    hotspots = pd.concat(hotspot_frames, ignore_index=True)
    hotspots.to_csv(paths["final"] / "forecast_hotspots_top100.csv", index=False)
    result = {
        "status": "ok",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "random_forest_week5_frozen",
        "forecast_dates": sorted(predictions),
        "reference_hour_local": int(config["weather"]["overpass_hour_local"]),
        "weather": weather_records,
        "common_color_scale_c": {"min": low, "max": high},
        "statistics": {
            date: {
                "min_c": float(np.nanmin(values)),
                "mean_c": float(np.nanmean(values)),
                "max_c": float(np.nanmax(values)),
                "valid_pixels": int(np.isfinite(values).sum()),
            }
            for date, values in predictions.items()
        },
        "source": provenance,
    }
    write_json(paths["final"] / "forecast_summary.json", result)
    return result


def verify_final_pipeline(config_path: str | Path = "configs/rabat_real_pilot.json") -> dict[str, Any]:
    config = load_real_config(config_path)
    paths = _paths(config)
    test_metrics = json.loads((paths["final"] / "test_metrics_final.json").read_text(encoding="utf-8"))
    forecast = json.loads((paths["final"] / "forecast_summary.json").read_text(encoding="utf-8"))
    if len(forecast["forecast_dates"]) != 2:
        raise AssertionError("Deux dates de prévision sont requises.")
    expected = [
        paths["final"] / "test_evaluation_marker.json",
        paths["final"] / "test_predictions_final.parquet",
        paths["final"] / "forecast_hotspots_top100.csv",
    ]
    for date in test_metrics["test_dates"]:
        expected.extend([paths["rasters"] / f"residual_test_{date}.tif", paths["figures"] / f"residual_test_{date}.png"])
    for date in forecast["forecast_dates"]:
        expected.extend([paths["rasters"] / f"lst_forecast_{date}.tif", paths["figures"] / f"lst_forecast_{date}.png"])
    missing = [path.as_posix() for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Livrables finaux manquants : {missing}")
    signatures = set()
    for date in forecast["forecast_dates"]:
        with rasterio.open(paths["rasters"] / f"lst_forecast_{date}.tif") as source:
            signatures.add((source.crs.to_string(), source.width, source.height, tuple(source.transform)[:6]))
    if len(signatures) != 1:
        raise AssertionError("Les deux GeoTIFF de prévision ne sont pas alignés.")
    return {
        "status": "ok",
        "test_dates_scored": len(test_metrics["test_dates"]),
        "forecast_dates": forecast["forecast_dates"],
        "forecast_rasters": 2,
        "test_rows": test_metrics["rows"],
        "model": forecast["model"],
    }


def run_complete_pipeline(config_path: str | Path = "configs/rabat_real_pilot.json") -> dict[str, Any]:
    try:
        verify_real_pipeline(config_path)
    except (FileNotFoundError, AssertionError):
        from .real_pipeline import run_real_pipeline

        run_real_pipeline(config_path)
    test = evaluate_final_test(config_path)
    forecast = generate_two_day_forecast(config_path)
    verification = verify_final_pipeline(config_path)
    from .final_report import generate_final_report

    report = generate_final_report(config_path)
    from .defense_report import generate_defense_guide

    defense_guide = generate_defense_guide(config_path)
    delivery = verify_complete_delivery(config_path)
    return {
        "status": "ok",
        "test": test,
        "forecast": forecast,
        "verification": verification,
        "report": report,
        "defense_guide": defense_guide,
        "delivery": delivery,
    }


def verify_complete_delivery(
    config_path: str | Path = "configs/rabat_real_pilot.json",
) -> dict[str, Any]:
    from pypdf import PdfReader

    config = load_real_config(config_path)
    real = verify_real_pipeline(config_path)
    final = verify_final_pipeline(config_path)
    write_final_model_card(config_path)
    paths = _paths(config)
    required = [
        Path("data/processed/real/pixel_date_real.parquet"),
        Path("data/processed/real/data_dictionary_real.csv"),
        paths["week5"] / "model_real_week5.pkl",
        paths["final"] / "MODEL_CARD_FINAL.md",
        paths["final"] / "forecast_hotspots_top100.csv",
        Path(config["paths"]["final_report_pdf"]),
        Path(config["paths"]["presentation_pptx"]),
        Path("scripts/run_complete.ps1"),
    ]
    required.extend(Path("docs") / f"{index:02d}_{name}.md" for index, name in [
        (1, "cadrage"), (2, "sources"), (3, "faisabilite"),
        (4, "dictionnaire_donnees"), (5, "modelisation"),
        (6, "bilan_semaines_1_5"), (7, "validation_finale"),
        (8, "previsions_j1_j2"), (9, "guide_installation_demo"),
        (10, "audit_final"),
    ])
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Livraison incomplète : {missing}")

    report_path = Path(config["paths"]["final_report_pdf"])
    report_pages = len(PdfReader(report_path).pages)
    if report_pages < 6:
        raise AssertionError("Le rapport final doit contenir au moins six pages.")
    presentation_path = Path(config["paths"]["presentation_pptx"])
    with zipfile.ZipFile(presentation_path) as archive:
        names = archive.namelist()
        slides = [name for name in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
        notes = [name for name in names if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", name)]
    if len(slides) < 9 or len(notes) != len(slides):
        raise AssertionError("Présentation incomplète ou notes de sources manquantes.")
    defense_path = Path(config["paths"]["defense_guide_pdf"])
    defense_pages = len(PdfReader(defense_path).pages) if defense_path.exists() else 0
    return {
        "status": "ok",
        "weeks_complete": 8,
        "dataset_rows": real["rows"],
        "aligned_source_rasters": real["aligned_rasters"],
        "test_dates": final["test_dates_scored"],
        "forecast_rasters": final["forecast_rasters"],
        "documentation_files": 10,
        "report_pages": report_pages,
        "presentation_slides": len(slides),
        "presentation_notes": len(notes),
        "defense_guide_pages": defense_pages,
    }
