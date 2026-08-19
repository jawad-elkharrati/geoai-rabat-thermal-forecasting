from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import ensure_directories, file_record, write_json


USER_AGENT = "geoai-rabat-stage/0.1 (educational project; contact via repository owner)"


def _request_json(
    url: str, *, method: str = "GET", payload: dict[str, Any] | None = None, timeout: int = 60
) -> Any:
    data = None
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_rabat_boundary(config: dict[str, Any], output_dir: Path) -> Path:
    query = urllib.parse.urlencode(
        {
            "q": config["study_area"]["boundary_query"],
            "format": "jsonv2",
            "polygon_geojson": 1,
            "polygon_threshold": 0.0003,
            "addressdetails": 1,
            "limit": 5,
        }
    )
    url = f"https://nominatim.openstreetmap.org/search?{query}"
    results = _request_json(url)
    polygon_result = next(
        (
            item
            for item in results
            if item.get("geojson", {}).get("type") in {"Polygon", "MultiPolygon"}
            and "Rabat" in item.get("display_name", "")
        ),
        None,
    )
    if polygon_result is None:
        raise RuntimeError("Nominatim n'a pas renvoyé de limite polygonale pour Rabat.")

    feature = {
        "type": "FeatureCollection",
        "name": "rabat_boundary_osm",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "display_name": polygon_result["display_name"],
                    "osm_type": polygon_result.get("osm_type"),
                    "osm_id": polygon_result.get("osm_id"),
                    "licence": polygon_result.get("licence"),
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "request_url": url,
                },
                "geometry": polygon_result["geojson"],
            }
        ],
    }
    path = output_dir / "boundaries" / "rabat_commune.geojson"
    write_json(path, feature)
    return path


def fetch_weather_sample(config: dict[str, Any], output_dir: Path) -> Path:
    lon, lat = config["study_area"]["centroid_wgs84"]
    query = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2024-07-01",
            "end_date": "2024-07-07",
            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "shortwave_radiation",
                ]
            ),
            "wind_speed_unit": "ms",
            "timezone": "Africa/Casablanca",
        }
    )
    url = f"https://archive-api.open-meteo.com/v1/archive?{query}"
    payload = _request_json(url)
    payload["_provenance"] = {
        "request_url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = output_dir / "weather" / "open_meteo_sample_2024-07-01_2024-07-07.json"
    write_json(path, payload)
    return path


def fetch_landsat_catalog_sample(config: dict[str, Any], output_dir: Path) -> Path:
    bbox = config["study_area"]["bbox_wgs84"]
    search = {
        "collections": ["landsat-c2l2-st"],
        "bbox": bbox,
        "datetime": "2024-06-01T00:00:00Z/2024-09-30T23:59:59Z",
        "limit": 20,
        "query": {
            "eo:cloud_cover": {
                "lte": config["quality"]["max_scene_cloud_percent"],
            }
        },
    }
    url = "https://landsatlook.usgs.gov/stac-server/search"
    response = _request_json(url, method="POST", payload=search)
    compact_items = []
    for feature in response.get("features", []):
        compact_items.append(
            {
                "id": feature.get("id"),
                "collection": feature.get("collection"),
                "datetime": feature.get("properties", {}).get("datetime"),
                "cloud_cover": feature.get("properties", {}).get("eo:cloud_cover"),
                "bbox": feature.get("bbox"),
                "asset_keys": sorted(feature.get("assets", {}).keys()),
            }
        )
    payload = {
        "request_url": url,
        "search": search,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "matched": response.get("context", {}).get("matched"),
        "returned": len(compact_items),
        "items": compact_items,
    }
    path = output_dir / "landsat" / "stac_sample_2024_summer.json"
    write_json(path, payload)
    return path


def fetch_samples(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["raw"]) / "samples"
    ensure_directories(output_dir)
    tasks = {
        "osm_boundary": fetch_rabat_boundary,
        "open_meteo": fetch_weather_sample,
        "landsat_stac": fetch_landsat_catalog_sample,
    }
    results: dict[str, Any] = {}
    records = []
    for name, task in tasks.items():
        try:
            path = task(config, output_dir)
            results[name] = {"status": "ok", "path": path.as_posix()}
            records.append(file_record(path, name, generated=False))
        except Exception as exc:  # Les APIs externes ne doivent pas casser le mode démo.
            results[name] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "files": records,
    }
    write_json(output_dir / "sample_manifest.json", manifest)
    return manifest
