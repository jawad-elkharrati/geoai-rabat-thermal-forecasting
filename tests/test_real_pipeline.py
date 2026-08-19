import numpy as np

from geoai_rabat.real_pipeline import REAL_FEATURES, _chamfer_distance, _grid, _osm_density_layers


def test_real_grid_is_aligned_at_30_metres():
    grid = _grid(
        {
            "bbox_wgs84": [-6.89, 33.94, -6.82, 34.01],
            "crs": "EPSG:32629",
            "resolution_m": 30,
        }
    )

    assert grid["resolution_m"] == 30
    assert grid["width"] == 222
    assert grid["height"] == 264
    assert grid["left"] % 30 == 0
    assert grid["top"] % 30 == 0


def test_chamfer_distance_uses_map_units():
    target = np.zeros((3, 3), dtype=bool)
    target[1, 1] = True

    distance = _chamfer_distance(target, resolution=30)

    assert distance[1, 1] == 0
    assert distance[1, 0] == 30
    assert np.isclose(distance[0, 0], np.sqrt(2) * 30)


def test_chamfer_distance_handles_absent_target():
    distance = _chamfer_distance(np.zeros((2, 2), dtype=bool), resolution=30)

    assert np.all(distance == 10_000)


def test_osm_buildings_and_roads_are_rasterized():
    grid = _grid(
        {
            "bbox_wgs84": [-6.89, 33.94, -6.82, 34.01],
            "crs": "EPSG:32629",
            "resolution_m": 30,
        }
    )
    payload = {
        "elements": [
            {
                "tags": {"building": "yes"},
                "geometry": [
                    {"lon": -6.8552, "lat": 33.9748},
                    {"lon": -6.8548, "lat": 33.9748},
                    {"lon": -6.8548, "lat": 33.9752},
                    {"lon": -6.8552, "lat": 33.9752},
                    {"lon": -6.8552, "lat": 33.9748},
                ],
            },
            {
                "tags": {"highway": "residential"},
                "geometry": [
                    {"lon": -6.856, "lat": 33.975},
                    {"lon": -6.854, "lat": 33.975},
                ],
            },
        ]
    }

    buildings, roads, summary = _osm_density_layers(
        payload,
        grid,
        np.ones((grid["height"], grid["width"]), dtype=bool),
    )

    assert float(np.nanmax(buildings)) > 0
    assert float(np.nanmax(roads)) > 0
    assert summary["building_ways"] == 1
    assert summary["road_ways"] == 1
    assert {"building_density", "road_density"}.issubset(REAL_FEATURES)
