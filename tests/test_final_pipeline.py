from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geoai_rabat.final_pipeline import _metrics, verify_complete_delivery, verify_final_pipeline


def test_metrics_are_exact_for_simple_values() -> None:
    result = _metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 2.0]))

    assert result["mae_c"] == 2 / 3
    assert np.isclose(result["rmse_c"], np.sqrt(2 / 3))
    assert result["r2"] == 0.0
    assert result["bias_c"] == 0.0


@pytest.mark.integration
def test_final_delivery_is_complete_and_test_was_opened_once() -> None:
    config_path = Path("configs/rabat_real_pilot.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    marker = json.loads(
        (Path(config["paths"]["final_artifacts"]) / "test_evaluation_marker.json").read_text(
            encoding="utf-8"
        )
    )

    result = verify_final_pipeline(config_path)

    assert result["status"] == "ok"
    assert result["test_dates_scored"] == 2
    assert result["forecast_rasters"] == 2
    assert marker["policy"].startswith("Evaluation unique")


@pytest.mark.integration
def test_complete_delivery_contains_all_eight_weeks() -> None:
    result = verify_complete_delivery("configs/rabat_real_pilot.json")

    assert result["status"] == "ok"
    assert result["weeks_complete"] == 8
    assert result["report_pages"] >= 6
    assert result["presentation_slides"] == result["presentation_notes"]
    assert result["defense_guide_pages"] >= 15
