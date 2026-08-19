from pathlib import Path

import pandas as pd

from geoai_rabat.config import load_config
from geoai_rabat.dataset import build_and_save_demo_dataset
from geoai_rabat.modeling import train_week5_models
from geoai_rabat.verify import verify_week5


def test_week5_pipeline_smoke(tmp_path):
    config = load_config(Path("configs/rabat.json"))
    config["grid"]["demo_rows"] = 14
    config["grid"]["demo_cols"] = 18
    config["models"]["random_forest"]["n_estimators"] = 12
    config["models"]["hist_gradient_boosting"]["max_iter"] = 20
    config["models"]["xgboost"]["n_estimators"] = 15
    config["paths"] = {
        "raw": str(tmp_path / "raw"),
        "interim": str(tmp_path / "interim"),
        "processed": str(tmp_path / "processed"),
        "artifacts": str(tmp_path / "artifacts"),
        "figures": str(tmp_path / "figures"),
    }

    build_and_save_demo_dataset(config)
    result = train_week5_models(config)
    check = verify_week5(config)

    assert check["status"] == "ok"
    assert result["metadata"]["test_scored"] is False
    metrics = pd.read_csv(tmp_path / "artifacts" / "metrics_validation.csv")
    assert {"global_mean_baseline", "linear_regression", "random_forest"}.issubset(
        set(metrics["model"])
    )
