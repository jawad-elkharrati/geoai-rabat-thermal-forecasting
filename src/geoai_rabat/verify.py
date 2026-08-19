from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from .schema import FEATURE_COLUMNS, TARGET_COLUMN
from .table_io import read_table, resolve_table_path


def verify_week5(config: dict[str, Any]) -> dict[str, Any]:
    processed = Path(config["paths"]["processed"])
    artifacts = Path(config["paths"]["artifacts"])
    expected = [
        resolve_table_path(processed / "pixel_date_demo.parquet"),
        processed / "data_dictionary.csv",
        processed / "quality_audit.json",
        processed / "dataset_manifest.json",
        artifacts / "metrics_validation.csv",
        artifacts / "model_week5.pkl",
        artifacts / "model_metadata.json",
        artifacts / "splits.json",
        artifacts / "feature_schema.json",
        resolve_table_path(artifacts / "validation_predictions.parquet"),
        artifacts / "MODEL_CARD.md",
    ]
    missing = [path.as_posix() for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Livrables manquants : {missing}")

    data = read_table(processed / "pixel_date_demo.parquet")
    if data.duplicated(["pixel_id", "date"]).any():
        raise AssertionError("Clés pixel-date dupliquées.")
    if data[TARGET_COLUMN].isna().any():
        raise AssertionError("La cible contient des valeurs manquantes.")

    with (artifacts / "splits.json").open(encoding="utf-8") as stream:
        splits = json.load(stream)
    split_sets = [set(splits[name]) for name in ["train", "validation", "test_reserved"]]
    if split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2]:
        raise AssertionError("Chevauchement de dates dans splits.json.")

    with (artifacts / "model_metadata.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("test_scored") is not False:
        raise AssertionError("Le test final ne doit pas être scoré en semaine 5.")
    if metadata["features"] != FEATURE_COLUMNS:
        raise AssertionError("Le schéma des features du modèle a changé.")

    with (artifacts / "model_week5.pkl").open("rb") as stream:
        model = pickle.load(stream)
    probe = data[FEATURE_COLUMNS].head(5)
    prediction = model.predict(probe)
    if len(prediction) != 5:
        raise AssertionError("Le modèle sérialisé ne prédit pas le nombre de lignes attendu.")

    metrics = pd.read_csv(artifacts / "metrics_validation.csv")
    selected = metadata["selected_model"]
    if selected not in set(metrics["model"]):
        raise AssertionError("Le modèle sélectionné n'existe pas dans les métriques.")

    return {
        "status": "ok",
        "dataset_rows": len(data),
        "dates": int(pd.to_datetime(data["date"]).nunique()),
        "selected_model": selected,
        "prediction_probe_min_c": float(prediction.min()),
        "prediction_probe_max_c": float(prediction.max()),
        "deliverables_checked": len(expected),
    }
