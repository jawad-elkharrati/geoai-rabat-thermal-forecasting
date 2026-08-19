from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/rabat.json")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Charge et valide la configuration JSON du projet."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    required = {"project", "study_area", "grid", "history", "quality", "dataset", "models", "paths"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Sections de configuration manquantes : {sorted(missing)}")

    if config["grid"]["resolution_m"] != 30:
        raise ValueError("La résolution de référence du projet doit rester fixée à 30 m.")
    if config["grid"]["crs"] != "EPSG:32629":
        raise ValueError("Le CRS de référence attendu pour Rabat est EPSG:32629.")
    if len(config["history"]["demo_dates"]) < 8:
        raise ValueError("Au moins huit dates sont nécessaires pour un split temporel utile.")
    return config


def project_path(config: dict[str, Any], key: str) -> Path:
    """Retourne un chemin de sortie configuré, relatif au dépôt."""
    try:
        return Path(config["paths"][key])
    except KeyError as exc:
        raise KeyError(f"Chemin inconnu dans la configuration : {key}") from exc
