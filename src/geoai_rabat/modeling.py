from __future__ import annotations

import math
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from .io_utils import ensure_directories, file_record, write_json
from .lite_models import (
    GradientBoostingLiteRegressor,
    LinearRegressorLite,
    RandomForestLiteRegressor,
)
from .schema import FEATURE_COLUMNS, TARGET_COLUMN
from .table_io import read_table, write_table


def split_dates(
    dates: pd.Series,
    validation_fraction: float,
    test_dates_reserved: int,
) -> dict[str, list[str]]:
    unique_dates = sorted(pd.to_datetime(dates).dt.normalize().unique())
    if len(unique_dates) < test_dates_reserved + 5:
        raise ValueError("Nombre de dates insuffisant pour entraînement, validation et test.")

    test = unique_dates[-test_dates_reserved:]
    development = unique_dates[:-test_dates_reserved]
    validation_count = max(2, int(math.ceil(len(development) * validation_fraction)))
    validation_count = min(validation_count, len(development) - 3)
    validation = development[-validation_count:]
    train = development[:-validation_count]

    split = {
        "train": [pd.Timestamp(value).date().isoformat() for value in train],
        "validation": [pd.Timestamp(value).date().isoformat() for value in validation],
        "test_reserved": [pd.Timestamp(value).date().isoformat() for value in test],
    }
    sets = [set(values) for values in split.values()]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise AssertionError("Chevauchement de dates entre les partitions.")
    return split


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    residual = true - prediction
    denominator = float(np.sum((true - true.mean()) ** 2))
    return {
        "mae_c": float(np.mean(np.abs(residual))),
        "rmse_c": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator else 0.0,
    }


def _candidate_models(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config["project"]["random_seed"])
    rf = config["models"]["random_forest"]
    hgb = config["models"]["hist_gradient_boosting"]

    return {
        "linear_regression": LinearRegressorLite(),
        "random_forest": RandomForestLiteRegressor(
            random_state=seed,
            n_estimators=int(rf["n_estimators"]),
            max_depth=int(rf["max_depth"]),
            min_samples_leaf=int(rf["min_samples_leaf"]),
        ),
        "gradient_boosting": GradientBoostingLiteRegressor(
            random_state=seed,
            n_estimators=int(hgb["max_iter"]),
            learning_rate=float(hgb["learning_rate"]),
            max_depth=3,
            min_samples_leaf=max(12, int(rf["min_samples_leaf"])),
        ),
    }


def _seasonal_predictions(
    train_dates: pd.Series,
    train_target: pd.Series,
    validation_dates: pd.Series,
) -> np.ndarray:
    monthly = train_target.groupby(pd.to_datetime(train_dates).dt.month.to_numpy()).mean()
    fallback = float(train_target.mean())
    months = pd.to_datetime(validation_dates).dt.month
    return months.map(monthly).fillna(fallback).to_numpy(dtype=np.float64)


def _plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    ordered = metrics.sort_values("mae_c")
    width, height = 1200, 160 + 90 * len(ordered)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), "Comparaison preliminaire - dates de validation completes", fill="#111827")
    draw.text((40, 55), "MAE de validation (deg C) - plus petit = meilleur", fill="#4b5563")
    maximum = max(float(ordered["mae_c"].max()), 0.1)
    for index, row in enumerate(ordered.itertuples(index=False)):
        y = 105 + index * 90
        draw.text((40, y), str(row.model), fill="#111827")
        bar_width = int(800 * float(row.mae_c) / maximum)
        color = "#8da0cb" if bool(row.is_baseline) else "#2563eb"
        draw.rectangle((300, y - 3, 300 + bar_width, y + 24), fill=color)
        draw.text((320 + bar_width, y), f"{float(row.mae_c):.3f}", fill="#111827")
    image.save(output)


def _plot_importance(importance: pd.DataFrame, output: Path) -> None:
    ordered = importance.sort_values("importance_mean", ascending=False).head(15)
    width, height = 1200, 140 + 62 * len(ordered)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((40, 25), "Importance des variables - permutation sur validation", fill="#111827")
    maximum = max(float(ordered["importance_mean"].clip(lower=0).max()), 1e-9)
    for index, row in enumerate(ordered.itertuples(index=False)):
        y = 80 + index * 62
        draw.text((40, y), str(row.feature), fill="#111827")
        positive = max(0.0, float(row.importance_mean))
        bar_width = int(720 * positive / maximum)
        draw.rectangle((300, y - 2, 300 + bar_width, y + 22), fill="#0f766e")
        draw.text((320 + bar_width, y), f"{float(row.importance_mean):.4f}", fill="#111827")
    image.save(output)


def _permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    baseline_r2 = _metrics(y.to_numpy(), model.predict(X))["r2"]
    importances = np.zeros((X.shape[1], repeats), dtype=np.float64)
    matrix = X.to_numpy(dtype=np.float64)
    for feature_index in range(X.shape[1]):
        for repeat in range(repeats):
            permuted = matrix.copy()
            permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
            score = _metrics(y.to_numpy(), model.predict(permuted))["r2"]
            importances[feature_index, repeat] = baseline_r2 - score
    return importances.mean(axis=1), importances.std(axis=1)


def _write_model_card(
    path: Path,
    selected_name: str,
    selected_metrics: dict[str, Any],
    split: dict[str, list[str]],
    rows: dict[str, int],
) -> None:
    content = f"""# Fiche modèle préliminaire - semaine 5

## Modèle sélectionné

`{selected_name}`, sélectionné sur la MAE des dates de validation.

| Métrique | Valeur |
|---|---:|
| MAE | {selected_metrics['mae_c']:.4f} °C |
| RMSE | {selected_metrics['rmse_c']:.4f} °C |
| R² | {selected_metrics['r2']:.4f} |
| Gain MAE vs baseline | {selected_metrics['gain_vs_global_baseline_pct']:.2f} % |

## Données

Jeu **synthétique de démonstration** : {rows['train']:,} lignes d'entraînement,
{rows['validation']:,} lignes de validation et {rows['test_reserved']:,} lignes
réservées. Ces métriques valident le pipeline logiciel ; elles ne mesurent pas encore
la performance scientifique sur Rabat.

## Partitions temporelles

- entraînement : {', '.join(split['train'])}
- validation : {', '.join(split['validation'])}
- test final réservé : {', '.join(split['test_reserved'])}

Le test final n'a pas été scoré.

## Usage prévu

Estimation de la température diurne de surface sur une grille 30 m, une fois le modèle
réentraîné et validé sur les observations réelles harmonisées.

## Usages interdits

Ne pas interpréter la sortie comme température de l'air, température ressentie,
diagnostic médical, mesure quotidienne satellitaire directe ou vérité terrain.

## Étape suivante

Semaine 6 : ouvrir le test final une seule fois, produire les résidus spatiaux,
analyser les erreurs par date et documenter l'incertitude.
"""
    path.write_text(content, encoding="utf-8")


def train_week5_models(config: dict[str, Any]) -> dict[str, Any]:
    processed = Path(config["paths"]["processed"])
    artifacts = Path(config["paths"]["artifacts"])
    figures = Path(config["paths"]["figures"])
    ensure_directories(artifacts, figures)

    dataset_path = processed / "pixel_date_demo.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError("Dataset absent. Exécutez d'abord build-demo.")
    data = read_table(dataset_path)
    data["date"] = pd.to_datetime(data["date"])

    split = split_dates(
        data["date"],
        float(config["dataset"]["validation_fraction_dates"]),
        int(config["dataset"]["test_dates_reserved"]),
    )
    write_json(artifacts / "splits.json", split)

    date_text = data["date"].dt.date.astype(str)
    masks = {name: date_text.isin(values) for name, values in split.items()}
    if (masks["train"] & masks["validation"]).any() or (masks["train"] & masks["test_reserved"]).any():
        raise AssertionError("Fuite temporelle détectée.")

    train = data.loc[masks["train"]].copy()
    validation = data.loc[masks["validation"]].copy()
    test_reserved = data.loc[masks["test_reserved"]].copy()
    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN]
    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation[TARGET_COLUMN]

    global_prediction = np.full(len(validation), y_train.mean(), dtype=np.float64)
    seasonal_prediction = _seasonal_predictions(train["date"], y_train, validation["date"])
    results = [
        {
            "model": "global_mean_baseline",
            **_metrics(y_validation, global_prediction),
            "is_baseline": True,
        },
        {
            "model": "monthly_mean_baseline",
            **_metrics(y_validation, seasonal_prediction),
            "is_baseline": True,
        },
    ]

    fitted_candidates: dict[str, Any] = {}
    validation_predictions: dict[str, np.ndarray] = {
        "global_mean_baseline": global_prediction,
        "monthly_mean_baseline": seasonal_prediction,
    }
    for name, model in _candidate_models(config).items():
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        results.append({"model": name, **_metrics(y_validation, prediction), "is_baseline": False})
        fitted_candidates[name] = model
        validation_predictions[name] = prediction

    metrics = pd.DataFrame(results)
    baseline_mae = float(
        metrics.loc[metrics["model"] == "global_mean_baseline", "mae_c"].iloc[0]
    )
    metrics["gain_vs_global_baseline_pct"] = 100.0 * (baseline_mae - metrics["mae_c"]) / baseline_mae
    metrics = metrics.sort_values(["is_baseline", "mae_c"], ascending=[True, True]).reset_index(drop=True)
    metrics.to_csv(artifacts / "metrics_validation.csv", index=False)

    candidate_metrics = metrics.loc[~metrics["is_baseline"]]
    selected_name = str(candidate_metrics.sort_values("mae_c").iloc[0]["model"])
    selected_validation_model = fitted_candidates[selected_name]

    # Importance sur un sous-échantillon de validation pour garder le run léger.
    sample_n = min(2500, len(validation))
    sample = validation.sample(n=sample_n, random_state=int(config["project"]["random_seed"]))
    importance_mean, importance_std = _permutation_importance(
        selected_validation_model,
        sample[FEATURE_COLUMNS],
        sample[TARGET_COLUMN],
        repeats=3,
        seed=int(config["project"]["random_seed"]),
    )
    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_mean": importance_mean,
            "importance_std": importance_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(artifacts / "feature_importance.csv", index=False)

    predictions_frame = validation[
        ["pixel_id", "date", "x_m", "y_m", "lon", "lat", TARGET_COLUMN]
    ].copy()
    for name, prediction in validation_predictions.items():
        predictions_frame[f"pred_{name}"] = prediction.astype(np.float32)
    predictions_frame[f"residual_{selected_name}"] = (
        predictions_frame[TARGET_COLUMN] - predictions_frame[f"pred_{selected_name}"]
    )
    write_table(predictions_frame, artifacts / "validation_predictions.parquet")

    # Réentraînement sur développement complet. Le test réservé n'est jamais prédit ici.
    development = pd.concat([train, validation], ignore_index=True)
    final_model = _candidate_models(config)[selected_name]
    final_model.fit(development[FEATURE_COLUMNS], development[TARGET_COLUMN])
    model_path = artifacts / "model_week5.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(final_model, stream, protocol=pickle.HIGHEST_PROTOCOL)

    selected_row = metrics.loc[metrics["model"] == selected_name].iloc[0].to_dict()
    metadata = {
        "stage": "week5_preliminary",
        "selected_model": selected_name,
        "selection_metric": "validation_mae_c",
        "selected_metrics": selected_row,
        "dataset_kind": "synthetic_geospatial_demo",
        "test_scored": False,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "rows": {
            "train": len(train),
            "validation": len(validation),
            "test_reserved": len(test_reserved),
            "development_refit": len(development),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "model_runtime": "geoai_rabat.lite_models (NumPy)",
        },
    }
    write_json(artifacts / "model_metadata.json", metadata)
    write_json(
        artifacts / "feature_schema.json",
        {
            "features_in_order": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "date_column": "date",
            "crs": config["grid"]["crs"],
            "resolution_m": config["grid"]["resolution_m"],
        },
    )

    metrics_figure = artifacts / "metrics_validation.png"
    importance_figure = artifacts / "feature_importance.png"
    _plot_metrics(metrics, metrics_figure)
    _plot_importance(importance, importance_figure)
    _write_model_card(
        artifacts / "MODEL_CARD.md",
        selected_name,
        selected_row,
        split,
        metadata["rows"],
    )

    output_files = [
        path
        for path in artifacts.iterdir()
        if path.is_file() and path.name != "training_manifest.json"
    ]
    manifest = {
        "stage": "week5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_name,
        "test_scored": False,
        "files": [file_record(path, "week5_training") for path in sorted(output_files)],
    }
    write_json(artifacts / "training_manifest.json", manifest)
    return {"metadata": metadata, "metrics": metrics.to_dict(orient="records")}
