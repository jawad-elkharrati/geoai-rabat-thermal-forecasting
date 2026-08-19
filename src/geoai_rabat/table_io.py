from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(frame: pd.DataFrame, parquet_path: Path) -> Path:
    """Écrit en Parquet si un moteur est présent, sinon en CSV gzip explicite."""
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(parquet_path, index=False, compression="zstd")
        return parquet_path
    except (ImportError, ModuleNotFoundError):
        fallback = parquet_path.with_suffix(".csv.gz")
        frame.to_csv(fallback, index=False, compression="gzip")
        return fallback


def read_table(parquet_path: Path) -> pd.DataFrame:
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    fallback = parquet_path.with_suffix(".csv.gz")
    if fallback.exists():
        return pd.read_csv(fallback, parse_dates=["date"] if "pixel_date" in fallback.name else None)
    raise FileNotFoundError(f"Table absente : {parquet_path} ou {fallback}")


def resolve_table_path(parquet_path: Path) -> Path:
    if parquet_path.exists():
        return parquet_path
    fallback = parquet_path.with_suffix(".csv.gz")
    if fallback.exists():
        return fallback
    return parquet_path
