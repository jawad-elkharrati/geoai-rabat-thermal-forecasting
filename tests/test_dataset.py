from pathlib import Path

from geoai_rabat.config import load_config
from geoai_rabat.dataset import build_demo_pixel_date
from geoai_rabat.schema import FEATURE_COLUMNS, TARGET_COLUMN


CONFIG = load_config(Path("configs/rabat.json"))


def test_demo_dataset_is_reproducible_and_valid():
    first, _, _ = build_demo_pixel_date(CONFIG)
    second, _, _ = build_demo_pixel_date(CONFIG)

    assert len(first) == len(second)
    assert first.head(50).equals(second.head(50))
    assert first["date"].nunique() == len(CONFIG["history"]["demo_dates"])
    assert not first.duplicated(["pixel_id", "date"]).any()
    assert not first[TARGET_COLUMN].isna().any()
    assert first[TARGET_COLUMN].between(10, 70).all()
    assert set(FEATURE_COLUMNS).issubset(first.columns)


def test_missingness_is_small_and_intentional():
    dataset, _, _ = build_demo_pixel_date(CONFIG)
    assert 0 < dataset["ndvi"].isna().mean() < 0.02
    assert 0 < dataset["ndbi"].isna().mean() < 0.02
    other_features = [name for name in FEATURE_COLUMNS if name not in {"ndvi", "ndbi"}]
    assert not dataset[other_features].isna().any().any()
