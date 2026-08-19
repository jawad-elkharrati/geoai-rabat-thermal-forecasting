import pandas as pd

from geoai_rabat.modeling import split_dates


def test_temporal_split_has_no_date_leakage():
    dates = pd.Series(pd.date_range("2019-01-01", periods=18, freq="90D").repeat(4))
    split = split_dates(dates, validation_fraction=0.2, test_dates_reserved=2)

    train = set(split["train"])
    validation = set(split["validation"])
    test = set(split["test_reserved"])
    assert train
    assert validation
    assert len(test) == 2
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert max(train) < min(validation) < min(test)
