import numpy as np

from geoai_rabat.quality import (
    landsat_clear_mask,
    normalized_difference,
    scale_landsat_surface_reflectance,
    scale_landsat_surface_temperature,
    validate_aligned_layers,
)


def test_landsat_qa_mask_excludes_documented_bits():
    qa = np.array([0, 1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5, 1 << 7])
    clear = landsat_clear_mask(qa)
    assert clear.tolist() == [True, False, False, False, False, False, False, True]


def test_landsat_temperature_scaling_and_fill():
    values = scale_landsat_surface_temperature(np.array([0, 44_947], dtype=np.uint16))
    assert np.isnan(values[0])
    assert np.isclose(values[1], 29.45, atol=0.1)


def test_surface_reflectance_scaling():
    value = scale_landsat_surface_reflectance(np.array([18_639], dtype=np.uint16))[0]
    assert np.isclose(value, 0.31257, atol=1e-4)


def test_normalized_difference_handles_zero_denominator():
    result = normalized_difference(np.array([0.8, 0.0]), np.array([0.2, 0.0]))
    assert np.isclose(result[0], 0.6)
    assert np.isnan(result[1])


def test_alignment_guard():
    assert validate_aligned_layers({"a": np.zeros((2, 3)), "b": np.ones((2, 3))}) == (2, 3)
