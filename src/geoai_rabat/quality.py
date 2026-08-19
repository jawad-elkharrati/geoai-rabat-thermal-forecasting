from __future__ import annotations

import numpy as np


LANDSAT_ST_SCALE = 0.00341802
LANDSAT_ST_OFFSET_K = 149.0
KELVIN_TO_CELSIUS = 273.15
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2


def landsat_clear_mask(
    qa_pixel: np.ndarray, excluded_bits: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
) -> np.ndarray:
    """Vrai pour les pixels non affectés par les drapeaux QA exclus.

    Bits Landsat 8/9 C2 QA_PIXEL :
    0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow.
    """
    qa = np.asarray(qa_pixel, dtype=np.uint16)
    invalid_mask = np.uint16(0)
    for bit in excluded_bits:
        if bit < 0 or bit > 15:
            raise ValueError(f"Bit QA hors plage uint16 : {bit}")
        invalid_mask |= np.uint16(1 << bit)
    return (qa & invalid_mask) == 0


def scale_landsat_surface_temperature(dn: np.ndarray) -> np.ndarray:
    """Convertit les DN Landsat C2 L2 ST en degrés Celsius."""
    values = np.asarray(dn, dtype=np.float32)
    result = values * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET_K - KELVIN_TO_CELSIUS
    return np.where(values == 0, np.nan, result).astype(np.float32)


def scale_landsat_surface_reflectance(dn: np.ndarray) -> np.ndarray:
    """Convertit les DN Landsat C2 L2 SR en réflectance."""
    values = np.asarray(dn, dtype=np.float32)
    result = values * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
    return np.where(values == 0, np.nan, result).astype(np.float32)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calcule (a-b)/(a+b), avec NaN pour les dénominateurs quasi nuls."""
    a_values = np.asarray(a, dtype=np.float32)
    b_values = np.asarray(b, dtype=np.float32)
    denominator = a_values + b_values
    with np.errstate(divide="ignore", invalid="ignore"):
        index = np.where(np.abs(denominator) > 1e-8, (a_values - b_values) / denominator, np.nan)
    return np.clip(index, -1.0, 1.0).astype(np.float32)


def validate_aligned_layers(layers: dict[str, np.ndarray]) -> tuple[int, ...]:
    """Vérifie que toutes les couches partagent exactement la même forme."""
    if not layers:
        raise ValueError("Aucune couche à contrôler.")
    shapes = {name: np.asarray(array).shape for name, array in layers.items()}
    unique_shapes = set(shapes.values())
    if len(unique_shapes) != 1:
        raise ValueError(f"Couches non alignées : {shapes}")
    return next(iter(unique_shapes))
