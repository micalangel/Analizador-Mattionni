"""Datos sintéticos compartidos: los tests funcionan sin conexión."""

import numpy as np
import pandas as pd
import pytest


def generar_ohlcv(n: int = 300, semilla: int = 0, precio_inicial: float = 100.0,
                  vol_diaria: float = 0.01, deriva: float = 0.0003) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2020-01-01", periods=n)
    log_ret = rng.normal(deriva, vol_diaria, n)
    cierre = precio_inicial * np.exp(np.cumsum(log_ret))
    apertura = cierre * (1 + rng.normal(0, 0.002, n))
    maximo = np.maximum(cierre, apertura) * (1 + np.abs(rng.normal(0, 0.003, n)))
    minimo = np.minimum(cierre, apertura) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volumen = rng.integers(1_000, 10_000, n).astype(float)
    return pd.DataFrame(
        {"Open": apertura, "High": maximo, "Low": minimo, "Close": cierre, "Volume": volumen},
        index=pd.DatetimeIndex(fechas, name="Date"),
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return generar_ohlcv()
