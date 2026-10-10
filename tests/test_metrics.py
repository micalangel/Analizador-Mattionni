import numpy as np
import pandas as pd
import pytest

from analizador import metrics as m


def serie(valores):
    return pd.Series(valores, index=pd.bdate_range("2021-01-01", periods=len(valores)), dtype=float)


def test_cagr_un_anio():
    r = serie([0.0] * 251 + [0.10])  # 252 días, +10 % total
    assert m.cagr(r) == pytest.approx(0.10)


def test_cagr_dos_anios():
    diario = 1.21 ** (1 / 504) - 1   # +21 % en dos años -> 10 % anual
    assert m.cagr(serie([diario] * 504)) == pytest.approx(0.10)


def test_volatilidad_y_sharpe():
    r = serie([0.01, -0.01] * 126)
    assert m.volatilidad_anualizada(r) == pytest.approx(np.std([0.01, -0.01] * 126, ddof=1) * np.sqrt(252))
    assert m.ratio_sharpe(r) == pytest.approx(0.0, abs=1e-12)
    r2 = serie([0.002, 0.0] * 126)
    esperado = r2.mean() / r2.std(ddof=1) * np.sqrt(252)
    assert m.ratio_sharpe(r2) == pytest.approx(esperado)
    assert np.isnan(m.ratio_sharpe(serie([0.001] * 10)))  # desviación nula


def test_max_drawdown():
    # Capital: 1 -> 1.2 -> 0.9 -> 1.0 : peor caída 0.9/1.2 - 1 = -25 %
    r = serie([0.2, 0.9 / 1.2 - 1, 1.0 / 0.9 - 1])
    assert m.max_drawdown(r) == pytest.approx(-0.25)
    assert m.max_drawdown(serie([-0.5])) == pytest.approx(-0.5)  # caída desde el capital inicial
    assert m.max_drawdown(serie([0.01, 0.02])) == 0.0


def test_tasa_acierto():
    pos = serie([1, 1, 0, 1, 1])
    ret = serie([0.0, 0.01, -0.02, 0.05, -0.01])
    # Días expuestos (posición del día anterior ≠ 0): índices 1, 2, 4 -> +, -, -
    assert m.tasa_acierto(ret, pos) == pytest.approx(1 / 3)


def test_operaciones_y_rotacion():
    pos = serie([0, 1, 1, 0, 0.5] + [0.5] * 247)  # 252 días
    assert m.numero_operaciones(pos) == 3
    assert m.rotacion_anual(pos) == pytest.approx(1 + 1 + 0.5)
    assert m.numero_operaciones(serie([1.0] * 5)) == 1  # la entrada inicial cuenta


def test_resumen_claves():
    claves = set(m.resumen(serie([0.01, -0.005] * 10), serie([1.0] * 20)))
    assert claves == {"CAGR", "Volatilidad anual", "Sharpe", "Máx. drawdown",
                      "Tasa de acierto", "Nº operaciones", "Rotación anual"}
