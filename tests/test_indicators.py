import numpy as np
import pandas as pd
import pytest

from analizador import indicators as ind
from conftest import generar_ohlcv


def serie(valores):
    return pd.Series(valores, index=pd.bdate_range("2021-01-01", periods=len(valores)), dtype=float)


def test_media_movil():
    s = serie([1, 2, 3, 4, 5])
    mm = ind.media_movil(s, 3)
    assert mm.iloc[:2].isna().all()
    assert mm.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_media_movil_exponencial_constante():
    s = serie([10.0] * 30)
    assert ind.media_movil_exponencial(s, 10).dropna().eq(10.0).all()


def test_rsi_extremos_y_rango():
    assert ind.rsi(serie(range(1, 40)), 14).dropna().eq(100).all()       # solo subidas
    assert ind.rsi(serie(range(40, 1, -1)), 14).dropna().eq(0).all()     # solo bajadas
    assert ind.rsi(serie([5.0] * 30), 14).dropna().eq(50).all()          # sin movimiento
    r = ind.rsi(generar_ohlcv(300)["Close"], 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_valor_conocido():
    # Alternancia +1/-1: ganancias y pérdidas medias tienden a igualarse -> RSI ~ 50.
    s = serie(np.cumsum([1, -1] * 100) + 100)
    assert ind.rsi(s, 14).iloc[-1] == pytest.approx(50, abs=3)


def test_macd():
    c = generar_ohlcv(200)["Close"]
    m = ind.macd(c)
    assert list(m.columns) == ["macd", "senal", "histograma"]
    validos = m.dropna()
    np.testing.assert_allclose(validos["histograma"], validos["macd"] - validos["senal"])
    assert ind.macd(serie([7.0] * 60)).dropna().abs().max().max() < 1e-12  # precio constante


def test_bandas_bollinger():
    s = serie([1, 2, 3, 4, 5])
    bb = ind.bandas_bollinger(s, ventana=5, num_desv=2)
    desv = np.std([1, 2, 3, 4, 5])
    assert bb["media"].iloc[-1] == pytest.approx(3)
    assert bb["superior"].iloc[-1] == pytest.approx(3 + 2 * desv)
    assert bb["inferior"].iloc[-1] == pytest.approx(3 - 2 * desv)


def test_volatilidad_movil():
    # Retornos log alternando +a y -a: desviación muestral conocida.
    a = 0.01
    logp = np.cumsum([0] + [a, -a] * 30)
    s = serie(100 * np.exp(logp))
    vol = ind.volatilidad_movil(s, ventana=20, anualizar=False).iloc[-1]
    esperado = np.std([a, -a] * 10, ddof=1)
    assert vol == pytest.approx(esperado)
    anual = ind.volatilidad_movil(s, ventana=20).iloc[-1]
    assert anual == pytest.approx(esperado * np.sqrt(252))


def test_drawdown():
    s = serie([100, 120, 90, 130, 65])
    dd = ind.drawdown(s)
    assert dd.tolist() == pytest.approx([0, 0, -0.25, 0, -0.5])
    assert (dd <= 0).all()


@pytest.mark.parametrize("funcion", [
    lambda c: ind.media_movil(c, 20),
    lambda c: ind.media_movil_exponencial(c, 20),
    lambda c: ind.rsi(c, 14),
    lambda c: ind.macd(c)["histograma"],
    lambda c: ind.bandas_bollinger(c)["superior"],
    lambda c: ind.volatilidad_movil(c, 20),
    lambda c: ind.drawdown(c),
])
def test_indicadores_sin_mirar_al_futuro(funcion):
    """Cambiar los precios a partir del día k no altera ningún valor anterior a k."""
    c = generar_ohlcv(250)["Close"]
    k = 150
    alterado = c.copy()
    alterado.iloc[k:] = alterado.iloc[k:] * 3 + 50
    pd.testing.assert_series_equal(funcion(c).iloc[:k], funcion(alterado).iloc[:k])


def test_calcular_todos(ohlcv):
    tabla = ind.calcular_todos(ohlcv)
    assert len(tabla) == len(ohlcv)
    assert {"MM20", "RSI14", "MACD", "BB_superior", "Vol20_anual", "Drawdown"} <= set(tabla.columns)
