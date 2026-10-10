import numpy as np
import pandas as pd
import pytest

from analizador import senales as sn
from conftest import generar_ohlcv


def paseo(n=3000, semilla=0, deriva=0.0004):
    rng = np.random.default_rng(semilla)
    c = 100 * np.exp(np.cumsum(rng.normal(deriva, 0.01, n)))
    return pd.DataFrame({"Close": c}, index=pd.bdate_range("2000-01-03", periods=n))


def test_senales_sin_mirar_al_futuro():
    c = generar_ohlcv(600, semilla=1)["Close"]
    k = 400
    alterado = c.copy()
    alterado.iloc[k:] *= 3
    for s in sn.SENALES:
        a, b = s.funcion(c).iloc[:k], s.funcion(alterado).iloc[:k]
        pd.testing.assert_series_equal(a, b, check_names=False, obj=s.nombre)


def test_senal_aleatoria_no_tiene_ventaja():
    """Con paseos aleatorios (sin patrón), ninguna señal debería superar al azar.

    Ojo: las señales que opinan pocos días pueden mostrar 'ventajas' de varios puntos por pura
    suerte; por eso el veredicto exige significación con corrección por pruebas múltiples.
    """
    datos = {f"I{i}": paseo(semilla=i) for i in range(6)}
    lab = sn.laboratorio(datos, horizontes={"1 semana": 5})
    assert (lab["veredicto"] != "Supera al azar").all()


def test_senal_con_informacion_si_se_detecta():
    """Una 'señal' que conoce el futuro (prohibida en la práctica) debe detectarse como ventaja:
    comprueba que la métrica funciona."""
    datos = {f"I{i}": paseo(semilla=10 + i) for i in range(4)}
    tramposa = sn.DefinicionSenal("trampa", "test", "", lambda c: np.sign(np.log(c.shift(-5) / c)))
    lab = sn.laboratorio(datos, horizontes={"1 semana": 5}, senales=[tramposa])
    assert lab.loc[0, "acierto_comp"] == pytest.approx(1.0)
    assert lab.loc[0, "veredicto"] == "Supera al azar"


def test_azar_considera_la_mezcla():
    """Decir siempre 'sube' en un mercado alcista no es ventaja: el azar con esa mezcla acierta igual."""
    datos = {f"I{i}": paseo(semilla=20 + i, deriva=0.002) for i in range(4)}
    siempre = sn.DefinicionSenal("siempre", "", "", lambda c: pd.Series(1.0, index=c.index))
    lab = sn.laboratorio(datos, horizontes={"1 mes": 21}, senales=[siempre])
    assert lab.loc[0, "acierto_comp"] > 0.6
    assert lab.loc[0, "ventaja_comp"] == pytest.approx(0.0, abs=1e-9)


def test_holm():
    p = np.array([0.01, 0.04, 0.03, np.nan])
    aj = sn._holm(p)
    assert aj[0] == pytest.approx(0.03) and aj[2] == pytest.approx(0.06) and aj[1] == pytest.approx(0.06)
    assert np.isnan(aj[3])


def test_cambio_de_mes():
    idx = pd.bdate_range("2021-01-01", "2021-02-28")
    s = sn._cambio_de_mes(idx)
    assert s.loc["2021-01-01"] == 1 and s.loc["2021-01-05"] == 1 and np.isnan(s.loc["2021-01-06"])
    assert s.loc["2021-01-29"] == 1 and s.loc["2021-01-28"] == 1 and np.isnan(s.loc["2021-01-27"])
