import numpy as np
import pandas as pd
import pytest

from analizador import situaciones as sit


def serie(valores, inicio="2000-01-03"):
    return pd.Series(valores, index=pd.bdate_range(inicio, periods=len(valores)), dtype=float)


def test_vix_usa_el_cierre_anterior():
    idx = pd.bdate_range("2020-01-01", periods=5)
    macro = pd.DataFrame({"vix": [10, 20, 30, 40, 50]}, index=idx)
    v = sit.vix_previo(idx, macro)
    assert np.isnan(v.iloc[0]) and v.iloc[3] == 30


def test_episodios_agrupa_dias_cercanos():
    activa = serie([False] * 10 + [True] * 3 + [False] * 5 + [True] * 2 + [False] * 100 + [True] + [False] * 5)
    ep = sit.episodios(activa, separacion=60)
    assert len(ep) == 2
    assert ep[0] == (activa.index[10], activa.index[19])
    assert ep[1][0] == activa.index[120]


def test_historial_cuenta_por_episodio():
    # 300 días subiendo, caída del 30 % y recuperación: un único episodio de caída fuerte.
    precios = serie(np.concatenate([np.linspace(100, 200, 300), np.linspace(200, 130, 40),
                                    np.linspace(130, 260, 400)]))
    h = sit.historial(precios, "caida", 21)
    assert len(h) == 1
    fila = h.iloc[0]
    assert fila["completo"] and fila["prop_mas_arriba"] > 0.5
    assert fila["caida_max"] <= -0.3
    r = sit.resumen(h)
    assert r["n"] == 1 and r["episodios_arriba"] == 1


def test_panico_con_macro():
    idx = pd.bdate_range("2005-01-03", periods=600)
    precios = pd.Series(np.linspace(100, 160, 600), index=idx)
    vix = pd.Series(15.0, index=idx)
    vix.iloc[100:110] = 40
    h = sit.historial(precios, "panico", 21, pd.DataFrame({"vix": vix}))
    assert len(h) == 1 and h.iloc[0]["vix_max"] == 40
    assert h.iloc[0]["prop_mas_arriba"] == 1.0


def test_condicion_sin_mirar_al_futuro():
    precios = serie(np.linspace(100, 50, 400))
    a = sit.condicion("caida", precios, None)
    alterado = precios.copy()
    alterado.iloc[300:] = 500
    pd.testing.assert_series_equal(a.iloc[:300], sit.condicion("caida", alterado, None).iloc[:300])


def test_situacion_hoy():
    precios = serie(np.concatenate([np.full(200, 100.0), np.full(60, 70.0)]))
    macro = pd.DataFrame({"vix": [35.0]}, index=[precios.index[-1]])
    hoy = sit.situacion_hoy(precios, macro)
    assert hoy["panico"] and hoy["caida_fuerte"] and hoy["caida"] == pytest.approx(-0.3)


def test_regla_desconocida():
    with pytest.raises(ValueError):
        sit.condicion("otra", serie([1.0, 2.0]), None)
