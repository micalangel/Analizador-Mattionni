import numpy as np
import pandas as pd
import pytest

from analizador import ficha as fc
from analizador import per_historico as ph
from analizador import tecnico as tc
from conftest import generar_ohlcv


def test_lecturas_basicas_y_tendencia_alcista():
    idx = pd.bdate_range("2020-01-01", periods=400)
    c = pd.Series(np.linspace(100, 200, 400), index=idx)
    datos = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c, "Volume": 1000.0})
    lecturas = {l.indicador: l for l in tc.analizar(datos)}
    assert lecturas["Precio frente a su media de 200 días"].tono == tc.ALCISTA
    assert lecturas["Cruce de medias 50 / 200"].tono == tc.ALCISTA
    assert "Último año" in lecturas
    assert sum(tc.resumen(list(lecturas.values()))) == len(lecturas)


def test_lecturas_sin_mirar_al_futuro():
    datos = generar_ohlcv(500, semilla=8)
    a = [(l.indicador, l.valor) for l in tc.analizar(datos.iloc[:400])]
    alterado = datos.copy()
    alterado.iloc[400:] *= 3
    b = [(l.indicador, l.valor) for l in tc.analizar(alterado.iloc[:400])]
    assert a == b


def test_niveles_encuentra_soporte_y_resistencia():
    idx = pd.bdate_range("2024-01-01", periods=260)
    onda = 100 + 10 * np.sin(np.arange(260) / 15)          # techos ~110 y suelos ~90 repetidos
    c = pd.Series(onda, index=idx)
    c.iloc[-1] = 100
    datos = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1.0})
    nv = tc.niveles(datos)
    tipos = {n.tipo for n in nv}
    assert tipos == {"soporte", "resistencia"}
    res = [n for n in nv if n.tipo == "resistencia"][0]
    assert res.precio == pytest.approx(110, abs=1) and res.toques >= 2


def test_dividendos():
    hoy = pd.Timestamp.today().normalize()
    idx = pd.DatetimeIndex([hoy - pd.DateOffset(months=m) for m in (40, 34, 28, 22, 16, 10, 4)])
    d = pd.Series([1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3], index=idx)
    assert fc.rentabilidad_dividendo(d, 50) == pytest.approx((1.2 + 1.3) / 50)
    por_ano = pd.Series([2.0, 2.2, 2.4, 2.3, 2.5], index=[2020, 2021, 2022, 2023, 2024])
    assert fc.anos_subiendo_dividendo(por_ano) == 1
    viejo = pd.Series([1.0], index=[hoy - pd.DateOffset(years=3)])
    assert fc.rentabilidad_dividendo(viejo, 50) == 0.0


def test_per_historico():
    idx = pd.bdate_range("2014-01-01", "2024-12-31")
    precios = pd.Series(100.0, index=idx)
    fechas = pd.date_range("2013-02-01", "2024-11-01", freq="QS-FEB") + pd.Timedelta(days=5)
    pub = pd.DataFrame({"BPA estimado": 1.0, "BPA publicado": 1.25, "Sorpresa %": 0.0},
                       index=pd.DatetimeIndex(fechas).tz_localize("America/New_York"))
    r = ph.calcular(precios, pub, per_oficial=20.0)
    assert r.actual == pytest.approx(20.0) and r.mediana == pytest.approx(20.0)  # 100 / (4 × 1,25)
    assert r.etiqueta == "En su media"
    assert ph.calcular(precios, pub, per_oficial=200.0) is None  # no cuadra con el oficial: se descarta


def test_bpa_doce_meses_exige_trimestres_seguidos():
    fechas = pd.DatetimeIndex(["2020-01-30", "2020-04-30", "2020-07-30", "2020-10-30", "2022-01-30"]).tz_localize("UTC")
    pub = pd.DataFrame({"BPA publicado": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=fechas)
    ttm = ph.bpa_doce_meses(pub)
    assert ttm.iloc[3] == 4.0 and pd.isna(ttm.iloc[4])  # hueco de un año: no se suma
