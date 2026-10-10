import numpy as np
import pytest

from analizador.fundamentals import COLUMNAS, obtener_fundamentales, ratios_desde_info


def test_ratios_desde_info():
    info = {"shortName": "Ejemplo", "trailingPE": 25.0, "priceToBook": 4.0,
            "debtToEquity": 150.0, "profitMargins": 0.2,
            "freeCashflow": 5e9, "marketCap": 100e9, "currency": "USD"}
    r = ratios_desde_info(info)
    assert r["PER"] == 25.0
    assert r["Deuda/Patrimonio"] == pytest.approx(1.5)
    assert r["Rentabilidad FCF"] == pytest.approx(0.05)
    assert r["Moneda"] == "USD"


def test_datos_incompletos_dan_nan():
    r = ratios_desde_info({"trailingPE": "Infinity", "marketCap": 0, "freeCashflow": 1e9})
    assert np.isnan(r["PER"])
    assert np.isnan(r["Rentabilidad FCF"])
    assert np.isnan(r["Deuda/Patrimonio"])


def test_obtener_fundamentales_con_fallos():
    def falso(t):
        if t == "ROTO":
            raise RuntimeError("sin datos")
        return {"trailingPE": 10.0}

    tabla = obtener_fundamentales("aaa, roto", obtener_info=falso)
    assert list(tabla.index) == ["AAA", "ROTO"]
    assert list(tabla.columns) == COLUMNAS
    assert tabla.loc["AAA", "PER"] == 10.0
    assert np.isnan(tabla.loc["ROTO", "PER"])
