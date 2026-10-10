import numpy as np
import pandas as pd
import pytest

from analizador import alertas as al
from analizador import calendario as cal
from analizador import mi_lista as ml


def test_mi_lista(tmp_path):
    ruta = tmp_path / "lista.json"
    assert ml.cargar(ruta) == []
    ml.anadir("AAPL", ruta)
    ml.anadir("SAN.MC", ruta)
    ml.anadir("AAPL", ruta)  # sin duplicados
    assert ml.cargar(ruta) == ["AAPL", "SAN.MC"]
    assert ml.quitar("AAPL", ruta) == ["SAN.MC"]
    ruta.write_text("esto no es json")
    assert ml.cargar(ruta) == []


def serie(n=300, semilla=0):
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2025-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Close": c, "Volume": np.exp(rng.normal(12, 0.1, n))}, index=idx)


def test_alertas_caida_volumen_movimiento_y_resultados():
    df = serie()
    df.iloc[-60:, 0] = df["Close"].iloc[-61] * 0.7       # cae un 30 %
    df.iloc[-1, 0] = df["Close"].iloc[-2] * 0.85          # último día −15 %
    df.iloc[-1, 1] = df["Volume"].iloc[-2] * 10           # volumen disparado
    hoy = df.index[-1]
    tipos = {a.tipo for a in al.evaluar("X", "Empresa X", df, None, hoy + pd.Timedelta(days=3), hoy=hoy)}
    assert tipos == {"caida", "volumen", "movimiento", "resultados"}


def test_sin_alertas_en_un_dia_normal():
    df = serie(semilla=3)
    df.iloc[-1, 1] = df["Volume"].iloc[-30:-1].median()
    hoy = df.index[-1]
    alertas = al.evaluar("X", "X", df, None, hoy + pd.Timedelta(days=40), hoy=hoy)
    assert [a.tipo for a in alertas if a.tipo != "caida"] == []


def test_panico_con_macro():
    df = serie()
    macro = pd.DataFrame({"vix": [35.0]}, index=[df.index[-2]])
    assert any(a.tipo == "panico" for a in al.evaluar("X", "X", df, macro, hoy=df.index[-1]))


def test_tabla_calendario_ordena_y_cuenta_dias():
    info = {"A": {"ticker": "A", "fecha": pd.Timestamp("2026-11-02 16:00", tz="America/New_York")},
            "B": {"ticker": "B", "fecha": pd.Timestamp("2026-10-20 07:00", tz="America/New_York")},
            "C": {"ticker": "C", "fecha": None}}
    t = cal.tabla_calendario(info, hoy=pd.Timestamp("2026-10-10"))
    assert list(t["ticker"]) == ["B", "A", "C"]
    assert t.loc[0, "dias"] == 10 and pd.isna(t.loc[2, "dias"])
