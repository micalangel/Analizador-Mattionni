import numpy as np
import pandas as pd

from analizador.data import (
    COLUMNAS_OHLCV,
    descargar_precios,
    descargar_varios,
    limpiar_ohlcv,
    normalizar_tickers,
    precios_cierre,
)
from conftest import generar_ohlcv


def test_normalizar_tickers():
    assert normalizar_tickers("aapl, msft  spy;AAPL") == ["AAPL", "MSFT", "SPY"]
    assert normalizar_tickers(["qqq", " vti "]) == ["QQQ", "VTI"]
    assert normalizar_tickers("") == []


def test_limpiar_trata_faltantes_y_duplicados():
    df = generar_ohlcv(10)
    df.iloc[2, df.columns.get_loc("Close")] = np.nan    # fila sin cierre: se elimina
    df.iloc[4, df.columns.get_loc("Open")] = np.nan     # se rellena con el cierre
    df.iloc[5, df.columns.get_loc("Volume")] = np.nan   # se rellena con 0
    df = pd.concat([df, df.iloc[[7]]])                  # fecha duplicada
    df = df.iloc[::-1]                                  # desordenado

    limpio = limpiar_ohlcv(df)
    assert list(limpio.columns) == COLUMNAS_OHLCV
    assert limpio.index.is_monotonic_increasing
    assert not limpio.index.duplicated().any()
    assert len(limpio) == 9
    assert not limpio.isna().any().any()
    fecha4 = df.sort_index().index[4]
    assert limpio.loc[fecha4, "Open"] == limpio.loc[fecha4, "Close"]


def test_limpiar_aplana_multiindex_y_quita_zona_horaria():
    df = generar_ohlcv(5)
    df.columns = pd.MultiIndex.from_product([df.columns, ["AAPL"]], names=["Price", "Ticker"])
    df.index = df.index.tz_localize("America/New_York")
    limpio = limpiar_ohlcv(df)
    assert list(limpio.columns) == COLUMNAS_OHLCV
    assert limpio.index.tz is None


def test_limpiar_vacio():
    assert limpiar_ohlcv(None).empty
    assert list(limpiar_ohlcv(pd.DataFrame()).columns) == COLUMNAS_OHLCV


def test_descarga_usa_cache(tmp_path):
    llamadas = []

    def falso(ticker, inicio, fin):
        llamadas.append(ticker)
        return generar_ohlcv(20)

    a = descargar_precios("spy", "2020-01-01", "2020-02-01", tmp_path, descargador=falso)
    b = descargar_precios("SPY", "2020-01-01", "2020-02-01", tmp_path, descargador=falso)
    assert llamadas == ["SPY"]  # la segunda vez se lee del CSV
    assert len(list(tmp_path.glob("*.csv"))) == 1
    pd.testing.assert_frame_equal(a, b, check_freq=False)


def test_descarga_fallida_devuelve_vacio(tmp_path):
    def roto(*_):
        raise ConnectionError("sin red")

    df = descargar_precios("XXX", "2020-01-01", "2020-02-01", tmp_path, descargador=roto)
    assert df.empty
    assert not list(tmp_path.glob("*.csv"))  # no se cachean descargas vacías


def test_descargar_varios_omite_sin_datos(tmp_path):
    def falso(ticker, *_):
        return generar_ohlcv(15) if ticker != "MALO" else pd.DataFrame()

    datos = descargar_varios("AAA, MALO, BBB", "2020-01-01", "2020-02-01",
                             directorio_cache=tmp_path, descargador=falso)
    assert list(datos) == ["AAA", "BBB"]
    cierres = precios_cierre(datos)
    assert list(cierres.columns) == ["AAA", "BBB"]
