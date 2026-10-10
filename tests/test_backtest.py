import numpy as np
import pandas as pd
import pytest

from analizador import backtest as bt
from conftest import generar_ohlcv


@pytest.fixture
def datos():
    return generar_ohlcv(320, semilla=3)


def test_estrategia_solo_recibe_datos_hasta_t(datos):
    """Test explícito de look-ahead: en cada llamada el último dato es el del día t."""
    vistos = []

    def espia(ventana):
        vistos.append((len(ventana), ventana.index[-1]))
        return 1.0

    bt.ejecutar_backtest(datos, espia, historia_minima=50)
    for t, (longitud, ultima_fecha) in zip(range(50, len(datos)), vistos):
        assert longitud == t + 1
        assert ultima_fecha == datos.index[t]
    assert len(vistos) == len(datos) - 50


def test_alterar_el_futuro_no_cambia_decisiones_pasadas(datos):
    estrategia = bt.cruce_medias(10, 40)
    k = 200
    alterado = datos.copy()
    alterado.iloc[k:, alterado.columns.get_loc("Close")] *= np.linspace(0.3, 3, len(datos) - k)

    a = bt.ejecutar_backtest(datos, estrategia, historia_minima=50)
    b = bt.ejecutar_backtest(alterado, estrategia, historia_minima=50)
    pd.testing.assert_series_equal(a.posiciones.loc[: datos.index[k - 1]],
                                   b.posiciones.loc[: datos.index[k - 1]])
    pd.testing.assert_series_equal(a.retornos.loc[: datos.index[k - 1]],
                                   b.retornos.loc[: datos.index[k - 1]])


def test_posicion_de_hoy_se_aplica_al_retorno_de_manana(datos):
    """Una señal basada en el retorno de hoy no puede capturar ese mismo retorno."""
    def momentum_un_dia(ventana):
        c = ventana["Close"]
        return 1.0 if c.iloc[-1] > c.iloc[-2] else -1.0

    res = bt.ejecutar_backtest(datos, momentum_un_dia, coste=0, deslizamiento=0, historia_minima=5)
    esperado = res.posiciones.shift(1).fillna(0) * res.retornos_activo
    pd.testing.assert_series_equal(res.retornos, esperado.rename("retorno"), check_names=False)
    # Si hubiera look-ahead, el retorno sería |r| todos los días (siempre positivo).
    assert not np.allclose(res.retornos.iloc[1:], res.retornos_activo.iloc[1:].abs())


def test_oraculo_no_es_posible():
    """Una estrategia que intenta usar el retorno del día siguiente no puede verlo."""
    datos = generar_ohlcv(100, semilla=1)

    def tramposa(ventana):
        siguiente = datos.index.get_loc(ventana.index[-1]) + 1
        # Solo puede saber cuántas filas recibe; las filas futuras no están en 'ventana'.
        assert len(ventana) == siguiente
        return 1.0

    bt.ejecutar_backtest(datos, tramposa, historia_minima=10)


def test_comprar_y_mantener_sin_costes_coincide_con_el_activo(datos):
    res = bt.ejecutar_backtest(datos, bt.comprar_y_mantener(), coste=0, deslizamiento=0, historia_minima=20)
    total = res.capital.iloc[-1]
    esperado = datos["Close"].iloc[-1] / datos["Close"].iloc[20]
    assert total == pytest.approx(esperado)


def test_costes_y_deslizamiento(datos):
    # Alterna 0/1 cada día: |Δpos| = 1 diario.
    def alterna(ventana):
        return float(len(ventana) % 2)

    sin = bt.ejecutar_backtest(datos, alterna, coste=0, deslizamiento=0, historia_minima=20)
    con = bt.ejecutar_backtest(datos, alterna, coste=0.001, deslizamiento=0.0005, historia_minima=20)
    diferencia = (sin.retornos - con.retornos)
    np.testing.assert_allclose(diferencia.iloc[1:], 0.0015)
    assert con.costes.sum() == pytest.approx(0.0015 * (len(datos) - 20) - 0.0015 * (1 - alterna(datos.iloc[:21])))
    assert con.capital.iloc[-1] < sin.capital.iloc[-1]


def test_posicion_constante_un_solo_coste_de_entrada(datos):
    res = bt.ejecutar_backtest(datos, bt.posicion_constante(0.5), coste=0.01, deslizamiento=0, historia_minima=20)
    assert res.costes.sum() == pytest.approx(0.005)
    assert res.metricas()["Nº operaciones"] == 1


def test_posiciones_no_finitas_y_limites(datos):
    res = bt.ejecutar_backtest(datos, lambda v: np.nan, historia_minima=20)
    assert (res.posiciones == 0).all()
    res = bt.ejecutar_backtest(datos, lambda v: 5.0, historia_minima=20)
    assert (res.posiciones == 1).all()


def test_comparar_con_referencias(datos):
    res = bt.comparar_con_referencias(datos, bt.cruce_medias(10, 50), nombre="Cruce",
                                      nivel_ingenuo=0.5, historia_minima=60)
    assert list(res) == ["Cruce", "Comprar y mantener", "Posición constante 0.5"]
    # Todas se evalúan exactamente en el mismo periodo.
    indices = [r.retornos.index for r in res.values()]
    assert all(i.equals(indices[0]) for i in indices)
    tabla = bt.tabla_metricas(res)
    assert {"CAGR", "Sharpe", "Rotación anual"} <= set(tabla.columns)


def test_reversion_rsi_devuelve_0_o_1(datos):
    res = bt.ejecutar_backtest(datos, bt.reversion_rsi(), historia_minima=60)
    assert set(res.posiciones.unique()) <= {0.0, 1.0}


def test_pocos_datos():
    with pytest.raises(ValueError):
        bt.ejecutar_backtest(generar_ohlcv(30), bt.comprar_y_mantener(), historia_minima=50)
