import numpy as np
import pandas as pd
import pytest

from analizador import simulador as sm
from analizador.data import descargar_intradia
from conftest import generar_ohlcv


@pytest.mark.parametrize("regla", list(sm.REGLAS))
def test_reglas_sin_mirar_al_futuro(regla):
    datos = generar_ohlcv(600, semilla=2)
    k = 400
    alterado = datos.copy()
    alterado.iloc[k:, alterado.columns.get_loc("Close")] *= np.linspace(1, 3, 600 - k)
    r = sm.REGLAS[regla]
    defecto = {p: v[2] for p, v in r.parametros.items()}
    a = r.funcion(datos, **defecto).iloc[:k]
    b = r.funcion(alterado, **defecto).iloc[:k]
    pd.testing.assert_series_equal(a, b, check_names=False)
    assert set(np.unique(a)) <= {0.0, 1.0}


def test_operaciones_y_costes():
    idx = pd.bdate_range("2024-01-01", periods=6)
    precios = pd.Series([100, 110, 121, 121, 110, 110.0], index=idx)
    pos = pd.Series([1, 1, 0, 0, 1, 1.0], index=idx)
    ops = sm.operaciones(precios, pos, coste=0.01)
    assert len(ops) == 2
    assert ops.loc[0, "resultado"] == pytest.approx(1.21 * 0.99 * 0.99 - 1)
    assert bool(ops.loc[1, "abierta"]) and ops.loc[1, "resultado"] == pytest.approx(0.99 - 1)


def test_posicion_se_aplica_a_la_vela_siguiente():
    idx = pd.bdate_range("2024-01-01", periods=4)
    precios = pd.Series([100, 200, 200, 200.0], index=idx)
    pos = pd.Series([0, 1, 1, 1.0], index=idx)   # decide comprar tras el salto: no lo captura
    capital = sm._capital(precios, pos, 0.0)
    assert capital.iloc[-1] == pytest.approx(1.0)


def test_simular_compara_con_mantener_y_azar():
    datos = generar_ohlcv(800, semilla=4)
    s = sm.simular(datos, "cruce", {"rapida": 10, "lenta": 50}, repeticiones_azar=50)
    assert s.total_mantener == pytest.approx(datos["Close"].iloc[-1] / datos["Close"].iloc[0] - 1)
    assert 0 <= s.percentil_azar <= 1 and len(s.resultados_azar) == 50
    assert 0 <= s.tiempo_dentro <= 1
    assert sm.max_caida(s.capital_regla) <= 0


def test_intradia_con_descargador_falso():
    idx = pd.date_range("2026-10-01 13:30", periods=10, freq="5min", tz="UTC")
    falso = lambda t, i, p: pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": np.arange(1, 11.0),
                                          "Volume": 5.0}, index=idx)
    df = descargar_intradia("X", "5 minutos", descargador=falso)
    assert len(df) == 10 and df.index.tz is None
    assert df.index[0].hour == 15  # 13:30 UTC = 15:30 en Madrid (horario de verano)
    assert descargar_intradia("X", "5 minutos", descargador=lambda *a: None).empty


def test_desde_usa_la_historia_previa_para_las_medias():
    datos = generar_ohlcv(900, semilla=6)
    desde = datos.index[600]
    s = sm.simular(datos, "cruce", {"rapida": 20, "lenta": 200}, repeticiones_azar=0, desde=desde)
    assert s.posicion.index[0] == desde and len(s.capital_regla) == 300
    # Con la historia previa, la media de 200 ya existe el primer día evaluado.
    completa = sm.REGLAS["cruce"].funcion(datos, rapida=20, lenta=200)
    assert s.posicion.iloc[0] == completa.loc[desde]
