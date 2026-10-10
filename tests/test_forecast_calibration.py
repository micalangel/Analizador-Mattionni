import numpy as np
import pandas as pd
import pytest

from analizador import calibration as cal
from analizador import forecast as fc
from conftest import generar_ohlcv

TODOS = list(fc.METODOS) + [fc.REFERENCIA]


@pytest.fixture(scope="module")
def precios():
    # Paseo aleatorio log-normal con volatilidad constante: los intervalos normales
    # deben estar bien calibrados.
    return generar_ohlcv(1200, semilla=7, vol_diaria=0.012, deriva=0.0)["Close"]


def test_cuantil_z():
    assert fc.cuantil_z(0.8) == pytest.approx(1.2815516, rel=1e-6)
    assert fc.cuantil_z(0.95) == pytest.approx(1.959964, rel=1e-6)
    with pytest.raises(ValueError):
        fc.cuantil_z(1.2)


@pytest.mark.parametrize("metodo", TODOS)
def test_intervalo_es_rango_valido(precios, metodo):
    tabla = fc.serie_intervalos(precios, horizonte=10, confianza=0.8, metodo=metodo).dropna()
    assert len(tabla) > 500
    assert (tabla["inferior"] < tabla["superior"]).all()
    assert (tabla["inferior"] > 0).all()


def test_mas_confianza_o_horizonte_ensancha(precios):
    a = fc.intervalo_actual(precios, 10, 0.8)
    b = fc.intervalo_actual(precios, 10, 0.95)
    c = fc.intervalo_actual(precios, 30, 0.8)
    ancho = lambda d: d["superior"] - d["inferior"]
    assert ancho(b) > ancho(a)
    assert ancho(c) > ancho(a)
    assert set(a) >= {"inferior", "superior", "confianza", "horizonte"}
    assert "centro" not in a  # nunca se entrega un precio único


def test_vol_historica_formula():
    r = np.array([0.01, -0.01] * 40)
    p = pd.Series(100 * np.exp(np.concatenate([[0], np.cumsum(r)])),
                  index=pd.bdate_range("2022-01-03", periods=len(r) + 1))
    d = fc.intervalo_actual(p, horizonte=4, confianza=0.8, metodo="vol_historica", ventana=20)
    sigma = np.std(r[-20:], ddof=1) * np.sqrt(4)
    z = fc.cuantil_z(0.8)
    assert d["inferior"] == pytest.approx(p.iloc[-1] * np.exp(-z * sigma))
    assert d["superior"] == pytest.approx(p.iloc[-1] * np.exp(z * sigma))


@pytest.mark.parametrize("metodo", TODOS)
def test_intervalos_sin_mirar_al_futuro(precios, metodo):
    """El intervalo emitido en t no cambia si se alteran los precios posteriores a t."""
    k = 800
    alterado = precios.copy()
    alterado.iloc[k:] *= 1.5
    a = fc.serie_intervalos(precios, 10, 0.8, metodo).iloc[:k]
    b = fc.serie_intervalos(alterado, 10, 0.8, metodo).iloc[:k]
    pd.testing.assert_frame_equal(a, b)


def test_ajustar_garch_recupera_persistencia():
    rng = np.random.default_rng(0)
    w, a, b = 1e-6, 0.08, 0.90
    s2, r = w / (1 - a - b), []
    for _ in range(3000):
        x = rng.normal(0, np.sqrt(s2))
        r.append(x)
        s2 = w + a * x * x + b * s2
    _, a_est, b_est = fc.ajustar_garch(np.array(r))
    assert a_est + b_est == pytest.approx(a + b, abs=0.05)


def test_cobertura_cercana_al_objetivo(precios):
    resumen, detalle = cal.informe_calibracion(precios, horizonte=5, confianza=0.8,
                                               metodos=["vol_historica"])
    fila = resumen.loc["vol_historica"]
    assert fila["cobertura"] == pytest.approx(0.8, abs=0.06)
    assert fila["calibrado"]
    assert set(resumen.index) == {"ingenuo", "vol_historica"}
    # Todos los métodos se evalúan en el mismo periodo.
    assert detalle["ingenuo"].index.equals(detalle["vol_historica"].index)


def test_realizado_es_el_precio_h_dias_despues(precios):
    tabla = cal.evaluar_metodo(precios, "vol_historica", horizonte=5, confianza=0.8)
    fecha = tabla.index[100]
    posicion = precios.index.get_loc(fecha)
    assert tabla.loc[fecha, "realizado"] == precios.iloc[posicion + 5]
    manual = (tabla["realizado"] >= tabla["inferior"]) & (tabla["realizado"] <= tabla["superior"])
    assert (tabla["dentro"] == manual).all()


def test_puntuacion_intervalo_winkler(precios):
    tabla = cal.evaluar_metodo(precios, "ewma", horizonte=5, confianza=0.8)
    lo, hi, y, p0 = (tabla[c].to_numpy() for c in ["inferior", "superior", "realizado", "precio_t"])
    manual = np.where(y < lo, (hi - lo) + 10 * (lo - y),
                      np.where(y > hi, (hi - lo) + 10 * (y - hi), hi - lo)) / p0  # 2/α = 10
    np.testing.assert_allclose(tabla["puntuacion_rel"], manual)


def test_tolerancia():
    assert cal.tolerancia_cobertura(0.8, 10_000, 1) == pytest.approx(0.02)
    assert cal.tolerancia_cobertura(0.8, 100, 10) > 0.2


def test_regimen_cambiante_descalibra_vol_constante():
    """Si la volatilidad se triplica de golpe, un intervalo de ventana larga cubre menos."""
    rng = np.random.default_rng(5)
    r = np.concatenate([rng.normal(0, 0.005, 700), rng.normal(0, 0.02, 500)])
    p = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2018-01-01", periods=len(r)))
    resumen, _ = cal.informe_calibracion(p, horizonte=5, confianza=0.8,
                                         metodos=["vol_historica", "ewma"], ventana=60)
    # La referencia ingenua usa cuantiles de toda la historia: se queda corta tras el cambio.
    assert resumen.loc["ingenuo", "cobertura"] < 0.8 - resumen.loc["ingenuo", "tolerancia"]
    # EWMA se adapta y mejora la puntuación frente al ingenuo.
    assert resumen.loc["ewma", "mejora vs ingenuo"]


def test_cobertura_movil(precios):
    _, detalle = cal.informe_calibracion(precios, 5, 0.8, metodos=["ewma"])
    cm = cal.cobertura_movil(detalle["ewma"], 100).dropna()
    assert ((cm >= 0) & (cm <= 1)).all()
