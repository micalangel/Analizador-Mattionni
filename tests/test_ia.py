import numpy as np
import pandas as pd
import pytest

from analizador import ia
from analizador.calibration import informe_calibracion
from conftest import generar_ohlcv

RAPIDO = dict(n_arboles=40, minimo_muestras=250, reentreno=200)


def datos_con_regimenes(n=1100, semilla=4):
    """Volatilidad que cambia por tramos: la IA tiene algo que aprender."""
    rng = np.random.default_rng(semilla)
    vol = np.where((np.arange(n) // 150) % 2 == 0, 0.006, 0.02)
    r = rng.normal(0, vol)
    precios = 100 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Open": precios, "High": precios * 1.005, "Low": precios * 0.995,
                         "Close": precios, "Volume": rng.integers(1e3, 1e4, n).astype(float)}, index=idx)


def test_caracteristicas_sin_mirar_al_futuro():
    datos = generar_ohlcv(400, semilla=2)
    k = 300
    alterado = datos.copy()
    alterado.iloc[k:] *= 2.5
    a = ia.construir_caracteristicas(datos).iloc[:k]
    b = ia.construir_caracteristicas(alterado).iloc[:k]
    pd.testing.assert_frame_equal(a, b)


def test_caracteristicas_de_resultados_solo_pasado():
    idx = pd.bdate_range("2020-01-01", periods=60)
    pub = pd.DataFrame({"BPA estimado": [1.0, 1.0], "BPA publicado": [1.1, np.nan],
                        "Sorpresa %": [10.0, np.nan]},
                       index=pd.DatetimeIndex(["2020-01-15 16:30", "2020-02-20 16:30"]).tz_localize("America/New_York"))
    f = ia._caracteristicas_resultados(idx, pub)
    assert f.loc["2020-01-15"].isna().all()                     # aún no se conocía
    assert f.loc["2020-01-16", "ultima_sorpresa"] == 10.0       # conocida al cierre siguiente
    assert f.loc["2020-01-16", "dias_desde_resultados"] == 0
    assert f.loc["2020-02-21", "ultima_sorpresa"] == 10.0       # la de febrero no tiene BPA publicado


def test_entrenar_solo_con_resultados_conocidos():
    datos = generar_ohlcv(600, semilla=3)
    X = ia.construir_caracteristicas(datos)
    y = ia.objetivo_escalado(datos["Close"], 10)
    hasta = datos.index[450]
    m = ia.entrenar(X, y, 10, 0.8, hasta=hasta, n_arboles=20, minimo_muestras=100)
    # La última muestra usable es la 440: su precio objetivo es el del día 450.
    usables = int((y.iloc[:441].notna() & X.iloc[:441].notna().sum(axis=1).ge(len(m.columnas) * 0.8)).sum())
    n_cal = int(usables * 0.25)
    assert m.n_calibracion == n_cal
    assert m.n_muestras == usables - n_cal - 10  # hueco de h sesiones entre ajuste y calibración
    assert m.entrenado_hasta == hasta


def test_ajuste_conformal_corrige_exceso_de_confianza():
    """Con árboles muy sobreajustados, el ajuste conformal ensancha los cuantiles."""
    datos = datos_con_regimenes(900)
    X = ia.construir_caracteristicas(datos)
    y = ia.objetivo_escalado(datos["Close"], 5)
    m = ia.entrenar(X, y, 5, 0.8, n_arboles=300, minimo_muestras=200)
    assert m.ajuste_conformal > 0


def test_walk_forward_sin_mirar_al_futuro():
    datos = datos_con_regimenes(900)
    k = 700
    alterado = datos.copy()
    alterado.iloc[k:] *= np.linspace(1, 3, len(datos) - k)[:, None]
    a, _ = ia.serie_intervalos_ia(datos, 10, 0.8, **RAPIDO)
    b, _ = ia.serie_intervalos_ia(alterado, 10, 0.8, **RAPIDO)
    pd.testing.assert_frame_equal(a.iloc[:k], b.iloc[:k])


def test_intervalos_validos_y_calibracion_razonable():
    datos = datos_con_regimenes()
    tabla, modelos = ia.serie_intervalos_ia(datos, 5, 0.8, **RAPIDO)
    validos = tabla.dropna()
    assert len(modelos) >= 3
    assert (validos["inferior"] < validos["superior"]).all()
    resumen, _ = informe_calibracion(datos["Close"], 5, 0.8, metodos=["ewma"],
                                     extra={"ia": (ia.DESCRIPCION, tabla)})
    assert "ia" in resumen.index
    assert resumen.loc["ia", "cobertura"] == pytest.approx(0.8, abs=0.1)


def test_importancias_y_guardar_cargar(tmp_path):
    datos = datos_con_regimenes(700)
    X = ia.construir_caracteristicas(datos)
    y = ia.objetivo_escalado(datos["Close"], 5)
    m = ia.entrenar(X, y, 5, 0.8, n_arboles=20, minimo_muestras=200)
    imp = m.importancias()
    assert imp.sum() == pytest.approx(1.0)
    assert "Volatilidad 5 días" in imp.index

    ruta = ia.guardar_modelo(m, "test", tmp_path)
    assert ruta.exists()
    cargado = ia.cargar_modelo("TEST", 5, 0.8, tmp_path)
    actual = ia.intervalo_actual_ia(datos, cargado)
    assert actual["inferior"] < actual["precio_ultimo"] < actual["superior"]
    assert "centro" not in actual
    assert ia.cargar_modelo("TEST", 10, 0.8, tmp_path) is None


def test_pocos_datos_no_entrena():
    datos = generar_ohlcv(200)
    tabla, modelos = ia.serie_intervalos_ia(datos, 10, 0.8, **RAPIDO)
    assert modelos == [] and tabla["inferior"].isna().all()
