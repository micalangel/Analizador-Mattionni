import numpy as np
import pandas as pd
import pytest

from analizador import entrenamiento as ent
from analizador import ia
from analizador.forecast import estado_garch, serie_intervalos
from analizador.proyeccion import ESTADOS, proyectar, proyectar_horizonte, tabla_resumen
from conftest import generar_ohlcv


def cesta_sintetica(n=2600, k=4):
    """Varios 'índices' con volatilidad por regímenes: suficiente para entrenar rápido."""
    datos = {}
    for i in range(k):
        rng = np.random.default_rng(100 + i)
        vol = np.where((np.arange(n) // (200 + 30 * i)) % 2 == 0, 0.007, 0.018)
        r = rng.normal(0.0002, vol)
        p = 100 * np.exp(np.cumsum(r))
        idx = pd.bdate_range("2000-01-03", periods=n)
        datos[f"IDX{i}"] = pd.DataFrame({"Open": p, "High": p * 1.004, "Low": p * 0.996, "Close": p,
                                         "Volume": 0.0}, index=idx)
    return datos


@pytest.fixture(scope="module")
def conjunto():
    datos = cesta_sintetica()
    mercado = datos["IDX0"]["Close"]
    # Menos años y una sola confianza para que el test sea rápido.
    c = ent.ConjuntoIA(10)
    panel = ent.preparar_panel(datos, mercado, 10)
    c.por_anio[0.8] = {a: m for a in (2004, 2006, 2008)
                       if (m := ent.entrenar_panel(panel, 10, 0.8, pd.Timestamp(f"{a}-01-01"), minimo=1000))}
    c.actual[0.8] = ent.entrenar_panel(panel, 10, 0.8, panel["fecha"].max() + pd.Timedelta(days=1), minimo=1000)
    return c, datos, mercado, panel


def test_mercado_usa_cierre_anterior():
    idx = pd.bdate_range("2021-01-04", periods=40)
    mercado = pd.Series(np.linspace(100, 140, 40), index=idx)
    datos = generar_ohlcv(40)
    datos.index = idx
    f = ia._caracteristicas_mercado(idx, mercado)
    esperado = np.log(mercado.iloc[29] / mercado.iloc[9])  # retorno 20 d al cierre ANTERIOR al día 30
    assert f["mercado_ret_20"].iloc[30] == pytest.approx(esperado)
    assert f["mercado_ret_20"].iloc[:21].isna().all()


def test_panel_y_corte_sin_mirar_al_futuro(conjunto):
    c, datos, mercado, panel = conjunto
    assert (panel["fecha_resultado"] > panel["fecha"]).all()
    corte = pd.Timestamp("2006-01-01")
    m = c.por_anio[0.8][2006]
    assert m.entrenado_hasta == corte
    usadas = panel[panel["fecha_resultado"] < corte]
    assert m.n_muestras + m.n_calibracion <= len(usadas)


def test_modelo_para_usa_el_ultimo_anterior(conjunto):
    c, *_ = conjunto
    assert c.modelo_para(0.8, 2007) is c.por_anio[0.8][2006]
    assert c.modelo_para(0.8, 2003) is None


def test_serie_global_fuera_de_muestra(conjunto):
    c, datos, mercado, _ = conjunto
    df = datos["IDX2"]
    tabla = ent.serie_intervalos_global(df, c, 0.8, mercado)
    assert tabla.loc[:"2003-12-31", "inferior"].isna().all()   # aún no había modelo
    validos = tabla.dropna()
    assert len(validos) > 500 and (validos["inferior"] < validos["superior"]).all()
    # Alterar precios posteriores a 2007 no cambia los intervalos de 2006-2007.
    alterado = df.copy()
    alterado.loc["2008-01-01":, ["Close", "High", "Low", "Open"]] *= 2
    otra = ent.serie_intervalos_global(alterado, c, 0.8, mercado)
    pd.testing.assert_frame_equal(tabla.loc["2006":"2007"], otra.loc["2006":"2007"])


def test_guardar_y_cargar(conjunto, tmp_path):
    c, datos, mercado, _ = conjunto
    ent.guardar(c, tmp_path)
    cargado = ent.cargar(10, tmp_path)
    assert cargado is not None and sorted(cargado.por_anio[0.8]) == [2004, 2006, 2008]
    actual = ent.intervalo_actual_global(datos["IDX1"], cargado, 0.8, mercado)
    assert actual["inferior"] < actual["precio_ultimo"] < actual["superior"]
    assert ent.cargar(21, tmp_path) is None


def test_garch_estado_reutilizable():
    p = generar_ohlcv(600, semilla=5)["Close"]
    estado = estado_garch(np.log(p / p.shift(1)))
    a = serie_intervalos(p, 10, 0.8, "garch")
    b = serie_intervalos(p, 10, 0.8, "garch", estado_garch_previo=estado)
    pd.testing.assert_frame_equal(a, b)


def test_proyeccion_elige_y_valida(conjunto):
    c, datos, mercado, _ = conjunto
    p = proyectar_horizonte(datos["IDX3"], 10, 0.8, c, mercado, "10 días")
    assert p.metodo in p.comparativa.index and p.comparativa["elegido"].sum() == 1
    assert p.rango["inferior"] < p.rango["precio_ultimo"] < p.rango["superior"]
    f = p.fiabilidad
    assert f["estado"] in ESTADOS and 0 <= f["cobertura"] <= 1
    # La validación es posterior a todo el tramo de selección.
    assert p.validacion.index.min() == f["desde"]
    assert "centro" not in p.rango


def test_proyeccion_con_poca_historia_avisa():
    p = proyectar_horizonte(generar_ohlcv(300), 21, 0.8)
    assert p.estado == "sin_datos"


def test_proyectar_todos_los_plazos():
    df = cesta_sintetica(1800, 1)["IDX0"]
    res = proyectar(df, 0.8, horizontes={"5 días": 5, "1 mes": 21})
    tabla = tabla_resumen(res)
    assert list(tabla.index) == ["5 días", "1 mes"]
    assert (tabla["Desde"] < tabla["Hasta"]).all()


def test_cargar_avisa_si_el_modelo_no_se_puede_leer(tmp_path):
    ent.ruta(5, tmp_path).write_bytes(b"esto no es un modelo")
    with pytest.warns(UserWarning, match="No se pudo cargar"):
        assert ent.cargar(5, tmp_path) is None


def test_modulo_ejecutable_delega_en_el_paquete():
    """Al ejecutar `python -m analizador.entrenamiento`, las clases deben guardarse con su nombre real."""
    import inspect

    fuente = inspect.getsource(ent)
    assert "from analizador.entrenamiento import main as _main" in fuente
    assert ent.ConjuntoIA.__module__ == "analizador.entrenamiento"


def test_camino_probable_coherente_con_rango_y_probabilidad():
    from statistics import NormalDist

    import numpy as np

    from analizador.proyeccion import camino_probable

    plazos = [(5, 95.0, 106.0, 0.55), (21, 90.0, 112.0, 0.58), (252, 75.0, 150.0, 0.68)]
    c = camino_probable(100.0, plazos, 0.8, ejemplos=3, semilla=1)
    assert len(c) == 253 and abs(c["central"].iloc[0] - 100) < 1e-9
    assert (c["inferior"] <= c["central"] + 1e-9).all() and (c["central"] <= c["superior"] + 1e-9).all()
    for h, inf, sup, p in plazos:
        fila = c.iloc[h]
        assert abs(fila["inferior"] - inf) < 1e-6 and abs(fila["superior"] - sup) < 1e-6
        # Con la anchura del rango, la probabilidad de acabar por encima de hoy es la estimada.
        sigma = (np.log(sup) - np.log(inf)) / (2 * NormalDist().inv_cdf(0.9))
        assert abs(NormalDist().cdf((np.log(fila["central"]) - np.log(100)) / sigma) - p) < 1e-6
    assert {"ejemplo_1", "ejemplo_2", "ejemplo_3"} <= set(c.columns)
    assert np.allclose(c[["ejemplo_1", "ejemplo_2", "ejemplo_3"]].iloc[0], 100)
    # Sin probabilidad, el centro del rango (media geométrica).
    c2 = camino_probable(100.0, [(10, 90.0, 110.0, None)], ejemplos=0)
    assert abs(c2["central"].iloc[-1] - np.sqrt(90 * 110)) < 1e-9
