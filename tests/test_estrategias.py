import numpy as np
import pandas as pd

from analizador import estrategias as es


def _datos(n_activos=12, anios=12, semilla=0):
    rng = np.random.default_rng(semilla)
    fechas = pd.bdate_range("2000-01-03", periods=252 * anios)
    datos = {}
    for i in range(n_activos):
        r = rng.normal(0.0003 * (i % 4), 0.01 + 0.002 * i, len(fechas))
        datos[f"A{i}"] = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(r))}, index=fechas)
    return datos


def test_limpiar_quita_picos_de_un_dia():
    c = pd.Series([10.0, 10.1, 100.0, 10.2, 10.3])
    assert es.limpiar(c).max() < 11


def test_serie_rota_descarta_historia_anterior():
    datos = _datos(3, 3)
    c = datos["A0"]["Close"].copy()
    c.iloc[400:] *= 50  # split mal ajustado
    datos["A0"] = pd.DataFrame({"Close": c})
    u = es.preparar(datos)
    assert u.rentabilidad["A0"].max() < es.MAXIMO_MENSUAL
    assert u.precios["A0"].loc[: c.index[399]].isna().all()


def test_pesos_no_miran_al_futuro():
    datos = _datos()
    u = es.preparar(datos)
    corte = u.precios.index[60]
    alterados = {t: df.copy() for t, df in datos.items()}
    for df in alterados.values():
        df.loc[df.index > corte + pd.Timedelta(days=5), "Close"] *= np.random.default_rng(1).uniform(0.2, 5)
    u2 = es.preparar(alterados)
    for v in es.generar_variantes(len(datos)):
        w1, w2 = es.pesos(v, u).loc[:corte], es.pesos(v, u2).loc[:corte]
        pd.testing.assert_frame_equal(w1, w2, check_names=False)


def test_pesos_validos():
    u = es.preparar(_datos())
    for v in es.generar_variantes(12):
        w = es.pesos(v, u)
        assert (w >= 0).all().all()
        assert (w.sum(axis=1) <= 1 + 1e-9).all()


def test_aprendiz_solo_usa_el_pasado():
    fechas = pd.date_range("1995-01-31", periods=12 * 12, freq="ME")
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"A": rng.normal(0.01, 0.02, len(fechas)), "B": rng.normal(0.0, 0.02, len(fechas))}, index=fechas)
    serie, elecciones = es.aprendiz(r, primer_anio=2001)
    # Cambiar el futuro (2004 en adelante) no cambia lo elegido antes.
    r2 = r.copy()
    r2.loc["2004":, "B"] += 0.5
    _, elecciones2 = es.aprendiz(r2, primer_anio=2001)
    antes = elecciones[elecciones["anio"] <= 2004]
    pd.testing.assert_frame_equal(antes.reset_index(drop=True),
                                  elecciones2[elecciones2["anio"] <= 2004][antes.columns].reset_index(drop=True))
    assert elecciones.iloc[0]["variante"] == "A"


def test_estudio_completo_sintetico():
    e = es.estudiar(_datos(), "prueba", simulaciones=20)
    assert {"Aprendiz", "Aprendiz ambicioso", "Comprar todo (igual peso)"} <= set(e.curvas.columns)
    assert len(e.tabla) == len(es.generar_variantes(12))
    assert e.actual["peso"].sum() <= 1 + 1e-9
