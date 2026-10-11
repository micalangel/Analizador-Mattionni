import numpy as np
import pandas as pd
import pytest

from analizador import salud as sl
from analizador import volumen as vol


def datos(n=400, semilla=0, picos=()):
    rng = np.random.default_rng(semilla)
    idx = pd.bdate_range("2015-01-01", periods=n)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    v = np.exp(rng.normal(12, 0.2, n))
    for p in picos:
        v[p] *= 8
    return pd.DataFrame({"Close": c, "Volume": v}, index=idx)


def test_z_volumen_detecta_picos_y_no_mira_al_futuro():
    df = datos(picos=(300,))
    z = vol.z_volumen(df)
    assert z.iloc[300] > 5
    alterado = df.copy()
    alterado.iloc[350:, alterado.columns.get_loc("Volume")] *= 50
    pd.testing.assert_series_equal(z.iloc[:350], vol.z_volumen(alterado).iloc[:350])


def test_volumen_relativo():
    df = datos()
    df["Volume"] = 1000.0
    df.iloc[-1, df.columns.get_loc("Volume")] = 3000.0
    assert vol.volumen_relativo(df).iloc[-1] == pytest.approx(3.0)


def test_dias_inusuales_y_lo_que_paso_despues():
    df = datos(picos=(320, 380))
    tabla = vol.dias_inusuales(df, horizonte=21)
    assert {df.index[380], df.index[320]} <= set(tabla.index)
    assert tabla.index.is_monotonic_decreasing  # más reciente primero
    assert pd.isna(tabla.loc[df.index[380], "despues"])         # aún no se sabe
    esperado = df["Close"].iloc[341] / df["Close"].iloc[320] - 1
    assert tabla.loc[df.index[320], "despues"] == pytest.approx(esperado)


def test_laboratorio_volumen_sin_patron_no_supera_al_azar():
    cesta = {f"E{i}": datos(1500, semilla=i, picos=range(100, 1500, 37)) for i in range(5)}
    lab = vol.laboratorio_volumen(cesta, {"1 mes": 21})
    assert len(lab) == len(vol.REGLAS_VOLUMEN)
    assert (lab["veredicto"] != "Supera al azar").all()


def test_salud_semaforos():
    fund = pd.Series({"PER": 20.0, "Margen de beneficio": 0.25, "Deuda/Patrimonio": 2.0, "Rentabilidad FCF": -0.01})
    idx = pd.to_datetime(["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"])
    tri = pd.DataFrame({"Beneficio neto": [10.0, 10, 10, -50]}, index=idx)
    crec = pd.DataFrame({"Crec. ingresos (interanual)": [np.nan, np.nan, np.nan, 0.12]}, index=idx)
    ind = {i.nombre: i for i in sl.indicadores(fund, tri, crec, "Tecnología")}
    assert ind["Margen de beneficio"].estado == "bien"
    assert ind["Deuda / patrimonio"].estado == "mal"
    assert ind["Rentabilidad del flujo de caja"].estado == "mal"
    assert ind["Crecimiento de las ventas"].estado == "bien"
    assert ind["Beneficio del último año"].estado == "mal"   # -20 en total
    assert ind["PER (precio / beneficio)"].estado == "info"
    # En un banco la deuda alta no se marca en rojo.
    banco = {i.nombre: i for i in sl.indicadores(fund, tri, crec, "Banca")}
    assert banco["Deuda / patrimonio"].estado == "info"


def test_salud_resumen():
    I = sl.Indicador
    solida = [I("a", "", "bien", ""), I("b", "", "bien", ""), I("c", "", "regular", ""), I("d", "", "info", "")]
    assert sl.resumen(solida)[1] == "bien"
    debil = [I("a", "", "mal", ""), I("b", "", "mal", ""), I("c", "", "bien", "")]
    assert sl.resumen(debil)[1] == "mal"
    assert sl.resumen([I("a", "", "info", "")])[1] == "info"


def test_per_de_referencia_usa_medianas_y_descarta_extremos():
    salud = pd.DataFrame({"ticker": list("ABCDEFG"), "per": [10, 12, 14, 16, 900, -5, None]})
    sectores = {"A": "Banca", "B": "Banca", "C": "Banca", "D": "Banca", "E": "Banca", "F": "Banca", "G": "Banca"}
    medianas, total = sl.per_de_referencia(salud, sectores)
    assert medianas["Banca"] == pytest.approx(13)  # sin el 900, el negativo ni el vacío
    assert total == pytest.approx(13)
    # Sectores con menos de 4 empresas no tienen referencia propia.
    medianas, _ = sl.per_de_referencia(salud.iloc[:3], {"A": "X", "B": "X", "C": "X"})
    assert "X" not in medianas


def test_comparar_per():
    assert sl.comparar_per(10, 20)[0] == "Por debajo de su sector"
    assert sl.comparar_per(21, 20)[0] == "En la media de su sector"
    assert sl.comparar_per(30, 20) == ("Por encima de su sector", pytest.approx(0.5))
    assert sl.comparar_per(None, 20)[0] == "—" and sl.comparar_per(-3, 20)[0] == "—"


def test_datos_incompletos_de_empresa_nueva():
    import numpy as np
    import pandas as pd

    from analizador.resultados import margenes_y_crecimiento
    from analizador.salud import _texto_per, completar, ultimo_anio

    # Trimestres con hueco (faltan los dos últimos de 2025), como SpaceX tras salir a bolsa.
    fechas = pd.to_datetime(["2025-03-31", "2025-06-30", "2026-03-31", "2026-06-30"])
    t = pd.DataFrame({"Ingresos": [4.0, 4.0, 4.7, 7.8], "Beneficio bruto": 1.0, "Beneficio operativo": 0.0,
                      "Beneficio neto": [-0.5, -1.0, -4.3, -0.5], "BPA diluido": -0.1,
                      "Flujo de caja libre": [-3.4, -3.2, -9.1, -16.8], "Deuda total": np.nan,
                      "Patrimonio neto": np.nan}, index=fechas)
    crec = margenes_y_crecimiento(t, 4)["Crec. ingresos (interanual)"]
    assert abs(crec.iloc[-1] - (7.8 / 4.0 - 1)) < 1e-9  # 2026-06 frente a 2025-06, aunque falten trimestres
    assert np.isnan(ultimo_anio(t["Beneficio neto"]))      # no son 4 trimestres seguidos
    seguidos = t.copy()
    seguidos.index = pd.to_datetime(["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"])
    assert abs(ultimo_anio(seguidos["Beneficio neto"]) - (-6.3)) < 1e-9

    anual = pd.DataFrame({"Flujo de caja libre": [-14.0]}, index=pd.to_datetime(["2025-12-31"]))
    fund = pd.Series({"PER": np.nan, "PER esperado": 84.4, "BPA 12m": -1.1, "Beneficio 12m": -8.9,
                      "Rentabilidad FCF": np.nan, "Capitalización": 2000.0})
    completo = completar(fund, t, anual)
    assert abs(completo["Rentabilidad FCF"] - (-14.0 / 2000)) < 1e-12
    assert completo["Beneficio 12m"] == -8.9  # se queda el de Yahoo si los trimestres no son seguidos
    assert _texto_per(completo).startswith("No tiene: pierde dinero") and "84,4" in _texto_per(completo)
    assert _texto_per(pd.Series({"PER": 20.0, "PER esperado": 18.0})) == "20,0 veces (esperado: 18,0)"
