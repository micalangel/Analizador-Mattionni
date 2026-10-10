import numpy as np
import pandas as pd
import pytest

from analizador import eventos, noticias, resultados
from conftest import generar_ohlcv

NY = "America/New_York"


@pytest.fixture
def precios():
    # Sesiones de lunes a viernes desde 2020-01-01 (miércoles).
    return generar_ohlcv(300, semilla=11)["Close"]


def pos(precios, fecha):
    return precios.index.get_loc(pd.Timestamp(fecha))


# --- Ventanas de evento ----------------------------------------------------------------

def test_evento_tras_el_cierre(precios):
    reac = eventos.reaccion_a_eventos(precios, [pd.Timestamp("2020-03-04 16:30", tz=NY)])
    fila = reac.iloc[0]
    assert fila["momento"] == "tras el cierre"
    assert fila["fecha_base"] == pd.Timestamp("2020-03-04")
    assert fila["fecha_reaccion"] == pd.Timestamp("2020-03-05")
    i = pos(precios, "2020-03-04")
    assert fila["reaccion"] == pytest.approx(precios.iloc[i + 1] / precios.iloc[i] - 1)


def test_evento_antes_de_la_apertura(precios):
    fila = eventos.reaccion_a_eventos(precios, [pd.Timestamp("2020-03-04 07:00", tz=NY)]).iloc[0]
    assert fila["momento"] == "antes de la apertura"
    assert fila["fecha_base"] == pd.Timestamp("2020-03-03")
    assert fila["fecha_reaccion"] == pd.Timestamp("2020-03-04")


def test_evento_en_fin_de_semana_y_utc(precios):
    # Sábado 2020-03-07 en UTC -> la reacción es del viernes al lunes.
    fila = eventos.reaccion_a_eventos(precios, [pd.Timestamp("2020-03-07 15:00", tz="UTC")]).iloc[0]
    assert fila["momento"] == "día sin sesión"
    assert fila["fecha_base"] == pd.Timestamp("2020-03-06")
    assert fila["fecha_reaccion"] == pd.Timestamp("2020-03-09")


def test_hora_desconocida_usa_dos_sesiones(precios):
    fila = eventos.reaccion_a_eventos(precios, [pd.Timestamp("2020-03-04 00:00", tz=NY)]).iloc[0]
    assert fila["sesiones"] == 2
    assert fila["fecha_base"] == pd.Timestamp("2020-03-03")


def test_evento_futuro_queda_pendiente(precios):
    fila = eventos.reaccion_a_eventos(precios, [precios.index[-1] + pd.Timedelta(days=30)]).iloc[0]
    assert fila["momento"] == "pendiente"
    assert np.isnan(fila["reaccion"])


def test_z_usa_solo_volatilidad_previa(precios):
    evento = pd.Timestamp("2020-06-03 17:00", tz=NY)
    a = eventos.reaccion_a_eventos(precios, [evento]).iloc[0]
    alterado = precios.copy()
    alterado.loc["2020-06-05":] *= np.linspace(1, 3, len(alterado.loc["2020-06-05":]))
    b = eventos.reaccion_a_eventos(alterado, [evento]).iloc[0]
    assert a["z"] == pytest.approx(b["z"])  # el futuro posterior a la reacción no influye


def test_z_es_reaccion_entre_volatilidad(precios):
    i = pos(precios, "2020-06-03")
    precios = precios.copy()
    precios.iloc[i + 1] = precios.iloc[i] * 1.10  # salto del 10 %
    fila = eventos.reaccion_a_eventos(precios, [pd.Timestamp("2020-06-03 17:00", tz=NY)]).iloc[0]
    sigma = np.log(precios / precios.shift(1)).iloc[i - 59: i + 1].std()
    assert fila["z"] == pytest.approx(np.log(1.10) / sigma)


# --- Resultados ------------------------------------------------------------------------

def estados_falsos():
    periodos = pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31", "2024-03-31"])
    cuenta = pd.DataFrame({
        "Total Revenue": [100, 110, 120, 130, 125],
        "Gross Profit": [40, 44, 48, 52, 50],
        "Operating Income": [20, 22, 24, 26, 25],
        "Net Income": [10, 11, -12, 13, 15],
        "Diluted EPS": [1.0, 1.1, -1.2, 1.3, 1.5],
    }, index=periodos).T
    caja = pd.DataFrame({"Free Cash Flow": [5, 6, 7, 8, 9]}, index=periodos).T
    balance = pd.DataFrame({"Total Debt": [50] * 5, "Stockholders Equity": [100] * 5}, index=periodos).T
    return cuenta, caja, balance


def test_tabla_y_margenes():
    tabla = resultados._tabla_partidas(*estados_falsos())
    assert tabla.index.is_monotonic_increasing
    assert tabla.loc["2023-03-31", "Ingresos"] == 100
    m = resultados.margenes_y_crecimiento(tabla, periodos_por_anio=4)
    assert m.loc["2023-03-31", "Margen bruto"] == pytest.approx(0.40)
    assert m.loc["2023-03-31", "Margen FCF"] == pytest.approx(0.05)
    assert m.loc["2024-03-31", "Crec. ingresos (interanual)"] == pytest.approx(0.25)  # vs 2023-03-31
    assert np.isnan(m.loc["2023-12-31", "Crec. ingresos (interanual)"])  # sin dato un año antes
    assert m.loc["2024-03-31", "Deuda / Patrimonio"] == pytest.approx(0.5)


def test_crecimiento_con_base_negativa_es_nan():
    tabla = resultados._tabla_partidas(*estados_falsos())
    m = resultados.margenes_y_crecimiento(tabla, periodos_por_anio=1)
    # 2023-12-31 frente a 2023-09-30 (beneficio negativo): sin porcentaje.
    assert np.isnan(m.loc["2023-12-31", "Crec. beneficio neto (interanual)"])


def test_estudio_y_resumen_publicaciones(precios):
    crudo = pd.DataFrame({
        "EPS Estimate": [1.0, 1.0, 1.0, 1.2],
        "Reported EPS": [1.1, 0.9, 1.05, np.nan],
        "Surprise(%)": [10.0, -10.0, 5.0, np.nan],
    }, index=pd.DatetimeIndex(["2020-04-01 16:05", "2020-07-01 16:05", "2020-10-01 07:00",
                               "2099-01-01 16:05"]).tz_localize(NY))
    pub = resultados.normalizar_publicaciones(crudo)
    assert list(pub.columns) == ["BPA estimado", "BPA publicado", "Sorpresa %"]
    estudio = resultados.estudio_publicaciones(precios, pub)
    assert estudio["reaccion"].notna().sum() == 3
    r = resultados.resumen_publicaciones(estudio)
    assert r["n"] == 3
    assert r["% superan estimación"] == pytest.approx(2 / 3)
    prox = resultados.proxima_publicacion(pub, ahora=pd.Timestamp("2021-01-01", tz="UTC"))
    assert prox["BPA estimado"] == 1.2


def test_obtener_resultados_tolera_fallos():
    def roto(_):
        raise RuntimeError("sin red")

    datos = resultados.obtener_resultados("XXX", descargador=roto)
    assert datos.trimestral.empty and datos.publicaciones.empty
    assert resultados.resumen_publicaciones(
        resultados.estudio_publicaciones(pd.Series(dtype=float), datos.publicaciones))["n"] == 0


# --- Noticias --------------------------------------------------------------------------

def test_noticias_ambos_formatos_y_duplicados(precios):
    items = [
        {"title": "Titular A", "publisher": "Medio 1", "link": "https://a",
         "providerPublishTime": int(pd.Timestamp("2020-05-05 14:00", tz="UTC").timestamp())},
        {"content": {"title": "Titular B", "pubDate": "2020-05-06T22:00:00Z",
                     "provider": {"displayName": "Medio 2"}, "canonicalUrl": {"url": "https://b"}}},
        {"title": "Titular A", "publisher": "Medio 1", "link": "https://a",
         "providerPublishTime": int(pd.Timestamp("2020-05-05 14:00", tz="UTC").timestamp())},
        {"title": None},
    ]
    df = noticias.obtener_noticias("abc", descargador=lambda t, n: items)
    assert list(df["titular"]) == ["Titular B", "Titular A"]  # más reciente primero
    estudio = noticias.estudio_noticias(precios, df)
    # B: 22:00 UTC = 18:00 NY -> tras el cierre del 6 de mayo.
    assert estudio.loc[estudio["titular"] == "Titular B", "momento"].iloc[0] == "tras el cierre"
    assert estudio["reaccion"].notna().all()


def test_noticias_sin_datos():
    df = noticias.obtener_noticias("abc", descargador=lambda t, n: [])
    assert df.empty
    assert noticias.estudio_noticias(pd.Series(dtype=float), df).empty
