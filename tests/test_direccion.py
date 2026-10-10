import numpy as np
import pandas as pd
import pytest

from analizador import direccion as d
from conftest import generar_ohlcv


def cesta(n=2600, k=4, deriva=0.0004):
    datos = {}
    for i in range(k):
        rng = np.random.default_rng(200 + i)
        p = 100 * np.exp(np.cumsum(rng.normal(deriva, 0.01, n)))
        datos[f"I{i}"] = pd.DataFrame({"Open": p, "High": p * 1.003, "Low": p * 0.997, "Close": p, "Volume": 0.0},
                                      index=pd.bdate_range("2000-01-03", periods=n))
    return datos


@pytest.fixture(scope="module")
def conjunto():
    datos = cesta()
    c = d.ConjuntoDireccion(21)
    panel = d.preparar_panel(datos, datos["I0"]["Close"], 21)
    c.por_anio = {a: m for a in (2004, 2006, 2008) if (m := d.entrenar(panel, pd.Timestamp(f"{a}-01-01"), 1000))}
    c.actual = d.entrenar(panel, panel["fecha"].max() + pd.Timedelta(days=1), 1000)
    return c, datos


def test_frecuencia_historica_solo_usa_periodos_terminados():
    idx = pd.bdate_range("2020-01-01", periods=600)
    p = pd.Series(np.arange(600, dtype=float) + 100, index=idx)  # siempre sube
    f = d.frecuencia_historica(p, 5)
    assert f.dropna().eq(1.0).all()
    # Primer valor: cuando ya hay 250 periodos de 5 sesiones comprobados.
    assert f.first_valid_index() == idx[5 + 249]
    alterado = p.copy()
    alterado.iloc[400:] = 50  # caída posterior
    pd.testing.assert_series_equal(f.iloc[:400], d.frecuencia_historica(alterado, 5).iloc[:400])


def test_probabilidades_sin_mirar_al_futuro(conjunto):
    c, datos = conjunto
    df = datos["I2"]
    a = d.serie_probabilidades(df, c, datos["I0"]["Close"])
    alterado = df.copy()
    alterado.loc["2008-01-01":, ["Open", "High", "Low", "Close"]] *= 0.5
    b = d.serie_probabilidades(alterado, c, datos["I0"]["Close"])
    pd.testing.assert_series_equal(a.loc[:"2007-12-31"], b.loc[:"2007-12-31"])
    validas = a.dropna()
    assert ((validas > 0) & (validas < 1)).all()


def test_estimar_elige_y_mide(conjunto):
    c, datos = conjunto
    e = d.estimar(datos["I3"], 21, c, datos["I0"]["Close"])
    assert e.fuente in ("ia", "frecuencia")
    assert 0 < e.probabilidad < 1 and 0 < e.frecuencia < 1
    assert 0 <= e.acierto_validacion <= 1
    assert e.desde < e.hasta
    assert set(e.calibracion.columns) == {"dijo", "subio", "veces"}


def test_sin_ia_usa_frecuencia():
    df = cesta(2000, 1)["I0"]
    e = d.estimar(df, 21)
    assert e.fuente == "frecuencia"
    assert e.probabilidad == pytest.approx(e.frecuencia)


def test_poca_historia():
    assert d.estimar(generar_ohlcv(200), 21) is None


def test_guardar_cargar(conjunto, tmp_path):
    c, _ = conjunto
    d.guardar(c, tmp_path)
    assert sorted(d.cargar(21, tmp_path).por_anio) == sorted(c.por_anio)
    d.ruta(5, tmp_path).write_bytes(b"roto")
    with pytest.warns(UserWarning):
        assert d.cargar(5, tmp_path) is None


def test_brier():
    assert d.brier([0.5, 0.5], [1, 0]) == pytest.approx(0.25)
    assert d.brier([1.0, 0.0], [1, 0]) == 0


def test_frecuencia_ajustada_corrige_supervivencia():
    # Poca historia: casi todo el peso para el mercado; mucha historia: para el activo.
    poca = d.frecuencia_ajustada(0.95, 2 * 252, 252)
    mucha = d.frecuencia_ajustada(0.95, 200 * 252, 252)
    assert d.BASE_MERCADO[252] < poca < mucha < 0.95
    assert poca == pytest.approx((2 * 0.95 + 20 * d.BASE_MERCADO[252]) / 22)
