import numpy as np
import pandas as pd

from analizador import analistas as an


def _precios(anios=6, crecimiento=0.10):
    fechas = pd.bdate_range("2015-01-01", periods=252 * anios)
    return pd.Series(100 * np.exp(np.arange(len(fechas)) * np.log(1 + crecimiento) / 252), index=fechas)


def _cambios(precios, prima=0.20, firmas=("A", "B", "C", "D")):
    filas = []
    for i, d in enumerate(precios.index[::21]):
        filas.append({"GradeDate": d, "Firm": firmas[i % len(firmas)], "currentPriceTarget": precios[d] * (1 + prima),
                      "priceTargetAction": "Raises", "Action": "main"})
    return pd.DataFrame(filas).set_index("GradeDate")


def test_split_se_ajusta_y_se_descarta_lo_absurdo():
    p = _precios()
    cambios = _cambios(p)
    # Un objetivo antiguo en precios de antes de un split 1:4 y otro absurdo.
    dia = p.index[100]
    cambios.loc[dia] = ["A", p[dia] * 4 * 1.2, "Raises", "main"]
    cambios.loc[p.index[101]] = ["B", p[p.index[101]] * 50, "Raises", "main"]
    splits = pd.Series([4.0], index=[p.index[500]])
    o = an.normalizar(cambios.sort_index(), p, splits)
    fila = o[o["fecha"] == dia].iloc[0]
    assert abs(fila["objetivo"] / p[dia] - 1.2) < 1e-9
    assert (o["objetivo"] / p.reindex(o["fecha"]).to_numpy()).max() < 5


def test_consenso_necesita_tres_firmas_y_no_mira_el_futuro():
    p = _precios()
    o = an.normalizar(_cambios(p), p)
    fechas = an.fin_de_mes(p)
    c = an.consenso(o, fechas)
    assert c.index[0] >= o["fecha"].iloc[2]
    # Cambiar objetivos futuros no altera el consenso de una fecha pasada.
    d = c.index[10]
    o2 = o.copy()
    o2.loc[o2["fecha"] > d, "objetivo"] *= 10
    assert an.consenso(o2, [d])["objetivo"].iloc[0] == c.loc[d, "objetivo"]


def test_fiabilidad_detecta_optimismo():
    p = _precios(crecimiento=0.05)
    o = an.normalizar(_cambios(p, prima=0.30), p)
    h = an.historial(p, an.consenso(o, an.fin_de_mes(p)))
    f = an.fiabilidad(h)
    assert f is not None
    assert f.prometido_medio > 0.25 and abs(f.real_medio - 0.05) < 0.01
    assert f.sesgo > 0.2 and f.llego_al_ano == 0 and f.acierto_direccion == 1


def test_revisiones_cuenta_ultimos_meses():
    o = pd.DataFrame({"fecha": pd.to_datetime(["2024-01-01", "2024-05-01", "2024-05-10", "2024-05-20"]),
                      "firma": list("ABCD"), "objetivo": [1, 2, 3, 4],
                      "accion": ["Raises", "Raises", "Lowers", "Maintains"], "opinion": ["up", "up", "down", "main"]})
    r = an.revisiones(o, pd.Timestamp("2024-06-01"))
    assert r == {"suben": 1, "bajan": 1, "mejoran": 1, "empeoran": 1, "total": 3}
