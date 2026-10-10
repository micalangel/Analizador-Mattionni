"""Laboratorio de estrategias de cartera: genera variantes, aprende cuál funciona y lo comprueba.

Las reglas sobre un solo activo (simulador) casi nunca ganan al azar. Las estrategias de
*cartera* eligen cada mes entre muchos activos; algunas familias (momentum, baja volatilidad,
tendencia) tienen décadas de estudios a favor, aunque con rachas largas malas.

Cómo funciona:

1. **Genera** variantes de cada familia con distintos parámetros (unas 30).
2. **Simula** cada una mes a mes: la cartera se decide con el cierre del último día de cada mes y
   se mantiene el mes siguiente. Cada cambio paga comisión. El dinero que no está invertido
   cobra el tipo a 3 meses de EE. UU.
3. **Aprende sin mirar al futuro** («el aprendiz»): cada 1 de enero elige la variante con mejor
   rentabilidad ajustada por riesgo (Sharpe) con los datos hasta ese día, y la usa ese año. Así su
   resultado es honesto: cada año se mide con datos que no había visto al elegir.
4. **Compara con el azar**: carteras con el mismo número de activos elegidos al azar cada mes.

Limitaciones: los índices están en su moneda local (sin efecto divisa) y sin dividendos en
algunos casos; la lista de empresas es la de hoy (sesgo de supervivencia: las que quebraron no
están), lo que infla todos los resultados con empresas, incluida la referencia.

Simulación educativa: no es una recomendación de inversión.
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import AVISO_LEGAL

COSTE = 0.002          # por unidad de rotación (comisión + deslizamiento)
PRIMER_ANIO_APRENDIZ = 2000
MESES_MINIMOS_ELEGIR = 60
SIMULACIONES_AZAR = 300

FAMILIAS = {
    "momentum": "Momentum: compra los N activos que más han subido en los últimos meses (sin contar el último).",
    "momentum_absoluto": "Momentum con freno: como el momentum, pero solo si además están en positivo; si no, liquidez.",
    "baja_volatilidad": "Baja volatilidad: compra los N activos más tranquilos de los últimos 6 meses.",
    "tendencia": "Tendencia: reparte entre todos los activos que están por encima de su media de M meses; "
                 "el resto, en liquidez.",
    "reversion": "Reversión: compra los N activos que más han caído el último mes, esperando rebote.",
    "combinada": "Combinada: compra los N activos con mejor mezcla de subida en 12 meses y poca volatilidad.",
    "momentum_filtro": "Momentum con filtro de mercado: momentum de 12 meses, pero solo cuando el conjunto del "
                       "mercado está por encima de su media de 10 meses; si no, todo en liquidez.",
    "revisiones": "Revisiones de analistas: compra las N empresas a las que más analistas han subido el precio "
                  "objetivo en los últimos 3 meses (frente a los que lo han bajado).",
    "potencial": "Potencial de analistas: compra las N empresas cuyo precio objetivo está más por encima del precio.",
    "sorpresa": "Sorpresa en resultados: compra las N empresas que más superaron lo esperado en sus últimos resultados.",
    "momentum_revisiones": "Momentum + revisiones: mezcla la subida de 12 meses con las revisiones de los analistas.",
    "momentum_sorpresa": "Momentum + sorpresa: mezcla la subida de 12 meses con la sorpresa en resultados.",
    "triple": "Triple: mezcla subida de 12 meses, revisiones de analistas y sorpresa en resultados.",
    "todos": "Comprar todo: reparte a partes iguales entre todos los activos (la referencia).",
}


@dataclass(frozen=True)
class Variante:
    familia: str
    parametros: tuple  # ((nombre, valor), ...)

    @property
    def p(self) -> dict:
        return dict(self.parametros)

    @property
    def nombre(self) -> str:
        p = self.p
        textos = {
            "momentum": f"Momentum {p.get('meses')} meses · {p.get('n')} activos",
            "momentum_absoluto": f"Momentum con freno {p.get('meses')} meses · {p.get('n')} activos",
            "baja_volatilidad": f"Baja volatilidad · {p.get('n')} activos",
            "tendencia": f"Tendencia (media de {p.get('meses')} meses)",
            "reversion": f"Reversión 1 mes · {p.get('n')} activos",
            "combinada": f"Combinada (subida + calma) · {p.get('n')} activos",
            "momentum_filtro": f"Momentum con filtro de mercado · {p.get('n')} activos",
            "revisiones": f"Revisiones de analistas · {p.get('n')} activos",
            "potencial": f"Potencial de analistas · {p.get('n')} activos",
            "sorpresa": f"Sorpresa en resultados · {p.get('n')} activos",
            "momentum_revisiones": f"Momentum + revisiones · {p.get('n')} activos",
            "momentum_sorpresa": f"Momentum + sorpresa · {p.get('n')} activos",
            "triple": f"Triple (momentum + revisiones + sorpresa) · {p.get('n')} activos",
            "todos": "Comprar todo (igual peso)",
        }
        return textos[self.familia]


COMBINACIONES_EXTRA = {"revisiones": ("revisiones",), "potencial": ("potencial",), "sorpresa": ("sorpresa",),
                       "momentum_revisiones": ("momentum", "revisiones"), "momentum_sorpresa": ("momentum", "sorpresa"),
                       "triple": ("momentum", "revisiones", "sorpresa")}


def generar_variantes(n_activos: int, extras: tuple[str, ...] = ()) -> list[Variante]:
    """Todas las variantes a probar; el número de activos se adapta al tamaño del universo."""
    ns = sorted({max(2, round(n_activos * f)) for f in (0.1, 0.2, 0.3)})
    v = [Variante("todos", ())]
    for meses, n in itertools.product((3, 6, 12), ns):
        v.append(Variante("momentum", (("meses", meses), ("n", n))))
        v.append(Variante("momentum_absoluto", (("meses", meses), ("n", n))))
    v += [Variante("baja_volatilidad", (("n", n),)) for n in ns]
    v += [Variante("reversion", (("n", n),)) for n in ns]
    v += [Variante("tendencia", (("meses", m),)) for m in (6, 10, 12)]
    v += [Variante("combinada", (("n", n),)) for n in ns]
    v += [Variante("momentum_filtro", (("n", n),)) for n in ns]
    for familia, piezas in COMBINACIONES_EXTRA.items():
        if all(p == "momentum" or p in extras for p in piezas):
            v += [Variante(familia, (("n", n),)) for n in ns]
    return v


# --- Datos mensuales ---------------------------------------------------------------------------

@dataclass
class Universo:
    precios: pd.DataFrame       # cierre de fin de mes
    rentabilidad: pd.DataFrame  # rentabilidad de cada mes (fila t = del fin de t-1 al fin de t)
    volatilidad: pd.DataFrame   # volatilidad diaria de los últimos 126 días, a fin de mes
    liquidez: pd.Series         # rentabilidad mensual de la liquidez (fila t = durante el mes t)
    extra: dict = field(default_factory=dict)  # puntuaciones mensuales adicionales (analistas, resultados…)


MAXIMO_MENSUAL = 3.0   # +300 % en un mes: casi siempre un error de datos (split mal ajustado)
MINIMO_MENSUAL = -0.9


def limpiar(cierre: pd.Series) -> pd.Series:
    """Quita los picos de un día que se deshacen al siguiente (errores de datos de Yahoo)."""
    c = cierre.astype(float).copy()
    for _ in range(3):
        x = c.pct_change()
        pico = (x.abs() > 0.5) & (x * x.shift(-1) < 0) & (x.shift(-1).abs() > 0.33)
        if not pico.any():
            break
        c[pico] = np.nan
        c = c.ffill()
    return c


def preparar(datos: dict[str, pd.DataFrame], tipo_3m: pd.Series | None = None) -> Universo:
    cierres = pd.DataFrame({t: limpiar(df["Close"].dropna()) for t, df in datos.items()}).sort_index()
    cierres.index = pd.DatetimeIndex(cierres.index).tz_localize(None) if cierres.index.tz else cierres.index
    meses = cierres.resample("ME")
    precios = meses.last()
    # Un activo sin cotizaciones en todo el mes no cuenta ese mes.
    precios = precios.where(meses.count() > 0)
    rent = precios / precios.shift(1) - 1
    # Meses imposibles: la serie está rota (p. ej. split mal ajustado) y su historia anterior no es
    # comparable con la posterior; se descarta todo lo anterior a la última rotura.
    imposible = (rent > MAXIMO_MENSUAL) | (rent < MINIMO_MENSUAL)
    for t in imposible.columns[imposible.any()]:
        rotura = imposible.index[imposible[t]][-1]
        precios.loc[:rotura, t] = np.nan
    rent = (precios / precios.shift(1) - 1)
    diaria = np.log(cierres / cierres.ffill(limit=5).shift(1))
    vol = diaria.rolling(126, min_periods=100).std().resample("ME").last().where(precios.notna())
    if tipo_3m is not None and not tipo_3m.dropna().empty:
        t = tipo_3m.dropna().astype(float)
        t.index = pd.DatetimeIndex(t.index).tz_localize(None) if t.index.tz else t.index
        anual = t.resample("ME").last().reindex(precios.index).ffill() / 100
        liquidez = (anual / 12).shift(1).fillna(0.0)  # el tipo conocido al empezar el mes
    else:
        liquidez = pd.Series(0.0, index=precios.index)
    return Universo(precios, rent, vol.reindex(precios.index), liquidez.clip(lower=0))


# --- Pesos de cada variante ------------------------------------------------------------------

def _top(puntuacion: pd.DataFrame, n: int, mayores: bool = True) -> pd.DataFrame:
    rango = puntuacion.rank(axis=1, ascending=not mayores, method="first")
    elegidos = (rango <= n) & puntuacion.notna()
    return elegidos.astype(float).div(n)  # cada elegido pesa 1/n; si hay menos de n, el resto es liquidez


def pesos(v: Variante, u: Universo) -> pd.DataFrame:
    """Pesos decididos al cierre de cada mes (fila t) para mantener durante el mes t+1."""
    p, pr = v.p, u.precios
    disponible = pr.notna() & pr.shift(12).notna()  # al menos un año de historia
    if v.familia == "todos":
        w = disponible.astype(float)
        return w.div(w.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if v.familia in ("momentum", "momentum_absoluto"):
        punt = (pr.shift(1) / pr.shift(p["meses"]) - 1).where(disponible)
        w = _top(punt, p["n"])
        if v.familia == "momentum_absoluto":
            w = w.where((pr / pr.shift(p["meses"]) - 1) > 0, 0.0)
        return w
    if v.familia == "baja_volatilidad":
        return _top(u.volatilidad.where(disponible), p["n"], mayores=False)
    if v.familia == "reversion":
        return _top(u.rentabilidad.where(disponible), p["n"], mayores=False)
    if v.familia == "combinada":
        momentum = (pr.shift(1) / pr.shift(12) - 1).where(disponible)
        calma = -u.volatilidad.where(disponible)
        return _top(momentum.rank(axis=1, pct=True) + calma.rank(axis=1, pct=True), p["n"])
    if v.familia == "momentum_filtro":
        momentum = (pr.shift(1) / pr.shift(12) - 1).where(disponible)
        # Índice del conjunto: media de las rentabilidades de los activos disponibles cada mes.
        conjunto = (1 + u.rentabilidad.where(disponible.shift(1, fill_value=False)).mean(axis=1).fillna(0)).cumprod()
        en_tendencia = conjunto > conjunto.rolling(10).mean()
        return _top(momentum, p["n"]).mul(en_tendencia.astype(float), axis=0)
    if v.familia in COMBINACIONES_EXTRA:
        rangos = []
        for pieza in COMBINACIONES_EXTRA[v.familia]:
            if pieza == "momentum":
                punt = pr.shift(1) / pr.shift(12) - 1
            else:
                punt = u.extra[pieza].reindex(index=pr.index, columns=pr.columns)
            rangos.append(punt.where(disponible).rank(axis=1, pct=True))
        # Solo cuentan los activos con todas las piezas disponibles ese mes.
        return _top(sum(rangos), p["n"])
    if v.familia == "tendencia":
        encima = (pr > pr.rolling(p["meses"]).mean()) & disponible
        total = disponible.sum(axis=1).replace(0, np.nan)
        return encima.astype(float).div(total, axis=0).fillna(0.0)
    raise ValueError(v.familia)


def rentabilidad_cartera(w: pd.DataFrame, u: Universo, coste: float = COSTE) -> pd.Series:
    """Rentabilidad mensual neta de costes. Los pesos de la fila t se aplican a la rentabilidad de t+1."""
    aplicados = w.shift(1).fillna(0.0)
    bruta = (aplicados * u.rentabilidad.fillna(0.0)).sum(axis=1)
    liquidez = (1 - aplicados.sum(axis=1)).clip(lower=0) * u.liquidez
    rotacion = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1).shift(1).fillna(0.0)
    return bruta + liquidez - rotacion * coste


# --- Métricas -----------------------------------------------------------------------------------

def metricas(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 12:
        return {}
    capital = (1 + r).cumprod()
    anios = len(r) / 12
    vol = r.std() * np.sqrt(12)
    return {"rent_anual": float(capital.iloc[-1] ** (1 / anios) - 1), "volatilidad": float(vol),
            "sharpe": float(r.mean() * 12 / vol) if vol > 1e-12 else np.nan,
            "max_caida": float((capital / capital.cummax() - 1).min()),
            "meses_positivos": float((r > 0).mean()), "anios": anios}


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    s = r.std()
    return float(r.mean() / s) if len(r) > 1 and s > 1e-12 else -np.inf


# --- Pruebas de resistencia --------------------------------------------------------------------

def mitad(desde: pd.Timestamp, hasta: pd.Timestamp) -> pd.Timestamp:
    """Fin de año más cercano a la mitad del periodo (para comprobar las dos mitades por separado)."""
    centro = desde + (hasta - desde) / 2
    return pd.Timestamp(f"{centro.year}-12-31")


def robustez(w: pd.DataFrame, u: Universo, desde: pd.Timestamp) -> dict:
    """Rentabilidad anual en condiciones peores: comisiones dobles, un mes de retraso y por mitades."""
    base = rentabilidad_cartera(w, u).loc[desde:]
    corte = mitad(desde, base.index[-1])
    salida = {"normal": metricas(base).get("rent_anual"),
              "comision_doble": metricas(rentabilidad_cartera(w, u, 2 * COSTE).loc[desde:]).get("rent_anual"),
              "un_mes_tarde": metricas(rentabilidad_cartera(w.shift(1).fillna(0.0), u).loc[desde:]).get("rent_anual")}
    salida["primera_mitad"] = metricas(base.loc[:corte]).get("rent_anual")
    salida["segunda_mitad"] = metricas(base.loc[corte + pd.Timedelta(days=1):]).get("rent_anual")
    return salida


# --- Aprendiz (selección año a año sin mirar al futuro) ---------------------------------------

def _rentabilidad_media(r: pd.Series) -> float:
    r = r.dropna()
    return float(np.log1p(r).mean()) if len(r) else -np.inf


CRITERIOS = {"sharpe": _sharpe, "rentabilidad": _rentabilidad_media}


def aprendiz(rentabilidades: pd.DataFrame, primer_anio: int = PRIMER_ANIO_APRENDIZ,
             minimo: int = MESES_MINIMOS_ELEGIR, criterio: str = "sharpe") -> tuple[pd.Series, pd.DataFrame]:
    """Cada año usa la variante con mejor puntuación hasta el 31 de diciembre anterior.

    ``rentabilidades``: columnas = variantes, filas = meses (rentabilidad neta del mes).
    ``criterio``: «sharpe» (rentabilidad por unidad de riesgo) o «rentabilidad» (crecimiento medio).
    """
    puntuar = CRITERIOS[criterio]
    salida = pd.Series(np.nan, index=rentabilidades.index)
    elecciones = []
    for anio in sorted(set(rentabilidades.index.year)):
        if anio < primer_anio:
            continue
        pasado = rentabilidades[rentabilidades.index.year < anio]
        validas = [c for c in pasado if pasado[c].dropna().shape[0] >= minimo]
        if not validas:
            continue
        puntuaciones = {c: puntuar(pasado[c].dropna()) for c in validas}
        mejor = max(puntuaciones, key=puntuaciones.get)
        filas = rentabilidades.index.year == anio
        salida[filas] = rentabilidades.loc[filas, mejor]
        elecciones.append({"anio": anio, "variante": mejor,
                           "rent_anio": float((1 + rentabilidades.loc[filas, mejor]).prod() - 1)})
    return salida.dropna(), pd.DataFrame(elecciones)


# --- Comparación con el azar ----------------------------------------------------------------

def azar(u: Universo, n: int, simulaciones: int = SIMULACIONES_AZAR, semilla: int = 0,
         desde: pd.Timestamp | None = None) -> np.ndarray:
    """Rentabilidad anual de carteras de ``n`` activos elegidos al azar cada mes (con la misma rotación media)."""
    rng = np.random.default_rng(semilla)
    disp = (u.precios.notna() & u.precios.shift(12).notna()).to_numpy()
    rent = u.rentabilidad.fillna(0.0).to_numpy()
    liq = u.liquidez.to_numpy()
    filas = np.arange(len(disp)) if desde is None else np.where(u.precios.index >= desde)[0]
    filas = filas[filas >= 1]
    resultados = np.empty(simulaciones)
    for s in range(simulaciones):
        r = np.empty(len(filas))
        for k, i in enumerate(filas):
            candidatos = np.flatnonzero(disp[i - 1])
            if len(candidatos) == 0:
                r[k] = liq[i]
                continue
            m = min(n, len(candidatos))
            elegidos = rng.choice(candidatos, m, replace=False)
            r[k] = rent[i, elegidos].sum() / n + (n - m) / n * liq[i] - 2 * COSTE * 0.5
        # Coste aproximado: la mitad de la cartera rota cada mes al azar (rotación alta, conservador).
        anios = len(r) / 12
        resultados[s] = np.prod(1 + r) ** (1 / anios) - 1 if anios > 0 else np.nan
    return resultados


# --- Estudio completo -----------------------------------------------------------------------

@dataclass
class Estudio:
    universo: str
    tabla: pd.DataFrame                 # métricas de cada variante (periodo completo)
    curvas: pd.DataFrame                # rentabilidades mensuales: variantes + «Aprendiz»
    elecciones: pd.DataFrame            # qué variante eligió el aprendiz cada año
    actual: pd.DataFrame                # cartera que tendría hoy la variante elegida para este año
    resumen: dict = field(default_factory=dict)


def estudiar(datos: dict[str, pd.DataFrame], universo: str, tipo_3m: pd.Series | None = None,
             nombres: dict[str, str] | None = None, simulaciones: int = SIMULACIONES_AZAR,
             extra: dict[str, pd.DataFrame] | None = None, primer_anio: int = PRIMER_ANIO_APRENDIZ,
             minimo: int = MESES_MINIMOS_ELEGIR) -> Estudio:
    u = preparar(datos, tipo_3m)
    u.extra = {k: v for k, v in (extra or {}).items() if v is not None and v.notna().any().any()}
    variantes = generar_variantes(len(datos), tuple(u.extra))
    curvas, todos_pesos = {}, {}
    for v in variantes:
        w = pesos(v, u)
        todos_pesos[v.nombre] = w
        curvas[v.nombre] = rentabilidad_cartera(w, u)
    curvas = pd.DataFrame(curvas)
    # Empieza cuando la referencia tiene activos invertidos.
    inicio = curvas.index[(todos_pesos["Comprar todo (igual peso)"].shift(1).sum(axis=1) > 0).to_numpy()][0]
    if u.extra:
        # Con datos adicionales, se empieza cuando todos cubren al menos a una cuarta parte de los activos.
        cobertura = [t.notna().sum(axis=1) >= 0.25 * len(datos) for t in u.extra.values()]
        listos = pd.concat(cobertura, axis=1).all(axis=1)
        if listos.any():
            inicio = max(inicio, listos.index[listos.to_numpy()][0] + pd.offsets.MonthEnd(1))
    curvas = curvas.loc[inicio:]
    # El último mes puede estar a medias: se incluye (rentabilidad hasta hoy).
    serie_aprendiz, elecciones = aprendiz(curvas, primer_anio, minimo)
    serie_ambicioso, elecciones_amb = aprendiz(curvas, primer_anio, minimo, criterio="rentabilidad")
    desde_aprendiz = serie_aprendiz.index[0]

    filas = []
    azar_cache: dict[int, np.ndarray] = {}
    for v in variantes:
        m = metricas(curvas[v.nombre])
        m_oos = metricas(curvas.loc[desde_aprendiz:, v.nombre])
        n = v.p.get("n")
        percentil = np.nan
        if n:
            if n not in azar_cache:
                azar_cache[n] = azar(u, n, simulaciones, desde=inicio)
            percentil = float((azar_cache[n] < m.get("rent_anual", np.nan)).mean())
        filas.append({"variante": v.nombre, "familia": v.familia, **m,
                      "rent_anual_desde_aprendiz": m_oos.get("rent_anual"), "vs_azar": percentil,
                      "veces_elegida": int((elecciones["variante"] == v.nombre).sum()) if len(elecciones) else 0})
    tabla = pd.DataFrame(filas).sort_values("sharpe", ascending=False).reset_index(drop=True)
    # Pruebas de resistencia de las mejores variantes y de la referencia.
    a_probar = list(dict.fromkeys(list(tabla["variante"].head(6)) + ["Comprar todo (igual peso)"]))
    rob = pd.DataFrame({n: robustez(todos_pesos[n], u, desde_aprendiz) for n in a_probar}).T
    tabla = tabla.merge(rob.add_prefix("rob_"), left_on="variante", right_index=True, how="left")

    curvas["Aprendiz"] = serie_aprendiz
    curvas["Aprendiz ambicioso"] = serie_ambicioso
    m_amb = metricas(serie_ambicioso)
    ref = "Comprar todo (igual peso)"
    m_apr = metricas(serie_aprendiz)
    m_ref = metricas(curvas.loc[desde_aprendiz:, ref])
    # Cartera de hoy: la variante con mejor Sharpe con TODOS los datos hasta hoy (la que usaría el año que viene).
    validas = [c for c in curvas.columns
               if not c.startswith("Aprendiz") and curvas[c].dropna().shape[0] >= minimo]
    elegida_hoy = max(validas, key=lambda c: _sharpe(curvas[c].dropna()))
    w_hoy = todos_pesos[elegida_hoy].iloc[-1]
    w_hoy = w_hoy[w_hoy > 0].sort_values(ascending=False)
    nombres = nombres or {}
    actual = pd.DataFrame({"ticker": w_hoy.index, "nombre": [nombres.get(t, t) for t in w_hoy.index],
                           "peso": w_hoy.to_numpy()})
    resumen = {"universo": universo, "activos": len(datos), "variantes": len(variantes),
               "desde": str(curvas.index[0].date()), "hasta": str(curvas.index[-1].date()),
               "desde_aprendiz": str(desde_aprendiz.date()), "elegida_hoy": elegida_hoy,
               "corte_mitades": str(mitad(desde_aprendiz, curvas.index[-1]).date()),
               "liquidez_hoy": float(max(0.0, 1 - w_hoy.sum())),
               **{f"aprendiz_{k}": v for k, v in m_apr.items()},
               **{f"ambicioso_{k}": v for k, v in m_amb.items()}, **{f"referencia_{k}": v for k, v in m_ref.items()}}
    if len(elecciones):
        # Percentil del aprendiz frente al azar con el número de activos de su variante más elegida.
        mas_elegida = elecciones["variante"].mode().iloc[0]
        apr_n = next((v.p.get("n") for v in variantes if v.nombre == mas_elegida), None)
        if apr_n:
            dist = azar(u, apr_n, simulaciones, semilla=1, desde=desde_aprendiz)
            resumen["aprendiz_vs_azar"] = float((dist < m_apr.get("rent_anual", np.nan)).mean())
    todas = pd.concat([elecciones.assign(criterio="sharpe"), elecciones_amb.assign(criterio="rentabilidad")],
                      ignore_index=True)
    return Estudio(universo, tabla, curvas, todas, actual, resumen)


# --- Guardar y cargar ---------------------------------------------------------------------------

def guardar(e: Estudio, directorio) -> None:
    d = Path(directorio) / "estrategias"
    d.mkdir(parents=True, exist_ok=True)
    e.tabla.to_csv(d / f"{e.universo}_tabla.csv", index=False)
    e.curvas.to_csv(d / f"{e.universo}_curvas.csv")
    e.elecciones.to_csv(d / f"{e.universo}_elecciones.csv", index=False)
    e.actual.to_csv(d / f"{e.universo}_actual.csv", index=False)
    pd.Series(e.resumen).to_csv(d / f"{e.universo}_resumen.csv", header=["valor"])


def cargar(universo: str, directorio) -> Estudio | None:
    d = Path(directorio) / "estrategias"
    if not (d / f"{universo}_tabla.csv").exists():
        return None
    resumen = pd.read_csv(d / f"{universo}_resumen.csv", index_col=0)["valor"].to_dict()
    for k, v in resumen.items():
        try:
            resumen[k] = float(v)
        except (TypeError, ValueError):
            pass
    return Estudio(universo, pd.read_csv(d / f"{universo}_tabla.csv"),
                   pd.read_csv(d / f"{universo}_curvas.csv", index_col=0, parse_dates=True),
                   pd.read_csv(d / f"{universo}_elecciones.csv"), pd.read_csv(d / f"{universo}_actual.csv"), resumen)


UNIVERSOS = {"bolsas": "Bolsas del mundo (índices)", "empresas": "Empresas de la lista",
             "otros": "Materias primas, divisas y criptomonedas",
             "todo": "Todo junto: bolsas, materias primas, divisas y cripto",
             "analistas": "Empresas de EE. UU. con datos de analistas y resultados"}


def main(argv: list[str] | None = None) -> int:
    from .catalogo import ACTIVOS
    from .data import descargar_varios
    from .entrenamiento import DIRECTORIO
    from .indices import INDICES

    p = argparse.ArgumentParser(prog="analizador.estrategias", description=__doc__.splitlines()[0], epilog=AVISO_LEGAL)
    p.add_argument("--universos", nargs="*", default=list(UNIVERSOS))
    p.add_argument("--directorio", default=str(DIRECTORIO))
    args = p.parse_args(argv)
    hoy = date.today()
    tipo = descargar_varios(["^IRX"], "1990-01-01", hoy).get("^IRX")
    tipo_3m = tipo["Close"] if tipo is not None else None
    nombres = {t: v[0] for t, v in ACTIVOS.items()}
    grupos = {"bolsas": list(INDICES),
              "empresas": [t for t, v in ACTIVOS.items() if v[1] == "Acción"],
              "otros": [t for t, v in ACTIVOS.items() if v[1] in ("Materia prima", "Divisa", "Criptomoneda")]}
    grupos["todo"] = grupos["bolsas"] + grupos["otros"]
    grupos["analistas"] = [t for t, v in ACTIVOS.items() if v[1] == "Acción" and v[2] == "Estados Unidos"]
    for clave in args.universos:
        t0 = time.time()
        datos = descargar_varios(grupos[clave], "1990-01-01", hoy)
        if clave == "analistas":
            from .datos_empresas import puntuaciones_mensuales

            meses = preparar(datos).precios.index
            extra = puntuaciones_mensuales(list(datos), meses)
            if not any(v.notna().any().any() for v in extra.values()):
                print("  Sin datos de analistas en la caché: ejecuta antes `python -m analizador.datos_empresas`.")
                continue
            print("  Cobertura de datos adicionales: " + ", ".join(
                f"{k} {int(v.notna().sum(axis=1).max())} empresas" for k, v in extra.items()))
            # Los datos de analistas empiezan hacia 2012: aprendiz desde 2016 con 3 años mínimos.
            e = estudiar(datos, clave, tipo_3m, nombres, extra=extra, primer_anio=2016, minimo=36)
        else:
            e = estudiar(datos, clave, tipo_3m, nombres)
        guardar(e, args.directorio)
        r = e.resumen
        print(f"{UNIVERSOS[clave]}: {len(datos)} activos, {r['variantes']} variantes, {time.time() - t0:.0f} s. "
              f"Aprendiz desde {r['desde_aprendiz']}: {r.get('aprendiz_rent_anual', np.nan):.1%}/año "
              f"(máx. caída {r.get('aprendiz_max_caida', np.nan):.0%}) frente a comprar todo "
              f"{r.get('referencia_rent_anual', np.nan):.1%}/año ({r.get('referencia_max_caida', np.nan):.0%}). "
              f"Ambicioso: {r.get('ambicioso_rent_anual', np.nan):.1%}/año ({r.get('ambicioso_max_caida', np.nan):.0%}).")
    return 0


if __name__ == "__main__":
    from analizador.estrategias import main as _main

    raise SystemExit(_main())
