"""Precios objetivo de los analistas y su historial de aciertos con cada empresa.

Yahoo Finance guarda, para muchas empresas de EE. UU., cada cambio de precio objetivo de cada banco
o casa de análisis desde 2012 aprox. Con eso se reconstruye, mes a mes, el «consenso»: la mediana
del último objetivo de cada firma publicado en los 12 meses anteriores (mínimo 3 firmas).

Después se comprueba qué pasó: el consenso de un mes cualquiera se compara con el precio real un
año después. Así se ve si con esta empresa los analistas suelen quedarse cortos o pasarse.

Detalles:
* Los objetivos antiguos a veces están en precios anteriores a un split y a veces ya ajustados; para
  cada objetivo se elige la versión (ajustada o no) más coherente con el precio de ese día, y se
  descartan los que no cuadran con ninguna (menos de 1/5 o más de 5 veces el precio).
* Solo se usa información disponible en cada fecha (sin mirar el futuro).
* Las muestras mensuales se solapan (cada una mira un año adelante), así que no son independientes:
  con pocos años de historia, las cifras orientan pero no son concluyentes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MINIMO_FIRMAS = 3
VENTANA_DIAS = 365
HORIZONTE_DIAS = 365


def _sin_zona(indice) -> pd.DatetimeIndex:
    indice = pd.DatetimeIndex(indice)
    return indice.tz_localize(None) if indice.tz is not None else indice


def normalizar(cambios: pd.DataFrame, precios: pd.Series, splits: pd.Series | None = None) -> pd.DataFrame:
    """Objetivos en precios de hoy (ajustados por splits): columnas fecha, firma, objetivo, accion, opinion.

    ``precios`` es el cierre ajustado por splits (no por dividendos).
    """
    columnas = ["fecha", "firma", "objetivo", "accion", "opinion"]
    if cambios is None or cambios.empty or "currentPriceTarget" not in cambios or precios.empty:
        return pd.DataFrame(columns=columnas)
    c = cambios.copy()
    c.index = _sin_zona(c.index)
    c = c[pd.to_numeric(c["currentPriceTarget"], errors="coerce") > 0].sort_index()
    if c.empty:
        return pd.DataFrame(columns=columnas)
    precios = precios.dropna().astype(float)
    precios.index = _sin_zona(precios.index)
    s = pd.Series(dtype=float) if splits is None else splits[splits > 0]
    s.index = _sin_zona(s.index)

    filas = []
    for fecha, f in c.iterrows():
        bruto = float(f["currentPriceTarget"])
        factor = float(np.prod(s[s.index > fecha].to_numpy())) if len(s) else 1.0
        pos = precios.index.searchsorted(fecha, side="right") - 1
        if pos < 0:
            continue
        p = precios.iloc[pos]
        candidatos = {bruto / factor, bruto}
        mejor = min(candidatos, key=lambda x: abs(np.log(x / p)))
        if not 0.2 <= mejor / p <= 5:
            continue
        filas.append((fecha, f.get("Firm", ""), mejor, f.get("priceTargetAction", ""), f.get("Action", "")))
    return pd.DataFrame(filas, columns=columnas)


def consenso(objetivos: pd.DataFrame, fechas, ventana_dias: int = VENTANA_DIAS,
             minimo: int = MINIMO_FIRMAS) -> pd.DataFrame:
    """Consenso en cada fecha: mediana del último objetivo de cada firma en la ventana anterior."""
    filas = []
    for d in pd.DatetimeIndex(fechas):
        tramo = objetivos[(objetivos["fecha"] <= d) & (objetivos["fecha"] > d - pd.Timedelta(days=ventana_dias))]
        ultimos = tramo.groupby("firma")["objetivo"].last()
        if len(ultimos) >= minimo:
            filas.append((d, float(ultimos.median()), float(ultimos.min()), float(ultimos.max()), len(ultimos)))
    return pd.DataFrame(filas, columns=["fecha", "objetivo", "minimo", "maximo", "firmas"]).set_index("fecha")


def fin_de_mes(precios: pd.Series) -> pd.DatetimeIndex:
    """Último día con cotización de cada mes."""
    s = precios.dropna()
    return pd.DatetimeIndex(s.groupby(s.index.to_period("M")).apply(lambda x: x.index[-1]).to_numpy())


def historial(precios: pd.Series, cons: pd.DataFrame, horizonte_dias: int = HORIZONTE_DIAS) -> pd.DataFrame:
    """Para cada consenso con un año ya cumplido: lo que prometía y lo que pasó."""
    precios = precios.dropna().astype(float)
    filas = []
    for d, fila in cons.iterrows():
        final = d + pd.Timedelta(days=horizonte_dias)
        if final > precios.index[-1]:
            continue
        p0 = float(precios.asof(d))
        p1 = float(precios.asof(final))
        tramo = precios.loc[(precios.index > d) & (precios.index <= final)]
        filas.append({"fecha": d, "precio": p0, "objetivo": fila["objetivo"], "precio_un_ano": p1,
                      "prometido": fila["objetivo"] / p0 - 1, "real": p1 / p0 - 1,
                      "tocado": bool(len(tramo) and tramo.max() >= fila["objetivo"])})
    return pd.DataFrame(filas)


@dataclass
class Fiabilidad:
    muestras: int
    anios: float
    prometido_medio: float     # subida media que anunciaban
    real_medio: float          # lo que hizo de media al año
    sesgo: float               # prometido − real (positivo = demasiado optimistas)
    acierto_direccion: float   # % de veces que acertaron si subiría o bajaría
    llego_al_ano: float        # % de veces que al año estaba en el objetivo o por encima
    lo_toco: float             # % de veces que lo tocó en algún momento del año
    error_medio: float         # error absoluto medio (en puntos de rentabilidad)


def fiabilidad(hist: pd.DataFrame) -> Fiabilidad | None:
    if hist is None or len(hist) < 12:
        return None
    prom, real = hist["prometido"], hist["real"]
    return Fiabilidad(
        muestras=len(hist),
        anios=(hist["fecha"].iloc[-1] - hist["fecha"].iloc[0]).days / 365.25,
        prometido_medio=float(prom.mean()), real_medio=float(real.mean()), sesgo=float((prom - real).mean()),
        acierto_direccion=float((np.sign(prom) == np.sign(real)).mean()),
        llego_al_ano=float((hist["precio_un_ano"] >= hist["objetivo"]).mean()),
        lo_toco=float(hist["tocado"].mean()), error_medio=float((prom - real).abs().mean()))


def revisiones(objetivos: pd.DataFrame, hoy: pd.Timestamp | None = None, dias: int = 90) -> dict[str, int]:
    """Cuántas firmas subieron o bajaron su objetivo (y su opinión) en los últimos ``dias`` días."""
    if objetivos is None or objetivos.empty:
        return {"suben": 0, "bajan": 0, "mejoran": 0, "empeoran": 0, "total": 0}
    hoy = hoy or objetivos["fecha"].max()
    r = objetivos[objetivos["fecha"] > hoy - pd.Timedelta(days=dias)]
    return {"suben": int((r["accion"] == "Raises").sum()), "bajan": int((r["accion"] == "Lowers").sum()),
            "mejoran": int((r["opinion"] == "up").sum()), "empeoran": int((r["opinion"] == "down").sum()),
            "total": len(r)}


def descargar(ticker: str):
    """Cierre ajustado solo por splits (toda la historia), cambios de los analistas y splits."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    precios = yf.download(ticker, period="max", auto_adjust=False, progress=False, threads=False)
    if isinstance(precios.columns, pd.MultiIndex):
        precios.columns = precios.columns.get_level_values(0)
    cierre = precios["Close"] if "Close" in precios else pd.Series(dtype=float)
    try:
        cambios = t.upgrades_downgrades
    except Exception:
        cambios = None
    try:
        splits = t.splits
    except Exception:
        splits = None
    return cierre, cambios if cambios is not None else pd.DataFrame(), splits
