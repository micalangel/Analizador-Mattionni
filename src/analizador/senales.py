"""Laboratorio de señales: ¿alguna señal predice la DIRECCIÓN mejor que el azar?

Para cada señal clásica (tendencia, momentum, reversión, volatilidad, calendario) se mide,
en muchos índices a la vez:

- **Acierto**: % de veces que la dirección indicada (sube/baja) coincide con la del
  precio h sesiones después.
- **Azar**: el acierto esperado de alguien que dijera "sube" y "baja" en las mismas
  proporciones que la señal pero al azar, sabiendo solo con qué frecuencia sube el
  mercado en general (p): azar = w₊·p + w₋·(1 − p). Si la señal no aporta información,
  su acierto es igual al del azar, aunque supere el 50 % (los mercados suben más días de
  los que bajan, así que decir "sube" acierta más de la mitad de las veces sin mérito).
- **Ventaja** = acierto − azar. Mide si la señal elige días mejores que uno cualquiera.

Rigor:
- Las señales solo usan datos hasta el cierre de t; el resultado es el precio de t + h.
- Periodo de **descubrimiento** (antes de 2010) y de **comprobación** (2010 en adelante).
  Una señal solo cuenta si su ventaja aparece en ambos y es significativa en la comprobación.
- Significación con tamaño efectivo reducido: los días solapados (h > 1) y los índices
  correlacionados no son observaciones independientes.
- Corrección de Holm por probar muchas señales a la vez (si pruebas 30 cosas, alguna
  saldrá "significativa" por pura suerte).
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd

from .indicators import bandas_bollinger, rsi

CORTE_COMPROBACION = pd.Timestamp("2010-01-01")
HORIZONTES_DIRECCION = {"1 día": 1, "1 semana": 5, "1 mes": 21}

# Una señal recibe el cierre y devuelve +1 (sube), -1 (baja) o 0/NaN (no dice nada) por día.
Senal = Callable[[pd.Series], pd.Series]


@dataclass(frozen=True)
class DefinicionSenal:
    nombre: str
    familia: str
    descripcion: str
    funcion: Senal
    usa_datos_completos: bool = False  # True: recibe el DataFrame (p. ej. para usar el volumen)


def _signo(cond_sube: pd.Series, cond_baja: pd.Series | None = None) -> pd.Series:
    s = pd.Series(np.where(cond_sube, 1.0, 0.0), index=cond_sube.index)
    if cond_baja is not None:
        s[cond_baja.fillna(False).to_numpy()] = -1.0
    return s.where(cond_sube.notna())


def _retorno(c, k):
    return np.log(c / c.shift(k))


SENALES: list[DefinicionSenal] = [
    DefinicionSenal("Momentum 1 día", "Momentum", "Sube si ayer subió; baja si bajó.",
                    lambda c: np.sign(_retorno(c, 1)).replace(0, np.nan)),
    DefinicionSenal("Momentum 1 mes", "Momentum", "Dirección de los últimos 21 días.",
                    lambda c: np.sign(_retorno(c, 21)).replace(0, np.nan)),
    DefinicionSenal("Momentum 12 meses", "Momentum", "Dirección de los últimos 252 días.",
                    lambda c: np.sign(_retorno(c, 252)).replace(0, np.nan)),
    DefinicionSenal("Tendencia (precio sobre media de 200)", "Tendencia",
                    "Sube si el cierre está por encima de su media de 200 días; baja si está por debajo.",
                    lambda c: np.sign(c - c.rolling(200).mean()).replace(0, np.nan)),
    DefinicionSenal("Cruce de medias 50/200", "Tendencia", "Sube si la media de 50 supera a la de 200.",
                    lambda c: np.sign(c.rolling(50).mean() - c.rolling(200).mean()).replace(0, np.nan)),
    DefinicionSenal("Reversión RSI (30/70)", "Reversión",
                    "Sube si RSI < 30 (sobreventa); baja si RSI > 70 (sobrecompra). El resto de días no opina.",
                    lambda c: (lambda x: pd.Series(np.select([x < 30, x > 70], [1.0, -1.0], np.nan),
                                                    index=c.index))(rsi(c, 14))),
    DefinicionSenal("Rebote tras caída fuerte", "Reversión",
                    "Sube tras una caída semanal mayor de 2 desviaciones típicas. El resto de días no opina.",
                    lambda c: (lambda z: pd.Series(np.where(z < -2, 1.0, np.nan), index=c.index))(
                        _retorno(c, 5) / (_retorno(c, 1).rolling(60).std() * np.sqrt(5)))),
    DefinicionSenal("Bandas de Bollinger", "Reversión",
                    "Sube si cierra bajo la banda inferior; baja si cierra sobre la superior.",
                    lambda c: (lambda b: pd.Series(np.select([c < b["inferior"], c > b["superior"]], [1.0, -1.0],
                                                             np.nan), index=c.index))(bandas_bollinger(c))),
    DefinicionSenal("Volatilidad baja", "Volatilidad",
                    "Sube si la volatilidad de 20 días es menor que la de 1 año; baja si es mayor.",
                    lambda c: np.sign(_retorno(c, 1).rolling(252).std() - _retorno(c, 1).rolling(20).std())
                    .replace(0, np.nan)),
    DefinicionSenal("Noviembre-abril", "Calendario",
                    "Sube de noviembre a abril; baja de mayo a octubre («sell in May»).",
                    lambda c: pd.Series(np.where(c.index.month.isin([11, 12, 1, 2, 3, 4]), 1.0, -1.0),
                                        index=c.index).where(c.notna())),
    DefinicionSenal("Cambio de mes", "Calendario",
                    "Sube en los últimos 2 y primeros 3 días hábiles de cada mes. El resto de días no opina.",
                    lambda c: _cambio_de_mes(c.index)),
]


def _cambio_de_mes(indice: pd.DatetimeIndex) -> pd.Series:
    meses = indice.to_period("M")
    pos = pd.Series(range(len(indice)), index=indice)
    primero = pos.groupby(meses).transform("min")
    ultimo = pos.groupby(meses).transform("max")
    activo = ((pos - primero) < 3) | ((ultimo - pos) < 2)
    return pd.Series(np.where(activo, 1.0, np.nan), index=indice)


# --- Evaluación -----------------------------------------------------------------------

def observaciones(datos: dict[str, pd.DataFrame], senal: DefinicionSenal, horizonte: int) -> pd.DataFrame:
    """Una fila por (índice, día) en que la señal opina, con su dirección y el resultado."""
    partes = []
    for ticker, df in datos.items():
        c = df["Close"].astype(float)
        s = senal.funcion(df if senal.usa_datos_completos else c)
        futuro = np.log(c.shift(-horizonte) / c)
        tabla = pd.DataFrame({"senal": s, "futuro": futuro}).dropna()
        tabla = tabla[(tabla["senal"] != 0) & (tabla["futuro"] != 0)]
        tabla["ticker"] = ticker
        partes.append(tabla)
    if not partes:
        return pd.DataFrame(columns=["senal", "futuro", "ticker"])
    obs = pd.concat(partes)
    obs["acierto"] = np.sign(obs["futuro"]) == obs["senal"]
    obs["sube"] = obs["futuro"] > 0
    return obs


def correlacion_media(datos: dict[str, pd.DataFrame]) -> float:
    """Correlación media entre los retornos semanales de los índices (para el tamaño efectivo)."""
    semanal = pd.DataFrame({t: df["Close"].resample("W").last() for t, df in datos.items()})
    r = np.log(semanal / semanal.shift(1))
    m = r.corr().to_numpy()
    n = m.shape[0]
    if n < 2:
        return 0.0
    return float(np.nanmean(m[~np.eye(n, dtype=bool)]))


def _tramo(obs: pd.DataFrame, horizonte: int, rho: float, p_sube: float) -> dict:
    """``p_sube`` = frecuencia con que sube el mercado en el periodo (todos los días)."""
    if obs.empty or not np.isfinite(p_sube):
        return {"n": 0}
    acierto = obs["acierto"].mean()
    w_sube = (obs["senal"] > 0).mean()
    base = w_sube * p_sube + (1 - w_sube) * (1 - p_sube)  # acierto del azar con la misma mezcla
    # Tamaño efectivo: fechas no solapadas × índices "independientes".
    fechas = obs.index.unique()
    n_indices = obs["ticker"].nunique()
    indices_ef = n_indices / (1 + (n_indices - 1) * max(rho, 0))
    tickers_por_fecha = len(obs) / max(len(fechas), 1)
    n_ef = max(len(fechas) / horizonte * min(indices_ef, tickers_por_fecha), 1)
    error = np.sqrt(base * (1 - base) / n_ef) if 0 < base < 1 else np.nan
    z = (acierto - base) / error if error and error > 0 else np.nan
    return {
        "n": len(obs), "n_efectivo": n_ef, "acierto": acierto, "base": base, "ventaja": acierto - base,
        "p_sube_global": p_sube, "dice_sube": w_sube,
        "p_sube_si_sube": obs.loc[obs["senal"] > 0, "sube"].mean() if w_sube > 0 else np.nan,
        "p_baja_si_baja": (~obs.loc[obs["senal"] < 0, "sube"]).mean() if w_sube < 1 else np.nan,
        "z": z, "p": 2 * (1 - NormalDist().cdf(abs(z))) if np.isfinite(z) else np.nan,
        "dias_con_opinion": len(fechas),
        "rentabilidad_media": float((obs["senal"] * obs["futuro"]).mean()),
    }


def laboratorio(datos: dict[str, pd.DataFrame], horizontes: dict[str, int] | None = None,
                senales: list[DefinicionSenal] | None = None, corte: pd.Timestamp = CORTE_COMPROBACION,
                alfa: float = 0.05) -> pd.DataFrame:
    """Tabla con una fila por (señal, horizonte): descubrimiento, comprobación y veredicto."""
    horizontes = horizontes or HORIZONTES_DIRECCION
    senales = senales or SENALES
    rho = correlacion_media(datos)
    siempre = DefinicionSenal("siempre", "", "", lambda c: pd.Series(1.0, index=c.index).where(c.notna()))
    filas = []
    for etiqueta, h in horizontes.items():
        todos = observaciones(datos, siempre, h)
        p_desc = todos.loc[todos.index < corte, "sube"].mean()
        p_comp = todos.loc[todos.index >= corte, "sube"].mean()
        for s in senales:
            obs = observaciones(datos, s, h)
            desc = _tramo(obs[obs.index < corte], h, rho, p_desc)
            comp = _tramo(obs[obs.index >= corte], h, rho, p_comp)
            filas.append({
                "señal": s.nombre, "familia": s.familia, "descripción": s.descripcion, "plazo": etiqueta,
                **{f"{k}_desc": v for k, v in desc.items()}, **{f"{k}_comp": v for k, v in comp.items()},
            })
    tabla = pd.DataFrame(filas)
    orden = {d.nombre: i for i, d in enumerate(senales)}
    tabla = tabla.sort_values(["señal", "plazo"], key=lambda c: c.map(orden) if c.name == "señal"
                              else c.map({e: i for i, e in enumerate(horizontes)})).reset_index(drop=True)
    tabla["p_holm"] = _holm(tabla["p_comp"].to_numpy())
    tabla["veredicto"] = [_veredicto(r, alfa) for _, r in tabla.iterrows()]
    return tabla


def _holm(p: np.ndarray) -> np.ndarray:
    """p-valores ajustados por Holm-Bonferroni (controla falsos descubrimientos por azar)."""
    p = np.asarray(p, dtype=float)
    validos = np.isfinite(p)
    m = validos.sum()
    ajustado = np.full_like(p, np.nan)
    orden = np.argsort(np.where(validos, p, np.inf))[:m]
    acumulado = 0.0
    for rango, i in enumerate(orden):
        acumulado = max(acumulado, min(1.0, (m - rango) * p[i]))
        ajustado[i] = acumulado
    return ajustado


def _veredicto(fila: pd.Series, alfa: float) -> str:
    vd, vc = fila.get("ventaja_desc"), fila.get("ventaja_comp")
    vd = np.nan if vd is None else float(vd)
    vc = np.nan if vc is None else float(vc)
    if not np.isfinite(vc):
        return "Sin datos"
    if not np.isfinite(vd):  # sin periodo de descubrimiento (activo con poca historia)
        return "Ventaja no concluyente" if vc > 0 else "No supera al azar"
    if vd > 0 and vc > 0 and fila["p_holm"] < alfa:
        return "Supera al azar"
    if vd > 0 and vc <= 0:
        return "Funcionaba y dejó de funcionar"
    if vc > 0:
        return "Ventaja no concluyente"
    return "No supera al azar"
