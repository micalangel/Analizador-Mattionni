"""Motor de backtesting walk-forward.

Reglas (ver docs/METHODOLOGY.md):
- En cada día t la estrategia recibe SOLO los datos hasta el cierre de t y devuelve
  una posición (fracción del capital: 1 = invertido, 0 = liquidez, -1 = corto).
- La posición decidida al cierre de t se aplica al retorno de t a t+1.
- Cada cambio de posición paga (coste + deslizamiento) × |Δposición|, cargado el día t.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics

Estrategia = Callable[[pd.DataFrame], float]


@dataclass
class ResultadoBacktest:
    nombre: str
    posiciones: pd.Series        # decidida al cierre de cada día
    retornos_activo: pd.Series   # retorno simple del activo de t-1 a t
    costes: pd.Series            # costes cargados cada día
    retornos: pd.Series          # retorno neto de la estrategia cada día
    inicio: pd.Timestamp         # primer día evaluado
    parametros: dict = field(default_factory=dict)

    @property
    def capital(self) -> pd.Series:
        return metrics.curva_capital(self.retornos)

    def metricas(self) -> dict[str, float]:
        return metrics.resumen(self.retornos, self.posiciones)


def ejecutar_backtest(
    datos: pd.DataFrame,
    estrategia: Estrategia,
    coste: float = 0.001,
    deslizamiento: float = 0.0005,
    historia_minima: int = 200,
    limite_posicion: float = 1.0,
    nombre: str = "Estrategia",
) -> ResultadoBacktest:
    """Evalúa ``estrategia`` día a día sin acceso a datos futuros.

    ``coste`` y ``deslizamiento`` se expresan como fracción del importe operado
    (0.001 = 0,1 % = 10 puntos básicos). Antes de ``historia_minima`` días la
    posición es 0 y esos días no cuentan en las métricas.
    """
    if "Close" not in datos.columns:
        raise ValueError("Los datos deben tener una columna 'Close'.")
    n = len(datos)
    if n <= historia_minima + 1:
        raise ValueError(f"Hacen falta más de {historia_minima + 1} días de datos (hay {n}).")

    posiciones = np.zeros(n)
    for t in range(historia_minima, n):
        ventana = datos.iloc[: t + 1]  # datos disponibles al cierre de t
        p = float(estrategia(ventana))
        posiciones[t] = 0.0 if not np.isfinite(p) else float(np.clip(p, -limite_posicion, limite_posicion))

    return _contabilizar(datos, posiciones, coste, deslizamiento, historia_minima, nombre)


def _contabilizar(datos, posiciones, coste, deslizamiento, historia_minima, nombre) -> ResultadoBacktest:
    indice = datos.index
    pos = pd.Series(posiciones, index=indice, name="posicion")
    r_activo = datos["Close"].pct_change().fillna(0.0)
    cambios = pos.diff().fillna(pos.iloc[0]).abs()
    costes = cambios * (coste + deslizamiento)
    neto = pos.shift(1).fillna(0.0) * r_activo - costes

    tramo = slice(historia_minima, None)
    return ResultadoBacktest(
        nombre=nombre,
        posiciones=pos.iloc[tramo],
        retornos_activo=r_activo.iloc[tramo],
        costes=costes.iloc[tramo],
        retornos=neto.iloc[tramo].rename("retorno"),
        inicio=indice[historia_minima],
        parametros={"coste": coste, "deslizamiento": deslizamiento, "historia_minima": historia_minima},
    )


# --- Estrategias de ejemplo y referencias ------------------------------------------

def comprar_y_mantener() -> Estrategia:
    return lambda datos: 1.0


def posicion_constante(nivel: float = 0.5) -> Estrategia:
    """Línea base ingenua: misma exposición todos los días, sin reaccionar a los datos."""
    return lambda datos: nivel


def cruce_medias(rapida: int = 50, lenta: int = 200) -> Estrategia:
    """1 si la media rápida está por encima de la lenta al cierre de hoy, si no 0."""
    if rapida >= lenta:
        raise ValueError("La media rápida debe ser más corta que la lenta.")

    def estrategia(datos: pd.DataFrame) -> float:
        c = datos["Close"]
        if len(c) < lenta:
            return 0.0
        return 1.0 if c.iloc[-rapida:].mean() > c.iloc[-lenta:].mean() else 0.0

    return estrategia


def reversion_rsi(ventana: int = 14, entrada: float = 30, salida: float = 70) -> Estrategia:
    """Entra cuando el RSI cae por debajo de ``entrada`` y sale cuando supera ``salida``.

    La decisión depende solo del RSI hasta hoy; el estado (dentro/fuera) se reconstruye
    a partir de esos mismos datos, por lo que no hay memoria oculta entre llamadas.
    """
    from .indicators import rsi

    def estrategia(datos: pd.DataFrame) -> float:
        valores = rsi(datos["Close"].iloc[-(ventana * 20):], ventana).dropna()
        dentro = False
        for v in valores:
            if not dentro and v < entrada:
                dentro = True
            elif dentro and v > salida:
                dentro = False
        return 1.0 if dentro else 0.0

    return estrategia


ESTRATEGIAS = {
    "Cruce de medias (50/200)": lambda: cruce_medias(50, 200),
    "Cruce de medias (20/100)": lambda: cruce_medias(20, 100),
    "Reversión RSI (30/70)": lambda: reversion_rsi(14, 30, 70),
}


def comparar_con_referencias(
    datos: pd.DataFrame,
    estrategia: Estrategia,
    nombre: str = "Estrategia",
    nivel_ingenuo: float = 0.5,
    **kwargs,
) -> dict[str, ResultadoBacktest]:
    """Ejecuta la estrategia junto a comprar y mantener y a una posición constante."""
    return {
        nombre: ejecutar_backtest(datos, estrategia, nombre=nombre, **kwargs),
        "Comprar y mantener": ejecutar_backtest(datos, comprar_y_mantener(),
                                                nombre="Comprar y mantener", **kwargs),
        f"Posición constante {nivel_ingenuo:g}": ejecutar_backtest(
            datos, posicion_constante(nivel_ingenuo),
            nombre=f"Posición constante {nivel_ingenuo:g}", **kwargs),
    }


def tabla_metricas(resultados: dict[str, ResultadoBacktest]) -> pd.DataFrame:
    return pd.DataFrame({k: r.metricas() for k, r in resultados.items()}).T
