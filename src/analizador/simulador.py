"""Simulador educativo de estrategias: dónde habría comprado y vendido una regla clásica.

Todo es simulado, sin dinero real ni órdenes a ningún bróker. Sirve para aprender cómo
funcionan las señales de compra y venta y, sobre todo, cuántas veces fallan.

Reglas sin mirar al futuro: la posición se decide con el cierre de cada vela y se aplica a la
vela siguiente. Cada compra o venta paga comisión y deslizamiento.

Comparación con el azar: se repite la simulación desplazando las entradas y salidas de la
regla a momentos al azar (mismo tiempo invertido y mismas duraciones de las operaciones). Si
la regla no queda por encima de la mayoría de esas versiones al azar, su resultado se explica
por estar invertido en un buen o mal momento, no por acertar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .indicators import rsi


@dataclass(frozen=True)
class Regla:
    nombre: str
    explicacion: str
    parametros: dict                     # nombre -> (mínimo, máximo, por defecto)
    funcion: Callable[..., pd.Series]    # (datos, **parametros) -> posición 0/1


def _cruce_medias(datos, rapida=50, lenta=200):
    c = datos["Close"]
    return (c.rolling(int(rapida)).mean() > c.rolling(int(lenta)).mean()).astype(float)


def _tendencia(datos, media=200):
    c = datos["Close"]
    return (c > c.rolling(int(media)).mean()).astype(float)


def _ruptura(datos, entrada=55, salida=20):
    """Compra al superar el máximo de las últimas N velas; vende al perder el mínimo de las últimas M."""
    c = datos["Close"]
    maximo = c.shift(1).rolling(int(entrada)).max()
    minimo = c.shift(1).rolling(int(salida)).min()
    pos, dentro = np.zeros(len(c)), False
    for i, (precio, mx, mn) in enumerate(zip(c.to_numpy(), maximo.to_numpy(), minimo.to_numpy())):
        if not dentro and np.isfinite(mx) and precio > mx:
            dentro = True
        elif dentro and np.isfinite(mn) and precio < mn:
            dentro = False
        pos[i] = float(dentro)
    return pd.Series(pos, index=c.index)


def _rsi(datos, entrada=30, salida=70):
    valores = rsi(datos["Close"], 14).to_numpy()
    pos, dentro = np.zeros(len(valores)), False
    for i, v in enumerate(valores):
        if np.isfinite(v):
            if not dentro and v < entrada:
                dentro = True
            elif dentro and v > salida:
                dentro = False
        pos[i] = float(dentro)
    return pd.Series(pos, index=datos.index)


def _rebote(datos, caida=20, objetivo=15, maximo_velas=120):
    """Compra cuando cae un X % desde su máximo reciente; vende al ganar un Y % o tras N velas."""
    c = datos["Close"].to_numpy()
    maximo = datos["Close"].rolling(252, min_periods=20).max().to_numpy()
    pos, dentro, precio_entrada, velas = np.zeros(len(c)), False, 0.0, 0
    for i in range(len(c)):
        if not dentro:
            if np.isfinite(maximo[i]) and c[i] <= maximo[i] * (1 - caida / 100):
                dentro, precio_entrada, velas = True, c[i], 0
        else:
            velas += 1
            if c[i] >= precio_entrada * (1 + objetivo / 100) or velas >= maximo_velas:
                dentro = False
        pos[i] = float(dentro)
    return pd.Series(pos, index=datos.index)


REGLAS = {
    "cruce": Regla("Cruce de medias", "Está dentro mientras la media rápida está por encima de la lenta "
                   "(la famosa «cruz dorada»).", {"rapida": (5, 100, 50), "lenta": (20, 300, 200)}, _cruce_medias),
    "tendencia": Regla("Seguir la tendencia", "Está dentro mientras el precio está por encima de su media.",
                       {"media": (10, 300, 200)}, _tendencia),
    "ruptura": Regla("Ruptura de máximos", "Compra cuando el precio supera el máximo de las últimas velas y vende "
                     "cuando pierde el mínimo reciente (reglas de las «tortugas»).",
                     {"entrada": (10, 120, 55), "salida": (5, 60, 20)}, _ruptura),
    "rsi": Regla("RSI: sobreventa y sobrecompra", "Compra cuando el RSI baja de un nivel (sobreventa) y vende "
                 "cuando sube por encima de otro (sobrecompra).", {"entrada": (10, 45, 30), "salida": (55, 90, 70)},
                 _rsi),
    "rebote": Regla("Comprar caídas fuertes", "Compra cuando cae un X % desde su máximo y vende al ganar un Y % o "
                    "pasadas N velas.", {"caida": (5, 50, 20), "objetivo": (5, 50, 15), "maximo_velas": (10, 500, 120)},
                    _rebote),
}


@dataclass
class Simulacion:
    posicion: pd.Series
    capital_regla: pd.Series
    capital_mantener: pd.Series
    operaciones: pd.DataFrame
    percentil_azar: float | None = None
    resultados_azar: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def total_regla(self) -> float:
        return float(self.capital_regla.iloc[-1] - 1)

    @property
    def total_mantener(self) -> float:
        return float(self.capital_mantener.iloc[-1] - 1)

    @property
    def tiempo_dentro(self) -> float:
        return float(self.posicion.shift(1).fillna(0).mean())


def _capital(precios: pd.Series, posicion: pd.Series, coste: float) -> pd.Series:
    r = precios.pct_change().fillna(0.0)
    cambios = posicion.diff().fillna(posicion.iloc[0]).abs()
    neto = posicion.shift(1).fillna(0.0) * r - cambios * coste
    return (1 + neto).cumprod()


def operaciones(precios: pd.Series, posicion: pd.Series, coste: float) -> pd.DataFrame:
    """Lista de operaciones: entrada y salida (o abierta), velas y resultado neto de costes."""
    p = posicion.fillna(0).to_numpy()
    filas, inicio = [], None
    for i in range(len(p)):
        if p[i] == 1 and (i == 0 or p[i - 1] == 0):
            inicio = i
        if inicio is not None and (p[i] == 0 and p[i - 1] == 1):
            filas.append((inicio, i, False))
            inicio = None
    if inicio is not None:
        filas.append((inicio, len(p) - 1, True))
    datos = []
    for a, b, abierta in filas:
        bruto = precios.iloc[b] / precios.iloc[a] - 1
        neto = (1 + bruto) * (1 - coste) * (1 - (0 if abierta else coste)) - 1
        datos.append({"entrada": precios.index[a], "precio_entrada": precios.iloc[a],
                      "salida": precios.index[b], "precio_salida": precios.iloc[b],
                      "velas": b - a, "resultado": neto, "abierta": abierta})
    return pd.DataFrame(datos)


def simular(datos: pd.DataFrame, regla: str, parametros: dict | None = None, coste: float = 0.0015,
            repeticiones_azar: int = 300, semilla: int = 0, desde: pd.Timestamp | None = None) -> Simulacion:
    """Aplica la regla, contabiliza costes y la compara con comprar y mantener y con el azar.

    La regla se calcula con toda la historia de ``datos`` (las medias necesitan días previos) y se
    evalúa solo desde ``desde``; si ese día ya estaba dentro, cuenta como una compra ese día.
    """
    r = REGLAS[regla]
    params = {k: v[2] for k, v in r.parametros.items()} | (parametros or {})
    posicion = r.funcion(datos, **params).reindex(datos.index).fillna(0.0)
    if desde is not None:
        datos, posicion = datos.loc[desde:], posicion.loc[desde:]
    precios = datos["Close"].astype(float)
    capital = _capital(precios, posicion, coste)
    mantener = precios / precios.iloc[0]
    sim = Simulacion(posicion, capital, mantener, operaciones(precios, posicion, coste))

    # Azar: la misma secuencia de entradas y salidas desplazada a un momento aleatorio (circular).
    if repeticiones_azar and posicion.diff().abs().sum() > 0 and len(precios) > 50:
        rng = np.random.default_rng(semilla)
        p = posicion.to_numpy()
        totales = []
        for _ in range(repeticiones_azar):
            desplazada = pd.Series(np.roll(p, rng.integers(1, len(p))), index=precios.index)
            totales.append(_capital(precios, desplazada, coste).iloc[-1] - 1)
        sim.resultados_azar = np.array(totales)
        sim.percentil_azar = float((sim.resultados_azar < sim.total_regla).mean())
    return sim


def max_caida(capital: pd.Series) -> float:
    return float((capital / capital.cummax() - 1).min())
