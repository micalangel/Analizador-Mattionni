"""Análisis técnico objetivo: describe la situación del precio con reglas fijas y explícitas.

Cada lectura dice qué se mide y qué valor tiene hoy («el precio está un 8 % por encima de su media
de 200 días»). No predice nada: en «Experimentos» se ve que estas señales, por sí solas, apenas
aciertan la dirección mejor que el azar. Sirven para describir el estado del precio con precisión.

Soportes y resistencias: niveles donde el precio se ha girado varias veces en el último año
(máximos y mínimos locales agrupados si están a menos de un 2 % entre sí). Cuantas más veces se
ha girado en un nivel, más «fuerte» se considera.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import bandas_bollinger, macd, rsi

ALCISTA, BAJISTA, NEUTRAL = "alcista", "bajista", "neutral"


@dataclass
class Lectura:
    grupo: str
    indicador: str
    valor: str
    lectura: str
    tono: str = NEUTRAL


@dataclass
class Nivel:
    precio: float
    tipo: str        # "soporte" o "resistencia"
    toques: int
    ultimo: pd.Timestamp
    distancia: float  # respecto al precio de hoy


def _pct(x: float) -> str:
    return f"{x:+.1%}".replace(".", ",")


def _num(x: float) -> str:
    """Número con formato español (1.234,56)."""
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def atr(datos: pd.DataFrame, ventana: int = 14) -> pd.Series:
    """Rango verdadero medio (volatilidad diaria en unidades de precio)."""
    c = datos["Close"]
    alto, bajo = datos.get("High", c), datos.get("Low", c)
    rango = pd.concat([alto - bajo, (alto - c.shift(1)).abs(), (bajo - c.shift(1)).abs()], axis=1).max(axis=1)
    return rango.rolling(ventana).mean()


def niveles(datos: pd.DataFrame, sesiones: int = 252, ventana: int = 10, agrupar: float = 0.02,
            maximo: int = 3) -> list[Nivel]:
    """Soportes (debajo del precio) y resistencias (encima) del último año."""
    d = datos.iloc[-sesiones:]
    c = d["Close"]
    alto, bajo = d.get("High", c), d.get("Low", c)
    n = len(d)
    puntos = []
    for i in range(ventana, n - ventana):
        tramo_alto = alto.iloc[i - ventana:i + ventana + 1]
        tramo_bajo = bajo.iloc[i - ventana:i + ventana + 1]
        if alto.iloc[i] == tramo_alto.max():
            puntos.append((float(alto.iloc[i]), d.index[i]))
        if bajo.iloc[i] == tramo_bajo.min():
            puntos.append((float(bajo.iloc[i]), d.index[i]))
    if not puntos:
        return []
    puntos.sort()
    grupos: list[list[tuple[float, pd.Timestamp]]] = [[puntos[0]]]
    for p in puntos[1:]:
        if p[0] <= grupos[-1][0][0] * (1 + agrupar):
            grupos[-1].append(p)
        else:
            grupos.append([p])
    hoy = float(c.iloc[-1])
    todos = []
    for g in grupos:
        precio = float(np.mean([p for p, _ in g]))
        tipo = "soporte" if precio < hoy else "resistencia"
        todos.append(Nivel(precio, tipo, len(g), max(f for _, f in g), precio / hoy - 1))
    soportes = sorted([x for x in todos if x.tipo == "soporte"], key=lambda x: -x.precio)[:maximo]
    resistencias = sorted([x for x in todos if x.tipo == "resistencia"], key=lambda x: x.precio)[:maximo]
    return resistencias[::-1] + soportes


def analizar(datos: pd.DataFrame, referencia: pd.Series | None = None, nombre_referencia: str = "el mercado"
             ) -> list[Lectura]:
    """Lecturas objetivas del estado técnico con el último cierre."""
    c = datos["Close"].astype(float)
    hoy = float(c.iloc[-1])
    lecturas: list[Lectura] = []

    # Tendencia
    for n in (50, 200):
        mm = c.rolling(n).mean()
        if pd.notna(mm.iloc[-1]):
            dist = hoy / mm.iloc[-1] - 1
            lecturas.append(Lectura("Tendencia", f"Precio frente a su media de {n} días", _pct(dist),
                                    f"Está {'por encima' if dist > 0 else 'por debajo'} de la media de {n} días "
                                    f"({'corto-medio' if n == 50 else 'largo'} plazo).", ALCISTA if dist > 0 else BAJISTA))
    mm200 = c.rolling(200).mean()
    if pd.notna(mm200.iloc[-1]) and pd.notna(mm200.iloc[-21]):
        pendiente = mm200.iloc[-1] / mm200.iloc[-21] - 1
        lecturas.append(Lectura("Tendencia", "Inclinación de la media de 200 días (último mes)", _pct(pendiente),
                                "La media de largo plazo " + ("sube." if pendiente > 0.002 else
                                                              "baja." if pendiente < -0.002 else "está plana."),
                                ALCISTA if pendiente > 0.002 else BAJISTA if pendiente < -0.002 else NEUTRAL))
    mm50 = c.rolling(50).mean()
    encima = (mm50 > mm200).dropna()
    if len(encima) > 1:
        cambios = encima[encima != encima.shift(1)].index[1:]
        ultimo = cambios[-1] if len(cambios) else None
        estado = bool(encima.iloc[-1])
        lecturas.append(Lectura("Tendencia", "Cruce de medias 50 / 200",
                                "50 sobre 200" if estado else "50 bajo 200",
                                ("«Cruz dorada»" if estado else "«Cruz de la muerte»")
                                + (f" desde el {ultimo:%d/%m/%Y}." if ultimo is not None else "."),
                                ALCISTA if estado else BAJISTA))

    # Impulso
    r = rsi(c, 14).iloc[-1]
    if pd.notna(r):
        zona = ("sobrecompra (por encima de 70)" if r > 70 else "sobreventa (por debajo de 30)" if r < 30
                else "zona neutral (entre 30 y 70)")
        lecturas.append(Lectura("Impulso", "RSI (14 días)", f"{r:.0f}", f"En {zona}.",
                                BAJISTA if r > 70 else ALCISTA if r < 30 else NEUTRAL))
    m = macd(c)
    if pd.notna(m["histograma"].iloc[-1]):
        h, h_prev = m["histograma"].iloc[-1], m["histograma"].iloc[-2]
        lecturas.append(Lectura("Impulso", "MACD frente a su señal", "por encima" if h > 0 else "por debajo",
                                ("Impulso positivo" if h > 0 else "Impulso negativo")
                                + (", aumentando." if abs(h) > abs(h_prev) else ", perdiendo fuerza."),
                                ALCISTA if h > 0 else BAJISTA))
    for etiqueta, n in (("1 mes", 21), ("3 meses", 63), ("6 meses", 126), ("1 año", 252), ("5 años", 1260)):
        if len(c) > n:
            rend = hoy / c.iloc[-n - 1] - 1
            texto = f"Ha {'subido' if rend > 0 else 'bajado'} un {_pct(abs(rend))[1:]}"
            if referencia is not None and len(referencia.dropna()) > n:
                ref = referencia.dropna()
                rend_ref = ref.iloc[-1] / ref.iloc[-n - 1] - 1
                texto += f"; {nombre_referencia}, un {_pct(rend_ref)}"
                tono = ALCISTA if rend > rend_ref else BAJISTA
            else:
                tono = ALCISTA if rend > 0 else BAJISTA
            lecturas.append(Lectura("Rendimiento", f"Últimos {etiqueta}" if n != 252 else "Último año",
                                    _pct(rend), texto + ".", tono))

    # Volatilidad
    lr = np.log(c / c.shift(1))
    v20, v252 = lr.iloc[-20:].std() * np.sqrt(252), lr.iloc[-252:].std() * np.sqrt(252)
    if pd.notna(v20) and pd.notna(v252) and v252 > 0:
        rel = v20 / v252
        lecturas.append(Lectura("Volatilidad", "Volatilidad del último mes (anualizada)", f"{v20:.0%}",
                                "Más agitada de lo normal para ella." if rel > 1.3 else
                                "Más tranquila de lo normal para ella." if rel < 0.75 else
                                f"Normal para ella (su media del año es {v252:.0%}).", NEUTRAL))
    a = atr(datos).iloc[-1]
    if pd.notna(a):
        lecturas.append(Lectura("Volatilidad", "Movimiento medio diario (ATR 14)", f"{a / hoy:.1%}".replace(".", ","),
                                f"En un día normal se mueve unos {_num(a)} de precio arriba o abajo.", NEUTRAL))
    bb = bandas_bollinger(c)
    if pd.notna(bb["superior"].iloc[-1]):
        pos = (hoy - bb["inferior"].iloc[-1]) / (bb["superior"].iloc[-1] - bb["inferior"].iloc[-1])
        lecturas.append(Lectura("Volatilidad", "Posición en las bandas de Bollinger", f"{pos:.0%}",
                                "Toca o supera la banda superior." if pos >= 1 else
                                "Toca o perfora la banda inferior." if pos <= 0 else
                                "Dentro de las bandas (0 % = banda inferior, 100 % = superior).",
                                BAJISTA if pos >= 1 else ALCISTA if pos <= 0 else NEUTRAL))

    # Máximos, mínimos y volumen
    ano = c.iloc[-252:]
    lecturas.append(Lectura("Niveles", "Distancia al máximo del último año", _pct(hoy / ano.max() - 1),
                            f"Máximo del año: {_num(ano.max())} ({ano.idxmax():%d/%m/%Y}).", NEUTRAL))
    lecturas.append(Lectura("Niveles", "Distancia al mínimo del último año", _pct(hoy / ano.min() - 1),
                            f"Mínimo del año: {_num(ano.min())} ({ano.idxmin():%d/%m/%Y}).", NEUTRAL))
    if "Volume" in datos and (datos["Volume"].iloc[-60:] > 0).all():
        v = datos["Volume"].astype(float)
        rel = v.iloc[-1] / v.iloc[-21:-1].mean()
        lecturas.append(Lectura("Volumen", "Volumen del último día frente a su media de 20 días",
                                f"{rel:.1f}×".replace(".", ","),
                                "Mucho más alto de lo normal." if rel > 2 else "Más bajo de lo normal." if rel < 0.5
                                else "Normal.", NEUTRAL))
    return lecturas


def resumen(lecturas: list[Lectura]) -> tuple[int, int, int]:
    """Número de lecturas alcistas, bajistas y neutrales (solo para dar una visión de conjunto)."""
    return (sum(l.tono == ALCISTA for l in lecturas), sum(l.tono == BAJISTA for l in lecturas),
            sum(l.tono == NEUTRAL for l in lecturas))
