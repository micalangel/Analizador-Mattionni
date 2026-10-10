"""Indicadores técnicos.

Todos son causales: el valor del día t solo usa datos hasta el cierre de t
(ventanas móviles y medias exponenciales hacia atrás, nunca centradas).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DIAS_HABILES_ANIO = 252


def media_movil(precios: pd.Series, ventana: int = 20) -> pd.Series:
    """Media móvil simple de ``ventana`` días."""
    return precios.rolling(ventana, min_periods=ventana).mean()


def media_movil_exponencial(precios: pd.Series, ventana: int = 20) -> pd.Series:
    """Media móvil exponencial con span = ``ventana``."""
    return precios.ewm(span=ventana, adjust=False, min_periods=ventana).mean()


def rsi(precios: pd.Series, ventana: int = 14) -> pd.Series:
    """Índice de fuerza relativa (RSI) con suavizado de Wilder, en escala 0-100."""
    delta = precios.diff()
    subidas = delta.clip(lower=0)
    bajadas = -delta.clip(upper=0)
    alfa = 1 / ventana
    media_sub = subidas.ewm(alpha=alfa, adjust=False, min_periods=ventana).mean()
    media_baj = bajadas.ewm(alpha=alfa, adjust=False, min_periods=ventana).mean()
    rs = media_sub / media_baj
    resultado = 100 - 100 / (1 + rs)
    # Sin bajadas en la ventana: RSI = 100; sin movimiento alguno: 50.
    resultado = resultado.mask((media_baj == 0) & (media_sub > 0), 100.0)
    resultado = resultado.mask((media_baj == 0) & (media_sub == 0), 50.0)
    return resultado.rename("RSI")


def macd(precios: pd.Series, rapida: int = 12, lenta: int = 26, senal: int = 9) -> pd.DataFrame:
    """MACD: diferencia de medias exponenciales, su línea de señal y el histograma."""
    ema_rapida = precios.ewm(span=rapida, adjust=False, min_periods=rapida).mean()
    ema_lenta = precios.ewm(span=lenta, adjust=False, min_periods=lenta).mean()
    linea = ema_rapida - ema_lenta
    linea_senal = linea.ewm(span=senal, adjust=False, min_periods=senal).mean()
    return pd.DataFrame({"macd": linea, "senal": linea_senal, "histograma": linea - linea_senal})


def bandas_bollinger(precios: pd.Series, ventana: int = 20, num_desv: float = 2.0) -> pd.DataFrame:
    """Bandas de Bollinger: media móvil ± ``num_desv`` desviaciones típicas móviles."""
    media = precios.rolling(ventana, min_periods=ventana).mean()
    desv = precios.rolling(ventana, min_periods=ventana).std(ddof=0)
    return pd.DataFrame({
        "media": media,
        "superior": media + num_desv * desv,
        "inferior": media - num_desv * desv,
    })


def retornos_log(precios: pd.Series) -> pd.Series:
    """Retornos logarítmicos diarios: ln(P_t / P_{t-1})."""
    return np.log(precios / precios.shift(1))


def volatilidad_movil(precios: pd.Series, ventana: int = 20, anualizar: bool = True) -> pd.Series:
    """Desviación típica móvil de los retornos logarítmicos (anualizada con √252)."""
    vol = retornos_log(precios).rolling(ventana, min_periods=ventana).std()
    return vol * np.sqrt(DIAS_HABILES_ANIO) if anualizar else vol


def drawdown(precios: pd.Series) -> pd.Series:
    """Caída relativa desde el máximo histórico acumulado hasta cada día (≤ 0)."""
    return precios / precios.cummax() - 1


def calcular_todos(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla con los indicadores habituales calculados sobre la columna Close."""
    c = df["Close"]
    bb = bandas_bollinger(c)
    m = macd(c)
    return pd.DataFrame({
        "Close": c,
        "MM20": media_movil(c, 20),
        "MM50": media_movil(c, 50),
        "MM200": media_movil(c, 200),
        "BB_media": bb["media"],
        "BB_superior": bb["superior"],
        "BB_inferior": bb["inferior"],
        "RSI14": rsi(c, 14),
        "MACD": m["macd"],
        "MACD_senal": m["senal"],
        "MACD_hist": m["histograma"],
        "Vol20_anual": volatilidad_movil(c, 20),
        "Drawdown": drawdown(c),
    })
