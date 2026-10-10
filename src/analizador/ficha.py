"""Datos de una empresa para el analizador: perfil, valoración, rentabilidad, balance, dividendos,
accionistas y opinión de los analistas (todo de Yahoo Finance).

La opinión y los precios objetivo de los analistas son datos de terceros: se muestran citando la
fuente y el número de analistas, no son una predicción de esta herramienta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SECTORES_ES = {"Technology": "Tecnología", "Consumer Cyclical": "Consumo cíclico",
               "Consumer Defensive": "Consumo básico", "Financial Services": "Servicios financieros",
               "Healthcare": "Salud", "Industrials": "Industria", "Energy": "Energía", "Utilities": "Servicios públicos",
               "Basic Materials": "Materiales", "Real Estate": "Inmobiliario",
               "Communication Services": "Comunicaciones"}
RECOMENDACIONES = {"strong_buy": "Compra fuerte", "buy": "Compra", "hold": "Mantener", "sell": "Venta",
                   "strong_sell": "Venta fuerte", "underperform": "Peor que el mercado", "outperform": "Mejor que el mercado"}


@dataclass
class Ficha:
    ticker: str
    info: dict
    dividendos: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))      # pagos individuales
    recomendaciones: pd.DataFrame = field(default_factory=pd.DataFrame)
    anual: pd.DataFrame = field(default_factory=pd.DataFrame)                          # estados anuales

    def v(self, clave: str):
        x = self.info.get(clave)
        try:
            return float(x) if x is not None and np.isfinite(float(x)) else None
        except (TypeError, ValueError):
            return x

    @property
    def nombre(self) -> str:
        return self.info.get("longName") or self.info.get("shortName") or self.ticker

    @property
    def sector(self) -> str:
        s = self.info.get("sector") or "—"
        return SECTORES_ES.get(s, s)

    @property
    def moneda(self) -> str:
        return self.info.get("currency") or ""


def rentabilidad_dividendo(dividendos: pd.Series, precio: float) -> float | None:
    """Dividendos cobrados en los últimos 12 meses / precio de hoy (más fiable que el dato de Yahoo)."""
    if dividendos is None or dividendos.empty or not precio:
        return None
    d = dividendos.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None) if d.index.tz else d.index
    ultimo_ano = d[d.index > d.index.max() - pd.DateOffset(years=1)]
    if d.index.max() < pd.Timestamp.today() - pd.DateOffset(months=18):
        return 0.0  # dejó de pagar
    return float(ultimo_ano.sum() / precio)


def dividendos_por_ano(dividendos: pd.Series) -> pd.Series:
    """Suma de dividendos pagados cada año natural (sin el año en curso, que está incompleto)."""
    if dividendos is None or dividendos.empty:
        return pd.Series(dtype=float)
    d = dividendos.copy()
    d.index = pd.DatetimeIndex(d.index).tz_localize(None) if d.index.tz else d.index
    por_ano = d.groupby(d.index.year).sum()
    return por_ano[por_ano.index < pd.Timestamp.today().year]


def anos_subiendo_dividendo(por_ano: pd.Series) -> int:
    """Años seguidos (hasta el último completo) en que el dividendo no ha bajado."""
    racha = 0
    valores = por_ano.to_numpy()
    for i in range(len(valores) - 1, 0, -1):
        if valores[i] >= valores[i - 1] * 0.995:
            racha += 1
        else:
            break
    return racha


def descargar(ticker: str) -> Ficha:
    import yfinance as yf

    t = yf.Ticker(ticker)

    def seguro(f, defecto):
        try:
            r = f()
            return defecto if r is None else r
        except Exception:
            return defecto

    info = seguro(lambda: t.info, {}) or {}
    ficha = Ficha(ticker, info, seguro(lambda: t.dividends, pd.Series(dtype=float)),
                  seguro(lambda: t.recommendations, pd.DataFrame()))
    from .resultados import obtener_resultados

    ficha.anual = seguro(lambda: obtener_resultados(ticker).anual, pd.DataFrame())
    return ficha
