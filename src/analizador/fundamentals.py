"""Ratios fundamentales (módulo opcional).

Los datos proceden de Yahoo Finance a través de yfinance: pueden estar retrasados,
incompletos o calculados con criterios distintos según la empresa. Los ETF y los
índices normalmente no tienen estos ratios.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .data import normalizar_tickers

AVISO_FUNDAMENTALES = (
    "Los datos fundamentales proceden de Yahoo Finance y pueden estar retrasados, "
    "incompletos o no ser comparables entre empresas. Verifícalos en los informes oficiales."
)

COLUMNAS = ["Nombre", "PER", "Precio/Valor contable", "Deuda/Patrimonio",
            "Margen de beneficio", "Rentabilidad FCF", "Moneda",
            "PER esperado", "BPA 12m", "Beneficio 12m", "Capitalización"]

ObtenerInfo = Callable[[str], dict]


def _info_yfinance(ticker: str) -> dict:
    import yfinance as yf

    return yf.Ticker(ticker).info or {}


def _numero(valor) -> float:
    try:
        x = float(valor)
    except (TypeError, ValueError):
        return np.nan
    return x if np.isfinite(x) else np.nan


def ratios_desde_info(info: dict) -> dict:
    """Extrae los ratios de un diccionario ``info`` de yfinance."""
    deuda = _numero(info.get("debtToEquity"))
    fcf = _numero(info.get("freeCashflow"))
    capitalizacion = _numero(info.get("marketCap"))
    return {
        "Nombre": info.get("shortName") or info.get("longName") or "",
        "PER": _numero(info.get("trailingPE")),
        "Precio/Valor contable": _numero(info.get("priceToBook")),
        # yfinance publica deuda/patrimonio en porcentaje (150 = 1,5 veces).
        "Deuda/Patrimonio": deuda / 100 if np.isfinite(deuda) else np.nan,
        "Margen de beneficio": _numero(info.get("profitMargins")),
        "Rentabilidad FCF": fcf / capitalizacion if capitalizacion and capitalizacion > 0 else np.nan,
        "Moneda": info.get("financialCurrency") or info.get("currency") or "",
        # Con pérdidas no hay PER; el esperado usa el beneficio que prevén los analistas.
        "PER esperado": _numero(info.get("forwardPE")),
        "BPA 12m": _numero(info.get("trailingEps")),
        "Beneficio 12m": _numero(info.get("netIncomeToCommon")),
        "Capitalización": capitalizacion,
    }


def obtener_fundamentales(
    tickers: str | Iterable[str],
    obtener_info: ObtenerInfo | None = None,
) -> pd.DataFrame:
    """Tabla comparativa de ratios, una fila por ticker. Los fallos dan filas vacías."""
    obtener_info = obtener_info or _info_yfinance
    filas = {}
    for t in normalizar_tickers(tickers):
        try:
            info = obtener_info(t)
        except Exception:
            info = {}
        filas[t] = ratios_desde_info(info)
    return pd.DataFrame.from_dict(filas, orient="index", columns=COLUMNAS)
