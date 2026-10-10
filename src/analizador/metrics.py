"""Métricas de rendimiento y riesgo sobre series de retornos diarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

DIAS_HABILES_ANIO = 252


def curva_capital(retornos: pd.Series, capital_inicial: float = 1.0) -> pd.Series:
    """Valor acumulado de una cartera a partir de retornos simples diarios."""
    return capital_inicial * (1 + retornos.fillna(0)).cumprod()


def cagr(retornos: pd.Series, periodos_anio: int = DIAS_HABILES_ANIO) -> float:
    """Tasa de crecimiento anual compuesta."""
    r = retornos.dropna()
    if len(r) == 0:
        return float("nan")
    total = float((1 + r).prod())
    if total <= 0:
        return -1.0
    return total ** (periodos_anio / len(r)) - 1


def volatilidad_anualizada(retornos: pd.Series, periodos_anio: int = DIAS_HABILES_ANIO) -> float:
    r = retornos.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(periodos_anio))


def ratio_sharpe(retornos: pd.Series, tasa_libre_riesgo: float = 0.0,
                 periodos_anio: int = DIAS_HABILES_ANIO) -> float:
    """Sharpe anualizado; ``tasa_libre_riesgo`` es anual."""
    r = retornos.dropna() - tasa_libre_riesgo / periodos_anio
    if len(r) < 2:
        return float("nan")
    desv = r.std(ddof=1)
    # Desviación prácticamente nula (p. ej. liquidez o retorno constante): Sharpe indefinido.
    if np.isnan(desv) or desv < 1e-12:
        return float("nan")
    return float(r.mean() / desv * np.sqrt(periodos_anio))


def max_drawdown(retornos: pd.Series) -> float:
    """Máxima caída desde un máximo de la curva de capital (valor ≤ 0)."""
    r = retornos.dropna()
    if len(r) == 0:
        return float("nan")
    capital = curva_capital(r)
    capital = pd.concat([pd.Series([1.0]), capital.reset_index(drop=True)])
    return float((capital / capital.cummax() - 1).min())


def tasa_acierto(retornos: pd.Series, posiciones: pd.Series) -> float:
    """Proporción de días con exposición (posición del día anterior ≠ 0) con retorno > 0."""
    expuesto = posiciones.shift(1).fillna(0) != 0
    r = retornos[expuesto].dropna()
    if len(r) == 0:
        return float("nan")
    return float((r > 0).mean())


def numero_operaciones(posiciones: pd.Series) -> int:
    """Número de días en los que cambia la posición (incluida la entrada inicial)."""
    cambios = posiciones.fillna(0).diff().fillna(posiciones.fillna(0).iloc[0] if len(posiciones) else 0)
    return int((cambios.abs() > 1e-12).sum())


def rotacion_anual(posiciones: pd.Series, periodos_anio: int = DIAS_HABILES_ANIO) -> float:
    """Rotación (turnover): suma de |Δposición| por año. 1.0 = una vez el capital al año."""
    p = posiciones.fillna(0)
    if len(p) == 0:
        return float("nan")
    cambios = p.diff().fillna(p.iloc[0]).abs().sum()
    return float(cambios / (len(p) / periodos_anio))


def resumen(retornos: pd.Series, posiciones: pd.Series) -> dict[str, float]:
    """Tabla de métricas estándar para una estrategia."""
    return {
        "CAGR": cagr(retornos),
        "Volatilidad anual": volatilidad_anualizada(retornos),
        "Sharpe": ratio_sharpe(retornos),
        "Máx. drawdown": max_drawdown(retornos),
        "Tasa de acierto": tasa_acierto(retornos, posiciones),
        "Nº operaciones": numero_operaciones(posiciones),
        "Rotación anual": rotacion_anual(posiciones),
    }
