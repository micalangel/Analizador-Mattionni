"""Calendario de resultados: cuándo publica cada empresa y cuánto suele moverse ese día.

Usa las fechas y estimaciones de beneficio por acción (BPA) de Yahoo Finance y la reacción del
precio en las publicaciones anteriores (``resultados.estudio_publicaciones``).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd

from .resultados import estudio_publicaciones, obtener_resultados, proxima_publicacion, resumen_publicaciones


def info_publicaciones(ticker: str, precios: pd.Series | None = None) -> dict:
    """Próxima publicación (fecha y BPA estimado) y reacción típica del precio en las anteriores."""
    res = obtener_resultados(ticker)
    prox = proxima_publicacion(res.publicaciones)
    salida = {"ticker": ticker, "fecha": prox["fecha"] if prox else None,
              "bpa_estimado": prox["BPA estimado"] if prox else None}
    if precios is not None and not res.publicaciones.empty:
        r = resumen_publicaciones(estudio_publicaciones(precios, res.publicaciones))
        if r.get("n"):
            salida.update({"publicaciones": r["n"], "movimiento_medio": r["Reacción absoluta media"],
                           "p10": r["Reacción p10"], "p90": r["Reacción p90"],
                           "supera_estimacion": r["% superan estimación"]})
    return salida


def proximas_publicaciones(tickers: list[str], precios: dict[str, pd.Series] | None = None,
                           hilos: int = 4) -> dict[str, dict]:
    """``info_publicaciones`` de varios tickers en paralelo; los que fallan se omiten."""
    precios = precios or {}

    def una(t):
        try:
            return info_publicaciones(t, precios.get(t))
        except Exception:
            return {"ticker": t, "fecha": None}

    with ThreadPoolExecutor(hilos) as ex:
        return {d["ticker"]: d for d in ex.map(una, tickers)}


def tabla_calendario(info: dict[str, dict], hoy: date | None = None) -> pd.DataFrame:
    """Tabla ordenada por fecha de la próxima publicación (las que no tienen fecha, al final)."""
    hoy = pd.Timestamp(hoy or date.today())
    filas = []
    for t, d in info.items():
        fecha = d.get("fecha")
        if fecha is not None and pd.notna(fecha):
            fecha = pd.Timestamp(fecha)
            fecha = fecha.tz_convert("America/New_York").tz_localize(None) if fecha.tzinfo else fecha
        filas.append({**d, "fecha": fecha,
                      "dias": (fecha.normalize() - hoy.normalize()).days if fecha is not None and pd.notna(fecha)
                      else None})
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    return tabla.sort_values("fecha", na_position="last").reset_index(drop=True)
