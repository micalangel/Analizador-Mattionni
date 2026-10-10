"""Estudio de resultados financieros.

1. Estados financieros resumidos (trimestrales y anuales): ingresos, beneficios, BPA,
   flujo de caja, deuda; márgenes y crecimiento interanual.
2. Historial de publicaciones de resultados: BPA estimado por los analistas frente al
   publicado (sorpresa) y la reacción del precio en la sesión de la publicación.

Fuente: Yahoo Finance vía yfinance. Los datos pueden estar retrasados, incompletos o
reexpresados, y los ETF no tienen estados financieros propios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .eventos import reaccion_a_eventos

AVISO_RESULTADOS = (
    "Estados financieros y estimaciones de analistas procedentes de Yahoo Finance: pueden estar "
    "retrasados, incompletos o reexpresados. Contrástalos con los informes oficiales de la empresa."
)

# Nombre en yfinance -> nombre mostrado
PARTIDAS = {
    "Total Revenue": "Ingresos",
    "Gross Profit": "Beneficio bruto",
    "Operating Income": "Beneficio operativo",
    "Net Income": "Beneficio neto",
    "Diluted EPS": "BPA diluido",
    "Operating Cash Flow": "Flujo de caja operativo",
    "Free Cash Flow": "Flujo de caja libre",
    "Total Debt": "Deuda total",
    "Stockholders Equity": "Patrimonio neto",
}


@dataclass
class DatosResultados:
    trimestral: pd.DataFrame       # filas = periodos (ascendente), columnas = PARTIDAS
    anual: pd.DataFrame
    publicaciones: pd.DataFrame    # índice = fecha/hora de publicación


def _tabla_partidas(*estados: pd.DataFrame) -> pd.DataFrame:
    """Une estados de yfinance (partidas en filas, periodos en columnas) y se queda con PARTIDAS."""
    partes = []
    for est in estados:
        if est is None or est.empty:
            continue
        filas = [p for p in PARTIDAS if p in est.index]
        parte = est.loc[filas].T.dropna(axis=1, how="all")
        if not parte.empty:
            partes.append(parte)
    if not partes:
        return pd.DataFrame(columns=list(PARTIDAS.values()))
    tabla = pd.concat(partes, axis=1)
    tabla = tabla.loc[:, ~tabla.columns.duplicated()]
    tabla.index = pd.to_datetime(tabla.index)
    tabla = tabla.sort_index().apply(pd.to_numeric, errors="coerce")
    return tabla.reindex(columns=list(PARTIDAS)).rename(columns=PARTIDAS)


def normalizar_publicaciones(df: pd.DataFrame | None) -> pd.DataFrame:
    """Convierte la tabla de ``get_earnings_dates`` a columnas en español, orden ascendente."""
    columnas = ["BPA estimado", "BPA publicado", "Sorpresa %"]
    if df is None or df.empty:
        return pd.DataFrame(columns=columnas)
    renombrar = {"EPS Estimate": "BPA estimado", "Reported EPS": "BPA publicado", "Surprise(%)": "Sorpresa %"}
    out = df.rename(columns=renombrar).reindex(columns=columnas)
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out[~out.index.duplicated()].sort_index()
    out.index.name = "Publicación"
    return out


def _descargar_yfinance(ticker: str) -> DatosResultados:
    import yfinance as yf

    t = yf.Ticker(ticker)

    def seguro(f):
        try:
            return f()
        except Exception:
            return None

    trimestral = _tabla_partidas(seguro(lambda: t.quarterly_income_stmt),
                                 seguro(lambda: t.quarterly_cashflow),
                                 seguro(lambda: t.quarterly_balance_sheet))
    anual = _tabla_partidas(seguro(lambda: t.income_stmt), seguro(lambda: t.cashflow),
                            seguro(lambda: t.balance_sheet))
    publicaciones = normalizar_publicaciones(seguro(lambda: t.get_earnings_dates(limit=24)))
    return DatosResultados(trimestral, anual, publicaciones)


def obtener_resultados(ticker: str, descargador: Callable[[str], DatosResultados] | None = None
                       ) -> DatosResultados:
    try:
        return (descargador or _descargar_yfinance)(ticker.strip().upper())
    except Exception:
        vacio = pd.DataFrame(columns=list(PARTIDAS.values()))
        return DatosResultados(vacio, vacio.copy(), normalizar_publicaciones(None))


def margenes_y_crecimiento(tabla: pd.DataFrame, periodos_por_anio: int) -> pd.DataFrame:
    """Márgenes sobre ingresos y crecimiento interanual de ingresos, beneficio neto y BPA.

    ``periodos_por_anio`` = 4 para datos trimestrales (compara con el mismo trimestre del año
    anterior) y 1 para anuales.
    """
    ingresos = tabla["Ingresos"]
    out = pd.DataFrame(index=tabla.index)
    out["Margen bruto"] = tabla["Beneficio bruto"] / ingresos
    out["Margen operativo"] = tabla["Beneficio operativo"] / ingresos
    out["Margen neto"] = tabla["Beneficio neto"] / ingresos
    out["Margen FCF"] = tabla["Flujo de caja libre"] / ingresos

    def crecimiento(s: pd.Series) -> pd.Series:
        previo = s.shift(periodos_por_anio)
        # Con base negativa o nula el porcentaje no tiene sentido.
        return ((s - previo) / previo).where(previo > 0)

    out["Crec. ingresos (interanual)"] = crecimiento(ingresos)
    out["Crec. beneficio neto (interanual)"] = crecimiento(tabla["Beneficio neto"])
    out["Crec. BPA (interanual)"] = crecimiento(tabla["BPA diluido"])
    out["Deuda / Patrimonio"] = tabla["Deuda total"] / tabla["Patrimonio neto"].where(tabla["Patrimonio neto"] > 0)
    return out.replace([np.inf, -np.inf], np.nan)


def estudio_publicaciones(precios: pd.Series, publicaciones: pd.DataFrame) -> pd.DataFrame:
    """Une cada publicación con la reacción del precio (ver ``eventos.reaccion_a_eventos``)."""
    if publicaciones.empty:
        return publicaciones.assign(reaccion=pd.Series(dtype=float))
    reac = reaccion_a_eventos(precios, publicaciones.index)
    reac.index = publicaciones.index
    return publicaciones.join(reac)


def resumen_publicaciones(estudio: pd.DataFrame) -> dict:
    """Estadísticas descriptivas de las publicaciones ya conocidas (con BPA y reacción)."""
    hechas = estudio.dropna(subset=["BPA publicado", "reaccion"])
    n = len(hechas)
    if n == 0:
        return {"n": 0}
    sorpresa = hechas["Sorpresa %"]
    reaccion = hechas["reaccion"]
    misma_direccion = (np.sign(sorpresa) == np.sign(reaccion))[sorpresa != 0]
    return {
        "n": n,
        "% superan estimación": float((sorpresa > 0).mean()),
        "Sorpresa media %": float(sorpresa.mean()),
        "Reacción media": float(reaccion.mean()),
        "Reacción absoluta media": float(reaccion.abs().mean()),
        "Reacción p10": float(reaccion.quantile(0.10)),
        "Reacción p90": float(reaccion.quantile(0.90)),
        "|z| medio": float(hechas["z"].abs().mean()),
        "% reacción en la dirección de la sorpresa": float(misma_direccion.mean()) if len(misma_direccion) else np.nan,
        "Correlación sorpresa-reacción": float(sorpresa.corr(reaccion)) if n >= 3 else np.nan,
    }


def proxima_publicacion(publicaciones: pd.DataFrame, ahora: pd.Timestamp | None = None) -> dict | None:
    """Próxima fecha publicada con su BPA estimado, si la hay."""
    if publicaciones.empty:
        return None
    ahora = ahora or pd.Timestamp.now(tz="UTC")
    indice = publicaciones.index
    if indice.tz is None:
        ahora = ahora.tz_localize(None) if ahora.tzinfo else ahora
    futuras = publicaciones[(indice > ahora) & publicaciones["BPA publicado"].isna()]
    if futuras.empty:
        return None
    fila = futuras.iloc[0]
    return {"fecha": futuras.index[0], "BPA estimado": fila["BPA estimado"]}
