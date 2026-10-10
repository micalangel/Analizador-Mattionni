"""PER histórico de una empresa: con qué PER suele cotizar y si el de hoy está por encima o por debajo.

PER de cada día = precio de cierre (ajustado por splits, NO por dividendos) / beneficio por acción
(BPA) de los cuatro últimos trimestres publicados hasta ese día. El BPA de un trimestre solo cuenta
desde el día de su publicación.

El BPA sale del calendario de resultados de Yahoo («Reported EPS»), que a menudo es el BPA ajustado
que publica la empresa; por eso el PER calculado puede diferir algo del PER oficial. Para comparar
con justicia, el PER de hoy y el histórico se calculan del mismo modo. Si el PER calculado hoy y el
oficial no se parecen (p. ej. empresas que cotizan en una moneda y presentan cuentas en otra), se
descarta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ANIOS = 10


@dataclass
class PerHistorico:
    actual: float
    mediana: float
    p10: float
    p90: float
    anios: float
    serie: pd.Series = field(repr=False)

    @property
    def diferencia(self) -> float:
        return self.actual / self.mediana - 1

    @property
    def etiqueta(self) -> str:
        if self.diferencia < -0.20:
            return "Por debajo de su media"
        if self.diferencia > 0.25:
            return "Por encima de su media"
        return "En su media"


def bpa_doce_meses(publicaciones: pd.DataFrame) -> pd.Series:
    """BPA de los 4 últimos trimestres, fechado el día de cada publicación (sin zona horaria)."""
    p = publicaciones.dropna(subset=["BPA publicado"]).sort_index()
    fechas = pd.DatetimeIndex(p.index)
    if fechas.tz is not None:
        fechas = fechas.tz_convert("America/New_York").tz_localize(None)
    bpa = pd.Series(p["BPA publicado"].to_numpy(dtype=float), index=fechas.normalize())
    bpa = bpa[~bpa.index.duplicated(keep="last")]
    # Cuatro publicaciones seguidas deben cubrir aproximadamente un año (sin huecos de trimestres).
    suma = bpa.rolling(4).sum()
    duracion = pd.Series(bpa.index, index=bpa.index).diff(3).dt.days
    return suma.where(duracion.between(250, 400))


def calcular(precios: pd.Series, publicaciones: pd.DataFrame, per_oficial: float | None = None,
             anios: int = ANIOS) -> PerHistorico | None:
    """PER histórico de los últimos ``anios`` años, o None si no hay datos fiables."""
    ttm = bpa_doce_meses(publicaciones).dropna()
    if ttm.empty or precios.empty:
        return None
    precios = precios.dropna().astype(float)
    precios.index = pd.DatetimeIndex(precios.index).tz_localize(None) if precios.index.tz else precios.index
    # El BPA publicado un día cuenta desde el cierre de ese día en adelante.
    bpa_diario = ttm.reindex(precios.index.union(ttm.index)).ffill().reindex(precios.index)
    per = (precios / bpa_diario).where(bpa_diario > 0)
    per = per[(per > 0) & (per < 300)]
    desde = precios.index[-1] - pd.DateOffset(years=anios)
    ventana = per.loc[desde:].dropna()
    if len(ventana) < 250 or pd.isna(per.iloc[-1]) if len(per) else True:
        return None
    actual = float(per.iloc[-1])
    if per_oficial is not None and np.isfinite(per_oficial) and per_oficial > 0:
        cociente = actual / per_oficial
        if not 0.5 <= cociente <= 2.0:  # monedas distintas o datos que no cuadran
            return None
    return PerHistorico(actual=actual, mediana=float(ventana.median()), p10=float(ventana.quantile(0.10)),
                        p90=float(ventana.quantile(0.90)), anios=(ventana.index[-1] - ventana.index[0]).days / 365.25,
                        serie=ventana)


def descargar(ticker: str, anios: int = ANIOS + 1):
    """Precio sin ajuste por dividendos, publicaciones con BPA (unos 20 años) y PER oficial de Yahoo."""
    import yfinance as yf

    from .resultados import normalizar_publicaciones

    t = yf.Ticker(ticker)
    precios = yf.download(ticker, period=f"{anios}y", auto_adjust=False, progress=False, threads=False)
    if isinstance(precios.columns, pd.MultiIndex):
        precios.columns = precios.columns.get_level_values(0)
    cierre = precios["Close"] if "Close" in precios else pd.Series(dtype=float)
    publicaciones = normalizar_publicaciones(t.get_earnings_dates(limit=100))
    try:
        per_oficial = (t.info or {}).get("trailingPE")
    except Exception:
        per_oficial = None
    return cierre, publicaciones, per_oficial
