"""Noticias recientes de un ticker y movimiento del precio en la sesión de cada una.

Los titulares proceden de Yahoo Finance (normalmente en inglés y de medios de terceros).
La herramienta no los resume ni los valora: solo los lista con su fecha, fuente y enlace,
y mide el movimiento del precio alrededor de su publicación. Coincidir en el tiempo no
implica que la noticia haya causado el movimiento.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from .eventos import reaccion_a_eventos

AVISO_NOTICIAS = (
    "Titulares de terceros vía Yahoo Finance. Que el precio se mueva el día de una noticia "
    "no significa que la noticia sea la causa."
)

COLUMNAS = ["titular", "fuente", "enlace"]


def _extraer(item: dict) -> dict | None:
    """Admite los dos formatos que ha usado yfinance (plano y con clave 'content')."""
    if "content" in item and isinstance(item["content"], dict):
        c = item["content"]
        fecha = c.get("pubDate") or c.get("displayTime")
        enlace = (c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url")
        fuente = (c.get("provider") or {}).get("displayName", "")
        titular = c.get("title")
    else:
        t = item.get("providerPublishTime")
        fecha = pd.Timestamp(t, unit="s", tz="UTC") if t else None
        enlace, fuente, titular = item.get("link"), item.get("publisher", ""), item.get("title")
    if not titular or fecha is None:
        return None
    fecha = pd.Timestamp(fecha)
    if fecha.tzinfo is None:
        fecha = fecha.tz_localize("UTC")
    return {"fecha": fecha, "titular": titular, "fuente": fuente, "enlace": enlace or ""}


def _descargar_yfinance(ticker: str, cantidad: int) -> list[dict]:
    import yfinance as yf

    items = []
    try:
        items = yf.Search(ticker, news_count=cantidad).news or []
    except Exception:
        pass
    if not items:
        try:
            items = yf.Ticker(ticker).get_news(count=cantidad) or []
        except Exception:
            items = []
    return items


def obtener_noticias(ticker: str, cantidad: int = 20,
                     descargador: Callable[[str, int], list[dict]] | None = None) -> pd.DataFrame:
    """Tabla de noticias ordenada de más reciente a más antigua, sin duplicados."""
    try:
        items = (descargador or _descargar_yfinance)(ticker.strip().upper(), cantidad)
    except Exception:
        items = []
    filas = [f for f in (_extraer(i) for i in items) if f]
    if not filas:
        return pd.DataFrame(columns=COLUMNAS, index=pd.DatetimeIndex([], tz="UTC", name="fecha"))
    df = pd.DataFrame(filas).drop_duplicates(subset=["titular"]).set_index("fecha")
    return df.sort_index(ascending=False)[COLUMNAS]


def estudio_noticias(precios: pd.Series, noticias: pd.DataFrame) -> pd.DataFrame:
    """Añade a cada noticia el movimiento del precio en su sesión y su tamaño en z."""
    if noticias.empty:
        return noticias.assign(reaccion=pd.Series(dtype=float), z=pd.Series(dtype=float))
    reac = reaccion_a_eventos(precios, noticias.index)
    reac.index = noticias.index
    return noticias.join(reac[["momento", "fecha_reaccion", "reaccion", "z"]])
