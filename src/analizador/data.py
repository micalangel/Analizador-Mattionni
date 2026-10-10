"""Descarga de datos diarios OHLCV con yfinance y caché local en CSV."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

COLUMNAS_OHLCV = ["Open", "High", "Low", "Close", "Volume"]
DIRECTORIO_CACHE = Path("data_cache")

# Un descargador recibe (ticker, inicio, fin) y devuelve un DataFrame OHLCV.
Descargador = Callable[[str, str, str], pd.DataFrame]


def normalizar_tickers(tickers: str | Iterable[str]) -> list[str]:
    """Convierte "aapl, msft spy" o una lista en ["AAPL", "MSFT", "SPY"] sin duplicados."""
    if isinstance(tickers, str):
        partes = re.split(r"[,\s;]+", tickers)
    else:
        partes = list(tickers)
    vistos: list[str] = []
    for t in partes:
        t = str(t).strip().upper()
        if t and t not in vistos:
            vistos.append(t)
    return vistos


def limpiar_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza un DataFrame OHLCV y trata los datos faltantes.

    - Aplana columnas MultiIndex (formato de yfinance con varios tickers).
    - Índice de fechas ordenado, sin zona horaria y sin duplicados.
    - Elimina filas sin precio de cierre (no se inventan precios).
    - Rellena Open/High/Low faltantes con el cierre del mismo día y Volume con 0.
    - Descarta precios no positivos, que harían imposibles los retornos logarítmicos.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNAS_OHLCV, index=pd.DatetimeIndex([], name="Date"))

    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance usa (campo, ticker); nos quedamos con el nivel del campo.
        niveles = [i for i in range(df.columns.nlevels)
                   if set(COLUMNAS_OHLCV) & set(df.columns.get_level_values(i))]
        df.columns = df.columns.get_level_values(niveles[0] if niveles else 0)
    df.columns = [str(c).strip().title() for c in df.columns]
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    for col in COLUMNAS_OHLCV:
        if col not in df.columns:
            df[col] = np.nan
    df = df[COLUMNAS_OHLCV]

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df["Close"].notna() & (df["Close"] > 0)]
    for col in ["Open", "High", "Low"]:
        df[col] = df[col].where(df[col] > 0).fillna(df["Close"])
    df["Volume"] = df["Volume"].fillna(0)
    return df


def _descargar_yfinance(ticker: str, inicio: str, fin: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        ticker,
        start=inicio,
        end=fin,
        auto_adjust=True,
        progress=False,
        threads=False,
    )


def ruta_cache(ticker: str, inicio: str, fin: str, directorio: Path | str = DIRECTORIO_CACHE) -> Path:
    nombre = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    return Path(directorio) / f"{nombre}_{inicio}_{fin}.csv"


def descargar_precios(
    ticker: str,
    inicio: str | date,
    fin: str | date,
    directorio_cache: Path | str = DIRECTORIO_CACHE,
    usar_cache: bool = True,
    descargador: Descargador | None = None,
) -> pd.DataFrame:
    """Descarga datos diarios OHLCV ajustados (auto_adjust=True) de un ticker.

    El resultado se guarda en CSV en ``directorio_cache``. Si la descarga falla o no
    devuelve datos, se devuelve un DataFrame vacío con las columnas OHLCV.
    """
    inicio, fin = str(pd.Timestamp(inicio).date()), str(pd.Timestamp(fin).date())
    ticker = ticker.strip().upper()
    ruta = ruta_cache(ticker, inicio, fin, directorio_cache)

    if usar_cache and ruta.exists():
        crudo = pd.read_csv(ruta, index_col=0, parse_dates=True)
        return limpiar_ohlcv(crudo)

    descargador = descargador or _descargar_yfinance
    try:
        crudo = descargador(ticker, inicio, fin)
    except Exception:  # red caída, ticker inexistente, cambios en la API...
        crudo = None
    df = limpiar_ohlcv(crudo)

    if usar_cache and not df.empty:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ruta)
    return df


def descargar_varios(
    tickers: str | Iterable[str],
    inicio: str | date,
    fin: str | date,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Descarga varios tickers. Los que no devuelven datos se omiten del resultado."""
    resultado: dict[str, pd.DataFrame] = {}
    for t in normalizar_tickers(tickers):
        df = descargar_precios(t, inicio, fin, **kwargs)
        if not df.empty:
            resultado[t] = df
    return resultado


def precios_cierre(datos: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Une los cierres de varios tickers en un DataFrame (una columna por ticker)."""
    if not datos:
        return pd.DataFrame()
    return pd.DataFrame({t: df["Close"] for t, df in datos.items()}).sort_index()


INTERVALOS_INTRADIA = {"1 hora": ("1h", "730d"), "5 minutos": ("5m", "60d")}


def descargar_intradia(ticker: str, intervalo: str, descargador=None) -> pd.DataFrame:
    """Velas intradía de Yahoo (5 minutos: últimos 60 días; 1 hora: últimos 2 años), en hora de Madrid.

    No se guardan en la caché CSV: cambian a lo largo del día.
    """
    codigo, periodo = INTERVALOS_INTRADIA[intervalo]
    try:
        if descargador is not None:
            crudo = descargador(ticker, codigo, periodo)
        else:
            import yfinance as yf

            crudo = yf.download(ticker, period=periodo, interval=codigo, auto_adjust=True,
                                progress=False, threads=False)
    except Exception:
        crudo = None
    if crudo is None or len(crudo) == 0:
        return pd.DataFrame(columns=COLUMNAS_OHLCV)
    df = crudo.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    indice = pd.to_datetime(df.index)
    if indice.tz is not None:
        indice = indice.tz_convert("Europe/Madrid").tz_localize(None)
    df.index = indice
    df = df.reindex(columns=COLUMNAS_OHLCV)
    df = df[df["Close"].notna() & (df["Close"] > 0)]
    df["Volume"] = df["Volume"].fillna(0)
    return df[~df.index.duplicated(keep="last")].sort_index()
