"""Datos históricos de empresas para el laboratorio de estrategias: analistas y resultados.

Para cada empresa se guardan en caché local (``data_cache/empresas``):

* los cambios de precio objetivo y de opinión de cada firma (sobre todo empresas de EE. UU.),
* las publicaciones de resultados con su sorpresa frente a lo esperado.

Con eso se construyen puntuaciones mensuales SIN mirar al futuro (cada fin de mes solo cuenta lo
publicado hasta ese día):

* ``revisiones``: (subidas − bajadas de precio objetivo) / total de cambios en los últimos 90 días.
* ``potencial``: consenso de precio objetivo / precio − 1.
* ``sorpresa``: sorpresa (%) de la última publicación de resultados de los últimos 100 días.

Uso: ``python -m analizador.datos_empresas`` (descarga; unos minutos).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analistas as an

CACHE = Path(__file__).resolve().parents[2] / "data_cache" / "empresas"


def _ruta(ticker: str, tipo: str) -> Path:
    return CACHE / f"{ticker.replace('^', '_').replace('/', '_')}_{tipo}.csv"


def descargar(ticker: str, pausa: float = 0.3) -> None:
    import yfinance as yf

    from .resultados import normalizar_publicaciones

    CACHE.mkdir(parents=True, exist_ok=True)
    try:
        precios, cambios, splits = an.descargar(ticker)
        objetivos = an.normalizar(cambios, precios, splits) if precios is not None and len(precios) else pd.DataFrame()
        objetivos.to_csv(_ruta(ticker, "analistas"), index=False)
        precios.rename("Close").to_csv(_ruta(ticker, "precio_sin_div"))
    except Exception as e:  # noqa: BLE001 - se sigue con las demás
        print(f"  {ticker}: analistas sin datos ({e.__class__.__name__})")
    try:
        pub = normalizar_publicaciones(yf.Ticker(ticker).get_earnings_dates(limit=100))
        pub.to_csv(_ruta(ticker, "resultados"))
    except Exception as e:  # noqa: BLE001
        print(f"  {ticker}: resultados sin datos ({e.__class__.__name__})")
    time.sleep(pausa)


def _leer(ticker: str, tipo: str, **kw) -> pd.DataFrame | None:
    ruta = _ruta(ticker, tipo)
    if not ruta.exists() or ruta.stat().st_size < 5:
        return None
    try:
        return pd.read_csv(ruta, **kw)
    except Exception:  # noqa: BLE001
        return None


def puntuaciones_mensuales(tickers: list[str], meses: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """{nombre: DataFrame meses × tickers} con las puntuaciones disponibles en cada fin de mes."""
    rev, pot, sor = {}, {}, {}
    for t in tickers:
        obj = _leer(t, "analistas", parse_dates=["fecha"])
        if obj is not None and len(obj):
            fechas = obj["fecha"].to_numpy()
            sube = (obj["accion"] == "Raises").to_numpy()
            baja = (obj["accion"] == "Lowers").to_numpy()
            fila_r = {}
            for m in meses:
                fin = np.datetime64(m)  # hasta el día anterior: lo del propio día puede ser tras el cierre
                ventana = (fechas <= fin) & (fechas > fin - np.timedelta64(90, "D"))
                total = ventana.sum()
                if total >= 2:
                    fila_r[m] = (sube[ventana].sum() - baja[ventana].sum()) / total
            rev[t] = pd.Series(fila_r, dtype=float)
            precio = _leer(t, "precio_sin_div", index_col=0, parse_dates=True)
            if precio is not None:
                p = precio.iloc[:, 0].astype(float)
                cons = an.consenso(obj, meses[meses >= obj["fecha"].min()])
                if len(cons):
                    pm = p.reindex(p.index.union(cons.index)).ffill().reindex(cons.index)
                    pot[t] = cons["objetivo"] / pm - 1
        pub = _leer(t, "resultados", index_col=0)
        if pub is not None and len(pub) and "Sorpresa %" in pub:
            pub.index = pd.to_datetime(pub.index, utc=True).tz_convert("America/New_York").tz_localize(None)
            pub = pub.dropna(subset=["Sorpresa %"]).sort_index()
            # Se conoce desde el día siguiente a la publicación (cubre las de después del cierre); para
            # operar al cierre de fin de mes tiene que haberse conocido antes de ese día.
            conocida = pub.index.normalize() + pd.Timedelta(days=1)
            fila_s = {}
            for m in meses:
                previas = conocida < m
                if previas.any():
                    ultima = np.flatnonzero(previas)[-1]
                    if (m - conocida[ultima]).days <= 100:
                        fila_s[m] = float(pub["Sorpresa %"].iloc[ultima])
            sor[t] = pd.Series(fila_s, dtype=float)
    def tabla(d):
        return pd.DataFrame(d).reindex(meses) if d else pd.DataFrame(index=meses)

    return {"revisiones": tabla(rev), "potencial": tabla(pot), "sorpresa": tabla(sor)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .catalogo import ACTIVOS

    p = argparse.ArgumentParser(prog="analizador.datos_empresas", description=__doc__.splitlines()[0])
    p.add_argument("--tickers", nargs="*")
    p.add_argument("--nuevos", action="store_true", help="Solo las empresas que aún no están en la caché")
    args = p.parse_args(argv)
    tickers = args.tickers or [t for t, v in ACTIVOS.items() if v[1] == "Acción"]
    if args.nuevos:
        tickers = [t for t in tickers if not _ruta(t, "resultados").exists()]
    t0 = time.time()
    for i, t in enumerate(tickers, 1):
        descargar(t)
        if i % 25 == 0:
            print(f"  {i}/{len(tickers)} ({time.time() - t0:.0f} s)", flush=True)
    print(f"Listo: {len(tickers)} empresas en {time.time() - t0:.0f} s.")
    return 0


if __name__ == "__main__":
    from analizador.datos_empresas import main as _main

    raise SystemExit(_main())
