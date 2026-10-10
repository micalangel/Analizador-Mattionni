"""Reacción del precio a eventos fechados (publicación de resultados, noticias).

La reacción se mide de cierre a cierre alrededor del evento:

- Evento después del cierre (≥ 16:00 Nueva York) o en día sin sesión: del último cierre
  anterior o igual al día del evento al cierre de la sesión siguiente.
- Evento antes de la apertura (< 9:30): del cierre de la sesión anterior al cierre de ese día.
- Durante la sesión: igual que antes de la apertura (cubre el día completo).
- Hora desconocida (medianoche exacta): ventana de dos sesiones, del cierre anterior al
  día del evento al cierre de la sesión siguiente.

Para comparar con lo "normal", la reacción se expresa también en desviaciones típicas
(z) usando la volatilidad diaria de los 60 retornos anteriores al evento: solo datos
disponibles antes de que ocurra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ZONA_MERCADO = "America/New_York"


def _a_hora_mercado(fecha) -> pd.Timestamp:
    ts = pd.Timestamp(fecha)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(ZONA_MERCADO)


def ventana_evento(indice: pd.DatetimeIndex, fecha) -> tuple[int, int, str] | None:
    """Posiciones (antes, después) en ``indice`` de los cierres que delimitan la reacción."""
    ts = _a_hora_mercado(fecha)
    dia = pd.Timestamp(ts.date())
    minutos = ts.hour * 60 + ts.minute
    hay_sesion = dia in indice

    if minutos == 0 and ts.second == 0:
        momento = "hora desconocida"
        antes = indice.searchsorted(dia, side="left") - 1          # último cierre < día
        despues = indice.searchsorted(dia, side="right")           # primera sesión > día
    elif minutos >= 16 * 60 or not hay_sesion:
        momento = "tras el cierre" if hay_sesion else "día sin sesión"
        antes = indice.searchsorted(dia, side="right") - 1         # último cierre ≤ día
        despues = antes + 1
    else:
        momento = "antes de la apertura" if minutos < 9 * 60 + 30 else "durante la sesión"
        despues = indice.get_loc(dia)
        antes = despues - 1

    if antes < 0 or despues >= len(indice):
        return None
    return int(antes), int(despues), momento


def reaccion_a_eventos(
    precios: pd.Series,
    fechas,
    ventana_vol: int = 60,
) -> pd.DataFrame:
    """Reacción del precio a cada fecha de ``fechas``.

    Columnas: momento, fecha_base, fecha_reaccion, reaccion (retorno simple),
    sesiones, z (reacción en desviaciones típicas de la volatilidad previa).
    Las fechas cuya reacción aún no se conoce aparecen con NaN.
    """
    precios = precios.dropna().sort_index()
    log_r = np.log(precios / precios.shift(1))
    vol = log_r.rolling(ventana_vol, min_periods=ventana_vol // 2).std()
    filas = []
    for fecha in fechas:
        v = ventana_evento(precios.index, fecha)
        if v is None:
            filas.append({"momento": "pendiente", "fecha_base": pd.NaT, "fecha_reaccion": pd.NaT,
                          "reaccion": np.nan, "sesiones": np.nan, "z": np.nan})
            continue
        antes, despues, momento = v
        p0, p1 = precios.iloc[antes], precios.iloc[despues]
        sesiones = despues - antes
        sigma = vol.iloc[antes]  # volatilidad conocida al cierre previo al evento
        z = np.log(p1 / p0) / (sigma * np.sqrt(sesiones)) if sigma and np.isfinite(sigma) else np.nan
        filas.append({"momento": momento, "fecha_base": precios.index[antes],
                      "fecha_reaccion": precios.index[despues], "reaccion": p1 / p0 - 1,
                      "sesiones": sesiones, "z": z})
    return pd.DataFrame(filas, index=pd.Index(list(fechas), name="fecha_evento"))
