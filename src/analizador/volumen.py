"""Volumen inusual: detectarlo y comprobar si anticipa subidas.

El volumen de un día se compara con el de los 60 días ANTERIORES (en logaritmos, porque el
volumen tiene picos muy grandes): z = (ln V_t − media) / desviación típica. Un z de 2 o
más es un volumen inusualmente alto. Solo usa datos conocidos al cierre de ese día.

Las reglas de volumen se prueban con el mismo método que el resto del laboratorio
(``senales.laboratorio``): periodo de descubrimiento y de comprobación, azar con la misma
mezcla y corrección por pruebas múltiples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .senales import DefinicionSenal, laboratorio

UMBRAL_Z = 2.0
VENTANA = 60
HORIZONTES_VOLUMEN = {"1 semana": 5, "1 mes": 21, "3 meses": 63}


def z_volumen(datos: pd.DataFrame, ventana: int = VENTANA) -> pd.Series:
    """Volumen de cada día frente a los ``ventana`` días anteriores (z en logaritmos)."""
    v = datos["Volume"].astype(float).where(datos["Volume"] > 0)
    lv = np.log(v)
    media = lv.shift(1).rolling(ventana, min_periods=ventana // 2).mean()
    desv = lv.shift(1).rolling(ventana, min_periods=ventana // 2).std()
    return (lv - media) / desv


def volumen_relativo(datos: pd.DataFrame, ventana: int = 20) -> pd.Series:
    """Volumen del día dividido por la media de los ``ventana`` días anteriores (1 = normal)."""
    v = datos["Volume"].astype(float).where(datos["Volume"] > 0)
    return v / v.shift(1).rolling(ventana, min_periods=ventana // 2).mean()


def _reglas() -> list[DefinicionSenal]:
    def con_subida(df):
        z, r = z_volumen(df), df["Close"].pct_change()
        return pd.Series(np.where((z >= UMBRAL_Z) & (r > 0), 1.0, np.nan), index=df.index)

    def con_bajada(df):
        z, r = z_volumen(df), df["Close"].pct_change()
        return pd.Series(np.where((z >= UMBRAL_Z) & (r < 0), 1.0, np.nan), index=df.index)

    def acumulacion(df):
        v = df["Volume"].astype(float).where(df["Volume"] > 0)
        ratio = v.rolling(20).mean() / v.shift(20).rolling(60, min_periods=30).mean()
        plano = np.log(df["Close"] / df["Close"].shift(20)).abs() < 0.03
        return pd.Series(np.where((ratio >= 1.5) & plano, 1.0, np.nan), index=df.index)

    def seco(df):
        return pd.Series(np.where(z_volumen(df) <= -UMBRAL_Z, 1.0, np.nan), index=df.index)

    return [
        DefinicionSenal("Volumen inusual en un día de subida", "Volumen",
                        "Volumen 2 desviaciones por encima de lo normal y el precio sube ese día.", con_subida, True),
        DefinicionSenal("Volumen inusual en un día de bajada", "Volumen",
                        "Volumen 2 desviaciones por encima de lo normal y el precio baja ese día.", con_bajada, True),
        DefinicionSenal("Acumulación silenciosa", "Volumen",
                        "El volumen de 20 días es 1,5 veces el habitual mientras el precio apenas se mueve.",
                        acumulacion, True),
        DefinicionSenal("Volumen muy bajo", "Volumen",
                        "Volumen 2 desviaciones por debajo de lo normal.", seco, True),
    ]


REGLAS_VOLUMEN = _reglas()


def laboratorio_volumen(datos: dict[str, pd.DataFrame], horizontes: dict[str, int] | None = None) -> pd.DataFrame:
    """¿Anticipa el volumen inusual subidas? (en las empresas con volumen publicado)."""
    con_volumen = {t: df for t, df in datos.items() if "Volume" in df and (df["Volume"] > 0).mean() > 0.9}
    return laboratorio(con_volumen, horizontes or HORIZONTES_VOLUMEN, REGLAS_VOLUMEN)


def dias_inusuales(datos: pd.DataFrame, horizonte: int = 21, umbral: float = UMBRAL_Z,
                   ultimos: int = 504) -> pd.DataFrame:
    """Días recientes de volumen inusual y qué hizo el precio después (si ya se sabe)."""
    datos = datos.iloc[-(ultimos + VENTANA + horizonte):]
    z = z_volumen(datos)
    c = datos["Close"].astype(float)
    tabla = pd.DataFrame({
        "z": z, "volumen_relativo": volumen_relativo(datos), "cambio_dia": c.pct_change(),
        "despues": c.shift(-horizonte) / c - 1,
    })
    return tabla[tabla["z"] >= umbral].iloc[::-1]
