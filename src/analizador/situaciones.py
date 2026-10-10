"""Situaciones especiales: pánico y caídas fuertes, contadas por EPISODIOS.

Reglas fijadas de antemano (conocidas en la literatura, no ajustadas a estos datos):

- **Pánico:** el índice de miedo VIX cierra en 30 o más.
- **Caída fuerte:** el activo está un 20 % o más por debajo de su máximo del último año.

Por qué por episodios: un pánico dura semanas o meses, y contar cada día por separado
haría que unos pocos episodios largos (2008-09, 2020) pesen como cientos de casos. Aquí
cada episodio cuenta una vez: su resultado es la proporción de sus días en los que el
precio estaba más arriba al cumplirse el plazo.

Las reglas usan solo datos del cierre anterior (VIX) o del propio día (caída del activo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

UMBRAL_VIX = 30.0
UMBRAL_CAIDA = -0.20
SEPARACION = 60  # sesiones sin la condición para considerar que empieza un episodio nuevo
CORTE_EPOCA = pd.Timestamp("2010-01-01")


@dataclass(frozen=True)
class Regla:
    clave: str
    nombre: str
    descripcion: str


REGLAS = {
    "panico": Regla("panico", "Pánico en los mercados", f"El índice de miedo VIX está en {UMBRAL_VIX:.0f} o más."),
    "caida": Regla("caida", "Caída fuerte", "El precio está un 20 % o más por debajo de su máximo del último año."),
}


def vix_previo(indice: pd.DatetimeIndex, macro: pd.DataFrame | None) -> pd.Series:
    """VIX del último cierre ANTERIOR a cada fecha (el VIX se publica con el cierre de EE. UU.)."""
    if macro is None or "vix" not in macro:
        return pd.Series(np.nan, index=indice)
    v = macro["vix"].dropna().sort_index()
    pos = v.index.searchsorted(indice, side="left") - 1
    valores = v.to_numpy()[np.clip(pos, 0, None)].astype(float)
    valores[pos < 0] = np.nan
    return pd.Series(valores, index=indice)


def condicion(regla: str, precios: pd.Series, macro: pd.DataFrame | None) -> pd.Series:
    """Serie booleana (NaN si no se puede saber) de días en que se cumple la regla."""
    if regla == "panico":
        v = vix_previo(precios.index, macro)
        return (v >= UMBRAL_VIX).where(v.notna())
    if regla == "caida":
        caida = precios / precios.rolling(252, min_periods=120).max() - 1
        return (caida <= UMBRAL_CAIDA).where(caida.notna())
    raise ValueError(f"Regla desconocida: {regla}")


def episodios(activa: pd.Series, separacion: int = SEPARACION) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Agrupa los días activos en episodios (inicio, fin)."""
    pos = np.flatnonzero(activa.fillna(False).to_numpy())
    if len(pos) == 0:
        return []
    cortes = np.flatnonzero(np.diff(pos) > separacion)
    inicios = np.concatenate([[pos[0]], pos[cortes + 1]])
    finales = np.concatenate([pos[cortes], [pos[-1]]])
    return [(activa.index[i], activa.index[f]) for i, f in zip(inicios, finales)]


def historial(precios: pd.Series, regla: str, horizonte: int, macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Una fila por episodio: fechas, % de sus días que acabaron más arriba y retorno desde el inicio."""
    precios = precios.dropna().astype(float)
    activa = condicion(regla, precios, macro)
    subio = (precios.shift(-horizonte) > precios).where(precios.shift(-horizonte).notna())
    filas = []
    for ini, fin in episodios(activa):
        dias = activa.loc[ini:fin].fillna(False)
        res = subio.loc[ini:fin][dias.to_numpy()]
        conocidos = res.dropna()
        i = precios.index.get_loc(ini)
        ret = precios.iloc[i + horizonte] / precios.iloc[i] - 1 if i + horizonte < len(precios) else np.nan
        filas.append({
            "inicio": ini, "fin": fin, "dias": int(dias.sum()),
            "prop_mas_arriba": float(conocidos.mean()) if len(conocidos) else np.nan,
            "completo": len(conocidos) == len(res) and len(res) > 0,
            "retorno_desde_inicio": ret,
            "vix_max": float(vix_previo(precios.loc[ini:fin].index, macro).max()) if regla == "panico" else np.nan,
            "caida_max": float((precios / precios.rolling(252, min_periods=120).max() - 1).loc[ini:fin].min()),
        })
    return pd.DataFrame(filas)


def resumen(hist: pd.DataFrame, frecuencia_normal: float | None = None) -> dict:
    """Estadísticas por episodio (cada episodio cuenta una vez)."""
    hechos = hist[hist["completo"]] if not hist.empty else hist
    if hechos.empty:
        return {"n": 0}
    mayoria = hechos["prop_mas_arriba"] > 0.5
    antes = hechos[hechos["inicio"] < CORTE_EPOCA]
    despues = hechos[hechos["inicio"] >= CORTE_EPOCA]
    return {
        "n": len(hechos),
        "episodios_arriba": int(mayoria.sum()),
        "prop_media": float(hechos["prop_mas_arriba"].mean()),
        "n_antes": len(antes), "arriba_antes": int((antes["prop_mas_arriba"] > 0.5).sum()),
        "n_despues": len(despues), "arriba_despues": int((despues["prop_mas_arriba"] > 0.5).sum()),
        "frecuencia_normal": frecuencia_normal,
    }


def situacion_hoy(precios: pd.Series, macro: pd.DataFrame | None = None) -> dict:
    """Qué reglas se cumplen con el último dato, y los valores que las determinan."""
    precios = precios.dropna().astype(float)
    caida = float(precios.iloc[-1] / precios.iloc[-252:].max() - 1)
    vix = float(macro["vix"].dropna().iloc[-1]) if macro is not None and "vix" in macro else np.nan
    return {
        "vix": vix, "caida": caida,
        "panico": bool(np.isfinite(vix) and vix >= UMBRAL_VIX),
        "caida_fuerte": caida <= UMBRAL_CAIDA,
    }
