"""Informe de calibración de los intervalos de pronóstico.

Para cada fecha t se emite un intervalo con datos hasta t y se compara con el precio
realmente observado en t + h. Se mide:

- Cobertura: proporción de veces que el precio real cae dentro del intervalo,
  comparada con la confianza objetivo.
- Ancho medio del intervalo, en % del precio de partida.
- Error: puntuación de intervalo (Winkler / Gneiting-Raftery), en % del precio. Premia
  intervalos estrechos y penaliza cada fallo en proporción a lo lejos que cae el precio.
- MAE del centro del modelo frente al precio real (informativo).

Un método solo se considera útil si (a) está bien calibrado y (b) su puntuación de
intervalo es mejor (menor) que la de la referencia ingenua en el mismo periodo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .forecast import METODOS, NOMBRE_REFERENCIA, REFERENCIA, serie_intervalos


def evaluar_metodo(
    precios: pd.Series,
    metodo: str,
    horizonte: int = 10,
    confianza: float = 0.8,
    ventana: int = 60,
) -> pd.DataFrame:
    """Tabla por fecha de emisión con el intervalo, el precio realizado y los errores."""
    tabla = serie_intervalos(precios, horizonte, confianza, metodo, ventana)
    return evaluar_tabla(tabla, precios, horizonte, confianza)


def evaluar_tabla(tabla: pd.DataFrame, precios: pd.Series, horizonte: int, confianza: float) -> pd.DataFrame:
    """Evalúa una tabla de intervalos ya calculada (columnas inferior, superior, centro)."""
    tabla = tabla.copy()
    tabla["precio_t"] = precios
    tabla["realizado"] = precios.shift(-horizonte)  # solo para evaluar, nunca para emitir
    tabla = tabla.dropna(subset=["inferior", "superior", "realizado"])

    alfa = 1 - confianza
    y, lo, hi, p0 = tabla["realizado"], tabla["inferior"], tabla["superior"], tabla["precio_t"]
    tabla["dentro"] = (y >= lo) & (y <= hi)
    tabla["ancho_rel"] = (hi - lo) / p0
    puntuacion = (hi - lo) + (2 / alfa) * ((lo - y).clip(lower=0) + (y - hi).clip(lower=0))
    tabla["puntuacion_rel"] = puntuacion / p0
    tabla["error_abs_rel"] = (y - tabla["centro"]).abs() / p0
    return tabla


def tolerancia_cobertura(confianza: float, n: int, horizonte: int) -> float:
    """Margen aceptable alrededor de la cobertura objetivo.

    Dos errores típicos binomiales con tamaño efectivo n / h (los intervalos solapados
    a h días no son independientes), con un mínimo de 2 puntos porcentuales.
    """
    n_efectivo = max(n / horizonte, 1)
    return max(0.02, 2 * np.sqrt(confianza * (1 - confianza) / n_efectivo))


def informe_calibracion(
    precios: pd.Series,
    horizonte: int = 10,
    confianza: float = 0.8,
    metodos: list[str] | None = None,
    ventana: int = 60,
    extra: dict[str, tuple[str, pd.DataFrame]] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compara los métodos con la referencia ingenua en un periodo común.

    ``extra`` permite añadir métodos cuyos intervalos ya están calculados (p. ej. la IA):
    {clave: (descripción, tabla_de_intervalos)}.

    Devuelve (resumen, detalle). ``resumen`` tiene una fila por método; ``detalle``
    contiene la tabla por fecha de cada método, recortada al periodo común.
    """
    metodos = list(metodos or METODOS)
    todos = [REFERENCIA] + [m for m in metodos if m != REFERENCIA]
    detalle = {m: evaluar_metodo(precios, m, horizonte, confianza, ventana) for m in todos}
    descripciones = dict(METODOS)
    for clave, (descripcion, tabla) in (extra or {}).items():
        detalle[clave] = evaluar_tabla(tabla, precios, horizonte, confianza)
        descripciones[clave] = descripcion

    comun = None
    for tabla in detalle.values():
        comun = tabla.index if comun is None else comun.intersection(tabla.index)
    if comun is None or len(comun) == 0:
        raise ValueError("No hay un periodo común con suficientes datos para evaluar.")
    detalle = {m: t.loc[comun] for m, t in detalle.items()}

    n = len(comun)
    tol = tolerancia_cobertura(confianza, n, horizonte)
    puntuacion_ref = detalle[REFERENCIA]["puntuacion_rel"].mean()
    filas = []
    for m, t in detalle.items():
        cobertura = t["dentro"].mean()
        puntuacion = t["puntuacion_rel"].mean()
        calibrado = abs(cobertura - confianza) <= tol
        es_ref = m == REFERENCIA
        mejora = not es_ref and puntuacion < puntuacion_ref
        filas.append({
            "método": m,
            "descripción": NOMBRE_REFERENCIA if es_ref else descripciones.get(m, m),
            "n": n,
            "cobertura": cobertura,
            "objetivo": confianza,
            "desviación": cobertura - confianza,
            "tolerancia": tol,
            "calibrado": bool(calibrado),
            "ancho medio %": 100 * t["ancho_rel"].mean(),
            "puntuación intervalo %": 100 * puntuacion,
            "MAE centro %": 100 * t["error_abs_rel"].mean(),
            "mejora vs ingenuo": None if es_ref else bool(mejora),
            "útil": None if es_ref else bool(calibrado and mejora),
        })
    resumen = pd.DataFrame(filas).set_index("método")
    return resumen, detalle


def cobertura_movil(detalle: pd.DataFrame, ventana: int = 252) -> pd.Series:
    """Cobertura en una ventana móvil de fechas de emisión (para ver su estabilidad)."""
    return detalle["dentro"].astype(float).rolling(ventana, min_periods=ventana // 2).mean()
