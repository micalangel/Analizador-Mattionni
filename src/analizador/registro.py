"""Diario de la IA: predicciones apuntadas ANTES de saber el resultado y su comprobación posterior.

Es la prueba más honesta de que la IA funciona: cada vez que se calcula un estudio, se apunta
lo que se dijo (probabilidad de estar más arriba y rango) con la fecha y el precio de ese día.
Cuando pasa el plazo, se compara con lo que ocurrió de verdad. Nada se puede retocar después.

También guarda el historial de entrenamientos (fecha, datos usados y acierto fuera de muestra)
para ver si la IA mejora o empeora al reentrenarla.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DIRECTORIO_REGISTRO = Path(__file__).resolve().parents[2] / "modelos" / "registro"
RUTA_PREDICCIONES = DIRECTORIO_REGISTRO / "predicciones.csv"
RUTA_ENTRENAMIENTOS = DIRECTORIO_REGISTRO / "entrenamientos.csv"

COLUMNAS = ["ticker", "nombre", "fecha", "plazo", "sesiones_plazo", "cierre", "prob_sube", "inferior", "superior",
            "confianza", "fuente_direccion"]
CLAVE = ["ticker", "fecha", "plazo"]


def registrar_predicciones(estudio: pd.DataFrame, ruta: Path | str = RUTA_PREDICCIONES) -> int:
    """Añade al diario las predicciones de un estudio. Las ya apuntadas (mismo activo, día y plazo) no se tocan."""
    nuevas = estudio.copy()
    if "sesiones_plazo" not in nuevas and "sesiones" in nuevas:  # el estudio de índices usa otro nombre
        nuevas["sesiones_plazo"] = nuevas["sesiones"]
    if "nombre" not in nuevas and "índice" in nuevas:
        nuevas["nombre"] = nuevas["índice"] + " (" + nuevas["país"] + ")"
    nuevas = nuevas.reindex(columns=COLUMNAS).dropna(subset=["cierre"])
    ruta = Path(ruta)
    if ruta.exists():
        previas = pd.read_csv(ruta)
        ya = set(map(tuple, previas[CLAVE].astype(str).to_numpy()))
        nuevas = nuevas[[tuple(map(str, k)) not in ya for k in nuevas[CLAVE].to_numpy()]]
        total = pd.concat([previas, nuevas], ignore_index=True)
    else:
        total = nuevas
    ruta.parent.mkdir(parents=True, exist_ok=True)
    total.to_csv(ruta, index=False)
    return len(nuevas)


def cargar_predicciones(ruta: Path | str = RUTA_PREDICCIONES) -> pd.DataFrame:
    ruta = Path(ruta)
    return pd.read_csv(ruta, parse_dates=["fecha"]) if ruta.exists() else pd.DataFrame(columns=COLUMNAS)


def fecha_comprobacion(fecha: pd.Timestamp, sesiones: int) -> pd.Timestamp:
    """Día hábil en que se cumple el plazo (aproximado: el calendario real de cada bolsa puede variar)."""
    return pd.bdate_range(fecha, periods=int(sesiones) + 1)[-1]


def comprobar(predicciones: pd.DataFrame, precios: dict[str, pd.Series]) -> pd.DataFrame:
    """Añade el resultado real a las predicciones cuyo plazo ya ha pasado (en sesiones de ese activo)."""
    tabla = predicciones.copy()
    tabla["precio_real"] = np.nan
    tabla["fecha_real"] = pd.NaT
    for i, fila in tabla.iterrows():
        serie = precios.get(fila["ticker"])
        if serie is None or serie.empty:
            continue
        pos = serie.index.searchsorted(pd.Timestamp(fila["fecha"]), side="right") - 1  # sesión de la predicción
        objetivo = pos + int(fila["sesiones_plazo"])
        if pos >= 0 and objetivo < len(serie):
            tabla.at[i, "precio_real"] = float(serie.iloc[objetivo])
            tabla.at[i, "fecha_real"] = serie.index[objetivo]
    hecho = tabla["precio_real"].notna()
    subio = tabla["precio_real"] > tabla["cierre"]
    tabla["comprobada"] = hecho
    tabla["subio"] = subio.where(hecho)
    tabla["acierto_lado"] = ((tabla["prob_sube"] > 0.5) == subio).where(hecho & tabla["prob_sube"].notna())
    tabla["dentro_rango"] = ((tabla["precio_real"] >= tabla["inferior"]) & (tabla["precio_real"] <= tabla["superior"])
                             ).where(hecho & tabla["inferior"].notna())
    return tabla


def resumen(comprobadas: pd.DataFrame) -> pd.DataFrame:
    """Marcador por plazo: acierto del lado frente a decir siempre «más arriba», y aciertos del rango."""
    hechas = comprobadas[comprobadas["comprobada"]]
    if hechas.empty:
        return pd.DataFrame()
    return hechas.groupby("plazo").agg(
        comprobadas=("ticker", "size"),
        dias_distintos=("fecha", "nunique"),
        acierto_lado=("acierto_lado", "mean"),
        siempre_sube=("subio", "mean"),
        dentro_rango=("dentro_rango", "mean"),
        confianza=("confianza", "mean"),
    )


def registrar_entrenamiento(filas: list[dict], ruta: Path | str = RUTA_ENTRENAMIENTOS) -> None:
    """Apunta una fila por horizonte y modelo con las métricas fuera de muestra del entrenamiento."""
    ruta = Path(ruta)
    nuevas = pd.DataFrame(filas)
    total = pd.concat([pd.read_csv(ruta), nuevas], ignore_index=True) if ruta.exists() else nuevas
    ruta.parent.mkdir(parents=True, exist_ok=True)
    total.to_csv(ruta, index=False)


def cargar_entrenamientos(ruta: Path | str = RUTA_ENTRENAMIENTOS) -> pd.DataFrame:
    ruta = Path(ruta)
    return pd.read_csv(ruta) if ruta.exists() else pd.DataFrame()


def filas_entrenamiento(conjuntos_rango: dict, direcciones: dict) -> list[dict]:
    """Construye las filas del historial a partir de los modelos recién entrenados (o cargados)."""
    filas = []
    for h, c in conjuntos_rango.items():
        if c is None:
            continue
        ev = c.metadatos.get("evaluacion", {}).get("0.8", {})
        filas.append({"fecha": c.metadatos.get("entrenado"), "modelo": "rangos", "sesiones_plazo": h,
                      "muestras": c.metadatos.get("muestras_totales"), "indices": len(c.metadatos.get("cesta", [])),
                      "metrica": "cobertura al 80 %", "valor": ev.get("cobertura"), "referencia": 0.8})
    for h, variantes in direcciones.items():
        for clave, c in (variantes or {}).items():
            if c is None:
                continue
            ev = c.metadatos.get("evaluacion", {})
            filas.append({"fecha": c.metadatos.get("entrenado"), "modelo": f"dirección ({clave})", "sesiones_plazo": h,
                          "muestras": c.metadatos.get("muestras_totales"), "indices": len(c.metadatos.get("cesta", [])),
                          "metrica": "acierto del lado", "valor": ev.get("acierto_ia"),
                          "referencia": ev.get("acierto_frecuencia")})
    return filas
