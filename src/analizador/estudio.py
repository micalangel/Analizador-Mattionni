"""Estudio de todos los activos del catálogo (empresas, divisas, materias primas, criptomonedas).

Calcula de antemano, para cada activo, la probabilidad de estar más arriba, el rango y su
fiabilidad a varios plazos, y si hoy está en una situación especial. La página «Empresas»
lee el resultado sin tener que calcular nada.

Uso:  python -m analizador.estudio            (unos minutos; usa todos los núcleos)
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import pandas as pd

from . import AVISO_LEGAL

PLAZOS_ESTUDIO = {"1 mes": 21, "6 meses": 126, "1 año": 252}
INICIO = "1990-01-01"

_MODELOS: dict = {}


def _cargar_modelos(directorio: str) -> None:
    """Se ejecuta una vez por proceso: carga los modelos de la IA."""
    from . import direccion as dr
    from . import entrenamiento as ent

    warnings.filterwarnings("ignore")
    _MODELOS["rangos"] = {h: ent.cargar(h, directorio) for h in PLAZOS_ESTUDIO.values()}
    _MODELOS["direcciones"] = {h: {v: dr.cargar(h, directorio, v) for v in dr.VARIANTES}
                               for h in PLAZOS_ESTUDIO.values()}


def estudiar_activo(ticker: str, datos: pd.DataFrame, mercado: pd.Series | None,
                    macro: pd.DataFrame | None, confianza: float = 0.8) -> list[dict]:
    """Filas del estudio (una por plazo) para un activo."""
    from . import situaciones as sit
    from .catalogo import ACTIVOS, nombre
    from .proyeccion import NOMBRES_CORTOS, proyectar

    nombre_activo, tipo, pais, sector = ACTIVOS.get(ticker, (ticker, "Otro", "—", "—"))
    proy = proyectar(datos, confianza, _MODELOS.get("rangos"), mercado, horizontes=PLAZOS_ESTUDIO,
                     direcciones=_MODELOS.get("direcciones"), macro=macro)
    hoy = sit.situacion_hoy(datos["Close"], macro) if len(datos) >= 260 else {}
    filas = []
    for etiqueta, p in proy.items():
        r, f, d = p.rango or {}, p.fiabilidad, p.direccion
        filas.append({
            "ticker": ticker, "nombre": nombre(ticker), "tipo": tipo, "país": pais, "sector": sector,
            "fecha": str(datos.index[-1].date()), "cierre": float(datos["Close"].iloc[-1]),
            "historia_desde": str(datos.index[0].date()), "sesiones": len(datos),
            "plazo": etiqueta, "sesiones_plazo": p.horizonte, "confianza": confianza,
            "inferior": r.get("inferior"), "superior": r.get("superior"),
            "estado_rango": p.estado, "aciertos_rango": f.get("cobertura"),
            "metodo_rango": NOMBRES_CORTOS.get(p.metodo, p.metodo),
            "prob_sube": d.probabilidad if d else None, "fuente_direccion": d.fuente if d else None,
            "acierto_direccion": d.acierto_validacion if d else None,
            "siempre_sube_validacion": d.acierto_siempre_sube if d else None,
            "caida_desde_maximo": hoy.get("caida"), "caida_fuerte": hoy.get("caida_fuerte"),
            "panico": hoy.get("panico"),
        })
    del nombre_activo
    return filas


def _tarea(args) -> list[dict]:
    ticker, datos, mercado, macro = args
    try:
        return estudiar_activo(ticker, datos, mercado, macro)
    except Exception as e:  # un activo con datos raros no debe parar el estudio
        return [{"ticker": ticker, "error": repr(e)}]


def _salud(tickers: list[str], directorio: str, t0: float) -> None:
    """Estudio de salud financiera; con pocos tickers se fusiona con el existente."""
    from .catalogo import ACTIVOS
    from .salud import estudio_salud

    print(f"Salud financiera de {len(tickers)} empresas…")
    tabla = estudio_salud(tickers, {t: v[3] for t, v in ACTIVOS.items()})
    destino = os.path.join(directorio, "estudio_salud.csv")
    if os.path.exists(destino):
        previo = pd.read_csv(destino)
        # Una consulta fallida (p. ej. por el límite de Yahoo) nunca sustituye a datos buenos anteriores.
        sin_datos = tabla["salud"].eq("info") & tabla.get("margen", pd.Series(dtype=float)).isna()
        conservar = previo[previo["ticker"].isin(tabla.loc[sin_datos, "ticker"]) & previo["salud"].ne("info")]
        tabla = tabla[~tabla["ticker"].isin(conservar["ticker"])]
        tabla = pd.concat([previo[~previo["ticker"].isin(tabla["ticker"])], tabla], ignore_index=True)
    tabla.to_csv(destino, index=False)
    print(f"Guardado {destino} ({len(tabla)} empresas, {int(tabla['salud'].eq('info').sum())} sin datos) "
          f"en {time.time() - t0:.0f} s.")


def main(argv: list[str] | None = None) -> int:
    from .catalogo import ACTIVOS
    from .data import descargar_varios
    from .entrenamiento import DIRECTORIO
    from .ia import TICKERS_MACRO, datos_macro
    from .indices import MERCADO

    p = argparse.ArgumentParser(prog="analizador.estudio", description=__doc__.splitlines()[0], epilog=AVISO_LEGAL)
    p.add_argument("--tickers", nargs="*", help="Por defecto, todo el catálogo salvo los índices")
    p.add_argument("--nuevos", action="store_true", help="Solo los activos del catálogo que aún no están en el estudio")
    p.add_argument("--solo-salud", action="store_true",
                   help="Solo descarga las cuentas de las empresas (PER, margen, deuda…) y guarda su salud")
    p.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--directorio", default=str(DIRECTORIO))
    args = p.parse_args(argv)

    t0 = time.time()
    if args.solo_salud:
        _salud(args.tickers or [t for t, v in ACTIVOS.items() if v[1] == "Acción"], args.directorio, t0)
        return 0
    tickers = args.tickers or [t for t, v in ACTIVOS.items() if v[1] != "Índice"]
    ruta_previa = os.path.join(args.directorio, "estudio_activos.csv")
    if args.nuevos and os.path.exists(ruta_previa):
        ya = set(pd.read_csv(ruta_previa)["ticker"])
        tickers = [t for t in tickers if t not in ya]
        args.tickers = tickers  # activa la fusión con el estudio existente
    print(f"Descargando {len(tickers)} activos…")
    hoy = date.today()
    datos = descargar_varios(tickers, INICIO, hoy)
    mercado = descargar_varios([MERCADO], INICIO, hoy)[MERCADO]["Close"]
    macro = datos_macro(descargar_varios(list(TICKERS_MACRO.values()), INICIO, hoy))
    print(f"  {len(datos)} con datos. Calculando con {args.procesos} procesos…")

    filas = []
    tareas = [(t, df, mercado, macro) for t, df in datos.items()]
    with ProcessPoolExecutor(args.procesos, initializer=_cargar_modelos, initargs=(args.directorio,)) as ex:
        for i, resultado in enumerate(ex.map(_tarea, tareas, chunksize=2), 1):
            filas.extend(resultado)
            if i % 25 == 0:
                print(f"  {i}/{len(tareas)} ({time.time() - t0:.0f} s)")
    tabla = pd.DataFrame(filas)
    errores = tabla[tabla.get("error").notna()] if "error" in tabla else pd.DataFrame()
    if not errores.empty:
        print(f"  {len(errores)} activos con error: {', '.join(errores['ticker'])}")
        tabla = tabla[tabla["error"].isna()].drop(columns="error")
    destino = os.path.join(args.directorio, "estudio_activos.csv")
    if args.tickers and os.path.exists(destino):  # solo unos activos: se actualizan dentro del estudio existente
        previo = pd.read_csv(destino)
        tabla = pd.concat([previo[~previo["ticker"].isin(tabla["ticker"])], tabla], ignore_index=True)
    tabla.to_csv(destino, index=False)
    print(f"Guardado {destino} ({tabla['ticker'].nunique()} activos) en {time.time() - t0:.0f} s.")
    if not args.tickers:  # estudio completo: también el experimento de volumen con todas las empresas
        from .volumen import laboratorio_volumen

        acciones = {t: df for t, df in datos.items() if ACTIVOS.get(t, ("", ""))[1] == "Acción" and len(df) > 500}
        laboratorio_volumen(acciones).to_csv(os.path.join(args.directorio, "estudio_volumen.csv"), index=False)
        print("Experimento de volumen guardado.")
    acciones_estudiadas = [t for t in tabla["ticker"].unique() if ACTIVOS.get(t, ("", ""))[1] == "Acción"]
    _salud(acciones_estudiadas, args.directorio, t0)
    from .registro import registrar_predicciones

    print(f"Diario de la IA: {registrar_predicciones(tabla)} predicciones nuevas apuntadas.")
    return 0


if __name__ == "__main__":
    from analizador.estudio import main as _main  # nombre real del módulo (ver entrenamiento.py)

    raise SystemExit(_main())
