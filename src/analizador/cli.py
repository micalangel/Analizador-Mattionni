"""Interfaz de línea de comandos.

Ejemplo:  python -m analizador AAPL MSFT --horizonte 10 --confianza 0.8
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from . import AVISO_LEGAL
from . import backtest as bt
from .calibration import informe_calibracion
from .data import descargar_varios
from .forecast import METODOS, intervalo_actual
from .indicators import calcular_todos


def crear_parser() -> argparse.ArgumentParser:
    hoy = date.today()
    p = argparse.ArgumentParser(
        prog="analizador",
        description="Análisis de acciones y ETF: indicadores, backtesting, "
                    "intervalos de pronóstico y calibración.",
        epilog=AVISO_LEGAL,
    )
    p.add_argument("tickers", nargs="+", help="Uno o varios tickers, p. ej. AAPL SPY")
    p.add_argument("--inicio", default=str(hoy - timedelta(days=365 * 10)), help="Fecha inicial AAAA-MM-DD")
    p.add_argument("--fin", default=str(hoy), help="Fecha final AAAA-MM-DD")
    p.add_argument("--horizonte", "--horizon", type=int, default=10, help="Horizonte en días hábiles (10)")
    p.add_argument("--confianza", type=float, default=0.8, help="Nivel de confianza del intervalo (0.8)")
    p.add_argument("--metodo", choices=list(METODOS), default="ewma", help="Método del intervalo")
    p.add_argument("--estrategia", choices=list(bt.ESTRATEGIAS), default=list(bt.ESTRATEGIAS)[0])
    p.add_argument("--coste", type=float, default=0.001, help="Coste por operación, fracción (0.001 = 0,1 %%)")
    p.add_argument("--deslizamiento", type=float, default=0.0005, help="Deslizamiento, fracción")
    p.add_argument("--fundamentales", action="store_true", help="Muestra también ratios fundamentales")
    p.add_argument("--ia", action="store_true",
                   help="Entrena la IA (walk-forward) y la añade al informe de calibración")
    p.add_argument("--resultados", action="store_true",
                   help="Estudio de publicaciones de resultados y noticias recientes")
    return p


def _entrenar_ia(ticker: str, df: pd.DataFrame, args) -> dict | None:
    from . import ia
    from .data import descargar_precios
    from .resultados import obtener_resultados

    publicaciones = obtener_resultados(ticker).publicaciones
    mercado = None if ticker == "SPY" else descargar_precios("SPY", args.inicio, args.fin)["Close"]
    print("\nEntrenando la IA (walk-forward)…")
    tabla, modelos = ia.serie_intervalos_ia(df, args.horizonte, args.confianza, publicaciones, mercado)
    if not modelos:
        print("  No hay datos suficientes para entrenar la IA.")
        return None
    X = ia.construir_caracteristicas(df, publicaciones, mercado)
    final = ia.entrenar(X, ia.objetivo_escalado(df["Close"], args.horizonte), args.horizonte, args.confianza)
    iv = ia.intervalo_actual_ia(df, final, publicaciones, mercado)
    ruta = ia.guardar_modelo(final, ticker)
    print(f"  {len(modelos)} reentrenos. Rango IA al {args.confianza:.0%} a {args.horizonte} días: "
          f"{iv['inferior']:.2f} – {iv['superior']:.2f}. Modelo guardado en {ruta}")
    print("  Datos más usados: " + ", ".join(final.importancias().head(5).index))
    return {ia.NOMBRE_METODO: (ia.DESCRIPCION, tabla)}


def _imprimir_resultados(ticker: str, df: pd.DataFrame) -> None:
    from .noticias import AVISO_NOTICIAS, estudio_noticias, obtener_noticias
    from .resultados import AVISO_RESULTADOS, estudio_publicaciones, obtener_resultados, resumen_publicaciones

    datos = obtener_resultados(ticker)
    estudio = estudio_publicaciones(df["Close"], datos.publicaciones)
    print("\nPublicaciones de resultados y reacción del precio:")
    if estudio.empty:
        print("  Sin datos (habitual en ETF).")
    else:
        print(estudio[["BPA estimado", "BPA publicado", "Sorpresa %", "momento", "reaccion", "z"]]
              .tail(12).round(3).to_string())
        for k, v in resumen_publicaciones(estudio).items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
    print(AVISO_RESULTADOS)

    nts = estudio_noticias(df["Close"], obtener_noticias(ticker, 10))
    print("\nNoticias recientes:")
    for fecha, fila in nts.iterrows():
        mov = "pendiente" if pd.isna(fila["reaccion"]) else f"{fila['reaccion']:+.2%}"
        print(f"  {fecha.tz_convert('America/New_York'):%d/%m/%Y %H:%M}  [{mov:>9}]  {fila['titular'][:90]}"
              f" ({fila['fuente']})")
    print(AVISO_NOTICIAS)


def main(argv: list[str] | None = None) -> int:
    args = crear_parser().parse_args(argv)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    print(f"\nAVISO: {AVISO_LEGAL}\n")
    datos = descargar_varios(args.tickers, args.inicio, args.fin)
    faltan = [t for t in args.tickers if t.upper() not in datos]
    if faltan:
        print(f"Sin datos para: {', '.join(faltan)}")
    if not datos:
        return 1

    for ticker, df in datos.items():
        print("=" * 70)
        print(f"{ticker}: {len(df)} sesiones, {df.index[0].date()} a {df.index[-1].date()}")
        ultimo = calcular_todos(df).iloc[-1]
        print("\nIndicadores al último cierre:")
        print(ultimo.round(4).to_string())

        try:
            iv = intervalo_actual(df["Close"], args.horizonte, args.confianza, args.metodo)
            print(f"\nRango al {args.confianza:.0%} a {args.horizonte} días ({METODOS[args.metodo]}):"
                  f" {iv['inferior']:.2f} – {iv['superior']:.2f} (último cierre {iv['precio_ultimo']:.2f})")
            extra = None
            if args.ia:
                extra = _entrenar_ia(ticker, df, args)
            resumen, _ = informe_calibracion(df["Close"], args.horizonte, args.confianza, extra=extra)
            print("\nInforme de calibración:")
            print(resumen.drop(columns=["descripción"]).round(3).to_string())
        except ValueError as e:
            print(f"\nIntervalo/calibración no disponible: {e}")

        try:
            res = bt.comparar_con_referencias(df, bt.ESTRATEGIAS[args.estrategia](), nombre=args.estrategia,
                                              coste=args.coste, deslizamiento=args.deslizamiento)
            print("\nBacktest (walk-forward, con costes):")
            print(bt.tabla_metricas(res).round(3).to_string())
        except ValueError as e:
            print(f"\nBacktest no disponible: {e}")

        if args.resultados:
            _imprimir_resultados(ticker, df)

    if args.fundamentales:
        from .fundamentals import AVISO_FUNDAMENTALES, obtener_fundamentales
        print("=" * 70)
        print(obtener_fundamentales(list(datos)).round(3).to_string())
        print(AVISO_FUNDAMENTALES)

    print(f"\nAVISO: {AVISO_LEGAL}")
    return 0
