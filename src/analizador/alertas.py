"""Alertas de «Mi lista»: hechos que conviene saber hoy de cada activo.

No son señales de compra o venta: avisan de situaciones (caída fuerte, volumen inusual, pánico en
el mercado, resultados cercanos, movimiento muy grande) para que el usuario las mire con calma.

Uso:  python -m analizador.alertas [--notificar]   (con --notificar, aviso del sistema en macOS)
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from . import AVISO_LEGAL

DIAS_RESULTADOS = 7


@dataclass
class Alerta:
    ticker: str
    tipo: str          # "caida", "volumen", "panico", "resultados", "movimiento"
    texto: str
    importancia: int   # 1 = informativa, 2 = a tener en cuenta


def evaluar(ticker: str, nombre: str, datos: pd.DataFrame, macro: pd.DataFrame | None = None,
            proxima_publicacion: pd.Timestamp | None = None, hoy: pd.Timestamp | None = None) -> list[Alerta]:
    """Alertas de un activo con los datos hasta el último cierre."""
    from . import situaciones as sit
    from . import volumen as vol

    alertas: list[Alerta] = []
    c = datos["Close"].dropna().astype(float)
    if len(c) < 2:
        return alertas
    hoy = pd.Timestamp(hoy or date.today())

    if len(c) >= 120:
        estado = sit.situacion_hoy(c, macro)
        if estado["caida_fuerte"]:
            alertas.append(Alerta(ticker, "caida", f"{nombre} está un {abs(estado['caida']):.0%} por debajo de su "
                                                     "máximo del último año (caída fuerte).", 2))
        if estado["panico"]:
            alertas.append(Alerta(ticker, "panico", f"Pánico en los mercados: el índice del miedo (VIX) está en "
                                                      f"{estado['vix']:.0f}.", 2))

    if "Volume" in datos and (datos["Volume"] > 0).mean() > 0.8:
        z = vol.z_volumen(datos).iloc[-1]
        rel = vol.volumen_relativo(datos).iloc[-1]
        if pd.notna(z) and z >= vol.UMBRAL_Z:
            alertas.append(Alerta(ticker, "volumen", f"Volumen inusual en {nombre}: {rel:.1f} veces lo normal en la "
                                                       "última sesión.", 1))

    r = np.log(c / c.shift(1))
    sigma = r.iloc[-61:-1].std()
    ultimo = r.iloc[-1]
    if pd.notna(sigma) and sigma > 0 and abs(ultimo) >= 3 * sigma:
        alertas.append(Alerta(ticker, "movimiento", f"{nombre} se movió un {np.expm1(ultimo):+.1%} en la última "
                                                      "sesión, mucho más de lo habitual.", 2))

    if proxima_publicacion is not None and pd.notna(proxima_publicacion):
        fecha = pd.Timestamp(proxima_publicacion)
        fecha = fecha.tz_convert(None) if fecha.tzinfo else fecha
        dias = (fecha.normalize() - hoy.normalize()).days
        if 0 <= dias <= DIAS_RESULTADOS:
            cuando = "hoy" if dias == 0 else "mañana" if dias == 1 else f"en {dias} días"
            alertas.append(Alerta(ticker, "resultados", f"{nombre} publica resultados {cuando} "
                                                          f"({fecha:%d/%m/%Y}).", 1))
    return alertas


def notificar_macos(titulo: str, texto: str) -> None:
    """Aviso del sistema en macOS (solo en el ordenador donde se ejecuta)."""
    texto = texto.replace('"', "'")
    subprocess.run(["osascript", "-e", f'display notification "{texto}" with title "{titulo}"'],
                   check=False, capture_output=True)


def main(argv: list[str] | None = None) -> int:
    from .calendario import proximas_publicaciones
    from .catalogo import nombre
    from .data import descargar_varios
    from .ia import TICKERS_MACRO, datos_macro
    from .mi_lista import cargar

    p = argparse.ArgumentParser(prog="analizador.alertas", description="Alertas de «Mi lista».", epilog=AVISO_LEGAL)
    p.add_argument("--notificar", action="store_true", help="Muestra un aviso del sistema (macOS) si hay alertas")
    args = p.parse_args(argv)

    lista = cargar()
    if not lista:
        print("«Mi lista» está vacía: añade activos desde la web.")
        return 0
    hoy = date.today()
    datos = descargar_varios(lista, "2015-01-01", hoy)
    macro = datos_macro(descargar_varios(list(TICKERS_MACRO.values()), "2015-01-01", hoy))
    proximas = proximas_publicaciones(lista)
    todas = []
    for t, df in datos.items():
        todas += evaluar(t, nombre(t), df, macro, proximas.get(t, {}).get("fecha"))
    for a in todas:
        print(f"{'⚠️' if a.importancia > 1 else 'ℹ️'}  {a.texto}")
    if not todas:
        print("Sin alertas hoy.")
    if args.notificar and todas:
        notificar_macos("Mattionni Analytics", f"{len(todas)} alertas en tu lista: " + " · ".join(a.texto for a in todas[:2]))
    return 0


if __name__ == "__main__":
    from analizador.alertas import main as _main  # nombre real del módulo (ver entrenamiento.py)

    raise SystemExit(_main())
