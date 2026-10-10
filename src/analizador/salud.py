"""Salud de la empresa: indicadores básicos con un semáforo orientativo.

Los umbrales son referencias generales para ayudar a leer las cifras, no reglas de inversión:
una empresa puede tener buenas razones para salirse de ellos (los bancos, por ejemplo, siempre
tienen mucha deuda porque es su negocio). Datos de Yahoo Finance, que pueden estar retrasados.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SECTORES_DEUDA_NORMAL = {"Banca", "Finanzas", "Inmobiliaria"}


@dataclass
class Indicador:
    nombre: str
    valor: str
    estado: str        # "bien", "regular", "mal" o "info"
    explicacion: str


def _es(texto: str) -> str:
    """Formato español: coma decimal."""
    return texto.replace(".", ",")


def _semaforo(valor, bien, mal, mayor_es_mejor=True) -> str:
    if valor is None or not np.isfinite(valor):
        return "info"
    if mayor_es_mejor:
        return "bien" if valor >= bien else "mal" if valor < mal else "regular"
    return "bien" if valor <= bien else "mal" if valor > mal else "regular"


def indicadores(fund: pd.Series, trimestral: pd.DataFrame, crecimiento: pd.DataFrame,
                sector: str = "—") -> list[Indicador]:
    """Construye los indicadores a partir de los ratios (``fundamentals``) y los estados (``resultados``)."""
    salida = []
    per = fund.get("PER")
    salida.append(Indicador(
        "PER (precio / beneficio)", "—" if pd.isna(per) else _es(f"{per:.1f} veces"), "info",
        "Cuántos años de beneficio actual paga el mercado por la empresa. La media histórica del S&P 500 ronda "
        "16; muy alto = se esperan grandes crecimientos, muy bajo = el mercado desconfía o está barata."))

    margen = fund.get("Margen de beneficio")
    salida.append(Indicador(
        "Margen de beneficio", "—" if pd.isna(margen) else _es(f"{margen:.1%}"), _semaforo(margen, 0.15, 0.05),
        "De cada 100 € que vende, cuántos le quedan de beneficio. Más del 15 % es muy rentable; menos del 5 %, poco."))

    deuda = fund.get("Deuda/Patrimonio")
    estado_deuda = "info" if sector in SECTORES_DEUDA_NORMAL else _semaforo(deuda, 0.5, 1.5, mayor_es_mejor=False)
    salida.append(Indicador(
        "Deuda / patrimonio", "—" if pd.isna(deuda) else _es(f"{deuda:.2f} veces"), estado_deuda,
        "Cuánta deuda tiene por cada euro de fondos propios. Menos de 0,5 es poca; más de 1,5, mucha."
        + (" En bancos, aseguradoras e inmobiliarias la deuda alta es normal en su negocio."
           if sector in SECTORES_DEUDA_NORMAL else "")))

    fcf = fund.get("Rentabilidad FCF")
    salida.append(Indicador(
        "Rentabilidad del flujo de caja", "—" if pd.isna(fcf) else _es(f"{fcf:.1%}"), _semaforo(fcf, 0.05, 0.0),
        "Dinero que genera de verdad cada año (después de invertir) en relación con lo que vale en bolsa. "
        "Por encima del 5 % es mucho; negativo significa que quema caja."))

    crec = np.nan
    if not crecimiento.empty and "Crec. ingresos (interanual)" in crecimiento:
        serie = crecimiento["Crec. ingresos (interanual)"].dropna()
        crec = float(serie.iloc[-1]) if len(serie) else np.nan
    salida.append(Indicador(
        "Crecimiento de las ventas", "—" if pd.isna(crec) else _es(f"{crec:+.1%}"), _semaforo(crec, 0.10, 0.0),
        "Ventas del último trimestre frente al mismo trimestre del año anterior. Más del 10 % es crecer mucho."))

    beneficio = np.nan
    if not trimestral.empty and "Beneficio neto" in trimestral:
        ultimos = trimestral["Beneficio neto"].dropna().iloc[-4:]
        beneficio = float(ultimos.sum()) if len(ultimos) == 4 else np.nan
    salida.append(Indicador(
        "Beneficio del último año", "—" if pd.isna(beneficio) else _millones(beneficio),
        "info" if pd.isna(beneficio) else ("bien" if beneficio > 0 else "mal"),
        "Suma del beneficio neto de los cuatro últimos trimestres. Negativo = está perdiendo dinero."))
    return salida


def resumen(lista: list[Indicador]) -> tuple[str, str]:
    """Frase de conjunto y su estado según cuántos indicadores salen bien o mal."""
    bien = sum(i.estado == "bien" for i in lista)
    mal = sum(i.estado == "mal" for i in lista)
    evaluados = sum(i.estado in {"bien", "regular", "mal"} for i in lista)
    if evaluados == 0:
        return "Sin datos suficientes para valorar su salud financiera", "info"
    regular = evaluados - bien - mal
    if mal == 0 and bien >= max(2, regular):
        return "Cuentas sólidas según estos indicadores", "bien"
    if mal >= 2 and mal >= bien:
        return "Cuentas con puntos débiles según estos indicadores", "mal"
    return "Cuentas con luces y sombras según estos indicadores", "regular"


def _millones(x: float) -> str:
    if abs(x) >= 1e9:
        return f"{x / 1e9:,.1f} mil millones".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{x / 1e6:,.0f} millones".replace(",", ".")


# --- Estudio de todas las empresas -------------------------------------------------------

def valores_numericos(fund: pd.Series, trimestral: pd.DataFrame, crecimiento: pd.DataFrame) -> dict:
    """Cifras en bruto (para tablas y filtros), además de los indicadores con semáforo."""
    crec = np.nan
    if not crecimiento.empty and "Crec. ingresos (interanual)" in crecimiento:
        serie = crecimiento["Crec. ingresos (interanual)"].dropna()
        crec = float(serie.iloc[-1]) if len(serie) else np.nan
    beneficio = np.nan
    if not trimestral.empty and "Beneficio neto" in trimestral:
        ultimos = trimestral["Beneficio neto"].dropna().iloc[-4:]
        beneficio = float(ultimos.sum()) if len(ultimos) == 4 else np.nan
    return {"per": fund.get("PER"), "precio_valor_contable": fund.get("Precio/Valor contable"),
            "margen": fund.get("Margen de beneficio"), "deuda_patrimonio": fund.get("Deuda/Patrimonio"),
            "rentabilidad_fcf": fund.get("Rentabilidad FCF"), "crecimiento_ventas": crec,
            "beneficio_anual": beneficio, "moneda": fund.get("Moneda")}


def estudiar_empresa(ticker: str, sector: str = "—") -> dict:
    """Descarga ratios y cuentas de una empresa y devuelve su fila del estudio de salud."""
    from .fundamentals import obtener_fundamentales
    from .resultados import margenes_y_crecimiento, obtener_resultados

    fund = obtener_fundamentales([ticker]).iloc[0]
    res = obtener_resultados(ticker)
    trimestral = res.trimestral.dropna(subset=["Ingresos"])
    crecimiento = margenes_y_crecimiento(trimestral, 4) if not trimestral.empty else trimestral
    lista = indicadores(fund, trimestral, crecimiento, sector)
    frase, estado = resumen(lista)
    return {"ticker": ticker, **valores_numericos(fund, trimestral, crecimiento),
            "salud": estado, "salud_texto": frase,
            "semaforos": ";".join(f"{i.nombre}={i.estado}" for i in lista)}


def estudio_salud(tickers: list[str], sectores: dict[str, str], hilos: int = 6, verbose: bool = True) -> pd.DataFrame:
    """Salud de muchas empresas en paralelo (la descarga es lo lento, así que se usan hilos)."""
    from concurrent.futures import ThreadPoolExecutor

    def una(t):
        try:
            return estudiar_empresa(t, sectores.get(t, "—"))
        except Exception as e:  # una empresa sin datos no debe parar el estudio
            return {"ticker": t, "salud": "info", "salud_texto": f"Sin datos ({type(e).__name__})"}

    import time

    filas = {}
    with ThreadPoolExecutor(hilos) as ex:
        for i, fila in enumerate(ex.map(una, tickers), 1):
            filas[fila["ticker"]] = fila
            if verbose and i % 50 == 0:
                print(f"  salud: {i}/{len(tickers)}")
    # Yahoo limita las consultas seguidas (error 429): las que fallaron se reintentan despacio.
    for intento in range(2):
        pendientes = [t for t, f in filas.items() if pd.isna(f.get("margen")) and pd.isna(f.get("per"))]
        if not pendientes:
            break
        if verbose:
            print(f"  reintentando {len(pendientes)} empresas sin datos (pausa {30 * (intento + 1)} s)…")
        time.sleep(30 * (intento + 1))
        for t in pendientes:
            filas[t] = una(t)
            time.sleep(1.0)
    return pd.DataFrame(filas.values())


# --- PER de referencia ------------------------------------------------------------------

MINIMO_EMPRESAS_SECTOR = 4


def per_de_referencia(salud: pd.DataFrame, sectores: dict[str, str]) -> tuple[dict[str, float], float]:
    """Mediana del PER por sector y de todas las empresas (solo PER positivos y razonables, < 200).

    Se usa la mediana y no la media porque unas pocas empresas con PER enormes (beneficio casi nulo)
    distorsionarían la media.
    """
    tabla = salud[["ticker", "per"]].copy()
    tabla["per"] = pd.to_numeric(tabla["per"], errors="coerce")
    tabla = tabla[(tabla["per"] > 0) & (tabla["per"] < 200)]
    tabla["sector"] = tabla["ticker"].map(sectores)
    por_sector = tabla.groupby("sector")["per"].agg(["median", "size"])
    medianas = por_sector.loc[por_sector["size"] >= MINIMO_EMPRESAS_SECTOR, "median"].to_dict()
    return {k: float(v) for k, v in medianas.items()}, float(tabla["per"].median())


def comparar_per(per: float | None, referencia: float | None) -> tuple[str, float | None]:
    """Etiqueta del PER frente a una referencia: por debajo (<−20 %), en la media o por encima (>+25 %)."""
    if per is None or referencia is None or not np.isfinite(per) or not np.isfinite(referencia) or per <= 0:
        return "—", None
    diferencia = per / referencia - 1
    if diferencia < -0.20:
        return "Por debajo de su sector", diferencia
    if diferencia > 0.25:
        return "Por encima de su sector", diferencia
    return "En la media de su sector", diferencia
