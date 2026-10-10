"""Entrenamiento global de la IA con los principales índices bursátiles de cada país.

Por qué una cesta: un solo ticker con 10 años tiene pocas observaciones independientes,
sobre todo a 6 meses o 1 año. Como el objetivo de la IA está escalado por la volatilidad
(z = retorno / σ), los patrones son comparables entre mercados y se pueden aprender
juntos. Con ~37 índices desde 1990 hay cientos de miles de muestras, incluidas las crisis
asiática (1997), puntocom (2000), financiera (2008), del euro (2011) y la COVID (2020).

Sin mirar al futuro
-------------------
Para cada año Y se entrena un modelo solo con muestras cuyo resultado (precio en s + h)
ya se conocía el 1 de enero de Y. Ese modelo es el que emite los intervalos de todas las
fechas del año Y, en cualquier ticker. Además se entrena un modelo "actual" con todo lo
disponible, que solo se usa para el rango de hoy.

Uso:  python -m analizador.entrenamiento            (unos minutos; guarda en modelos/)
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import AVISO_LEGAL
from .data import descargar_varios
from .forecast import HORIZONTES, cuantil_z
from .ia import ModeloIA, construir_caracteristicas, objetivo_escalado, sigma_ewma_h
from .indices import INDICES, MERCADO

CESTA = list(INDICES)
CONFIANZAS = (0.5, 0.8, 0.95)
DIRECTORIO = Path(__file__).resolve().parents[2] / "modelos" / "global"
PRIMER_ANIO = 1996
PASO_ANIOS = 2  # un modelo nuevo cada 2 años (cada fecha usa el último entrenado antes de su año)


@dataclass
class ConjuntoIA:
    """Modelos de un horizonte: {confianza: {año: ModeloIA}} más el modelo actual por confianza."""
    horizonte: int
    por_anio: dict = field(default_factory=dict)
    actual: dict = field(default_factory=dict)
    metadatos: dict = field(default_factory=dict)

    def modelo_para(self, confianza: float, anio: int) -> ModeloIA | None:
        modelos = self.por_anio.get(confianza, {})
        validos = [a for a in modelos if a <= anio]
        return modelos[max(validos)] if validos else None


def _estimador(cuantil: float, semilla: int = 0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="quantile", quantile=cuantil, max_iter=150, learning_rate=0.05,
        max_leaf_nodes=4, min_samples_leaf=500, l2_regularization=1.0, random_state=semilla,
    )


def preparar_panel(datos: dict[str, pd.DataFrame], mercado: pd.Series | None, horizonte: int) -> pd.DataFrame:
    """Une las características y el objetivo de todos los tickers en una sola tabla."""
    partes = []
    for ticker, df in datos.items():
        X = construir_caracteristicas(df, None, mercado)
        X["z"] = objetivo_escalado(df["Close"], horizonte)
        fechas = pd.Series(df.index, index=df.index)
        X["fecha"] = df.index
        X["fecha_resultado"] = fechas.shift(-horizonte)  # cuándo se conoce el resultado
        X["ticker"] = ticker
        partes.append(X.dropna(subset=["z", "vol_60"]))
    panel = pd.concat(partes, ignore_index=True)
    return panel.sort_values("fecha", kind="stable").reset_index(drop=True)


def columnas_caracteristicas(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if c not in {"z", "fecha", "fecha_resultado", "ticker"}]


def entrenar_panel(panel: pd.DataFrame, horizonte: int, confianza: float, corte: pd.Timestamp,
                   fraccion_calibracion: float = 0.25, minimo: int = 3000) -> ModeloIA | None:
    """Entrena con las muestras cuyo resultado se conocía antes de ``corte``.

    El último 25 % (por fecha) se reserva para la calibración conformal; las muestras de
    ajuste deben tener su resultado antes de que empiece ese tramo (sin solapamiento).
    """
    usable = panel[panel["fecha_resultado"] < corte]
    if len(usable) < minimo:
        return None
    columnas = [c for c in columnas_caracteristicas(usable) if usable[c].notna().mean() > 0.5]
    inicio_cal = usable["fecha"].quantile(1 - fraccion_calibracion)
    cal = usable[usable["fecha"] >= inicio_cal]
    ajuste = usable[usable["fecha_resultado"] < inicio_cal]
    if len(ajuste) < minimo * 0.6 or len(cal) < 200:
        return None

    alfa = 1 - confianza
    modelo = ModeloIA(horizonte, confianza, columnas, n_muestras=len(ajuste), n_calibracion=len(cal),
                      entrenado_hasta=pd.Timestamp(corte))
    for nombre, q in (("inferior", alfa / 2), ("superior", 1 - alfa / 2)):
        modelo.modelos[nombre] = _estimador(q).fit(ajuste[columnas], ajuste["z"])
    q_cal = modelo.predecir_cuantiles(cal[columnas])
    errores = np.maximum(q_cal["q_inf"] - cal["z"], cal["z"] - q_cal["q_sup"]).to_numpy()
    nivel = min(1.0, np.ceil((len(cal) + 1) * (1 - alfa)) / len(cal))
    modelo.ajuste_conformal = float(np.quantile(errores, nivel, method="higher"))
    return modelo


def _importancias(modelo: ModeloIA, muestra: pd.DataFrame, semilla: int = 0) -> dict:
    """Importancia por permutación: cuánto empeora la pérdida cuantílica al barajar cada dato."""
    rng = np.random.default_rng(semilla)
    X = muestra[modelo.columnas]
    y = muestra["z"].to_numpy()
    alfa = 1 - modelo.confianza

    def perdida(Xp):
        q = modelo.predecir_cuantiles(Xp)
        total = 0.0
        for col, tau in (("q_inf", alfa / 2), ("q_sup", 1 - alfa / 2)):
            e = y - q[col].to_numpy()
            total += np.mean(np.maximum(tau * e, (tau - 1) * e))
        return total

    base = perdida(X)
    imp = {}
    for c in modelo.columnas:
        Xp = X.copy()
        Xp[c] = rng.permutation(Xp[c].to_numpy())
        imp[c] = max(perdida(Xp) - base, 0.0)
    suma = sum(imp.values()) or 1.0
    return {c: v / suma for c, v in imp.items()}


def entrenar_horizonte(datos: dict[str, pd.DataFrame], mercado: pd.Series | None, horizonte: int,
                       confianzas=CONFIANZAS, primer_anio: int = PRIMER_ANIO,
                       hoy: pd.Timestamp | None = None, verbose: bool = True) -> ConjuntoIA:
    panel = preparar_panel(datos, mercado, horizonte)
    hoy = pd.Timestamp(hoy or date.today())
    conjunto = ConjuntoIA(horizonte, metadatos={
        "cesta": sorted(datos), "entrenado": str(hoy.date()), "muestras_totales": len(panel),
        "desde": str(panel["fecha"].min().date()), "hasta": str(panel["fecha"].max().date()),
    })
    for conf in confianzas:
        conjunto.por_anio[conf] = {}
        for anio in range(primer_anio, hoy.year + 1, PASO_ANIOS):
            m = entrenar_panel(panel, horizonte, conf, pd.Timestamp(f"{anio}-01-01"))
            if m is not None:
                conjunto.por_anio[conf][anio] = m
        actual = entrenar_panel(panel, horizonte, conf, hoy + pd.Timedelta(days=1))
        if actual is not None:
            muestra = panel.tail(20_000)
            actual.metadatos["importancias"] = _importancias(actual, muestra)
            conjunto.actual[conf] = actual
        if verbose:
            print(f"  h={horizonte:>3} conf={conf:.0%}: {len(conjunto.por_anio[conf])} modelos anuales "
                  f"+ actual ({len(panel):,} muestras)")
    conjunto.metadatos["evaluacion"] = evaluar_conjunto(panel, conjunto)
    return conjunto


def evaluar_conjunto(panel: pd.DataFrame, conjunto: ConjuntoIA) -> dict:
    """Cobertura fuera de muestra de la cesta: cada año evaluado con el modelo de ese año."""
    resultado = {}
    anios = panel["fecha"].dt.year
    for conf, modelos in conjunto.por_anio.items():
        dentro = []
        for anio, m in modelos.items():
            filas = panel[anios == anio]
            if filas.empty:
                continue
            q = m.predecir_cuantiles(filas[m.columnas])
            dentro.append(((filas["z"] >= q["q_inf"]) & (filas["z"] <= q["q_sup"])).to_numpy())
        if dentro:
            d = np.concatenate(dentro)
            resultado[str(conf)] = {"cobertura": float(d.mean()), "n": int(len(d))}
    return resultado


def ruta(horizonte: int, directorio: Path | str = DIRECTORIO) -> Path:
    return Path(directorio) / f"ia_global_h{horizonte}.joblib"


def guardar(conjunto: ConjuntoIA, directorio: Path | str = DIRECTORIO) -> Path:
    import joblib
    import sklearn

    conjunto.metadatos["sklearn"] = sklearn.__version__
    destino = ruta(conjunto.horizonte, directorio)
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(conjunto, destino, compress=3)
    return destino


def cargar(horizonte: int, directorio: Path | str = DIRECTORIO) -> ConjuntoIA | None:
    """Carga los modelos de un horizonte; None si no existen o son de otra versión de sklearn."""
    import joblib
    import sklearn

    origen = ruta(horizonte, directorio)
    if not origen.exists():
        return None
    try:
        conjunto = joblib.load(origen)
    except Exception as e:  # no se oculta: un modelo que no carga es un fallo que hay que ver
        warnings.warn(f"No se pudo cargar {origen.name}: {e!r}. Reentrena con "
                      "`python -m analizador.entrenamiento`.", stacklevel=2)
        return None
    if conjunto.metadatos.get("sklearn") != sklearn.__version__:
        warnings.warn(f"{origen.name} se entrenó con scikit-learn {conjunto.metadatos.get('sklearn')} y está "
                      f"instalada la {sklearn.__version__}: reentrena la IA.", stacklevel=2)
        return None
    return conjunto


# --- Aplicar a un ticker ---------------------------------------------------------------

def serie_intervalos_global(datos: pd.DataFrame, conjunto: ConjuntoIA, confianza: float,
                            mercado: pd.Series | None = None) -> pd.DataFrame:
    """Intervalos walk-forward de un ticker: cada fecha usa el modelo de su año (fuera de muestra)."""
    precios = datos["Close"].astype(float)
    h = conjunto.horizonte
    X = construir_caracteristicas(datos, None, mercado)
    sigma = sigma_ewma_h(precios, h)
    q_inf = pd.Series(np.nan, index=precios.index)
    q_sup = pd.Series(np.nan, index=precios.index)
    for anio in sorted(set(precios.index.year)):
        m = conjunto.modelo_para(confianza, anio)
        if m is None:
            continue
        filas = precios.index.year == anio
        bloque = X.loc[filas].reindex(columns=m.columnas)
        q = m.predecir_cuantiles(bloque)
        q_inf[filas] = q["q_inf"].to_numpy()
        q_sup[filas] = q["q_sup"].to_numpy()
    return pd.DataFrame({
        "inferior": precios * np.exp(q_inf * sigma),
        "superior": precios * np.exp(q_sup * sigma),
        "centro": precios.where(q_inf.notna() & sigma.notna()),
        "sigma_h": (q_sup - q_inf) * sigma / (2 * cuantil_z(confianza)),
    })


def intervalo_actual_global(datos: pd.DataFrame, conjunto: ConjuntoIA, confianza: float,
                            mercado: pd.Series | None = None) -> dict | None:
    modelo = conjunto.actual.get(confianza)
    if modelo is None:
        return None
    precios = datos["Close"].astype(float)
    X = construir_caracteristicas(datos, None, mercado).iloc[[-1]].reindex(columns=modelo.columnas)
    q = modelo.predecir_cuantiles(X).iloc[0]
    sigma = sigma_ewma_h(precios, conjunto.horizonte).iloc[-1]
    if not np.isfinite(sigma):
        return None
    p = float(precios.iloc[-1])
    return {"fecha": precios.index[-1], "precio_ultimo": p,
            "inferior": float(p * np.exp(q["q_inf"] * sigma)), "superior": float(p * np.exp(q["q_sup"] * sigma)),
            "confianza": confianza, "horizonte": conjunto.horizonte, "metodo": "ia"}


# --- Estudio de índices ----------------------------------------------------------------

RUTA_ESTUDIO = DIRECTORIO / "estudio_indices.csv"


def estudio_indices(datos: dict[str, pd.DataFrame], conjuntos: dict, mercado: pd.Series | None,
                    confianza: float = 0.8, verbose: bool = True, direcciones: dict | None = None,
                    macro: pd.DataFrame | None = None) -> pd.DataFrame:
    """Proyección y fiabilidad comprobada de cada índice a todos los plazos (formato largo)."""
    from .proyeccion import NOMBRES_CORTOS, proyectar

    filas = []
    for ticker, df in datos.items():
        pais, nombre, region = INDICES.get(ticker, (ticker, ticker, "Otros"))
        proy = proyectar(df, confianza, conjuntos, None if ticker == MERCADO else mercado, direcciones=direcciones,
                         macro=macro)
        for etiqueta, p in proy.items():
            r, f, d = p.rango or {}, p.fiabilidad, p.direccion
            filas.append({
                "ticker": ticker, "país": pais, "índice": nombre, "región": region,
                "fecha": str(df.index[-1].date()), "cierre": float(df["Close"].iloc[-1]),
                "plazo": etiqueta, "sesiones": p.horizonte, "confianza": confianza,
                "inferior": r.get("inferior"), "superior": r.get("superior"),
                "cobertura_validacion": f.get("cobertura"), "n_independientes": f.get("n_independientes"),
                "estado": p.estado, "método": NOMBRES_CORTOS.get(p.metodo, p.metodo),
                "historia_desde": str(df.index[0].date()),
                "prob_sube": d.probabilidad if d else None,
                "fuente_direccion": d.fuente if d else None,
                "acierto_direccion": d.acierto_validacion if d else None,
            })
        if verbose:
            print(f"  estudio {ticker:<11} {pais}")
    return pd.DataFrame(filas)


# --- Línea de comandos -----------------------------------------------------------------

def _entrenar_direccion(datos, mercado, h, directorio, macro=None):
    """Entrena las dos variantes (sin y con VIX/tipos); cada activo elegirá luego la suya."""
    from . import direccion

    resultado = {}
    for clave, mc in (("ia", None), ("ia_macro", macro)):
        if clave == "ia_macro" and macro is None:
            continue
        conjunto = direccion.entrenar_conjunto(datos, mercado, h, macro=mc)
        destino = direccion.guardar(conjunto, directorio)
        ev = conjunto.metadatos["evaluacion"]
        print(f"  Guardado {destino.name}: acierto de dirección {ev['acierto_ia']:.1%} frente a "
              f"{ev['acierto_frecuencia']:.1%} de la frecuencia histórica (mejora Brier {ev['mejora']:+.2%})")
        resultado[clave] = conjunto
    return resultado


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="analizador.entrenamiento",
                                description="Entrena la IA global con una cesta de valores y ETF.",
                                epilog=AVISO_LEGAL)
    p.add_argument("--inicio", default="1990-01-01")
    p.add_argument("--cesta", nargs="*", default=CESTA)
    p.add_argument("--horizontes", nargs="*", type=int, default=list(HORIZONTES.values()))
    p.add_argument("--directorio", default=str(DIRECTORIO))
    p.add_argument("--solo-estudio", action="store_true",
                   help="No reentrena: rehace el estudio de índices con los modelos guardados")
    p.add_argument("--solo-direccion", action="store_true",
                   help="Reentrena solo la IA de dirección (usa los modelos de rangos guardados)")
    args = p.parse_args(argv)

    t0 = time.time()
    print(f"Descargando {len(args.cesta)} tickers desde {args.inicio}…")
    datos = descargar_varios(args.cesta, args.inicio, date.today())
    print(f"  {len(datos)} con datos: {', '.join(datos)}")
    mercado = datos[MERCADO]["Close"] if MERCADO in datos else None
    from . import direccion
    from .ia import TICKERS_MACRO, datos_macro

    macro = datos_macro(descargar_varios(list(TICKERS_MACRO.values()), args.inicio, date.today()))

    conjuntos, direcciones = {}, {}
    for h in args.horizontes:
        if args.solo_estudio or args.solo_direccion:
            conjuntos[h] = cargar(h, args.directorio)
            if args.solo_direccion:
                direcciones[h] = _entrenar_direccion(datos, mercado, h, args.directorio, macro)
            else:
                direcciones[h] = {v: direccion.cargar(h, args.directorio, v) for v in direccion.VARIANTES}
            continue
        conjunto = entrenar_horizonte(datos, mercado, h)
        destino = guardar(conjunto, args.directorio)
        conjuntos[h] = conjunto
        ev = conjunto.metadatos["evaluacion"]
        resumen = ", ".join(f"{float(c):.0%}→{v['cobertura']:.1%}" for c, v in ev.items())
        print(f"  Guardado {destino.name} ({destino.stat().st_size / 1e6:.1f} MB). "
              f"Cobertura fuera de muestra en la cesta: {resumen}")
        direcciones[h] = _entrenar_direccion(datos, mercado, h, args.directorio, macro)

    if set(args.cesta) <= set(INDICES):
        print("Estudio de índices…")
        estudio = estudio_indices(datos, {h: c for h, c in conjuntos.items() if c is not None}, mercado,
                                  direcciones=direcciones, macro=macro)
        destino = Path(args.directorio) / RUTA_ESTUDIO.name
        estudio.to_csv(destino, index=False)
        print(f"  Guardado {destino}")
        from .registro import registrar_predicciones

        print(f"  Diario de la IA: {registrar_predicciones(estudio)} predicciones nuevas apuntadas.")
    if not args.solo_estudio:
        from .registro import filas_entrenamiento, registrar_entrenamiento

        registrar_entrenamiento(filas_entrenamiento(
            {} if args.solo_direccion else conjuntos, direcciones))
    print(f"Listo en {time.time() - t0:.0f} s.")
    return 0


if __name__ == "__main__":
    # Se delega en el módulo importado por su nombre real: si no, las clases se guardarían como
    # «__main__.ConjuntoIA» y la web no podría cargar los modelos.
    from analizador.entrenamiento import main as _main

    raise SystemExit(_main())
