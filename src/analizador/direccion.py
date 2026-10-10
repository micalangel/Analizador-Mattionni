"""¿Subirá o bajará? Probabilidad de que el precio esté más arriba dentro de h sesiones.

Dos fuentes, y se usa la que mejor haya funcionado en datos que no se usaron para elegir:

1. **Frecuencia histórica** del propio activo: de todas las veces que se pudo comprobar
   hasta hoy, ¿en qué proporción estaba más arriba h sesiones después? Es sencilla pero
   muy informativa: los índices suben más a menudo de lo que bajan, sobre todo a largo plazo.
2. **IA de dirección**: clasificador (gradient boosting) entrenado con los principales
   índices mundiales desde 1990 con las mismas características que la IA de rangos, más
   la frecuencia histórica. Como la IA de rangos, se entrena un modelo cada dos años solo
   con resultados ya conocidos, y cada fecha usa el modelo de su año (fuera de muestra).

Medida de calidad: **puntuación de Brier** (error cuadrático de la probabilidad; menor es
mejor) y su **mejora** frente a la frecuencia histórica: 1 − Brier_IA / Brier_frecuencia.
Una mejora de 0 significa que la IA no aporta nada sobre la frecuencia histórica.

Es una estimación con incertidumbre, no una recomendación.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .ia import construir_caracteristicas

MINIMO_FRECUENCIA = 250  # comprobaciones mínimas para fiarse de la frecuencia histórica

# Frecuencia con que los 37 índices mundiales estuvieron más arriba al cumplirse cada plazo (1990-2026).
# Se usa para corregir el sesgo de supervivencia de la historia de cada empresa (ver ``frecuencia_ajustada``).
BASE_MERCADO = {5: 0.553, 10: 0.567, 21: 0.582, 126: 0.627, 252: 0.654}
PESO_MERCADO = 20  # equivale a 20 periodos independientes de evidencia a favor de la media del mercado


def frecuencia_ajustada(frecuencia: float, sesiones_historia: int, horizonte: int) -> float:
    """Acerca la frecuencia de subida de un activo a la del mercado cuanto menos historia tiene.

    Las empresas que hoy están en la lista son las que sobrevivieron y crecieron, así que su
    historia es demasiado optimista (sesgo de supervivencia). Con n periodos independientes
    (sesiones / horizonte) la estimación es (n·f + 20·f_mercado) / (n + 20): con 10 años a
    1 año vista pesa 1/3 la empresa y 2/3 el mercado; con 30 días a 1 mes, al revés.
    """
    base = BASE_MERCADO.get(horizonte, 0.6)
    n = max(sesiones_historia / horizonte, 0)
    return float((n * frecuencia + PESO_MERCADO * base) / (n + PESO_MERCADO))


def frecuencia_historica(precios: pd.Series, horizonte: int) -> pd.Series:
    """En cada t, proporción de periodos de h sesiones terminados hasta t que acabaron al alza."""
    subio = (np.log(precios / precios.shift(horizonte)) > 0).astype(float)
    subio[precios.shift(horizonte).isna()] = np.nan  # periodo aún sin comprobar
    return subio.expanding(min_periods=MINIMO_FRECUENCIA).mean()


def caracteristicas_direccion(datos: pd.DataFrame, horizonte: int, mercado: pd.Series | None,
                              macro: pd.DataFrame | None = None) -> pd.DataFrame:
    X = construir_caracteristicas(datos, None, mercado, macro)
    X["frecuencia_subida"] = frecuencia_historica(datos["Close"].astype(float), horizonte)
    return X


@dataclass
class ModeloDireccion:
    columnas: list[str]
    modelo: object
    entrenado_hasta: pd.Timestamp
    n_muestras: int

    def probabilidad(self, X: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(X.reindex(columns=self.columnas))[:, 1]


@dataclass
class ConjuntoDireccion:
    horizonte: int
    por_anio: dict = field(default_factory=dict)
    actual: ModeloDireccion | None = None
    metadatos: dict = field(default_factory=dict)

    def modelo_para(self, anio: int) -> ModeloDireccion | None:
        validos = [a for a in self.por_anio if a <= anio]
        return self.por_anio[max(validos)] if validos else None


def preparar_panel(datos: dict[str, pd.DataFrame], mercado: pd.Series | None, horizonte: int,
                   macro: pd.DataFrame | None = None) -> pd.DataFrame:
    partes = []
    for ticker, df in datos.items():
        c = df["Close"].astype(float)
        X = caracteristicas_direccion(df, horizonte, mercado, macro)
        futuro = np.log(c.shift(-horizonte) / c)
        X["sube"] = (futuro > 0).astype(float).where(futuro.notna())
        X["fecha"] = df.index
        X["fecha_resultado"] = pd.Series(df.index, index=df.index).shift(-horizonte)
        X["ticker"] = ticker
        partes.append(X.dropna(subset=["sube", "vol_60", "frecuencia_subida"]))
    return pd.concat(partes, ignore_index=True).sort_values("fecha", kind="stable").reset_index(drop=True)


def _columnas(panel: pd.DataFrame) -> list[str]:
    fuera = {"sube", "fecha", "fecha_resultado", "ticker"}
    return [c for c in panel.columns if c not in fuera and panel[c].notna().mean() > 0.5]


def entrenar(panel: pd.DataFrame, corte: pd.Timestamp, minimo: int = 3000) -> ModeloDireccion | None:
    """Clasificador con las muestras cuyo resultado se conocía antes de ``corte``."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    usable = panel[panel["fecha_resultado"] < corte]
    if len(usable) < minimo:
        return None
    columnas = _columnas(usable)
    # Muy prudente a propósito: con 150 iteraciones y hojas pequeñas la IA aprendía ruido y
    # empeoraba a la frecuencia histórica. Configuración elegida comparando tres variantes en
    # la misma cesta (pequeño sesgo optimista, ver docs/METHODOLOGY.md).
    modelo = HistGradientBoostingClassifier(
        max_iter=40, learning_rate=0.03, max_leaf_nodes=3, min_samples_leaf=3000, random_state=0,
    ).fit(usable[columnas], usable["sube"].astype(int))
    return ModeloDireccion(columnas, modelo, pd.Timestamp(corte), len(usable))


def brier(prob: np.ndarray, resultado: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(resultado)) ** 2))


def evaluar_panel(panel: pd.DataFrame, conjunto: ConjuntoDireccion) -> dict:
    """Calidad fuera de muestra en la cesta: cada año con el modelo de su año."""
    anios = panel["fecha"].dt.year
    p_ia, p_frec, y = [], [], []
    for anio in sorted(anios.unique()):
        m = conjunto.modelo_para(anio)
        if m is None:
            continue
        filas = panel[anios == anio]
        p_ia.append(m.probabilidad(filas))
        p_frec.append(filas["frecuencia_subida"].to_numpy())
        y.append(filas["sube"].to_numpy())
    if not y:
        return {}
    p_ia, p_frec, y = map(np.concatenate, (p_ia, p_frec, y))
    b_ia, b_frec = brier(p_ia, y), brier(p_frec, y)
    return {"brier_ia": b_ia, "brier_frecuencia": b_frec, "mejora": 1 - b_ia / b_frec,
            "acierto_ia": float(((p_ia > 0.5) == y).mean()),
            "acierto_frecuencia": float(((p_frec > 0.5) == y).mean()), "n": int(len(y))}


def entrenar_conjunto(datos: dict[str, pd.DataFrame], mercado: pd.Series | None, horizonte: int,
                      primer_anio: int = 1996, paso: int = 2, hoy: pd.Timestamp | None = None,
                      macro: pd.DataFrame | None = None) -> ConjuntoDireccion:
    panel = preparar_panel(datos, mercado, horizonte, macro)
    hoy = pd.Timestamp(hoy or pd.Timestamp.today().normalize())
    conjunto = ConjuntoDireccion(horizonte, metadatos={"cesta": sorted(datos), "entrenado": str(hoy.date()),
                                                       "muestras_totales": len(panel),
                                                       "usa_macro": macro is not None})
    for anio in range(primer_anio, hoy.year + 1, paso):
        m = entrenar(panel, pd.Timestamp(f"{anio}-01-01"))
        if m is not None:
            conjunto.por_anio[anio] = m
    conjunto.actual = entrenar(panel, hoy + pd.Timedelta(days=1))
    conjunto.metadatos["evaluacion"] = evaluar_panel(panel, conjunto)
    return conjunto


# --- Guardar y cargar ------------------------------------------------------------------

VARIANTES = {"ia": "", "ia_macro": "macro_"}  # clave -> sufijo del archivo
NOMBRES_FUENTE = {"frecuencia": "frecuencia histórica", "ia": "IA de dirección",
                  "ia_macro": "IA de dirección con miedo (VIX) y tipos de interés"}


def ruta(horizonte: int, directorio, variante: str = "ia") -> Path:
    return Path(directorio) / f"ia_direccion_{VARIANTES[variante]}h{horizonte}.joblib"


def guardar(conjunto: ConjuntoDireccion, directorio) -> Path:
    import joblib
    import sklearn

    conjunto.metadatos["sklearn"] = sklearn.__version__
    variante = "ia_macro" if conjunto.metadatos.get("usa_macro") else "ia"
    destino = ruta(conjunto.horizonte, directorio, variante)
    destino.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(conjunto, destino, compress=3)
    return destino


def cargar(horizonte: int, directorio, variante: str = "ia") -> ConjuntoDireccion | None:
    import joblib
    import sklearn

    origen = ruta(horizonte, directorio, variante)
    if not origen.exists():
        return None
    try:
        conjunto = joblib.load(origen)
    except Exception as e:
        warnings.warn(f"No se pudo cargar {origen.name}: {e!r}. Reentrena la IA.", stacklevel=2)
        return None
    if conjunto.metadatos.get("sklearn") != sklearn.__version__:
        warnings.warn(f"{origen.name} es de otra versión de scikit-learn: reentrena la IA.", stacklevel=2)
        return None
    return conjunto


# --- Aplicar a un activo ---------------------------------------------------------------

def serie_probabilidades(datos: pd.DataFrame, conjunto: ConjuntoDireccion,
                         mercado: pd.Series | None = None, macro: pd.DataFrame | None = None) -> pd.Series:
    """Probabilidad de subida emitida cada día con el modelo de su año (fuera de muestra)."""
    X = caracteristicas_direccion(datos, conjunto.horizonte, mercado, macro)
    prob = pd.Series(np.nan, index=datos.index)
    for anio in sorted(set(datos.index.year)):
        m = conjunto.modelo_para(anio)
        if m is None:
            continue
        filas = (datos.index.year == anio) & X["frecuencia_subida"].notna().to_numpy()
        if filas.any():
            prob[filas] = m.probabilidad(X.loc[filas])
    return prob


@dataclass
class EstimacionDireccion:
    probabilidad: float                 # de estar más arriba al final del plazo
    fuente: str                         # "ia" o "frecuencia"
    frecuencia: float                   # frecuencia histórica del activo a este plazo
    mejora_validacion: float = np.nan   # 1 − Brier_IA / Brier_frecuencia en validación
    acierto_validacion: float = np.nan  # % de veces que acertó el lado (> 50 % = sube)
    acierto_siempre_sube: float = np.nan
    n_independientes: float = np.nan
    desde: pd.Timestamp | None = None
    hasta: pd.Timestamp | None = None
    calibracion: pd.DataFrame = field(default_factory=pd.DataFrame)  # dijo ~X % → subió Y %


def estimar(datos: pd.DataFrame, horizonte: int,
            conjuntos: "dict[str, ConjuntoDireccion] | ConjuntoDireccion | None" = None,
            mercado: pd.Series | None = None, fraccion_validacion: float = 0.3,
            macro: pd.DataFrame | None = None) -> EstimacionDireccion | None:
    """Probabilidad actual de subida y su fiabilidad comprobada en el tramo de validación.

    ``conjuntos`` = {"ia": ConjuntoDireccion, "ia_macro": ConjuntoDireccion}; compiten con la
    frecuencia histórica y se elige la de menor Brier en el tramo de selección.
    """
    if isinstance(conjuntos, ConjuntoDireccion):
        conjuntos = {"ia": conjuntos}
    conjuntos = {k: c for k, c in (conjuntos or {}).items() if c is not None}
    c = datos["Close"].astype(float)
    frec = frecuencia_historica(c, horizonte)
    if frec.dropna().empty:
        return None
    futuro = np.log(c.shift(-horizonte) / c)
    y = (futuro > 0).astype(float).where(futuro.notna())
    frecuencia_hoy = frecuencia_ajustada(float(frec.dropna().iloc[-1]), len(c.dropna()), horizonte)

    candidatos = {"frecuencia": frec}
    for clave, conj in conjuntos.items():
        if conj.metadatos.get("usa_macro") and macro is None:
            continue
        candidatos[clave] = serie_probabilidades(datos, conj, mercado,
                                                 macro if conj.metadatos.get("usa_macro") else None)
    tabla = pd.DataFrame({**candidatos, "y": y}).dropna()
    if len(tabla) < 500:
        return EstimacionDireccion(frecuencia_hoy, "frecuencia", frecuencia_hoy)

    corte = int(len(tabla) * (1 - fraccion_validacion))
    seleccion = tabla.iloc[: max(corte - horizonte, 0)]
    validacion = tabla.iloc[corte:]
    errores = {k: brier(seleccion[k], seleccion["y"]) for k in candidatos}
    fuente = min(errores, key=errores.get)

    p_val = validacion[fuente]
    b_fuente = brier(p_val, validacion["y"])
    b_frec = brier(validacion["frecuencia"], validacion["y"])
    tramos = pd.cut(p_val, [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0], include_lowest=True)
    calib = validacion.groupby(tramos, observed=True).agg(dijo=(fuente, "mean"), subio=("y", "mean"),
                                                          veces=("y", "size"))

    if fuente != "frecuencia":
        prob = None
        conj = conjuntos[fuente]
        if conj.actual is not None:
            usa_macro = macro if conj.metadatos.get("usa_macro") else None
            X = caracteristicas_direccion(datos, horizonte, mercado, usa_macro).iloc[[-1]]
            if X["frecuencia_subida"].notna().all():
                prob = float(conj.actual.probabilidad(X)[0])
        if prob is None:
            fuente, prob = "frecuencia", frecuencia_hoy
    else:
        prob = frecuencia_hoy

    return EstimacionDireccion(
        probabilidad=prob, fuente=fuente, frecuencia=frecuencia_hoy,
        mejora_validacion=(1 - b_fuente / b_frec) if b_frec > 0 else np.nan,
        acierto_validacion=float(((p_val > 0.5) == validacion["y"]).mean()),
        acierto_siempre_sube=float(validacion["y"].mean()),
        n_independientes=len(validacion) / horizonte,
        desde=validacion.index[0], hasta=validacion.index[-1], calibracion=calib,
    )
