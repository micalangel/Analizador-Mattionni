"""IA entrenable: rangos de precio con gradient boosting de cuantiles.

Qué aprende
-----------
Para cada día t se calculan características conocidas al cierre de t (volatilidad,
tendencia, RSI, volumen, días desde la última publicación de resultados, volatilidad del
mercado...). El objetivo es el retorno logarítmico a h días **escalado por la volatilidad
EWMA** de ese día:

    z_t = ln(P_{t+h} / P_t) / σ_h,EWMA(t)

Se entrenan dos modelos de gradient boosting con pérdida cuantílica, uno para el cuantil
α/2 y otro para 1 − α/2 de z. El intervalo es P_t · exp(q · σ_h,EWMA). Así la IA no tiene
que aprender el nivel de volatilidad (lo aporta EWMA), sino cuándo los movimientos suelen
ser más grandes, más pequeños o asimétricos de lo que EWMA supone.

Calibración conformal (CQR)
---------------------------
Los modelos de árboles tienden a ser demasiado confiados: sus cuantiles se ajustan a los
datos de entrenamiento y fuera de ellos aciertan menos de lo que dicen. Por eso el último
25 % de las muestras de entrenamiento se reserva (con un hueco de h sesiones para que los
resultados no se solapen) y se mide el error real en ese tramo:

    E_s = max(q_inf(s) − z_s, z_s − q_sup(s))

Los cuantiles se ensanchan (o estrechan) en el cuantil (1 − α) de E. Es el método de
"conformalized quantile regression" (Romano, Patterson y Candès, 2019).

Sin mirar al futuro
-------------------
Una muestra s solo sirve para entrenar si su resultado (precio en s + h) ya se conocía en
el momento del entrenamiento: s + h ≤ fecha de entrenamiento. En la evaluación
walk-forward el modelo se reentrena cada ``reentreno`` sesiones y cada intervalo se emite
con el último modelo entrenado antes de esa fecha.

Nunca produce un precio único ni recomendaciones; solo rangos que se evalúan en el mismo
informe de calibración que los demás métodos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .forecast import cuantil_z
from .indicators import bandas_bollinger, rsi

DIRECTORIO_MODELOS = Path("modelos")
NOMBRE_METODO = "ia"
DESCRIPCION = "IA: gradient boosting de cuantiles sobre volatilidad EWMA (entrenada walk-forward)"

NOMBRES_CARACTERISTICAS = {
    "vol_5": "Volatilidad 5 días",
    "vol_20": "Volatilidad 20 días",
    "vol_60": "Volatilidad 60 días",
    "vol_252": "Volatilidad 1 año",
    "ratio_vol_5_60": "Volatilidad 5 d / 60 d",
    "ratio_vol_60_252": "Volatilidad 60 d / 1 año",
    "ret_1": "Retorno 1 día",
    "ret_5": "Retorno 5 días",
    "ret_20": "Retorno 20 días",
    "ret_60": "Retorno 60 días",
    "ret_252": "Retorno 1 año",
    "rsi_14": "RSI 14",
    "dist_mm50": "Distancia a la media de 50",
    "dist_mm200": "Distancia a la media de 200",
    "bollinger_b": "Posición en Bollinger (%B)",
    "drawdown_252": "Caída desde el máximo de 1 año",
    "rango_10": "Rango diario medio (10 d)",
    "volumen_z": "Volumen anómalo (z, 20 d)",
    "dias_desde_resultados": "Días desde la última publicación",
    "ultima_sorpresa": "Última sorpresa de BPA (%)",
    "mercado_vol_20": "Volatilidad del mercado (20 d)",
    "vix": "Miedo del mercado (VIX)",
    "vix_relativo": "VIX frente a su media de 1 año",
    "vix_cambio_20": "Cambio del VIX en 20 días",
    "curva_tipos": "Curva de tipos (10 años − 3 meses)",
    "curva_cambio_60": "Cambio de la curva en 60 días",
    "tipo_10a_cambio_252": "Cambio del tipo a 10 años en 1 año",
    "mercado_ret_20": "Retorno del mercado (20 d)",
}


# --- Características -------------------------------------------------------------------

def construir_caracteristicas(
    datos: pd.DataFrame,
    publicaciones: pd.DataFrame | None = None,
    mercado: pd.Series | None = None,
    macro: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tabla de características; la fila t solo usa información disponible al cierre de t.

    ``macro`` (opcional): columnas «vix», «tipo_10a» y «tipo_3m» con sus cierres diarios
    (ver ``datos_macro``).
    """
    c = datos["Close"].astype(float)
    r = np.log(c / c.shift(1))
    f = pd.DataFrame(index=datos.index)
    for v in (5, 20, 60):
        f[f"vol_{v}"] = r.rolling(v).std()
    f["vol_252"] = r.rolling(252, min_periods=120).std()
    f["ratio_vol_5_60"] = f["vol_5"] / f["vol_60"]
    f["ratio_vol_60_252"] = f["vol_60"] / f["vol_252"]
    for k in (1, 5, 20, 60, 252):
        f[f"ret_{k}"] = np.log(c / c.shift(k))
    f["rsi_14"] = rsi(c, 14)
    f["dist_mm50"] = c / c.rolling(50).mean() - 1
    f["dist_mm200"] = c / c.rolling(200).mean() - 1
    bb = bandas_bollinger(c, 20, 2)
    f["bollinger_b"] = (c - bb["inferior"]) / (bb["superior"] - bb["inferior"])
    f["drawdown_252"] = c / c.rolling(252, min_periods=60).max() - 1

    if {"High", "Low"} <= set(datos.columns):
        f["rango_10"] = ((datos["High"] - datos["Low"]) / c).rolling(10).mean()
    if "Volume" in datos.columns and (datos["Volume"] > 0).any():
        lv = np.log(datos["Volume"].where(datos["Volume"] > 0))
        f["volumen_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()

    if publicaciones is not None and not publicaciones.empty:
        f = f.join(_caracteristicas_resultados(datos.index, publicaciones))
    if mercado is not None and not mercado.empty:
        f = f.join(_caracteristicas_mercado(datos.index, mercado))
    if macro is not None and not macro.empty:
        f = f.join(_caracteristicas_macro(datos.index, macro))
    return f.replace([np.inf, -np.inf], np.nan)


def _caracteristicas_mercado(indice: pd.DatetimeIndex, mercado: pd.Series) -> pd.DataFrame:
    """Volatilidad y retorno de 20 días del mercado según su último cierre ANTERIOR a cada fecha.

    Se usa el cierre estrictamente anterior porque muchos mercados (Asia, Europa) cierran
    antes que el de EE. UU.: usar el cierre estadounidense del mismo día sería mirar al futuro.
    """
    m = mercado.dropna().astype(float).sort_index()
    rm = np.log(m / m.shift(1))
    propias = pd.DataFrame({"mercado_vol_20": rm.rolling(20).std(),
                            "mercado_ret_20": np.log(m / m.shift(20))})
    pos = m.index.searchsorted(indice, side="left") - 1
    valores = propias.to_numpy()[np.clip(pos, 0, None)]
    valores[pos < 0] = np.nan
    return pd.DataFrame(valores, index=indice, columns=propias.columns)


TICKERS_MACRO = {"vix": "^VIX", "tipo_10a": "^TNX", "tipo_3m": "^IRX"}


def datos_macro(descargas: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Une los cierres de VIX y tipos de interés (descargados con ``TICKERS_MACRO``)."""
    columnas = {c: descargas[t]["Close"] for c, t in TICKERS_MACRO.items() if t in descargas}
    return pd.DataFrame(columnas).sort_index() if columnas else None


def _caracteristicas_macro(indice: pd.DatetimeIndex, macro: pd.DataFrame) -> pd.DataFrame:
    """Miedo (VIX) y tipos de interés de EE. UU., según su último cierre ANTERIOR a cada fecha."""
    m = macro.sort_index().ffill()
    propias = pd.DataFrame(index=m.index)
    if "vix" in m:
        propias["vix"] = m["vix"]
        propias["vix_relativo"] = m["vix"] / m["vix"].rolling(252, min_periods=120).mean()
        propias["vix_cambio_20"] = np.log(m["vix"] / m["vix"].shift(20))
    if {"tipo_10a", "tipo_3m"} <= set(m.columns):
        curva = m["tipo_10a"] - m["tipo_3m"]
        propias["curva_tipos"] = curva
        propias["curva_cambio_60"] = curva - curva.shift(60)
    if "tipo_10a" in m:
        propias["tipo_10a_cambio_252"] = m["tipo_10a"] - m["tipo_10a"].shift(252)
    pos = m.index.searchsorted(indice, side="left") - 1
    valores = propias.to_numpy()[np.clip(pos, 0, None)]
    valores[pos < 0] = np.nan
    return pd.DataFrame(valores, index=indice, columns=propias.columns)


def _caracteristicas_resultados(indice: pd.DatetimeIndex, publicaciones: pd.DataFrame) -> pd.DataFrame:
    """Días desde la última publicación ya ocurrida y su sorpresa (solo pasado)."""
    pub = publicaciones.dropna(subset=["BPA publicado"]).copy()
    out = pd.DataFrame(index=indice, columns=["dias_desde_resultados", "ultima_sorpresa"], dtype=float)
    if pub.empty:
        return out
    fechas = pd.DatetimeIndex(pub.index)
    if fechas.tz is not None:
        fechas = fechas.tz_convert("America/New_York").tz_localize(None)
    # Una publicación se conoce a partir del cierre del día siguiente (cubre las de tras el cierre).
    conocida = (fechas.normalize() + pd.Timedelta(days=1)).to_numpy()
    orden = np.argsort(conocida)
    conocida, sorpresa = conocida[orden], pub["Sorpresa %"].to_numpy()[orden]
    pos = np.searchsorted(conocida, indice.to_numpy(), side="right") - 1
    validos = pos >= 0
    dias = np.full(len(indice), np.nan)
    sorp = np.full(len(indice), np.nan)
    dias[validos] = (indice.to_numpy()[validos] - conocida[pos[validos]]) / np.timedelta64(1, "D")
    sorp[validos] = sorpresa[pos[validos]]
    out["dias_desde_resultados"] = dias
    out["ultima_sorpresa"] = sorp
    return out


def sigma_ewma_h(precios: pd.Series, horizonte: int, lam: float = 0.94, minimo: int = 60) -> pd.Series:
    r = np.log(precios / precios.shift(1))
    var = (r ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=minimo).mean()
    return np.sqrt(var * horizonte)


def objetivo_escalado(precios: pd.Series, horizonte: int) -> pd.Series:
    """z_t = ln(P_{t+h}/P_t) / σ_h(t). Solo se usa como etiqueta de entrenamiento."""
    return np.log(precios.shift(-horizonte) / precios) / sigma_ewma_h(precios, horizonte)


# --- Modelo ----------------------------------------------------------------------------

@dataclass
class ModeloIA:
    horizonte: int
    confianza: float
    columnas: list[str]
    modelos: dict = field(default_factory=dict)       # {"inferior": est, "superior": est}
    rellenos: dict = field(default_factory=dict)      # mediana de entrenamiento por columna
    ajuste_conformal: float = 0.0                     # ensanchamiento en unidades de z
    n_calibracion: int = 0
    n_muestras: int = 0
    entrenado_hasta: pd.Timestamp | None = None
    metadatos: dict = field(default_factory=dict)

    def importancias(self) -> pd.Series:
        if "importancias" in self.metadatos:  # calculadas al entrenar (modelos sin feature_importances_)
            imp = pd.Series(self.metadatos["importancias"])
        else:
            imp = pd.Series(sum(m.feature_importances_ for m in self.modelos.values()) / len(self.modelos),
                            index=self.columnas)
        imp.index = [NOMBRES_CARACTERISTICAS.get(c, c) for c in imp.index]
        return imp.sort_values(ascending=False)

    def predecir_cuantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[self.columnas].fillna(self.rellenos)
        lo = self.modelos["inferior"].predict(X) - self.ajuste_conformal
        hi = self.modelos["superior"].predict(X) + self.ajuste_conformal
        # Evita cuantiles cruzados.
        return pd.DataFrame({"q_inf": np.minimum(lo, hi), "q_sup": np.maximum(lo, hi)}, index=X.index)


def _estimador(cuantil: float, n_arboles: int, semilla: int):
    """Árboles de una sola división y hojas grandes: con datos financieros, ruidosos y con
    pocas muestras independientes, los modelos más simples generalizan mejor.

    Estos valores se eligieron comparando tres configuraciones en AAPL, SPY y MSFT
    (2016-2026); esa elección usa los mismos datos que luego se muestran, lo que introduce
    un pequeño sesgo optimista (ver docs/METHODOLOGY.md).
    """
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        loss="quantile", alpha=cuantil, n_estimators=n_arboles, learning_rate=0.03,
        max_depth=1, min_samples_leaf=150, subsample=0.8, random_state=semilla,
    )


def entrenar(
    caracteristicas: pd.DataFrame,
    objetivo: pd.Series,
    horizonte: int,
    confianza: float,
    hasta: pd.Timestamp | None = None,
    n_arboles: int = 100,
    semilla: int = 0,
    minimo_muestras: int = 300,
    fraccion_calibracion: float = 0.25,
) -> ModeloIA | None:
    """Entrena con las muestras cuyo resultado se conocía en ``hasta`` (s + h ≤ hasta)."""
    indice = caracteristicas.index
    corte = len(indice) if hasta is None else indice.searchsorted(hasta, side="right")
    ultima_muestra = corte - 1 - horizonte  # su resultado se conoce justo en 'hasta'
    if ultima_muestra < 0:
        return None
    X = caracteristicas.iloc[: ultima_muestra + 1]
    y = objetivo.iloc[: ultima_muestra + 1]
    columnas = [c for c in X.columns if X[c].notna().mean() > 0.5]
    validas = y.notna() & X[columnas].notna().sum(axis=1).ge(len(columnas) * 0.8)
    X, y = X.loc[validas, columnas], y[validas]
    if len(y) < minimo_muestras:
        return None
    # Tramo de ajuste (primeras muestras) y de calibración (últimas), separados por h sesiones.
    n_cal = int(len(y) * fraccion_calibracion)
    fin_ajuste = len(y) - n_cal - horizonte
    if fin_ajuste < minimo_muestras * 0.6 or n_cal < 50:
        return None
    X_aj, y_aj = X.iloc[:fin_ajuste], y.iloc[:fin_ajuste]
    X_cal, y_cal = X.iloc[len(y) - n_cal:], y.iloc[len(y) - n_cal:]

    rellenos = X_aj.median().to_dict()
    alfa = 1 - confianza
    modelo = ModeloIA(horizonte, confianza, columnas, rellenos=rellenos, n_muestras=len(X_aj),
                      entrenado_hasta=indice[corte - 1], n_calibracion=n_cal)
    for nombre, q in (("inferior", alfa / 2), ("superior", 1 - alfa / 2)):
        modelo.modelos[nombre] = _estimador(q, n_arboles, semilla).fit(X_aj.fillna(rellenos), y_aj)

    q_cal = modelo.predecir_cuantiles(X_cal)  # aún sin ajuste conformal
    errores = np.maximum(q_cal["q_inf"] - y_cal, y_cal - q_cal["q_sup"]).to_numpy()
    nivel = min(1.0, np.ceil((n_cal + 1) * (1 - alfa)) / n_cal)
    modelo.ajuste_conformal = float(np.quantile(errores, nivel, method="higher"))
    return modelo


def serie_intervalos_ia(
    datos: pd.DataFrame,
    horizonte: int = 10,
    confianza: float = 0.8,
    publicaciones: pd.DataFrame | None = None,
    mercado: pd.Series | None = None,
    reentreno: int = 126,
    n_arboles: int = 100,
    minimo_muestras: int = 300,
) -> tuple[pd.DataFrame, list[ModeloIA]]:
    """Evaluación walk-forward: intervalos emitidos cada día con el último modelo disponible.

    Devuelve la tabla de intervalos (mismo formato que ``forecast.serie_intervalos``) y la
    lista de modelos entrenados por el camino.
    """
    cuantil_z(confianza)  # valida
    precios = datos["Close"].astype(float)
    X = construir_caracteristicas(datos, publicaciones, mercado)
    y = objetivo_escalado(precios, horizonte)
    sigma = sigma_ewma_h(precios, horizonte)

    q_inf = pd.Series(np.nan, index=precios.index)
    q_sup = pd.Series(np.nan, index=precios.index)
    modelos: list[ModeloIA] = []
    modelo = None
    n = len(precios)
    t = minimo_muestras + horizonte
    while t < n:
        candidato = entrenar(X, y, horizonte, confianza, hasta=precios.index[t],
                             n_arboles=n_arboles, minimo_muestras=minimo_muestras)
        if candidato is not None:
            modelo = candidato
            modelos.append(modelo)
        fin = min(t + reentreno, n)
        if modelo is not None:
            bloque = X.iloc[t:fin]
            q = modelo.predecir_cuantiles(bloque)
            q_inf.iloc[t:fin] = q["q_inf"].to_numpy()
            q_sup.iloc[t:fin] = q["q_sup"].to_numpy()
        t = fin

    tabla = pd.DataFrame({
        "inferior": precios * np.exp(q_inf * sigma),
        "superior": precios * np.exp(q_sup * sigma),
        "centro": precios.where(q_inf.notna() & sigma.notna()),
        "sigma_h": (q_sup - q_inf) * sigma / (2 * cuantil_z(confianza)),
    })
    return tabla, modelos


def intervalo_actual_ia(datos: pd.DataFrame, modelo: ModeloIA,
                        publicaciones: pd.DataFrame | None = None,
                        mercado: pd.Series | None = None) -> dict:
    """Rango para dentro de h días con el modelo dado y los datos del último cierre."""
    precios = datos["Close"].astype(float)
    X = construir_caracteristicas(datos, publicaciones, mercado).iloc[[-1]]
    for col in modelo.columnas:
        if col not in X.columns:
            X[col] = np.nan
    q = modelo.predecir_cuantiles(X).iloc[0]
    sigma = sigma_ewma_h(precios, modelo.horizonte).iloc[-1]
    p = precios.iloc[-1]
    return {"fecha": precios.index[-1], "precio_ultimo": float(p),
            "inferior": float(p * np.exp(q["q_inf"] * sigma)),
            "superior": float(p * np.exp(q["q_sup"] * sigma)),
            "confianza": modelo.confianza, "horizonte": modelo.horizonte, "metodo": NOMBRE_METODO}


# --- Guardar y cargar ------------------------------------------------------------------

def ruta_modelo(ticker: str, horizonte: int, confianza: float,
                directorio: Path | str = DIRECTORIO_MODELOS) -> Path:
    return Path(directorio) / f"{ticker.upper()}_h{horizonte}_c{int(round(confianza * 100))}.joblib"


def guardar_modelo(modelo: ModeloIA, ticker: str, directorio: Path | str = DIRECTORIO_MODELOS) -> Path:
    import joblib
    import sklearn

    modelo.metadatos.update({"ticker": ticker.upper(), "guardado": datetime.now().isoformat(timespec="seconds"),
                             "sklearn": sklearn.__version__})
    ruta = ruta_modelo(ticker, modelo.horizonte, modelo.confianza, directorio)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, ruta)
    return ruta


def cargar_modelo(ticker: str, horizonte: int, confianza: float,
                  directorio: Path | str = DIRECTORIO_MODELOS) -> ModeloIA | None:
    """Carga un modelo guardado; devuelve None si no existe o es de otra versión de sklearn."""
    import joblib
    import sklearn

    ruta = ruta_modelo(ticker, horizonte, confianza, directorio)
    if not ruta.exists():
        return None
    try:
        modelo = joblib.load(ruta)
    except Exception:
        return None
    if modelo.metadatos.get("sklearn") != sklearn.__version__:
        return None
    return modelo
