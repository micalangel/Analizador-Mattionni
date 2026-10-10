"""Intervalos de pronóstico basados en la volatilidad histórica.

Nunca se da un precio único: cada método produce un rango [inferior, superior] con un
nivel de confianza declarado. Se modela el retorno logarítmico a ``h`` días como normal:

    ln(P_{t+h} / P_t) ~ N(μ·h, σ²_h)
    intervalo = P_t · exp(μ·h ± z · σ_h),   z = cuantil normal de (1 + confianza) / 2

La columna ``centro`` (mediana del modelo) se usa solo internamente para medir el error
en el informe de calibración; la interfaz muestra únicamente el rango.

Todos los cálculos son causales: el intervalo emitido en t usa datos hasta el cierre de t.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

METODOS = {
    "vol_historica": "Volatilidad histórica (desv. típica móvil, sin deriva)",
    "vol_historica_deriva": "Volatilidad histórica con deriva (media móvil de retornos)",
    "ewma": "EWMA / RiskMetrics (λ = 0,94)",
    "garch": "GARCH(1,1) reestimado cada 63 días",
}
REFERENCIA = "ingenuo"

# Plazos de proyección en sesiones bursátiles (≈ 21 por mes, 252 por año).
HORIZONTES = {"5 días": 5, "10 días": 10, "1 mes": 21, "6 meses": 126, "1 año": 252}
NOMBRE_REFERENCIA = "Ingenuo: sin cambio + cuantiles históricos de retornos a h días"


def cuantil_z(confianza: float) -> float:
    if not 0 < confianza < 1:
        raise ValueError("La confianza debe estar entre 0 y 1 (p. ej. 0.8).")
    return NormalDist().inv_cdf(0.5 + confianza / 2)


def _log_retornos(precios: pd.Series) -> pd.Series:
    return np.log(precios / precios.shift(1))


# --- GARCH(1,1) con numpy -------------------------------------------------------------

def _rejilla_garch():
    alfas = np.linspace(0.02, 0.30, 15)
    betas = np.linspace(0.60, 0.97, 20)
    a, b = np.meshgrid(alfas, betas)
    a, b = a.ravel(), b.ravel()
    validos = a + b < 0.995
    return a[validos], b[validos]


def ajustar_garch(r: np.ndarray) -> tuple[float, float, float]:
    """Estima (ω, α, β) por máxima verosimilitud gaussiana en una rejilla de (α, β).

    ω se fija por "variance targeting": ω = varianza_muestral · (1 − α − β).
    Se evalúan todas las combinaciones a la vez (vectorizado), lo que basta para fines
    educativos y evita depender de librerías de optimización.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)] - np.nanmean(r)
    var_muestral = r.var()
    a, b = _rejilla_garch()
    w = var_muestral * (1 - a - b)
    s2 = np.full_like(a, var_muestral)
    loglik = np.zeros_like(a)
    for x in r:
        loglik -= 0.5 * (np.log(s2) + x * x / s2)
        s2 = w + a * x * x + b * s2
    i = int(np.argmax(loglik))
    return float(w[i]), float(a[i]), float(b[i])


def estado_garch(r: pd.Series, minimo: int = 250, reajuste: int = 63,
                 max_historia: int = 2520) -> pd.DataFrame:
    """Estado GARCH conocido al cierre de cada t: varianza del día siguiente y parámetros.

    Los parámetros se reestiman cada ``reajuste`` sesiones con los últimos ``max_historia``
    retornos hasta t (10 años por defecto) y se reconstruye la varianza en esa ventana.
    No depende del horizonte, así que se calcula una vez y sirve para todos.
    """
    valores = r.to_numpy()
    n = len(valores)
    s2_sig = np.full(n, np.nan)
    pw, pa, pb = (np.full(n, np.nan) for _ in range(3))
    params, s2, media = None, np.nan, 0.0
    for t in range(n):
        x = valores[t]
        if not np.isfinite(x) or t < minimo:
            continue
        if params is None or (t - minimo) % reajuste == 0:
            historia = valores[max(1, t + 1 - max_historia): t + 1]
            params = ajustar_garch(historia)
            media = float(np.nanmean(historia))
            w, a, b = params
            s2 = float(np.nanvar(historia))
            for y in historia[np.isfinite(historia)] - media:
                s2 = w + a * y * y + b * s2
        else:
            w, a, b = params
            y = x - media
            s2 = w + a * y * y + b * s2
        s2_sig[t], pw[t], pa[t], pb[t] = s2, w, a, b
    return pd.DataFrame({"s2": s2_sig, "w": pw, "a": pa, "b": pb}, index=r.index)


def sigma_garch_h(estado: pd.DataFrame, horizonte: int) -> pd.Series:
    """√(Σ varianzas previstas de t+1 a t+h) en forma cerrada."""
    p = estado["a"] + estado["b"]
    v_largo = estado["w"] / (1 - p)
    suma = horizonte * v_largo + (estado["s2"] - v_largo) * (1 - p ** horizonte) / (1 - p)
    return np.sqrt(suma.clip(lower=0))


def _sigma_garch(r: pd.Series, horizonte: int) -> pd.Series:
    return sigma_garch_h(estado_garch(r), horizonte)


# --- Intervalos ------------------------------------------------------------------------

def serie_intervalos(
    precios: pd.Series,
    horizonte: int = 10,
    confianza: float = 0.8,
    metodo: str = "vol_historica",
    ventana: int = 60,
    estado_garch_previo: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Intervalo emitido en cada fecha t para el precio de t + ``horizonte`` días hábiles.

    ``estado_garch_previo`` (de ``estado_garch``) evita reestimar GARCH en cada horizonte.
    """
    if horizonte < 1:
        raise ValueError("El horizonte debe ser de al menos 1 día.")
    precios = precios.astype(float)
    r = _log_retornos(precios)
    z = cuantil_z(confianza)
    alfa = 1 - confianza

    if metodo == REFERENCIA:
        # Retornos a h días ya conocidos en t (de t − h a t) y sus cuantiles acumulados.
        r_h = np.log(precios / precios.shift(horizonte))
        minimo = max(250, ventana)
        q_inf = r_h.expanding(min_periods=minimo).quantile(alfa / 2)
        q_sup = r_h.expanding(min_periods=minimo).quantile(1 - alfa / 2)
        resultado = pd.DataFrame({
            "inferior": precios * np.exp(q_inf),
            "superior": precios * np.exp(q_sup),
            "centro": precios.where(q_inf.notna()),
        })
        resultado["sigma_h"] = (q_sup - q_inf) / (2 * z)
        return resultado

    if metodo == "vol_historica":
        sigma_d = r.rolling(ventana, min_periods=ventana).std()
        mu = pd.Series(0.0, index=precios.index)
        sigma_h = sigma_d * np.sqrt(horizonte)
    elif metodo == "vol_historica_deriva":
        sigma_d = r.rolling(ventana, min_periods=ventana).std()
        mu = r.rolling(ventana, min_periods=ventana).mean()
        sigma_h = sigma_d * np.sqrt(horizonte)
    elif metodo == "ewma":
        lam = 0.94
        var_d = (r ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=ventana).mean()
        mu = pd.Series(0.0, index=precios.index)
        sigma_h = np.sqrt(var_d * horizonte)
    elif metodo == "garch":
        mu = pd.Series(0.0, index=precios.index)
        estado = estado_garch_previo if estado_garch_previo is not None else estado_garch(r)
        sigma_h = sigma_garch_h(estado, horizonte)
    else:
        raise ValueError(f"Método desconocido: {metodo}. Opciones: {list(METODOS) + [REFERENCIA]}")

    deriva = mu * horizonte
    return pd.DataFrame({
        "inferior": precios * np.exp(deriva - z * sigma_h),
        "superior": precios * np.exp(deriva + z * sigma_h),
        "centro": (precios * np.exp(deriva)).where(sigma_h.notna()),
        "sigma_h": sigma_h,
    })


def intervalo_actual(
    precios: pd.Series,
    horizonte: int = 10,
    confianza: float = 0.8,
    metodo: str = "vol_historica",
    ventana: int = 60,
) -> dict:
    """Rango para el precio dentro de ``horizonte`` días emitido con el último dato."""
    tabla = serie_intervalos(precios, horizonte, confianza, metodo, ventana).dropna()
    if tabla.empty:
        raise ValueError("No hay suficientes datos para calcular el intervalo.")
    fila = tabla.iloc[-1]
    return {
        "fecha": tabla.index[-1],
        "precio_ultimo": float(precios.loc[tabla.index[-1]]),
        "inferior": float(fila["inferior"]),
        "superior": float(fila["superior"]),
        "confianza": confianza,
        "horizonte": horizonte,
        "metodo": metodo,
    }
