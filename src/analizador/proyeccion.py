"""Proyección de rangos a varios plazos con fiabilidad comprobada.

Para cada plazo (5 días, 10 días, 1 mes, 6 meses, 1 año):

1. Se calculan, día a día y sin mirar al futuro, los intervalos de varios métodos:
   EWMA, GARCH, volatilidad histórica, referencia ingenua y la IA global (si hay modelos).
2. **Selección** (primer 70 % del histórico evaluable): se elige el método bien
   calibrado con menor error (puntuación de intervalo). Si ninguno está calibrado, el
   de cobertura más cercana al objetivo.
3. **Validación** (último 30 %, que no se usó para elegir): se mide cuántas veces el
   precio real cayó dentro del rango del método elegido. Esa es la fiabilidad que se
   muestra, para no presumir de un acierto conseguido eligiendo a posteriori.
4. El rango de hoy se calcula con el método elegido.

Solo se informa de rangos e incertidumbre; nunca de un precio previsto ni de qué hacer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import evaluar_tabla, tolerancia_cobertura
from .forecast import HORIZONTES, METODOS, REFERENCIA, NOMBRE_REFERENCIA, estado_garch, intervalo_actual, serie_intervalos

CANDIDATOS = ["ewma", "garch", "vol_historica", REFERENCIA]
NOMBRES_CORTOS = {
    "ewma": "EWMA", "garch": "GARCH", "vol_historica": "Volatilidad histórica",
    REFERENCIA: "Histórico de movimientos", "ia": "IA",
}
DESCRIPCIONES = {**METODOS, REFERENCIA: NOMBRE_REFERENCIA,
                 "ia": "IA entrenada con los principales índices mundiales (gradient boosting de cuantiles)"}

ESTADOS = {
    "fiable": "Fiable en el histórico",
    "optimista": "Optimista: acertó menos de lo prometido",
    "prudente": "Prudente: acertó más de lo prometido",
    "poca_evidencia": "Poca evidencia histórica",
    "sin_datos": "Sin datos suficientes",
}


@dataclass
class ProyeccionHorizonte:
    etiqueta: str
    horizonte: int
    confianza: float
    metodo: str | None = None
    rango: dict | None = None
    fiabilidad: dict = field(default_factory=dict)
    comparativa: pd.DataFrame = field(default_factory=pd.DataFrame)
    validacion: pd.DataFrame = field(default_factory=pd.DataFrame)  # tabla del método elegido
    direccion: object | None = None  # direccion.EstimacionDireccion

    @property
    def estado(self) -> str:
        return self.fiabilidad.get("estado", "sin_datos")


def _metricas(tabla: pd.DataFrame) -> dict:
    return {"n": len(tabla), "cobertura": float(tabla["dentro"].mean()) if len(tabla) else np.nan,
            "puntuacion": float(tabla["puntuacion_rel"].mean()) if len(tabla) else np.nan,
            "ancho": float(tabla["ancho_rel"].mean()) if len(tabla) else np.nan}


def proyectar_horizonte(
    datos: pd.DataFrame,
    horizonte: int,
    confianza: float = 0.8,
    conjunto_ia=None,
    mercado: pd.Series | None = None,
    etiqueta: str | None = None,
    fraccion_validacion: float = 0.3,
    minimo_evaluable: int = 500,
    estado_garch_previo: pd.DataFrame | None = None,
) -> ProyeccionHorizonte:
    cierre = datos["Close"].astype(float)
    res = ProyeccionHorizonte(etiqueta or f"{horizonte} sesiones", horizonte, confianza)
    if estado_garch_previo is None:
        estado_garch_previo = estado_garch(np.log(cierre / cierre.shift(1)))

    tablas = {m: serie_intervalos(cierre, horizonte, confianza, m, estado_garch_previo=estado_garch_previo)
              for m in CANDIDATOS}
    if conjunto_ia is not None:
        from .entrenamiento import serie_intervalos_global
        tablas["ia"] = serie_intervalos_global(datos, conjunto_ia, confianza, mercado)
    evaluadas = {m: evaluar_tabla(t, cierre, horizonte, confianza) for m, t in tablas.items()}
    evaluadas = {m: e for m, e in evaluadas.items() if len(e) >= minimo_evaluable // 2}

    comun = None
    for e in evaluadas.values():
        comun = e.index if comun is None else comun.intersection(e.index)
    if comun is None or len(comun) < minimo_evaluable:
        res.fiabilidad = {"estado": "sin_datos", "n": 0 if comun is None else len(comun)}
        try:  # aun así damos el rango con EWMA, avisando de que no se pudo comprobar
            res.metodo, res.rango = "ewma", intervalo_actual(cierre, horizonte, confianza, "ewma")
        except ValueError:
            pass
        return res

    # Validación: último 30 % de fechas de emisión. Selección: fechas cuyo resultado se
    # conocía antes de empezar la validación (sin solapamiento entre ambos tramos).
    corte = int(len(comun) * (1 - fraccion_validacion))
    fechas_val = comun[corte:]
    fechas_sel = comun[: max(corte - horizonte, 0)]
    tol_sel = tolerancia_cobertura(confianza, len(fechas_sel), horizonte)

    filas = []
    for m, e in evaluadas.items():
        sel, val = _metricas(e.loc[fechas_sel]), _metricas(e.loc[fechas_val])
        filas.append({"método": m, "cobertura selección": sel["cobertura"], "error selección": sel["puntuacion"],
                      "calibrado en selección": abs(sel["cobertura"] - confianza) <= tol_sel,
                      "cobertura validación": val["cobertura"], "error validación": val["puntuacion"],
                      "ancho medio": val["ancho"]})
    comparativa = pd.DataFrame(filas).set_index("método")

    # La referencia ingenua (cuantiles históricos) solo sirve para comparar: con activos que han subido
    # mucho en el pasado da rangos irreales (p. ej. «como mucho caería +2 %»), así que no se elige.
    elegibles = comparativa.drop(index=[REFERENCIA], errors="ignore")
    if elegibles.empty:
        elegibles = comparativa
    calibrados = elegibles[elegibles["calibrado en selección"]]
    if not calibrados.empty:
        elegido = calibrados["error selección"].idxmin()
    else:
        elegido = (elegibles["cobertura selección"] - confianza).abs().idxmin()
    comparativa["elegido"] = comparativa.index == elegido

    validacion = evaluadas[elegido].loc[fechas_val]
    n_val = len(validacion)
    cobertura = float(validacion["dentro"].mean())
    tol = tolerancia_cobertura(confianza, n_val, horizonte)
    n_indep = n_val / horizonte
    if n_indep < 8:
        estado = "poca_evidencia"
    elif abs(cobertura - confianza) <= tol:
        estado = "fiable"
    elif cobertura < confianza:
        estado = "optimista"
    else:
        estado = "prudente"

    res.metodo = elegido
    res.comparativa = comparativa
    res.validacion = validacion
    res.fiabilidad = {
        "estado": estado, "cobertura": cobertura, "objetivo": confianza, "tolerancia": tol,
        "n": n_val, "n_independientes": n_indep,
        "desde": fechas_val[0], "hasta": fechas_val[-1],
        "aciertos_de_100": int(round(cobertura * 100)),
    }
    if elegido == "ia":
        from .entrenamiento import intervalo_actual_global
        res.rango = intervalo_actual_global(datos, conjunto_ia, confianza, mercado)
    else:
        res.rango = intervalo_actual(cierre, horizonte, confianza, elegido)
    return res


def proyectar(
    datos: pd.DataFrame,
    confianza: float = 0.8,
    conjuntos_ia: dict | None = None,
    mercado: pd.Series | None = None,
    horizontes: dict[str, int] | None = None,
    direcciones: dict | None = None,
    macro: pd.DataFrame | None = None,
) -> dict[str, ProyeccionHorizonte]:
    """Proyección para todos los plazos.

    ``conjuntos_ia`` = {horizonte: ConjuntoIA} (rangos); ``direcciones`` = {horizonte:
    {"ia": ConjuntoDireccion, "ia_macro": ConjuntoDireccion}} (probabilidad de subida);
    ``macro`` = VIX y tipos (``ia.datos_macro``). Todos opcionales.
    """
    from .direccion import estimar

    horizontes = horizontes or HORIZONTES
    cierre = datos["Close"].astype(float)
    estado = estado_garch(np.log(cierre / cierre.shift(1)))
    resultado = {}
    for etiqueta, h in horizontes.items():
        p = proyectar_horizonte(datos, h, confianza, (conjuntos_ia or {}).get(h), mercado,
                                etiqueta, estado_garch_previo=estado)
        p.direccion = estimar(datos, h, (direcciones or {}).get(h), mercado, macro=macro)
        resultado[etiqueta] = p
    return resultado


def tabla_resumen(proyecciones: dict[str, ProyeccionHorizonte]) -> pd.DataFrame:
    filas = []
    for etiqueta, p in proyecciones.items():
        r, f = p.rango or {}, p.fiabilidad
        precio = r.get("precio_ultimo", np.nan)
        filas.append({
            "Plazo": etiqueta,
            "Desde": r.get("inferior", np.nan),
            "Hasta": r.get("superior", np.nan),
            "Variación mín.": r.get("inferior", np.nan) / precio - 1 if precio else np.nan,
            "Variación máx.": r.get("superior", np.nan) / precio - 1 if precio else np.nan,
            "Aciertos comprobados": f.get("cobertura", np.nan),
            "Fiabilidad": ESTADOS[p.estado],
            "Método": NOMBRES_CORTOS.get(p.metodo, p.metodo or "—"),
        })
    return pd.DataFrame(filas).set_index("Plazo")


# --- Camino probable ---------------------------------------------------------------------

def camino_probable(precio: float, plazos: list[tuple[int, float, float, float | None]], confianza: float = 0.8,
                    ejemplos: int = 3, semilla: int = 0) -> pd.DataFrame:
    """Prolonga el gráfico: camino central, cono de rango y caminos de ejemplo, sesión a sesión.

    ``plazos``: [(sesiones, inferior, superior, prob_subir o None), ...] del rango calibrado de cada plazo.

    * **Cono**: el rango de cada plazo, interpolado en escala logarítmica entre plazos (en t = 0 vale el precio).
    * **Camino central**: a cada plazo, el precio que deja un 50 % de probabilidad por encima y por debajo.
      Suponiendo una distribución lognormal con la anchura del rango, se coloca de modo que la probabilidad
      de acabar por encima del precio de hoy sea la ``prob_subir`` estimada (si no la hay, el centro del rango).
      No es una predicción del precio exacto: el precio real se moverá en zigzag alrededor.
    * **Caminos de ejemplo**: recorridos simulados (paseo aleatorio con la volatilidad del rango, centrado en
      el camino central) para mostrar cómo suele moverse de verdad un precio. Son ilustrativos.
    """
    from statistics import NormalDist

    normal = NormalDist()
    z = normal.inv_cdf((1 + confianza) / 2)
    lp = np.log(precio)
    nudos_t, nudos_inf, nudos_sup, nudos_mu, nudos_sigma = [0], [lp], [lp], [lp], [0.0]
    for h, inf, sup, prob in sorted(plazos):
        if not (inf and sup) or inf <= 0 or sup <= inf:
            continue
        li, ls = np.log(inf), np.log(sup)
        sigma = (ls - li) / (2 * z)
        if prob is not None and np.isfinite(prob):
            p = float(np.clip(prob, 0.02, 0.98))
            mu = lp + sigma * normal.inv_cdf(p)
            mu = float(np.clip(mu, li, ls))  # nunca fuera del rango calibrado
        else:
            mu = (li + ls) / 2
        nudos_t.append(h)
        nudos_inf.append(li)
        nudos_sup.append(ls)
        nudos_mu.append(mu)
        nudos_sigma.append(sigma)
    t = np.arange(0, nudos_t[-1] + 1)
    salida = pd.DataFrame({"sesion": t,
                           "inferior": np.exp(np.interp(t, nudos_t, nudos_inf)),
                           "superior": np.exp(np.interp(t, nudos_t, nudos_sup)),
                           "central": np.exp(np.interp(t, nudos_t, nudos_mu))})
    if len(t) > 1 and ejemplos:
        # Volatilidad diaria implícita en cada tramo (la varianza crece con el tiempo).
        varianza = np.interp(t, nudos_t, np.square(nudos_sigma))
        paso = np.sqrt(np.clip(np.diff(varianza, prepend=0.0), 0, None))
        rng = np.random.default_rng(semilla)
        mu_t = np.interp(t, nudos_t, nudos_mu)
        for k in range(ejemplos):
            ruido = np.cumsum(rng.standard_normal(len(t)) * paso)
            ruido[0] = 0.0
            salida[f"ejemplo_{k + 1}"] = np.exp(mu_t + ruido)
    return salida
