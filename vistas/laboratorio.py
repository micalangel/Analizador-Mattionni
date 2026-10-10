"""Página «Laboratorio de señales»: ¿alguna señal adivina la dirección mejor que el azar?"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import numpy as np

from analizador import situaciones as sit
from analizador.forecast import HORIZONTES
from analizador.indices import INDICES, MERCADO
from analizador.senales import CORTE_COMPROBACION, HORIZONTES_DIRECCION, laboratorio
from vistas.comun import COLORES, GRIS, cargar, cargar_macro, estilo, numero, seccion

ICONOS = {"Supera al azar": "✅", "Funcionaba y dejó de funcionar": "📉", "Ventaja no concluyente": "❔",
          "No supera al azar": "✖️", "Sin datos": "—"}


@st.cache_data(show_spinner="Repasando los episodios de pánico…", ttl=60 * 60 * 6)
def episodios_panico(horizonte: int, hoy: date) -> pd.DataFrame:
    """Cada episodio de VIX ≥ 30: qué parte de los índices mundiales estaba más arriba al cumplirse el plazo."""
    datos = cargar(tuple(INDICES), date(1990, 1, 1), hoy)
    macro = cargar_macro(hoy)
    base = sit.historial(datos[MERCADO]["Close"], "panico", horizonte, macro)
    filas = []
    for _, ep in base.iterrows():
        mayoria = []
        for df in datos.values():
            c = df["Close"]
            tramo = c.loc[ep["inicio"]:ep["fin"]]
            if len(tramo) == 0:
                continue
            futuro = c.shift(-horizonte).loc[tramo.index]
            if futuro.isna().any():
                continue
            mayoria.append((futuro > tramo).mean() > 0.5)
        filas.append({**ep.to_dict(), "indices_arriba": np.mean(mayoria) if mayoria else np.nan,
                      "n_indices": len(mayoria)})
    return pd.DataFrame(filas)


@st.cache_data(show_spinner="Probando todas las señales en los índices mundiales…", ttl=60 * 60 * 6)
def calcular(hoy: date) -> tuple[pd.DataFrame, int]:
    datos = cargar(tuple(INDICES), date(1990, 1, 1), hoy)
    return laboratorio(datos), len(datos)


st.title("Experimentos: ¿qué reglas funcionan?")
st.markdown("<div class='subtitulo'>¿Hay alguna regla clásica que adivine si el mercado subirá o bajará mejor que el "
            "azar? Se prueba cada una en los principales índices del mundo, con datos que no se usaron para "
            "descubrirla.</div>", unsafe_allow_html=True)
st.write("")

lab, n_indices = calcular(date.today())
anio = CORTE_COMPROBACION.year

superan = lab[lab["veredicto"] == "Supera al azar"]
dejaron = lab[lab["veredicto"] == "Funcionaba y dejó de funcionar"]
mejor = lab.loc[lab["acierto_comp"].idxmax()]
with seccion("resumen", "azul"):
    k = st.columns(3)
    k[0].metric("Pruebas realizadas", f"{len(lab)}", help="Señales × plazos")
    k[1].metric("Superan al azar de forma fiable", f"{len(superan)}")
    k[2].metric(f"Mejor acierto desde {anio}", f"{mejor['acierto_comp']:.0%}",
                f"azar: {mejor['base_comp']:.0%}", delta_color="off",
                help=f"{mejor['señal']} a {mejor['plazo']}")

    if superan.empty:
        st.info(f"**Conclusión actual:** ninguna de estas señales adivina la dirección mejor que el azar de forma "
                f"fiable desde {anio}. {len(dejaron)} pruebas daban ventaja antes de {anio} y la perdieron después: "
                "es lo habitual cuando una regla se hace conocida. Lo que sí resulta predecible es **cuánto** se "
                "moverá el precio (los rangos de la página «¿Subirá o bajará?»), no **hacia dónde**.")
    else:
        st.success("**Superan al azar:** " + "; ".join(f"{r['señal']} a {r['plazo']} "
                                                        f"({r['acierto_comp']:.0%} frente a {r['base_comp']:.0%})"
                                                        for _, r in superan.iterrows()))

with seccion("senales", "blanco"):
    st.markdown("<div class='titulo-bloque'>Regla por regla</div>", unsafe_allow_html=True)
    plazo = st.segmented_control("Plazo", list(HORIZONTES_DIRECCION), default="1 semana", key="plazo_lab") or "1 semana"
    tabla = lab[lab["plazo"] == plazo]

    fig = go.Figure()
    y = tabla["señal"][::-1]
    for _, r in tabla.iloc[::-1].iterrows():
        fig.add_trace(go.Scatter(x=[r["ventaja_desc"], r["ventaja_comp"]], y=[r["señal"]] * 2, mode="lines",
                                 line=dict(color=GRIS, width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tabla["ventaja_desc"][::-1], y=y, mode="markers", name=f"Antes de {anio} (descubrimiento)",
                             marker=dict(color=COLORES[1], size=11, line=dict(width=2, color="white")),
                             hovertemplate="%{y}<br>Ventaja: %{x:+.1%}<extra>antes de " + str(anio) + "</extra>"))
    fig.add_trace(go.Scatter(x=tabla["ventaja_comp"][::-1], y=y, mode="markers", name=f"Desde {anio} (comprobación)",
                             marker=dict(color=COLORES[0], size=11, line=dict(width=2, color="white")),
                             hovertemplate="%{y}<br>Ventaja: %{x:+.1%}<extra>desde " + str(anio) + "</extra>"))
    fig.add_vline(x=0, line=dict(color=GRIS, width=1, dash="dot"))
    fig.update_layout(title=f"Ventaja sobre el azar a {plazo} (puntos de acierto)", hovermode="closest")
    fig.update_xaxes(tickformat="+.0%", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(estilo(fig, 60 + 38 * len(tabla) + 80), width="stretch")
    st.caption("A la derecha de la línea, la señal acertó más que el azar; a la izquierda, menos. Si el punto azul "
               "(datos nuevos) se acerca a cero, la ventaja del pasado no se mantuvo.")

    vista = pd.DataFrame({
        "Señal": tabla["señal"],
        f"Acierto desde {anio}": tabla["acierto_comp"].map("{:.1%}".format),
        "Azar": tabla["base_comp"].map("{:.1%}".format),
        "Ventaja": tabla["ventaja_comp"].map("{:+.1%}".format),
        f"Ventaja antes de {anio}": tabla["ventaja_desc"].map("{:+.1%}".format),
        "Veredicto": [f"{ICONOS.get(v, '')} {v}" for v in tabla["veredicto"]],
        "Qué dice": tabla["descripción"],
    })
    st.dataframe(vista, hide_index=True, width="stretch")

    with st.expander("Cómo se hace la prueba (y por qué no basta con acertar más del 50 %)"):
        st.markdown(
            f"""
    - **Datos:** {n_indices} índices bursátiles de todo el mundo desde 1990. Cada señal usa solo información
      disponible al cierre del día; el resultado es si el índice estaba más arriba o más abajo al cumplirse el plazo.
    - **Azar:** los mercados suben más días de los que bajan, así que decir siempre «sube» ya acierta más de la
      mitad de las veces. La referencia es alguien que dice «sube» y «baja» en las mismas proporciones que la
      señal, pero al azar. Si la señal no sabe nada, acierta lo mismo que él.
    - **Dos periodos:** antes de {anio} para descubrir y desde {anio} para comprobar. Una señal que solo funciona
      en el periodo en que se descubrió probablemente fue casualidad o dejó de funcionar al hacerse conocida.
    - **Significación:** los índices se mueven juntos (correlación media alta) y los plazos se solapan, así que
      hay muchas menos observaciones independientes de las que parece. Además, al probar {len(lab)} cosas a la vez
      alguna saldría «ganadora» por suerte: se aplica la corrección de Holm.
    - **Veredictos:** ✅ supera al azar en ambos periodos y es significativa; 📉 funcionaba y dejó de funcionar;
      ❔ ventaja positiva pero no concluyente; ✖️ no supera al azar.

    Un acierto del 80 % en la dirección no aparece en estos datos. Si alguna vez un estudio lo muestra, lo más
    probable es que use información del futuro sin darse cuenta o que haya memorizado el pasado.
    """
        )

with seccion("panico", "ambar"):
    st.markdown("<div class='titulo-bloque'>¿Qué pasa después de un pánico?</div>", unsafe_allow_html=True)
    st.caption("Una de las pocas pistas con respaldo: históricamente, los momentos de miedo extremo (VIX ≥ 30) han sido "
               "seguidos más a menudo por subidas. Aquí está cada episodio desde 1990, uno a uno, para que se vea en qué "
               "se basa y dónde falló.")
    plazo_p = st.segmented_control("Plazo", ["1 mes", "6 meses", "1 año"], default="1 año", key="plazo_panico") or "1 año"
    eps = episodios_panico(HORIZONTES[plazo_p], date.today())
    hechos = eps.dropna(subset=["indices_arriba"])
    if not hechos.empty:
        spx_arriba = (hechos["prop_mas_arriba"] > 0.5)
        antes = hechos["inicio"] < CORTE_COMPROBACION
        k = st.columns(3)
        k[0].metric("Episodios de pánico cerrados", len(hechos))
        k[1].metric(f"S&P 500 más arriba a {plazo_p}", f"{spx_arriba.sum()} de {len(hechos)}")
        k[2].metric("Media de índices del mundo más arriba", f"{hechos['indices_arriba'].mean():.0%}")
        st.info(f"Antes de {anio}: el S&P 500 acabó más arriba en {spx_arriba[antes].sum()} de {antes.sum()} episodios. "
                f"Desde {anio}: en {spx_arriba[~antes].sum()} de {(~antes).sum()}. La regla ha funcionado mucho mejor en la "
                "última época (con los bancos centrales rescatando los mercados), y con tan pocos episodios cualquier "
                "porcentaje es poco firme.")
    vista = pd.DataFrame({
        "Empieza": eps["inicio"].dt.strftime("%d/%m/%Y"),
        "Termina": eps["fin"].dt.strftime("%d/%m/%Y"),
        "VIX máximo": eps["vix_max"].map(lambda v: numero(v, 0)),
        f"S&P 500: días que acabaron más arriba a {plazo_p}": eps["prop_mas_arriba"].map(
            lambda v: "pendiente" if pd.isna(v) else f"{v:.0%}"),
        f"Índices del mundo más arriba a {plazo_p}": [
            "pendiente" if pd.isna(v) else f"{v:.0%} de {n}" for v, n in zip(eps["indices_arriba"], eps["n_indices"])],
    })
    st.dataframe(vista.iloc[::-1], hide_index=True, width="stretch")


@st.cache_data(ttl=60 * 60 * 6)
def cargar_volumen() -> pd.DataFrame | None:
    from analizador.entrenamiento import DIRECTORIO

    ruta = DIRECTORIO / "estudio_volumen.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


exp_vol = cargar_volumen()
if exp_vol is not None:
    with seccion("volumen", "verde"):
        st.markdown("<div class='titulo-bloque'>¿Anticipa subidas el volumen inusual?</div>", unsafe_allow_html=True)
        st.caption("Probado con todas las empresas de la web desde 1990. «Más arriba» = el precio estaba por encima "
                   "al cumplirse el plazo; «un día cualquiera» = lo mismo sin ninguna condición.")
        plazo_v = st.segmented_control("Plazo", list(exp_vol["plazo"].unique()), default="3 meses",
                                       key="plazo_vol") or "3 meses"
        tv = exp_vol[exp_vol["plazo"] == plazo_v]
        st.dataframe(pd.DataFrame({
            "Regla": tv["señal"],
            f"Más arriba desde {anio}": tv["acierto_comp"].map("{:.1%}".format),
            "Un día cualquiera": tv["base_comp"].map("{:.1%}".format),
            "Ventaja": tv["ventaja_comp"].map("{:+.1%}".format),
            f"Ventaja antes de {anio}": tv["ventaja_desc"].map("{:+.1%}".format),
            "Veredicto": [f"{ICONOS.get(v, '')} {v}" for v in tv["veredicto"]],
            "Qué dice": tv["descripción"],
        }), hide_index=True, width="stretch")
        st.caption("Ninguna regla de volumen supera al azar de forma fiable. La que más se acerca es el volumen alto en "
                   "un día de caída (ligero rebote posterior), pero la diferencia es pequeña y puede ser casualidad.")

