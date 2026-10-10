"""Página «Índices del mundo»: estudio de los principales índices de cada país."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analizador.entrenamiento import DIRECTORIO, RUTA_ESTUDIO
from analizador.forecast import HORIZONTES
from analizador.proyeccion import ESTADOS
from vistas.comun import COLORES, GRIS, estilo, numero, seccion

ICONOS = {"fiable": "✅", "prudente": "🔵", "optimista": "⚠️", "poca_evidencia": "◻️", "sin_datos": "—"}


@st.cache_data(ttl=60 * 10)
def cargar_estudio() -> pd.DataFrame | None:
    ruta = DIRECTORIO / RUTA_ESTUDIO.name
    return pd.read_csv(ruta) if ruta.exists() else None


st.title("Bolsas del mundo")
st.markdown("<div class='subtitulo'>Los principales índices bursátiles de cada país: hasta dónde podrían moverse en "
            "cada plazo y cuántas veces ha acertado ese tipo de rango en el pasado.</div>", unsafe_allow_html=True)
st.write("")

estudio = cargar_estudio()
if estudio is None:
    st.info("Aún no hay estudio de índices. Se genera al entrenar la IA: `python -m analizador.entrenamiento`.")
    st.stop()

with seccion("filtros", "suave"):
    c1, c2 = st.columns([3, 2])
    with c1:
        plazo = st.segmented_control("Plazo", list(HORIZONTES), default="1 mes", key="plazo_indices") or "1 mes"
    with c2:
        regiones = ["Todas"] + sorted(estudio["región"].unique())
        region = st.selectbox("Región", regiones)


tabla = estudio[estudio["plazo"] == plazo].copy()
if region != "Todas":
    tabla = tabla[tabla["región"] == region]
tabla["var_min"] = tabla["inferior"] / tabla["cierre"] - 1
tabla["var_max"] = tabla["superior"] / tabla["cierre"] - 1
orden_regiones = {"América": 0, "Europa": 1, "Asia-Pacífico": 2, "Oriente Medio y África": 3}
tabla["_orden"] = tabla["región"].map(orden_regiones).fillna(9)
tabla = tabla.sort_values(["_orden", "país", "índice"])

confianza = tabla["confianza"].iloc[0] if len(tabla) else 0.8
fiables = (tabla["estado"] == "fiable").sum()
with seccion("resumen", "azul"):
    k = st.columns(3)
    k[0].metric("Índices estudiados", len(tabla))
    k[1].metric("Con fiabilidad comprobada", f"{fiables} de {len(tabla)}")
    k[2].metric("Datos al", pd.to_datetime(tabla["fecha"]).max().strftime("%d/%m/%Y") if len(tabla) else "—")

with seccion("grafico", "blanco"):
    # Rango de variación por índice (puntos = mínimo y máximo, línea = rango).
    etiquetas = [f"{r['índice']} ({r['país']})" for _, r in tabla.iterrows()][::-1]
    fig = go.Figure()
    for (_, r), etq in zip(tabla.iloc[::-1].iterrows(), etiquetas):
        color = COLORES[0] if r["estado"] == "fiable" else GRIS
        fig.add_trace(go.Scatter(
            x=[r["var_min"], r["var_max"]], y=[etq, etq], mode="lines+markers",
            line=dict(color=color, width=4), marker=dict(size=8, color=color),
            showlegend=False,
            hovertemplate=(f"<b>{etq}</b><br>Rango: {r['var_min']:+.1%} / {r['var_max']:+.1%}<br>"
                           f"{ESTADOS[r['estado']]}<extra></extra>")))
    fig.add_vline(x=0, line=dict(color=GRIS, width=1))
    fig.update_layout(title=f"Variación posible a {plazo} (rango al {confianza:.0%})", hovermode="closest")
    fig.update_xaxes(tickformat="+.0%", showgrid=True, gridcolor="rgba(128,128,128,0.15)")
    st.plotly_chart(estilo(fig, max(360, 26 * len(tabla) + 120)), width="stretch")
    st.caption("Azul: rango con fiabilidad comprobada en el histórico reciente. Gris: menos evidencia o calibración "
               "peor (ver la columna «Fiabilidad»). Un rango más ancho significa más incertidumbre, no más "
               "oportunidad.")

    with seccion("tabla", "blanco"):
        st.markdown("<div class='titulo-bloque'>Detalle por país</div>", unsafe_allow_html=True)

    vista = pd.DataFrame({
        "Prob. más arriba": tabla["prob_sube"].map(lambda v: "—" if pd.isna(v) else f"{v:.0%}")
        if "prob_sube" in tabla else "—",
        "Región": tabla["región"],
        "País": tabla["país"],
        "Índice": tabla["índice"],
        "Último cierre": tabla["cierre"].map(numero),
        "Rango": [f"{numero(a)} – {numero(b)}" for a, b in zip(tabla["inferior"], tabla["superior"])],
        "Variación": [f"{a:+.1%} / {b:+.1%}" for a, b in zip(tabla["var_min"], tabla["var_max"])],
        "Aciertos comprobados": tabla["cobertura_validacion"].map(lambda v: "—" if pd.isna(v) else f"{v:.0%}"),
        "Fiabilidad": [f"{ICONOS[e]} {ESTADOS[e]}" for e in tabla["estado"]],
        "Método": tabla["método"],
        "Historia desde": pd.to_datetime(tabla["historia_desde"]).dt.year,
        "Ver": "/proyeccion?activo=" + tabla["ticker"],
    })
    st.dataframe(vista, hide_index=True, width="stretch",
                 column_config={"Ver": st.column_config.LinkColumn("Ver", display_text="Ver →")})
    st.caption("Pulsa «Ver →» para el detalle y el gráfico de cada índice. "
               "El estudio se recalcula al reentrenar la IA.")
