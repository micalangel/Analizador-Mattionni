"""Página «Comparar»: 2 a 4 activos lado a lado."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analizador import mi_lista as ml
from analizador.catalogo import descripcion, nombre, opciones_ordenadas
from analizador.catalogo import sector as sector_de
from analizador.salud import comparar_per
from vistas.comun import COLORES, cargar, cargar_estudios, estilo, nota, numero, per_referencia, seccion

SALUD = {"bien": ("Sólida", "#047857"), "regular": ("Mixta", "#B45309"), "mal": ("Débil", "#B91C1C")}
PERIODOS = {"6 meses": pd.DateOffset(months=6), "1 año": pd.DateOffset(years=1), "5 años": pd.DateOffset(years=5),
            "10 años": pd.DateOffset(years=10), "Todo": None}


def cifra(valor, formato: str) -> str:
    return "—" if valor is None or pd.isna(valor) else formato.format(valor)


st.title("Comparar")
st.markdown("<div class='subtitulo'>Elige de 2 a 4 empresas, índices o activos y compáralos lado a lado: "
            "probabilidad, riesgo, salud de sus cuentas y cómo se han movido.</div>", unsafe_allow_html=True)
st.write("")

lista = ml.cargar()
inicial = (lista[:3] if len(lista) >= 2 else ["AAPL", "MSFT", "ITX.MC"])
with seccion("elegir", "suave"):
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        st.markdown("<div class='titulo-bloque'>Qué comparar</div>", unsafe_allow_html=True)
        elegidos = st.multiselect("Qué comparar", opciones_ordenadas(), default=inicial, max_selections=4,
                                  format_func=descripcion, label_visibility="collapsed")
    with c2:
        st.markdown("<div class='titulo-bloque'>Plazo</div>", unsafe_allow_html=True)
        plazo = st.segmented_control("Plazo", ["1 mes", "6 meses", "1 año"], default="1 año", key="plazo_comp",
                                     label_visibility="collapsed") or "1 año"

if len(elegidos) < 2:
    nota("Elige al menos dos para comparar.")
    st.stop()

estudios = cargar_estudios()
medianas_per, _ = per_referencia()

# --- Tarjetas lado a lado ------------------------------------------------------------------

cajas = []
for i, t in enumerate(elegidos):
    e = estudios[(estudios["ticker"] == t) & (estudios["plazo"] == plazo)] if not estudios.empty else pd.DataFrame()
    e = e.iloc[0] if len(e) else pd.Series(dtype=object)
    prob = e.get("prob_sube")
    caida = e.get("inferior") / e.get("cierre") - 1 if pd.notna(e.get("inferior")) else None
    subida = e.get("superior") / e.get("cierre") - 1 if pd.notna(e.get("superior")) else None
    salud_txt, salud_color = SALUD.get(e.get("salud"), ("Sin datos", "#6B7280"))
    filas = [
        ("Prob. más arriba", cifra(prob, "{:.0%}")),
        ("Caída posible", cifra(caida, "{:+.0%}")),
        ("Subida posible", cifra(subida, "{:+.0%}")),
        ("Desde su máximo", cifra(e.get("caida_desde_maximo"), "{:+.0%}")),
        ("Salud", f"<span style='color:{salud_color};font-weight:500'>{salud_txt}</span>"
         if e.get("tipo") == "Acción" else "—"),
        ("PER", cifra(e.get("per"), "{:.1f}")),
        ("PER de su sector", cifra(medianas_per.get(sector_de(t)), "{:.1f}")),
        ("PER frente al sector", comparar_per(e.get("per"), medianas_per.get(sector_de(t)))[0]
         .replace(" de su sector", "")),
        ("Margen", cifra(e.get("margen"), "{:.1%}")),
        ("Deuda / patrimonio", cifra(e.get("deuda_patrimonio"), "{:.2f}")),
        ("Crec. ventas", cifra(e.get("crecimiento_ventas"), "{:+.1%}")),
    ]
    cuerpo = "".join(f"<div style='display:flex;justify-content:space-between;border-top:1px solid #F0F0F2;"
                     f"padding:.35rem 0;font-size:.92rem'><span style='color:#6B6B6B'>{k}</span><span>{v}</span></div>"
                     for k, v in filas)
    cajas.append(f"<a class='caja' href='/proyeccion?activo={t}' target='_self' style='border-top:5px solid "
                 f"{COLORES[i % len(COLORES)]}'><h3>{nombre(t)}</h3>"
                 f"<div class='etiqueta'>Último cierre: {numero(e.get('cierre'))}</div>{cuerpo}</a>")
st.markdown("<div class='rejilla'>" + "".join(cajas) + "</div>", unsafe_allow_html=True)
st.caption(f"Probabilidad y rango a {plazo} del último estudio calculado. Pulsa una tarjeta para ver su ficha. "
           "Si falta algún dato, ese activo aún no está en el estudio (calcúlalo con `python -m analizador.estudio`).")

# --- Gráfico comparado ------------------------------------------------------------------

with seccion("grafico", "blanco"):
    st.markdown("<div class='titulo-bloque'>Cómo se han movido (base 100)</div>", unsafe_allow_html=True)
    periodo = st.segmented_control("Periodo", list(PERIODOS), default="5 años", key="periodo_comp",
                                   label_visibility="collapsed") or "5 años"
    datos = cargar(tuple(elegidos), date(1990, 1, 1), date.today())
    fig = go.Figure()
    for i, t in enumerate(elegidos):
        if t not in datos:
            continue
        c = datos[t]["Close"]
        if PERIODOS[periodo] is not None:
            c = c.loc[c.index[-1] - PERIODOS[periodo]:]
        base = c / c.iloc[0] * 100
        fig.add_trace(go.Scatter(x=base.index, y=base, name=nombre(t), line=dict(color=COLORES[i % len(COLORES)],
                                                                               width=2)))
    fig.add_hline(y=100, line=dict(color="#9CA3AF", width=1, dash="dot"))
    st.plotly_chart(estilo(fig, 400), width="stretch")
    st.caption("Todos empiezan en 100 al principio del periodo: así se ve qué ha subido más en porcentaje, aunque sus "
               "precios sean muy distintos. Lo que pasó antes no predice lo que pasará.")
