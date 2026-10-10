"""Página «Calendario de resultados»: cuándo publican sus cuentas y cuánto suelen moverse ese día."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from analizador import mi_lista as ml
from analizador.calendario import proximas_publicaciones, tabla_calendario
from analizador.catalogo import ACTIVOS, descripcion, nombre, opciones_ordenadas
from vistas.comun import cargar, nota, seccion

POR_DEFECTO = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "SAN.MC", "BBVA.MC", "ITX.MC", "IBE.MC"]


@st.cache_data(show_spinner="Consultando las fechas de resultados…", ttl=60 * 60 * 12)
def calendario(tickers: tuple[str, ...], hoy: date) -> pd.DataFrame:
    datos = cargar(tickers, date(2010, 1, 1), hoy)
    info = proximas_publicaciones(list(tickers), {t: d["Close"] for t, d in datos.items()})
    return tabla_calendario(info, hoy)


st.title("Calendario de resultados")
st.markdown("<div class='subtitulo'>Cuándo presentan sus cuentas las empresas y cuánto suele moverse su precio ese "
            "día. Los días de resultados son de los que más se mueve una acción.</div>", unsafe_allow_html=True)
st.write("")

mi = [t for t in ml.cargar() if ACTIVOS.get(t, ("", "Acción"))[1] == "Acción"]
with seccion("elegir", "suave"):
    st.markdown("<div class='titulo-bloque'>Empresas</div>", unsafe_allow_html=True)
    acciones = [t for t in opciones_ordenadas() if ACTIVOS[t][1] == "Acción"]
    elegidas = st.multiselect("Empresas", acciones, default=mi or POR_DEFECTO, format_func=descripcion,
                              label_visibility="collapsed", placeholder="Elige empresas…")
    st.caption("Por defecto, las empresas de «Mi lista»." if mi else
               "Mostrando unas cuantas conocidas. Añade empresas a «Mi lista» para que salgan aquí por defecto.")

if not elegidas:
    nota("Elige al menos una empresa.")
    st.stop()

tabla = calendario(tuple(elegidas), date.today())
proximas = tabla[tabla["dias"].notna() & (tabla["dias"] >= 0)]
sin_fecha = tabla[tabla["dias"].isna()]

with seccion("pronto", "azul"):
    semana = proximas[proximas["dias"] <= 7]
    st.markdown("<div class='titulo-bloque'>En los próximos 7 días</div>", unsafe_allow_html=True)
    if semana.empty:
        st.write("Ninguna de estas empresas presenta resultados esta semana.")
    for _, f in semana.iterrows():
        cuando = "hoy" if f["dias"] == 0 else "mañana" if f["dias"] == 1 else f"en {int(f['dias'])} días"
        mov = f" · suele moverse un ±{f['movimiento_medio']:.0%} ese día" if pd.notna(f.get("movimiento_medio")) else ""
        st.markdown(f"**{nombre(f['ticker'])}** — {cuando} ({f['fecha']:%d/%m/%Y}){mov} "
                    f"[Ver →](/proyeccion?activo={f['ticker']})")

with seccion("tabla", "blanco"):
    st.markdown("<div class='titulo-bloque'>Próximas presentaciones</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Empresa": proximas["ticker"].map(nombre),
        "Fecha": proximas["fecha"].dt.strftime("%d/%m/%Y"),
        "Faltan": proximas["dias"].map(lambda d: f"{int(d)} días"),
        "BPA esperado": proximas["bpa_estimado"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
        "Suele moverse": proximas.get("movimiento_medio", pd.Series(dtype=float)) * 100,
        "Rango típico": [("—" if pd.isna(a) else f"{a:+.0%} / {b:+.0%}")
                         for a, b in zip(proximas.get("p10", [None] * len(proximas)),
                                         proximas.get("p90", [None] * len(proximas)))],
        "Supera lo esperado": proximas.get("supera_estimacion", pd.Series(dtype=float)) * 100,
        "Ver": "/proyeccion?activo=" + proximas["ticker"],
    }), hide_index=True, width="stretch", column_config={
        "Suele moverse": st.column_config.NumberColumn(
            "Suele moverse", format="±%.1f %%", help="Movimiento medio del precio (subiendo o bajando) en sus "
                                                     "últimas presentaciones de resultados."),
        "Supera lo esperado": st.column_config.NumberColumn(
            "Supera lo esperado", format="%.0f %%", help="% de veces que su beneficio superó lo que esperaban los "
                                                          "analistas."),
        "Ver": st.column_config.LinkColumn("Ver", display_text="Ver →"),
    })
    st.caption("BPA esperado = beneficio por acción que esperan los analistas. «Rango típico» = cómo se movió el precio "
               "en 8 de cada 10 presentaciones anteriores. Fechas de Yahoo Finance; las empresas pueden cambiarlas.")
    if not sin_fecha.empty:
        st.caption("Sin fecha anunciada: " + ", ".join(sin_fecha["ticker"].map(nombre)) + ".")
