"""Página «Mi lista»: los activos que sigues, sus alertas de hoy y su resumen."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from analizador.alertas import evaluar
from analizador.calendario import proximas_publicaciones
from analizador.catalogo import descripcion, nombre, opciones_ordenadas
from vistas.comun import cargar, cargar_estudios, cargar_macro, lista_cargar, lista_guardar, nota, numero, seccion

SALUD = {"bien": "🟢 Sólida", "regular": "🟡 Mixta", "mal": "🔴 Débil", "info": "⚪ Sin datos"}


@st.cache_data(show_spinner="Revisando tu lista…", ttl=60 * 60)
def revisar(tickers: tuple[str, ...], hoy: date):
    datos = cargar(tickers, date(2015, 1, 1), hoy)
    macro = cargar_macro(hoy)
    info = proximas_publicaciones(list(datos), {t: d["Close"] for t, d in datos.items()})
    alertas = []
    for t, df in datos.items():
        alertas += evaluar(t, nombre(t), df, macro, info.get(t, {}).get("fecha"))
    return datos, info, alertas


st.title("Mi lista")
st.markdown("<div class='subtitulo'>Los activos que sigues: qué conviene saber hoy de cada uno y sus cifras "
            "principales. También puedes añadirlos con la estrella ☆ en «¿Subirá o bajará?».</div>",
            unsafe_allow_html=True)
st.write("")

lista = lista_cargar()
from analizador.mi_lista import en_la_nube  # noqa: E402

if en_la_nube():
    st.caption("En la versión online, tu lista se guarda solo mientras tengas la página abierta. Para guardarla de "
               "forma permanente, usa la app en tu ordenador.")
with seccion("editar", "suave"):
    st.markdown("<div class='titulo-bloque'>Activos que sigues</div>", unsafe_allow_html=True)
    nueva = st.multiselect("Activos que sigues", opciones_ordenadas(), default=[t for t in lista],
                           format_func=descripcion, label_visibility="collapsed",
                           placeholder="Busca y añade empresas, índices, divisas…")
    if nueva != lista:
        lista_guardar(nueva)
        lista = nueva
        st.rerun()

if not lista:
    nota("Tu lista está vacía. Añade activos arriba o con la estrella ☆ en «¿Subirá o bajará?».")
    st.stop()

datos, info, alertas = revisar(tuple(lista), date.today())

# --- Alertas -----------------------------------------------------------------------------

with seccion("alertas", "ambar" if alertas else "verde"):
    st.markdown(f"<div class='titulo-bloque'>Alertas de hoy ({len(alertas)})</div>", unsafe_allow_html=True)
    if not alertas:
        st.write("Nada especial hoy en tu lista.")
    for a in sorted(alertas, key=lambda x: -x.importancia):
        icono = "⚠️" if a.importancia > 1 else "ℹ️"
        st.markdown(f"{icono} {a.texto} [Ver →](/proyeccion?activo={a.ticker})")
    st.caption("Las alertas avisan de situaciones (caída fuerte, volumen inusual, pánico, resultados cercanos o un "
               "movimiento muy grande) para que las mires con calma. No son señales de compra ni de venta.")

# --- Resumen -----------------------------------------------------------------------------

estudios = cargar_estudios()
plazo = st.segmented_control("Plazo de la probabilidad", ["1 mes", "6 meses", "1 año"], default="1 año",
                             key="plazo_lista") or "1 año"
filas = []
for t in lista:
    df = datos.get(t)
    fila_e = estudios[(estudios["ticker"] == t) & (estudios["plazo"] == plazo)] if not estudios.empty else pd.DataFrame()
    e = fila_e.iloc[0] if len(fila_e) else None
    prox = info.get(t, {}).get("fecha")
    prox = pd.Timestamp(prox) if prox is not None and pd.notna(prox) else None
    filas.append({
        "Nombre": nombre(t),
        "Último cierre": numero(df["Close"].iloc[-1]) if df is not None else "—",
        "Último día": df["Close"].pct_change().iloc[-1] * 100 if df is not None and len(df) > 1 else None,
        "Prob. más arriba": round(e["prob_sube"] * 100) if e is not None and pd.notna(e.get("prob_sube")) else None,
        "Caída posible": (e["inferior"] / e["cierre"] - 1) * 100 if e is not None and pd.notna(e.get("inferior")) else None,
        "Salud": SALUD.get(e.get("salud"), "—") if e is not None and isinstance(e.get("salud"), str) else "—",
        "Próximos resultados": prox.strftime("%d/%m/%Y") if prox is not None else "—",
        "Ver": f"/proyeccion?activo={t}",
    })
with seccion("resumen", "blanco"):
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch", column_config={
        "Último día": st.column_config.NumberColumn("Último día", format="%+.1f %%"),
        "Prob. más arriba": st.column_config.ProgressColumn("Prob. más arriba", format="%d %%",
                                                            min_value=0, max_value=100),
        "Caída posible": st.column_config.NumberColumn("Caída posible", format="%+.0f %%"),
        "Ver": st.column_config.LinkColumn("Ver", display_text="Ver →"),
    })
    st.caption("Probabilidad y caída posible del último estudio calculado. «Próximos resultados» según Yahoo Finance.")
