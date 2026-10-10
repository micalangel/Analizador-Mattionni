"""Página «La IA»: cómo aprende, su diario de predicciones comprobadas y su historial de entrenamientos."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from analizador import registro as rg
from analizador.indices import INDICES
from vistas.comun import cargar, numero, seccion

ORDEN_PLAZOS = ["5 días", "10 días", "1 mes", "6 meses", "1 año"]


@st.cache_data(show_spinner="Comprobando las predicciones con los precios reales…", ttl=60 * 60 * 6)
def diario(hoy: date) -> pd.DataFrame:
    pred = rg.cargar_predicciones()
    if pred.empty:
        return pred
    pred["vence"] = [rg.fecha_comprobacion(f, s) for f, s in zip(pred["fecha"], pred["sesiones_plazo"])]
    vencidas = pred[pred["vence"] <= pd.Timestamp(hoy)]
    precios = {}
    if not vencidas.empty:
        inicio = vencidas["fecha"].min().date() - timedelta(days=10)
        datos = cargar(tuple(sorted(vencidas["ticker"].unique())), inicio, hoy)
        precios = {t: df["Close"] for t, df in datos.items()}
    return rg.comprobar(pred, precios).assign(vence=pred["vence"])


st.title("¿Acierta la IA?")
st.markdown("<div class='subtitulo'>Aquí se ve si la IA funciona de verdad: sus predicciones se apuntan antes de "
            "saber el resultado y se comprueban cuando pasa el plazo.</div>", unsafe_allow_html=True)
st.write("")

with seccion("como", "azul"):
    st.markdown("<div class='titulo-bloque'>Cómo aprende</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**1. Estudia el pasado**  \n{len(INDICES)} índices de 30 países desde 1990, con crisis incluidas, "
                "más el miedo del mercado (VIX) y los tipos de interés.")
    c2.markdown("**2. Se examina con años que no ha visto**  \nCada modelo se entrena solo con datos anteriores al año "
                "que luego predice. Así se mide su acierto sin trampas.")
    c3.markdown("**3. Compite con métodos sencillos**  \nEn cada activo, la IA solo se usa si en sus años recientes "
                "acertó más que la estadística simple. Si no, se usa la estadística.")
    c4.markdown("**4. Se reentrena con lo nuevo**  \nAl reentrenarla incorpora todo lo ocurrido hasta ese día. "
                "El diario de abajo comprueba si sus predicciones reales se cumplen.")

# --- Diario de predicciones ---------------------------------------------------------------

with seccion("diario", "blanco"):
    st.markdown("<div class='titulo-bloque'>Diario de predicciones</div>", unsafe_allow_html=True)
    tabla = diario(date.today())
    if tabla.empty:
        st.info("Todavía no hay predicciones apuntadas. Se apuntan solas cada vez que se calcula el estudio "
                "(`python -m analizador.estudio` o `python -m analizador.entrenamiento`).")
    else:
        hechas = tabla[tabla["comprobada"]]
        pendientes = tabla[~tabla["comprobada"]]
        k = st.columns(3)
        k[0].metric("Predicciones apuntadas", f"{len(tabla):,}".replace(",", "."))
        k[1].metric("Ya comprobadas", f"{len(hechas):,}".replace(",", "."))
        proxima = pendientes["vence"].min() if not pendientes.empty else None
        k[2].metric("Próxima comprobación", proxima.strftime("%d/%m/%Y") if proxima is not None else "—")

        marcador = rg.resumen(tabla)
        if marcador.empty:
            st.info(f"Aún no ha vencido ninguna predicción. Las primeras (a 5 días) se podrán comprobar hacia el "
                    f"{proxima:%d/%m/%Y}; las de 1 año, dentro de un año. El marcador irá creciendo solo.")
        else:
            marcador = marcador.reindex([p for p in ORDEN_PLAZOS if p in marcador.index])
            st.markdown("**Marcador real** (predicciones hechas antes de conocer el resultado)")
            st.dataframe(pd.DataFrame({
                "Comprobadas": marcador["comprobadas"],
                "Días distintos": marcador["dias_distintos"],
                "Acertó el lado (más arriba / más abajo)": marcador["acierto_lado"].map("{:.0%}".format),
                "Decir siempre «más arriba»": marcador["siempre_sube"].map("{:.0%}".format),
                "Precio dentro del rango": marcador["dentro_rango"].map("{:.0%}".format),
                "Se prometía": marcador["confianza"].map("{:.0%}".format),
            }), width="stretch")
            st.caption("Ojo: las predicciones del mismo día se parecen entre sí (si cae todo el mercado, fallan juntas). "
                       "Hasta tener muchos días distintos, el marcador es poco firme.")
            ultimas = hechas.sort_values("fecha_real", ascending=False).head(50)
            with st.expander("Últimas predicciones comprobadas"):
                st.dataframe(pd.DataFrame({
                    "Activo": ultimas["nombre"],
                    "Plazo": ultimas["plazo"],
                    "Dicho el": ultimas["fecha"].dt.strftime("%d/%m/%Y"),
                    "Prob. más arriba": ultimas["prob_sube"].map(lambda v: "—" if pd.isna(v) else f"{v:.0%}"),
                    "Precio entonces": ultimas["cierre"].map(numero),
                    "Precio al vencer": ultimas["precio_real"].map(numero),
                    "¿Lado?": ultimas["acierto_lado"].map({True: "✅", False: "❌"}).fillna("—"),
                    "¿Dentro del rango?": ultimas["dentro_rango"].map({True: "✅", False: "❌"}).fillna("—"),
                }), hide_index=True, width="stretch")

        with st.expander("Predicciones pendientes"):
            pend = pendientes.sort_values("vence")
            st.dataframe(pd.DataFrame({
                "Activo": pend["nombre"], "Plazo": pend["plazo"], "Dicho el": pend["fecha"].dt.strftime("%d/%m/%Y"),
                "Se comprueba hacia": pend["vence"].dt.strftime("%d/%m/%Y"),
                "Prob. más arriba": pend["prob_sube"].map(lambda v: "—" if pd.isna(v) else f"{v:.0%}"),
                "Rango": [f"{numero(a)} – {numero(b)}" for a, b in zip(pend["inferior"], pend["superior"])],
            }), hide_index=True, width="stretch", height=400)

# --- Historial de entrenamientos ----------------------------------------------------------

with seccion("historial", "suave"):
    st.markdown("<div class='titulo-bloque'>Historial de entrenamientos</div>", unsafe_allow_html=True)
    ent = rg.cargar_entrenamientos()
    if ent.empty:
        st.info("Aún no hay entrenamientos registrados.")
    else:
        nombres_plazo = {5: "5 días", 10: "10 días", 21: "1 mes", 126: "6 meses", 252: "1 año"}
        ent["Plazo"] = ent["sesiones_plazo"].map(nombres_plazo)
        st.dataframe(pd.DataFrame({
            "Fecha": ent["fecha"], "Modelo": ent["modelo"], "Plazo": ent["Plazo"],
            "Muestras": ent["muestras"].map(lambda v: f"{v:,.0f}".replace(",", ".")),
            "Qué se mide": ent["metrica"],
            "Resultado (años no vistos)": ent["valor"].map("{:.1%}".format),
            "Referencia": ent["referencia"].map("{:.1%}".format),
        }).iloc[::-1], hide_index=True, width="stretch")
        st.caption("Rangos: el resultado debe parecerse al 80 % prometido. Dirección: la referencia es la estadística "
                   "simple (frecuencia histórica); la IA aporta si su resultado es mayor.")

    with st.expander("¿Cómo se reentrena?"):
        st.markdown(
            """
    En el ordenador donde está instalada la app:

    ```
    python -m analizador.entrenamiento     # reentrena la IA con los últimos datos (≈ 15 min)
    python -m analizador.estudio           # recalcula el estudio de empresas y apunta predicciones (≈ 10 min)
    ```

    Lo ideal es hacerlo cada mes (entrenamiento) y cada día o semana (estudio), para que el diario tenga
    predicciones frecuentes y la IA incorpore lo último. Se puede automatizar con una tarea programada.
    """
        )
