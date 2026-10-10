"""Página «Simulador»: señales de compra y venta de reglas clásicas, simuladas y sin dinero real."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analizador.catalogo import descripcion, nombre, opciones_ordenadas
from analizador.data import descargar_intradia
from analizador.simulador import REGLAS, max_caida, simular
from vistas.comun import COLORES, cargar, estilo, nota, numero, seccion, tarjeta

VELAS = {"Diarias": None, "1 hora": "1 hora", "5 minutos": "5 minutos"}
PERIODOS = {"1 año": pd.DateOffset(years=1), "5 años": pd.DateOffset(years=5), "10 años": pd.DateOffset(years=10),
            "Todo": None}
VERDE, ROJO = "#0E9F6E", "#DC2626"


@st.cache_data(show_spinner="Descargando velas intradía…", ttl=60 * 15)
def intradia(ticker: str, intervalo: str) -> pd.DataFrame:
    return descargar_intradia(ticker, intervalo)


@st.cache_data(show_spinner="Simulando…", ttl=60 * 15)
def simular_cache(datos: pd.DataFrame, regla: str, parametros: tuple, coste: float, desde):
    return simular(datos, regla, dict(parametros), coste, desde=desde)


st.title("Simulador de estrategias")
st.markdown("<div class='subtitulo'>Elige una regla clásica de compra y venta y mira dónde habría comprado (▲) y "
            "vendido (▼), cuánto habría ganado o perdido y si lo hace mejor que el azar. Todo simulado, sin dinero "
            "real.</div>", unsafe_allow_html=True)
st.write("")

with seccion("elegir", "suave"):
    c1, c2, c3 = st.columns([1.3, 1.3, 1.5], gap="medium")
    with c1:
        st.markdown("<div class='titulo-bloque'>Activo</div>", unsafe_allow_html=True)
        lista = opciones_ordenadas()
        ticker = st.selectbox("Activo", lista, index=lista.index("^GSPC"), format_func=descripcion,
                              label_visibility="collapsed")
    with c2:
        st.markdown("<div class='titulo-bloque'>Velas</div>", unsafe_allow_html=True)
        velas = st.segmented_control("Velas", list(VELAS), default="Diarias", key="velas_sim",
                                     label_visibility="collapsed") or "Diarias"
    with c3:
        st.markdown("<div class='titulo-bloque'>Periodo</div>", unsafe_allow_html=True)
        if velas == "Diarias":
            periodo = st.segmented_control("Periodo", list(PERIODOS), default="10 años", key="periodo_sim",
                                           label_visibility="collapsed") or "10 años"
        else:
            st.caption("Yahoo da los últimos 60 días en velas de 5 minutos y unos 2 años en velas de 1 hora.")
            periodo = None
    c4, c5 = st.columns([1.6, 2.4], gap="large")
    with c4:
        st.markdown("<div class='titulo-bloque'>Regla</div>", unsafe_allow_html=True)
        clave = st.selectbox("Regla", list(REGLAS), format_func=lambda k: REGLAS[k].nombre,
                             label_visibility="collapsed")
        st.caption(REGLAS[clave].explicacion)
    with c5:
        st.markdown("<div class='titulo-bloque'>Ajustes</div>", unsafe_allow_html=True)
        params = {}
        columnas = st.columns(len(REGLAS[clave].parametros) + 1)
        nombres_param = {"rapida": "Media rápida", "lenta": "Media lenta", "media": "Media de", "entrada": "Entrada",
                         "salida": "Salida", "caida": "Caída %", "objetivo": "Objetivo %", "maximo_velas": "Máx. velas"}
        for col, (k, (mn, mx, defecto)) in zip(columnas, REGLAS[clave].parametros.items()):
            params[k] = col.number_input(nombres_param.get(k, k), mn, mx, defecto, key=f"{clave}_{k}")
        coste_pb = columnas[-1].number_input("Costes (pb)", 0, 100, 15,
                                             help="Comisión + deslizamiento por compra o venta, en puntos básicos "
                                                  "(15 pb = 0,15 %).")

desde = None
if velas == "Diarias":
    datos = cargar((ticker,), date(1990, 1, 1), date.today()).get(ticker)
    if datos is not None and PERIODOS[periodo] is not None:
        desde = datos.index[-1] - PERIODOS[periodo]
else:
    datos = intradia(ticker, VELAS[velas])
if datos is None or len(datos) < 60:
    nota("No hay datos suficientes para este activo con estas velas.", "ambar")
    st.stop()

sim = simular_cache(datos, clave, tuple(sorted(params.items())), coste_pb / 1e4, desde)
if desde is not None:
    datos = datos.loc[desde:]
ops = sim.operaciones
unidad = "días" if velas == "Diarias" else "velas"

# --- Resultado ---------------------------------------------------------------------------

with tarjeta("resultado", "azul"):
    hoy_dentro = bool(sim.posicion.iloc[-1])
    cambios = sim.posicion.diff().fillna(0)
    ultimo_cambio = cambios[cambios != 0].index[-1] if (cambios != 0).any() else None
    estado = "DENTRO (comprada)" if hoy_dentro else "FUERA (sin comprar)"
    color = VERDE if hoy_dentro else "#6B6B6B"
    desde = f" desde el {ultimo_cambio:%d/%m/%Y %H:%M}" if ultimo_cambio is not None and velas != "Diarias" else (
        f" desde el {ultimo_cambio:%d/%m/%Y}" if ultimo_cambio is not None else "")
    st.markdown(f"<div class='etiqueta'>{nombre(ticker)} · {REGLAS[clave].nombre} · velas {velas.lower()}</div>"
                f"<div style='font-size:1.6rem;font-weight:600;color:{color}'>Hoy la regla está {estado}{desde}</div>",
                unsafe_allow_html=True)
    k = st.columns(4)
    k[0].metric("Resultado de la regla", f"{sim.total_regla:+.1%}")
    k[1].metric("Comprar y mantener", f"{sim.total_mantener:+.1%}")
    k[2].metric("Operaciones", f"{len(ops)}", f"{(ops['resultado'] > 0).mean():.0%} ganadoras" if len(ops) else None,
                delta_color="off")
    k[3].metric("Mejor que el azar", f"{sim.percentil_azar:.0%}" if sim.percentil_azar is not None else "—",
                help="De 300 versiones con las mismas operaciones pero en momentos al azar, % que lo hicieron peor.")
    gana = sim.total_regla > sim.total_mantener
    if sim.percentil_azar is not None and sim.percentil_azar >= 0.95 and gana:
        veredicto = "En este periodo la regla ganó a comprar y mantener y a casi todas las versiones al azar."
    elif gana:
        veredicto = ("En este periodo ganó a comprar y mantener, pero muchas versiones al azar lo hicieron igual o mejor: "
                     "puede ser suerte.")
    else:
        veredicto = "En este periodo la regla quedó por debajo de simplemente comprar y mantener."
    st.markdown(f"<div class='suave' style='margin-top:.4rem'>{veredicto} Estuvo invertida el "
                f"{sim.tiempo_dentro:.0%} del tiempo; peor caída de su capital: {max_caida(sim.capital_regla):.0%} "
                f"(comprar y mantener: {max_caida(sim.capital_mantener):.0%}).</div>", unsafe_allow_html=True)

# --- Gráfico con entradas y salidas --------------------------------------------------------

with seccion("grafico", "blanco"):
    st.markdown("<div class='titulo-bloque'>Dónde habría comprado (▲) y vendido (▼)</div>", unsafe_allow_html=True)
    precios = datos["Close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=precios.index, y=precios, name="Precio", line=dict(color=COLORES[1], width=1.5),
                             hovertemplate="%{x|%d/%m/%Y %H:%M}: %{y:,.2f}<extra></extra>"))
    if len(ops):
        fig.add_trace(go.Scatter(x=ops["entrada"], y=ops["precio_entrada"], mode="markers", name="Compra",
                                 marker=dict(symbol="triangle-up", size=11, color=VERDE, line=dict(width=1, color="white")),
                                 hovertemplate="Compra %{x|%d/%m/%Y %H:%M}: %{y:,.2f}<extra></extra>"))
        cerradas = ops[~ops["abierta"]]
        fig.add_trace(go.Scatter(x=cerradas["salida"], y=cerradas["precio_salida"], mode="markers", name="Venta",
                                 marker=dict(symbol="triangle-down", size=11, color=ROJO, line=dict(width=1, color="white")),
                                 hovertemplate="Venta %{x|%d/%m/%Y %H:%M}: %{y:,.2f}<extra></extra>"))
    if velas != "Diarias":  # sin huecos de noches y fines de semana
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(hovermode="closest", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(estilo(fig, 420), width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim.capital_regla.index, y=sim.capital_regla * 100, name="Regla",
                             line=dict(color=COLORES[0], width=2)))
    fig.add_trace(go.Scatter(x=sim.capital_mantener.index, y=sim.capital_mantener * 100, name="Comprar y mantener",
                             line=dict(color="#9CA3AF", width=2)))
    fig.update_layout(title="Cuánto valdrían 100 € (con costes)", legend=dict(orientation="h", y=1.12))
    if velas != "Diarias":
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(estilo(fig, 300), width="stretch")

# --- Operaciones -------------------------------------------------------------------------

with seccion("operaciones", "suave"):
    st.markdown("<div class='titulo-bloque'>Operaciones simuladas</div>", unsafe_allow_html=True)
    if ops.empty:
        st.write("La regla no hizo ninguna operación en este periodo.")
    else:
        formato = "%d/%m/%Y" if velas == "Diarias" else "%d/%m %H:%M"
        st.dataframe(pd.DataFrame({
            "Compra": ops["entrada"].dt.strftime(formato),
            "Precio compra": ops["precio_entrada"].map(numero),
            "Venta": [("abierta" if a else s.strftime(formato)) for s, a in zip(ops["salida"], ops["abierta"])],
            "Precio venta": [("—" if a else numero(p)) for p, a in zip(ops["precio_salida"], ops["abierta"])],
            f"Duración ({unidad})": ops["velas"],
            "Resultado": ops["resultado"] * 100,
        }).iloc[::-1], hide_index=True, width="stretch", height=min(38 * len(ops) + 40, 420),
            column_config={"Resultado": st.column_config.NumberColumn("Resultado", format="%+.1f %%")})
        ganadoras, perdedoras = ops[ops["resultado"] > 0], ops[ops["resultado"] <= 0]
        st.caption(f"Ganancia media de las ganadoras: {ganadoras['resultado'].mean():+.1%} · pérdida media de las "
                   f"perdedoras: {perdedoras['resultado'].mean():+.1%}. Cada compra y cada venta pagan "
                   f"{coste_pb} pb de costes." if len(ganadoras) and len(perdedoras) else "")

with st.expander("Cómo leer el simulador", icon=":material/help:"):
    st.markdown(
        """
- **▲ / ▼**: donde la regla habría comprado y vendido, decidiendo con el cierre de cada vela y operando en la siguiente
  (nunca con información del futuro).
- **Mejor que el azar**: se repite la simulación con las mismas operaciones movidas a momentos al azar. Si la regla no
  supera a la mayoría, su resultado se debe a haber estado invertida en buenos o malos momentos, no a acertar.
- **Costes**: en intradía, con muchas operaciones, las comisiones se comen casi cualquier ventaja. Prueba a subir los
  costes y verás cómo cambia.
- **Probar muchos ajustes hasta encontrar el que mejor sale es hacerse trampas**: lo que funcionó en el pasado con unos
  números concretos rara vez se repite. Esto es para aprender cómo se comportan las reglas, no para operar.
"""
    )
