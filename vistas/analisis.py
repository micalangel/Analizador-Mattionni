"""Página «Análisis avanzado»: indicadores, fundamentales, resultados, backtest y calibración."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analizador import backtest as bt
from analizador import ia
from analizador import noticias as nt
from analizador import resultados as rs
from analizador.calibration import cobertura_movil
from analizador.catalogo import ACTIVOS, buscar, descripcion, nombre, opciones_ordenadas
from analizador.forecast import METODOS, REFERENCIA, intervalo_actual
from analizador.indicators import calcular_todos
from vistas.comun import (COLORES, GRIS, calcular_backtest, calcular_calibracion, cargar,
                          cargar_fundamentales, cargar_noticias, cargar_resultados, entrenar_ia, estilo)

# --- Barra lateral ---------------------------------------------------------------------

st.sidebar.title("Parámetros")
elegidos = st.sidebar.multiselect("Qué quieres analizar", opciones_ordenadas(), default=["^GSPC", "AAPL", "MSFT"],
                                  format_func=descripcion, accept_new_options=True,
                                  help="Busca por nombre o escribe cualquier ticker de Yahoo Finance.")
hoy = date.today()
inicio = st.sidebar.date_input("Fecha inicial", hoy - timedelta(days=365 * 10), max_value=hoy)
fin = st.sidebar.date_input("Fecha final", hoy, max_value=hoy)

st.sidebar.subheader("Pronóstico")
horizonte = st.sidebar.slider("Horizonte (días hábiles)", 1, 60, 10)
confianza = st.sidebar.select_slider("Nivel de confianza", [0.5, 0.6, 0.7, 0.8, 0.9, 0.95], value=0.8,
                                     format_func=lambda x: f"{x:.0%}")
metodo = st.sidebar.selectbox("Método del intervalo", list(METODOS), index=2, format_func=METODOS.get)
ventana_vol = st.sidebar.number_input("Ventana de volatilidad (días)", 20, 250, 60, step=10)

st.sidebar.subheader("Backtest")
estrategia = st.sidebar.selectbox("Estrategia", list(bt.ESTRATEGIAS))
coste_pb = st.sidebar.number_input("Coste por operación (puntos básicos)", 0.0, 100.0, 10.0, step=1.0)
desliz_pb = st.sidebar.number_input("Deslizamiento (puntos básicos)", 0.0, 100.0, 5.0, step=1.0)
nivel_ingenuo = st.sidebar.slider("Posición constante de la línea base ingenua", 0.0, 1.0, 0.5, 0.05)
st.sidebar.caption("1 punto básico = 0,01 % del importe operado.")

# --- Cabecera ------------------------------------------------------------------------

st.title("Avanzado")
st.caption("Indicadores, fundamentales, resultados y noticias, backtest y calibración detallada. "
           "Solo se muestran números e incertidumbre: sin recomendaciones ni precios objetivo.")

tickers = []
for e in elegidos:
    t = e if e in ACTIVOS else (buscar(e) or [str(e).strip().upper()])[0]
    if t not in tickers:
        tickers.append(t)
if not tickers:
    st.info("Elige al menos un activo en la barra lateral.")
    st.stop()
if inicio >= fin:
    st.error("La fecha inicial debe ser anterior a la final.")
    st.stop()

datos = cargar(tuple(tickers), inicio, fin)
sin_datos = [t for t in tickers if t not in datos]
if sin_datos:
    st.error(f"Sin datos para: {', '.join(nombre(t) for t in sin_datos)}. Revisa el nombre o el rango de fechas.")
if not datos:
    st.stop()

color_de = {t: COLORES[i % len(COLORES)] for i, t in enumerate(tickers)}
ticker = st.selectbox("Analizado en detalle", list(datos), format_func=nombre)
df = datos[ticker]
st.caption(f"{nombre(ticker)} ({ticker}): {len(df)} sesiones del {df.index[0]:%d/%m/%Y} al {df.index[-1]:%d/%m/%Y}. "
           "Precios ajustados por dividendos y splits (auto_adjust=True).")

tab_precio, tab_fund, tab_res, tab_bt, tab_pron = st.tabs(
    ["📉 Precio e indicadores", "📋 Fundamentales", "🧾 Resultados y noticias", "🔁 Backtest",
     "🎯 Pronóstico y calibración"])

# --- Precio e indicadores -------------------------------------------------------------

with tab_precio:
    ind = calcular_todos(df)
    c1, c2 = st.columns([3, 1])
    with c2:
        medias = st.multiselect("Medias móviles", ["MM20", "MM50", "MM200"], ["MM50", "MM200"])
        bollinger = st.checkbox("Bandas de Bollinger (20, 2σ)", True)
    with c1:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.5, 0.16, 0.17, 0.17],
                            subplot_titles=("Precio", "RSI (14)", "MACD (12, 26, 9)", "Drawdown"))
        if bollinger:
            fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_superior"], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"), 1, 1)
            fig.add_trace(go.Scatter(x=ind.index, y=ind["BB_inferior"], line=dict(width=0),
                                     fill="tonexty", fillcolor="rgba(138,137,132,0.18)",
                                     name="Bollinger", hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["Close"], name="Cierre",
                                 line=dict(color=COLORES[0], width=2)), 1, 1)
        for i, mm in enumerate(medias):
            fig.add_trace(go.Scatter(x=ind.index, y=ind[mm], name=mm,
                                     line=dict(color=COLORES[i + 1], width=1.5)), 1, 1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["RSI14"], name="RSI", showlegend=False,
                                 line=dict(color=COLORES[6], width=1.5)), 2, 1)
        for nivel in (30, 70):
            fig.add_hline(y=nivel, line=dict(color=GRIS, width=1, dash="dot"), row=2, col=1)
        fig.add_trace(go.Bar(x=ind.index, y=ind["MACD_hist"], name="Histograma MACD", showlegend=False,
                             marker_color=GRIS, opacity=0.5), 3, 1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD"], name="MACD",
                                 line=dict(color=COLORES[0], width=1.5)), 3, 1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["MACD_senal"], name="Señal",
                                 line=dict(color=COLORES[1], width=1.5)), 3, 1)
        fig.add_trace(go.Scatter(x=ind.index, y=ind["Drawdown"], name="Drawdown", fill="tozeroy", showlegend=False,
                                 line=dict(color=COLORES[7], width=1.5)), 4, 1)
        fig.update_yaxes(tickformat=".0%", row=4, col=1)
        estilo(fig, 840).update_layout(margin=dict(t=80), legend=dict(y=1.06))
        st.plotly_chart(fig, width="stretch")

    ultimo = ind.iloc[-1]
    m = st.columns(4)
    m[0].metric("Último cierre", f"{ultimo['Close']:.2f}")
    m[1].metric("RSI (14)", f"{ultimo['RSI14']:.1f}")
    m[2].metric("Volatilidad 20 d (anual)", f"{ultimo['Vol20_anual']:.1%}")
    m[3].metric("Drawdown actual", f"{ultimo['Drawdown']:.1%}")

    if len(datos) > 1:
        st.subheader("Comparación de rendimiento (base 100)")
        fig = go.Figure()
        for t, d in datos.items():
            base = d["Close"] / d["Close"].iloc[0] * 100
            fig.add_trace(go.Scatter(x=base.index, y=base, name=nombre(t), line=dict(color=color_de[t], width=2)))
        st.plotly_chart(estilo(fig), width="stretch")
        st.caption("Cada serie se reescala a 100 en su primer día disponible. "
                   "El rendimiento pasado no predice resultados futuros.")

    with st.expander("Tabla de indicadores (últimos 30 días)"):
        st.dataframe(ind.tail(30).iloc[::-1].round(4), width="stretch")

# --- Fundamentales --------------------------------------------------------------------

with tab_fund:
    st.subheader("Ratios fundamentales")
    if st.button("Consultar fundamentales"):
        st.session_state["fund"] = tuple(datos)
    if st.session_state.get("fund") == tuple(datos):
        tabla = cargar_fundamentales(tuple(datos))
        formatos = {"PER": "{:.1f}", "Precio/Valor contable": "{:.2f}", "Deuda/Patrimonio": "{:.2f}",
                    "Margen de beneficio": "{:.1%}", "Rentabilidad FCF": "{:.2%}"}
        vista = tabla.copy().astype(object)
        for col, fmt in formatos.items():
            vista[col] = [fmt.format(v) if pd.notna(v) else "—" for v in tabla[col]]
        st.dataframe(vista, width="stretch")
        st.caption("PER = precio / beneficio por acción de los últimos 12 meses. Rentabilidad FCF = "
                   "flujo de caja libre / capitalización. Deuda/Patrimonio en veces. "
                   "«—» = dato no disponible (habitual en ETF). La tabla no ordena ni puntúa valores.")

# --- Resultados y noticias ---------------------------------------------------------------

with tab_res:
    st.subheader(f"Estudio de resultados financieros: {nombre(ticker)}")
    datos_res = cargar_resultados(ticker)

    def millones(v):
        return "—" if pd.isna(v) else f"{v / 1e6:,.0f}"

    periodo = st.radio("Periodicidad", ["Trimestral", "Anual"], horizontal=True)
    tabla_fin = datos_res.trimestral if periodo == "Trimestral" else datos_res.anual
    tabla_fin = tabla_fin.dropna(subset=["Ingresos"])  # periodos con solo balance: fuera
    if tabla_fin.dropna(how="all").empty:
        st.warning("No hay estados financieros para este ticker (habitual en ETF e índices).")
    else:
        fig = go.Figure()
        for i, col in enumerate(["Ingresos", "Beneficio operativo", "Beneficio neto", "Flujo de caja libre"]):
            fig.add_trace(go.Bar(x=tabla_fin.index.strftime("%Y-%m"), y=tabla_fin[col] / 1e6, name=col,
                                 marker_color=COLORES[i]))
        fig.update_layout(title="Cifras principales (millones)", barmode="group", bargap=0.25,
                          bargroupgap=0.05, xaxis_type="category")
        st.plotly_chart(estilo(fig, 380), width="stretch")

        vista = tabla_fin.T.copy()
        vista.columns = tabla_fin.index.strftime("%Y-%m")
        bpa = vista.loc["BPA diluido"]
        vista = vista.map(millones).astype(object)
        vista.loc["BPA diluido"] = ["—" if pd.isna(v) else f"{v:.2f}" for v in bpa]
        st.caption("Importes en millones de la moneda de los estados; BPA por acción.")
        st.dataframe(vista, width="stretch")

        ratios = rs.margenes_y_crecimiento(tabla_fin, 4 if periodo == "Trimestral" else 1)
        vista_r = ratios.T.copy()
        vista_r.columns = ratios.index.strftime("%Y-%m")
        vista_r = vista_r.astype(object)
        for fila in vista_r.index:
            fmt = "{:.2f}" if fila == "Deuda / Patrimonio" else "{:.1%}"
            vista_r.loc[fila] = ["—" if pd.isna(v) else fmt.format(v) for v in ratios[fila]]
        st.markdown("**Márgenes, crecimiento interanual y endeudamiento**")
        st.dataframe(vista_r, width="stretch")
        st.caption("Crecimiento interanual: frente al mismo trimestre del año anterior (o al año anterior). "
                   "«—» si no hay dato comparable o la base es negativa.")

    st.markdown("#### Publicaciones de resultados y reacción del precio")
    estudio_pub = rs.estudio_publicaciones(df["Close"], datos_res.publicaciones)
    prox = rs.proxima_publicacion(datos_res.publicaciones)
    if prox:
        st.write(f"Próxima publicación anunciada: **{prox['fecha']:%d/%m/%Y %H:%M}** (hora de Nueva York). "
                 f"BPA estimado por el consenso: **{prox['BPA estimado']:.2f}**.")
    hechas = estudio_pub.dropna(subset=["BPA publicado", "reaccion"]) if not estudio_pub.empty else estudio_pub
    if hechas.empty:
        st.warning("No hay historial de publicaciones con precio disponible en el rango elegido.")
    else:
        r = rs.resumen_publicaciones(estudio_pub)
        k = st.columns(4)
        k[0].metric("Publicaciones analizadas", r["n"])
        k[1].metric("Superan la estimación", f"{r['% superan estimación']:.0%}")
        k[2].metric("Movimiento absoluto medio", f"{r['Reacción absoluta media']:.1%}")
        k[3].metric("Rango típico (p10 – p90)", f"{r['Reacción p10']:+.1%} / {r['Reacción p90']:+.1%}")
        st.caption(f"En el {r['% reacción en la dirección de la sorpresa']:.0%} de los casos el precio se movió "
                   f"en la misma dirección que la sorpresa. Correlación sorpresa-reacción: "
                   f"{r['Correlación sorpresa-reacción']:.2f} (con n = {r['n']}, muy incierta). "
                   f"|z| medio = {r['|z| medio']:.1f}: tamaño del movimiento en desviaciones típicas de un "
                   "día normal previo.")

        fig = go.Figure(go.Scatter(
            x=hechas["Sorpresa %"], y=hechas["reaccion"], mode="markers",
            marker=dict(color=COLORES[0], size=10, line=dict(width=2, color="white")),
            text=[f"{i:%d/%m/%Y}" for i in hechas.index],
            hovertemplate="%{text}<br>Sorpresa %{x:.1f} %<br>Reacción %{y:.1%}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=GRIS, width=1))
        fig.add_vline(x=0, line=dict(color=GRIS, width=1))
        fig.update_layout(title="Sorpresa de BPA frente a reacción del precio", hovermode="closest",
                          xaxis_title="Sorpresa del BPA (%)", yaxis_title="Reacción del precio")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(estilo(fig, 380), width="stretch")

        tabla_pub = estudio_pub.sort_index(ascending=False)
        vista_p = pd.DataFrame({
            "Publicación (NY)": [f"{i:%d/%m/%Y %H:%M}" for i in tabla_pub.index],
            "Momento": tabla_pub["momento"].fillna("—"),
            "BPA estimado": tabla_pub["BPA estimado"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
            "BPA publicado": tabla_pub["BPA publicado"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
            "Sorpresa": tabla_pub["Sorpresa %"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f} %"),
            "Reacción": tabla_pub["reaccion"].map(lambda v: "—" if pd.isna(v) else f"{v:+.2%}"),
            "z": tabla_pub["z"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}"),
        })
        st.dataframe(vista_p, width="stretch", hide_index=True)
        with st.expander("Cómo se mide la reacción"):
            st.markdown(
                "- Publicación **tras el cierre**: del cierre de ese día al cierre de la sesión siguiente.\n"
                "- **Antes de la apertura**: del cierre anterior al cierre de ese día.\n"
                "- **Hora desconocida**: ventana de dos sesiones.\n"
                "- **z** = reacción logarítmica / volatilidad diaria de los 60 días previos (× √sesiones). "
                "Solo usa datos anteriores a la publicación.\n"
                "- Son estadísticas descriptivas del pasado; no anticipan la próxima reacción.")

    st.divider()
    st.subheader(f"Noticias recientes: {nombre(ticker)}")
    cantidad = st.slider("Número de noticias", 5, 40, 15, step=5)
    estudio_nt = nt.estudio_noticias(df["Close"], cargar_noticias(ticker, cantidad))
    if estudio_nt.empty:
        st.warning("Yahoo Finance no ha devuelto noticias para este ticker.")
    else:
        vista_n = pd.DataFrame({
            "Fecha (NY)": [f"{i.tz_convert('America/New_York'):%d/%m/%Y %H:%M}" for i in estudio_nt.index],
            "Titular": estudio_nt["titular"],
            "Fuente": estudio_nt["fuente"],
            "Movimiento en su sesión": estudio_nt["reaccion"].map(
                lambda v: "pendiente" if pd.isna(v) else f"{v:+.2%}"),
            "z": estudio_nt["z"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}"),
            "Enlace": estudio_nt["enlace"],
        })
        st.dataframe(vista_n, width="stretch", hide_index=True,
                     column_config={"Enlace": st.column_config.LinkColumn("Enlace", display_text="Abrir")})
        st.caption("«pendiente» = la sesión de reacción aún no ha cerrado o queda fuera del rango de fechas. "
                   "|z| > 2 indica un movimiento poco habitual para ese valor, sin implicar causa.")

# --- Backtest ---------------------------------------------------------------------------

with tab_bt:
    st.subheader(f"Backtest walk-forward: {estrategia}")
    st.caption("Cada día la estrategia solo ve los datos hasta el cierre; la posición decidida se "
               "aplica al día siguiente. Se descuentan costes y deslizamiento en cada cambio de posición.")
    historia = 200
    if len(df) <= historia + 30:
        st.warning(f"Hacen falta más de {historia + 30} sesiones para el backtest. Amplía el rango de fechas.")
    else:
        capitales, posiciones, tabla = calcular_backtest(df, estrategia, coste_pb / 1e4, desliz_pb / 1e4,
                                                         nivel_ingenuo, historia)
        colores_bt = dict(zip(capitales, [COLORES[0], COLORES[1], COLORES[2]]))
        fig = go.Figure()
        for serie, cap in capitales.items():
            fig.add_trace(go.Scatter(x=cap.index, y=cap, name=serie,
                                     line=dict(color=colores_bt[serie], width=2)))
        fig.update_layout(title="Valor de 1 unidad invertida", yaxis_title="Capital")
        st.plotly_chart(estilo(fig), width="stretch")

        formato = {"CAGR": "{:.2%}", "Volatilidad anual": "{:.2%}", "Sharpe": "{:.2f}",
                   "Máx. drawdown": "{:.2%}", "Tasa de acierto": "{:.1%}",
                   "Nº operaciones": "{:.0f}", "Rotación anual": "{:.2f}"}
        st.dataframe(tabla.style.format(formato, na_rep="—"), width="stretch")

        fig = go.Figure()
        for serie, cap in capitales.items():
            dd = cap / cap.cummax() - 1
            fig.add_trace(go.Scatter(x=dd.index, y=dd, name=serie,
                                     line=dict(color=colores_bt[serie], width=1.5)))
        fig.update_layout(title="Drawdown de cada curva")
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(estilo(fig, 320), width="stretch")

        pos = posiciones[estrategia]
        fig = go.Figure(go.Scatter(x=pos.index, y=pos, line=dict(color=COLORES[0], width=1.5, shape="hv"),
                                   name="Posición"))
        fig.update_layout(title="Posición de la estrategia (1 = invertido, 0 = liquidez)")
        st.plotly_chart(estilo(fig, 240), width="stretch")
        with st.expander("Cómo leer estos resultados"):
            st.markdown(
                "- **Comprar y mantener**: posición 1 todo el periodo (un solo coste de entrada).\n"
                f"- **Posición constante {nivel_ingenuo:g}**: línea base ingenua; misma exposición todos "
                "los días sin mirar los datos. Si la estrategia no la supera en Sharpe, su lógica de "
                "*timing* no aporta frente a simplemente reducir la exposición.\n"
                "- Un buen resultado histórico puede deberse al azar o al sobreajuste de parámetros.\n"
                "- Tasa de acierto = % de días con exposición y retorno neto positivo.\n"
                "- Rotación anual = suma de |cambios de posición| por año.")

# --- Pronóstico y calibración ----------------------------------------------------------

with tab_pron:
    st.subheader(f"Rango de precios a {horizonte} días hábiles con confianza del {confianza:.0%}")
    cierre = df["Close"]
    try:
        iv = intervalo_actual(cierre, horizonte, confianza, metodo, int(ventana_vol))
    except ValueError as e:
        st.warning(str(e))
        iv = None

    if iv:
        k = st.columns(3)
        k[0].metric("Límite inferior", f"{iv['inferior']:.2f}")
        k[1].metric("Límite superior", f"{iv['superior']:.2f}")
        k[2].metric("Último cierre (referencia)", f"{iv['precio_ultimo']:.2f}")
        st.caption(f"Método: {METODOS[metodo]}. Si el modelo estuviera bien calibrado, el precio quedaría "
                   f"dentro de este rango aproximadamente {confianza:.0%} de las veces, y fuera el "
                   f"{1 - confianza:.0%} restante. No es un precio previsto.")

        historico = cierre.iloc[-250:]
        futuras = pd.bdate_range(iv["fecha"], periods=horizonte + 1)
        pasos = range(horizonte + 1)
        log_inf = np.log(iv["inferior"] / iv["precio_ultimo"])
        log_sup = np.log(iv["superior"] / iv["precio_ultimo"])
        inf = [iv["precio_ultimo"] * np.exp(log_inf * np.sqrt(s / horizonte)) for s in pasos]
        sup = [iv["precio_ultimo"] * np.exp(log_sup * np.sqrt(s / horizonte)) for s in pasos]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historico.index, y=historico, name="Cierre histórico",
                                 line=dict(color=COLORES[0], width=2)))
        fig.add_trace(go.Scatter(x=futuras, y=sup, mode="lines", line=dict(width=1, color=COLORES[1]),
                                 name="Límite superior"))
        fig.add_trace(go.Scatter(x=futuras, y=inf, mode="lines", line=dict(width=1, color=COLORES[1]), fill="tonexty",
                                 fillcolor="rgba(235,104,52,0.15)", name="Límite inferior"))
        fig.update_layout(title=f"Último año y rango al {confianza:.0%} (ensanchamiento ∝ √tiempo, ilustrativo)")
        st.plotly_chart(estilo(fig), width="stretch")

    st.subheader("🤖 IA entrenable")
    st.caption("Modelo de aprendizaje automático (gradient boosting de cuantiles) que aprende de este ticker "
               "cuándo los movimientos suelen ser mayores o menores de lo que indica la volatilidad reciente. "
               "Se entrena solo con el pasado, se reentrena cada 126 sesiones y se corrige con calibración "
               "conformal. Compite con los demás métodos en el informe de calibración de abajo.")
    clave_ia = (ticker, horizonte, confianza, str(inicio), str(fin))
    if st.button(f"Entrenar IA con {nombre(ticker)}", type="primary"):
        st.session_state["ia_clave"] = clave_ia
    tabla_ia = None
    if st.session_state.get("ia_clave") == clave_ia:
        mercado = None if ticker == "SPY" else cargar(("SPY",), inicio, fin).get("SPY", pd.DataFrame()).get("Close")
        publicaciones = cargar_resultados(ticker).publicaciones
        tabla_ia, modelo_ia, historial_ia = entrenar_ia(ticker, df, horizonte, confianza, publicaciones, mercado)
        if modelo_ia is None or tabla_ia["inferior"].isna().all():
            st.warning("No hay datos suficientes para entrenar la IA (hacen falta unos 3 años de sesiones).")
            tabla_ia = None
        else:
            iv_ia = ia.intervalo_actual_ia(df, modelo_ia, publicaciones, mercado)
            k = st.columns(3)
            k[0].metric("IA: límite inferior", f"{iv_ia['inferior']:.2f}")
            k[1].metric("IA: límite superior", f"{iv_ia['superior']:.2f}")
            k[2].metric("Muestras de entrenamiento", f"{modelo_ia.n_muestras + modelo_ia.n_calibracion:,}")
            st.caption(f"Rango a {horizonte} días con confianza declarada del {confianza:.0%} según el modelo "
                       f"entrenado con datos hasta el {modelo_ia.entrenado_hasta:%d/%m/%Y}. Compruébalo con su "
                       "calibración antes de darle crédito.")

            imp = modelo_ia.importancias().head(10).iloc[::-1]
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=COLORES[0],
                                   hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
            fig.update_layout(title="Qué datos ha usado más la IA (importancia relativa)", hovermode="closest")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(estilo(fig, 360), width="stretch")
            st.caption("Importancia = cuánto ha reducido el error cada dato en los árboles del modelo. "
                       "Indica en qué se fija la IA, no qué causa los precios.")
            with st.expander(f"Historial de entrenamientos ({len(historial_ia)} reentrenos walk-forward)"):
                st.dataframe(historial_ia.style.format({"Entrenado hasta": "{:%d/%m/%Y}",
                                                        "Ajuste conformal (z)": "{:+.2f}"}),
                             width="stretch", hide_index=True)
                st.caption("Ajuste conformal > 0: el modelo era demasiado confiado y se ensanchan sus rangos; "
                           "< 0: era demasiado prudente y se estrechan.")

    st.subheader("Informe de calibración")
    st.caption("Para cada día del histórico se emitió un intervalo con los datos disponibles ese día y se "
               f"comparó con el precio real {horizonte} días después.")
    try:
        resumen, detalle = calcular_calibracion(cierre, horizonte, confianza, int(ventana_vol), tabla_ia)
    except ValueError as e:
        st.warning(f"No hay datos suficientes para la calibración: {e}")
        resumen = None

    if resumen is not None:
        vista = resumen.copy()
        vista.index = resumen["descripción"]
        vista = vista.drop(columns=["descripción"])
        si_no = lambda v: "—" if v is None or pd.isna(v) else ("Sí" if v else "No")
        for col in ["calibrado", "mejora vs ingenuo", "útil"]:
            vista[col] = vista[col].map(si_no)
        st.dataframe(vista.style.format({
            "cobertura": "{:.1%}", "objetivo": "{:.0%}", "desviación": "{:+.1%}", "tolerancia": "±{:.1%}",
            "ancho medio %": "{:.2f}", "puntuación intervalo %": "{:.2f}", "MAE centro %": "{:.2f}",
        }), width="stretch")

        fila = resumen.loc[metodo]
        ref = resumen.loc[REFERENCIA]
        texto = (f"**{METODOS[metodo]}**: cobertura {fila['cobertura']:.1%} frente a un objetivo del "
                 f"{confianza:.0%} (tolerancia ±{fila['tolerancia']:.1%}) → "
                 f"{'bien calibrado' if fila['calibrado'] else 'mal calibrado'}. "
                 f"Puntuación de intervalo {fila['puntuación intervalo %']:.2f} % frente a "
                 f"{ref['puntuación intervalo %']:.2f} % de la referencia ingenua → "
                 f"{'mejora' if fila['mejora vs ingenuo'] else 'no mejora'} a la referencia. ")
        texto += ("Cumple ambos criterios en este periodo." if fila["útil"] else
                  "No cumple ambos criterios en este periodo, así que no se considera útil.")
        (st.success if fila["útil"] else st.info)(texto)
        if ia.NOMBRE_METODO in resumen.index:
            f_ia = resumen.loc[ia.NOMBRE_METODO]
            mejores = resumen.drop(index=[REFERENCIA, ia.NOMBRE_METODO])["puntuación intervalo %"]
            texto_ia = (f"**IA**: cobertura {f_ia['cobertura']:.1%} → "
                        f"{'bien calibrada' if f_ia['calibrado'] else 'mal calibrada'}; puntuación "
                        f"{f_ia['puntuación intervalo %']:.2f} % frente a {ref['puntuación intervalo %']:.2f} % de la "
                        f"referencia ingenua y {mejores.min():.2f} % del mejor método estadístico. ")
            texto_ia += ("Cumple ambos criterios en este periodo." if f_ia["útil"] else
                         "No cumple ambos criterios en este periodo, así que no se considera útil.")
            (st.success if f_ia["útil"] else st.info)(texto_ia)

        cm = cobertura_movil(detalle[metodo]).dropna()
        cm_ref = cobertura_movil(detalle[REFERENCIA]).dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cm.index, y=cm, name=METODOS[metodo], line=dict(color=COLORES[0], width=2)))
        fig.add_trace(go.Scatter(x=cm_ref.index, y=cm_ref, name="Referencia ingenua",
                                 line=dict(color=COLORES[1], width=1.5)))
        fig.add_hline(y=confianza, line=dict(color=GRIS, width=1, dash="dash"),
                      annotation_text=f"Objetivo {confianza:.0%}", annotation_position="top left")
        fig.update_layout(title="Cobertura móvil (ventana de 252 fechas de emisión)")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(estilo(fig, 340), width="stretch")

        det = detalle[metodo].iloc[-500:]
        # Se dibuja cada intervalo en la fecha en que se conoce el resultado (t + h).
        fechas_obj = cierre.index[cierre.index.get_indexer(det.index) + horizonte]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fechas_obj, y=det["superior"], line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fechas_obj, y=det["inferior"], line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(42,120,214,0.18)", name="Intervalo emitido h días antes"))
        fig.add_trace(go.Scatter(x=fechas_obj, y=det["realizado"], name="Precio real",
                                 line=dict(color=COLORES[0], width=1.5)))
        fuera = det[~det["dentro"]]
        fig.add_trace(go.Scatter(x=fechas_obj[~det["dentro"].to_numpy()], y=fuera["realizado"], mode="markers",
                                 name="Fuera del intervalo",
                                 marker=dict(color=COLORES[7], size=8, symbol="x")))
        fig.update_layout(title="Intervalos históricos frente al precio real (últimos 500)")
        st.plotly_chart(estilo(fig), width="stretch")

        with st.expander("Cómo leer el informe de calibración"):
            st.markdown(
                "- **Cobertura**: % de veces que el precio real cayó dentro del intervalo. Debe estar "
                "cerca del objetivo: muy por debajo = intervalos demasiado estrechos (exceso de confianza); "
                "muy por encima = demasiado anchos.\n"
                "- **Tolerancia**: margen estadístico aceptado (2 errores típicos con tamaño efectivo n/h, "
                "mínimo 2 puntos), porque los intervalos solapados no son independientes.\n"
                "- **Ancho medio %**: anchura media del intervalo en % del precio de partida.\n"
                "- **Puntuación de intervalo %** (Winkler): ancho + penalización 2/α por cada unidad que el "
                "precio cae fuera. Menor es mejor. Es el *error* con el que se compara cada método.\n"
                "- **MAE centro %**: error absoluto medio del centro del modelo, solo informativo.\n"
                "- **Útil** = bien calibrado **y** mejor puntuación que la referencia ingenua en el mismo "
                "periodo. Que un método sea útil en el pasado no garantiza que lo siga siendo.")
