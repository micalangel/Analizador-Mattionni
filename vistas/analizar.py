"""Página «Analizar una empresa»: toda la información útil de una empresa y su análisis técnico objetivo."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analizador import ficha as fc
from analizador import noticias as nt
from analizador import per_historico as ph
from analizador import tecnico as tc
from analizador.catalogo import ACTIVOS, buscar, descripcion, nombre, opciones_ordenadas
from analizador.catalogo import pais as pais_de
from analizador.catalogo import sector as sector_de
from analizador.indices import INDICES, MERCADO
from analizador.resultados import estudio_publicaciones, obtener_resultados, proxima_publicacion, resumen_publicaciones
from analizador.salud import comparar_per
from vistas.comun import COLORES, cargar, estilo, nota, numero, per_referencia, seccion, tarjeta

INICIO = date(2000, 1, 1)
VERDE, ROJO, GRIS_T = "#0E9F6E", "#DC2626", "#6B6B6B"
INDICE_DEL_PAIS = {}
for _t, (_pais, _nombre, _) in INDICES.items():
    INDICE_DEL_PAIS.setdefault(_pais, (_t, _nombre))


@st.cache_data(show_spinner="Consultando los datos de la empresa…", ttl=60 * 60 * 6)
def datos_ficha(ticker: str) -> fc.Ficha:
    return fc.descargar(ticker)


@st.cache_data(show_spinner="Calculando su PER histórico…", ttl=60 * 60 * 12)
def datos_per(ticker: str):
    try:
        precios, publicaciones, oficial = ph.descargar(ticker)
        return ph.calcular(precios, publicaciones, oficial)
    except Exception:
        return None


@st.cache_data(show_spinner="Consultando resultados y noticias…", ttl=60 * 60)
def datos_resultados(ticker: str, hoy: date):
    datos = cargar((ticker,), INICIO, hoy).get(ticker)
    res = obtener_resultados(ticker)
    estudio = estudio_publicaciones(datos["Close"], res.publicaciones) if datos is not None else pd.DataFrame()
    noticias = nt.estudio_noticias(datos["Close"], nt.obtener_noticias(ticker, 10)) if datos is not None else pd.DataFrame()
    return res.publicaciones, estudio, noticias


def cifra(valor, formato="{:.2f}") -> str:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return "—"
    return formato.format(valor).replace(",", "X").replace(".", ",").replace("X", ".")


def grande(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    for limite, sufijo in ((1e12, " billones"), (1e9, " mil millones"), (1e6, " millones")):
        if abs(x) >= limite:
            return cifra(x / limite, "{:,.1f}") + sufijo
    return cifra(x, "{:,.0f}")


def cajas(elementos: list[tuple[str, str, str]]) -> None:
    """Rejilla de tarjetas (título, cifra, explicación)."""
    html = "".join(f"<div class='caja'><div class='etiqueta'>{t}</div><div class='cifra-media' style='font-size:1.45rem'>"
                   f"{v}</div><p style='font-size:.85rem;margin-top:.3rem'>{e}</p></div>" for t, v, e in elementos)
    st.markdown(f"<div class='rejilla'>{html}</div>", unsafe_allow_html=True)


# --- Selección -----------------------------------------------------------------------------

st.title("Analizar una empresa")
st.markdown("<div class='subtitulo'>Toda la información útil de una empresa en un solo sitio: qué hace, cuánto vale, "
            "cómo van sus cuentas, sus dividendos, quién la tiene, qué dicen los analistas y su situación técnica.</div>",
            unsafe_allow_html=True)
st.write("")

acciones = [t for t in opciones_ordenadas() if ACTIVOS[t][1] == "Acción"]
if "empresa_analizar" not in st.session_state:
    enlace = st.query_params.get("activo")
    st.session_state["empresa_analizar"] = enlace if enlace in acciones else "ITX.MC"
with seccion("elegir", "suave"):
    eleccion = st.selectbox("Empresa", acciones, format_func=descripcion, key="empresa_analizar",
                            accept_new_options=True, help="Busca por nombre o escribe cualquier ticker de Yahoo.")
ticker = eleccion if eleccion in ACTIVOS else (buscar(eleccion) or [str(eleccion).upper()])[0]
st.query_params["activo"] = ticker

hoy = date.today()
precios = cargar((ticker,), INICIO, hoy).get(ticker)
if precios is None or precios.empty:
    st.error(f"No se han encontrado datos de «{eleccion}».")
    st.stop()
f = datos_ficha(ticker)
cierre = float(precios["Close"].iloc[-1])
cambio = precios["Close"].pct_change().iloc[-1]

# --- Cabecera ----------------------------------------------------------------------------------

with tarjeta("cabecera", "azul"):
    c1, c2 = st.columns([2, 1])
    with c1:
        industria = f.info.get("industry") or ""
        st.markdown(f"<div class='etiqueta'>{f.sector} · {industria} · {pais_de(ticker) if ticker in ACTIVOS else f.info.get('country', '')}"
                    f" · {ticker}</div><div style='font-size:1.9rem;font-weight:600'>{nombre(ticker, f.nombre)}</div>",
                    unsafe_allow_html=True)
        color = VERDE if cambio >= 0 else ROJO
        st.markdown(f"<span class='cifra-media'>{numero(cierre)} {f.moneda}</span> "
                    f"<span style='color:{color};font-weight:500;margin-left:.5rem'>{cambio:+.2%} último día</span>",
                    unsafe_allow_html=True)
    with c2:
        alto, bajo = f.v("fiftyTwoWeekHigh"), f.v("fiftyTwoWeekLow")
        if alto and bajo and alto > bajo:
            pos = min(max((cierre - bajo) / (alto - bajo), 0), 1)
            st.markdown(f"<div class='etiqueta'>Rango del último año</div>"
                        f"<div style='position:relative;height:8px;background:#E5E7EB;border-radius:4px;margin:.6rem 0'>"
                        f"<div style='position:absolute;left:calc({pos:.0%} - 6px);top:-4px;width:12px;height:16px;"
                        f"background:#2563EB;border-radius:3px'></div></div>"
                        f"<div style='display:flex;justify-content:space-between;font-size:.85rem;color:{GRIS_T}'>"
                        f"<span>{numero(bajo)}</span><span>{numero(alto)}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='etiqueta' style='margin-top:.6rem'>Capitalización</div>"
                    f"<div style='font-weight:500'>{grande(f.v('marketCap'))} {f.moneda}</div>", unsafe_allow_html=True)
    st.markdown(f"[Ver probabilidad y rango de precio →](/proyeccion?activo={ticker}) · "
                f"[Simular estrategias →](/simulador)")

pestanas = st.tabs(["Resumen", "Valoración", "Rentabilidad y deuda", "Crecimiento", "Dividendos",
                    "Accionistas y analistas", "Análisis técnico", "Resultados y noticias"])

# --- Resumen ------------------------------------------------------------------------------------

with pestanas[0]:
    resumen_negocio = f.info.get("longBusinessSummary")
    if resumen_negocio:
        st.markdown("<div class='titulo-bloque'>Qué hace la empresa</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='nota'>{resumen_negocio}</div>", unsafe_allow_html=True)
        st.caption("Descripción de Yahoo Finance, en inglés.")
    cajas([
        ("Empleados", cifra(f.v("fullTimeEmployees"), "{:,.0f}"), "Plantilla a tiempo completo."),
        ("Beta", cifra(f.v("beta")), "Cuánto se mueve frente al mercado: 1 = igual; más de 1 = más que el mercado."),
        ("Ventas anuales", grande(f.v("totalRevenue")) + f" {f.moneda}", "Ingresos de los últimos 12 meses."),
        ("Beneficio neto anual", grande(f.v("netIncomeToCommon")) + f" {f.moneda}", "Lo que le queda después de todo."),
    ])
    web = f.info.get("website")
    if web:
        st.caption(f"Web oficial: {web}")

# --- Valoración ----------------------------------------------------------------------------------

with pestanas[1]:
    per = datos_per(ticker)
    medianas, _ = per_referencia()
    sector_cat = sector_de(ticker)
    per_actual = f.v("trailingPE")
    if per is not None:
        etiqueta = per.etiqueta
        color = VERDE if per.diferencia < -0.2 else ROJO if per.diferencia > 0.25 else GRIS_T
        nota(f"<b>¿Está cara respecto a su historia?</b> Su PER hoy es <b>{cifra(per.actual, '{:.1f}')}</b> y su media de "
             f"los últimos {per.anios:.0f} años es <b>{cifra(per.mediana, '{:.1f}')}</b> (normalmente entre "
             f"{cifra(per.p10, '{:.0f}')} y {cifra(per.p90, '{:.0f}')}). "
             f"<span style='color:{color};font-weight:600'>{etiqueta} ({per.diferencia:+.0%})</span>.")
    rel_sector, dif_sector = comparar_per(per_actual, medianas.get(sector_cat))
    cajas([
        ("PER actual", cifra(per_actual, "{:.1f}"), "Precio / beneficio de los últimos 12 meses."),
        ("Su PER medio (10 años)", cifra(per.mediana if per else None, "{:.1f}"),
         "Con qué PER suele cotizar esta empresa." if per else "No hay datos fiables para calcularlo."),
        ("PER medio de su sector", cifra(medianas.get(sector_cat), "{:.1f}"),
         f"{rel_sector}{f' ({dif_sector:+.0%})' if dif_sector is not None else ''}." if rel_sector != "—"
         else "Mediana de las empresas de su sector en esta web."),
        ("PER esperado", cifra(f.v("forwardPE"), "{:.1f}"), "Con el beneficio que esperan los analistas para el próximo año."),
        ("PEG", cifra(f.v("trailingPegRatio") or f.v("pegRatio")),
         "PER dividido entre el crecimiento esperado. Alrededor de 1 se considera equilibrado."),
        ("Precio / ventas", cifra(f.v("priceToSalesTrailing12Months")), "Cuántas veces sus ventas anuales vale en bolsa."),
        ("Precio / valor contable", cifra(f.v("priceToBook")), "Cuántas veces sus fondos propios vale en bolsa."),
        ("EV / EBITDA", cifra(f.v("enterpriseToEbitda"), "{:.1f}"),
         "Valor de la empresa (con su deuda) frente a su beneficio operativo antes de amortizaciones."),
    ])
    if per is not None:
        fig = go.Figure()
        s = per.serie
        fig.add_trace(go.Scatter(x=s.index, y=s, name="PER", line=dict(color=COLORES[0], width=1.5)))
        fig.add_hrect(y0=per.p10, y1=per.p90, fillcolor="rgba(37,99,235,0.08)", line_width=0)
        fig.add_hline(y=per.mediana, line=dict(color=COLORES[1], dash="dot"),
                      annotation_text=f"Media {cifra(per.mediana, '{:.1f}')}", annotation_position="top left")
        fig.update_layout(title="Su PER a lo largo de los años (zona azul: donde suele estar)", showlegend=False)
        st.plotly_chart(estilo(fig, 320), width="stretch")
        st.caption("PER calculado con el precio de cada día y el beneficio por acción de los cuatro trimestres anteriores "
                   "(según el calendario de resultados de Yahoo), por eso puede diferir un poco del PER oficial.")

# --- Rentabilidad y deuda ---------------------------------------------------------------------

with pestanas[2]:
    caja_neta = (f.v("totalCash") or 0) - (f.v("totalDebt") or 0) if f.v("totalCash") is not None else None
    cajas([
        ("Margen bruto", cifra(f.v("grossMargins"), "{:.1%}"), "Lo que queda de cada venta tras el coste de lo vendido."),
        ("Margen operativo", cifra(f.v("operatingMargins"), "{:.1%}"), "Lo que queda tras los gastos del negocio."),
        ("Margen neto", cifra(f.v("profitMargins"), "{:.1%}"), "Beneficio final por cada 100 de ventas."),
        ("Rentabilidad sobre fondos propios (ROE)", cifra(f.v("returnOnEquity"), "{:.1%}"),
         "Beneficio que saca del dinero de sus accionistas. Más del 15 % es alto."),
        ("Rentabilidad sobre activos (ROA)", cifra(f.v("returnOnAssets"), "{:.1%}"), "Beneficio que saca de todo lo que tiene."),
        ("Caja", grande(f.v("totalCash")) + f" {f.moneda}", "Dinero disponible."),
        ("Deuda", grande(f.v("totalDebt")) + f" {f.moneda}", "Deuda financiera total."),
        ("Caja neta (caja − deuda)", grande(caja_neta) + f" {f.moneda}",
         "Positiva: tiene más caja que deuda. Negativa: debe más de lo que tiene en caja."),
        ("Ratio de liquidez", cifra(f.v("currentRatio")),
         "Activos a corto plazo / deudas a corto plazo. Por debajo de 1 puede tener apuros para pagar a corto."),
        ("Flujo de caja libre anual", grande(f.v("freeCashflow")) + f" {f.moneda}",
         "Dinero que genera de verdad cada año después de invertir."),
    ])

# --- Crecimiento --------------------------------------------------------------------------------

with pestanas[3]:
    cajas([
        ("Crecimiento de ventas", cifra(f.v("revenueGrowth"), "{:+.1%}"), "Último trimestre frente al mismo del año anterior."),
        ("Crecimiento del beneficio", cifra(f.v("earningsGrowth"), "{:+.1%}"), "Último trimestre frente al mismo del año anterior."),
    ])
    anual = f.anual.dropna(subset=["Ingresos"]) if not f.anual.empty and "Ingresos" in f.anual else pd.DataFrame()
    if not anual.empty:
        fig = go.Figure()
        anos = anual.index.year.astype(str)
        fig.add_trace(go.Bar(x=anos, y=anual["Ingresos"] / 1e6, name="Ventas", marker_color="#BFDBFE"))
        fig.add_trace(go.Bar(x=anos, y=anual["Beneficio neto"] / 1e6, name="Beneficio neto", marker_color=COLORES[0]))
        fig.update_layout(title="Ventas y beneficio por año (millones)", barmode="group", xaxis_type="category")
        st.plotly_chart(estilo(fig, 320), width="stretch")
    if per is not None:
        bpa = ph.bpa_doce_meses(obtener_resultados(ticker).publicaciones).dropna()
        if len(bpa) > 4:
            fig = go.Figure(go.Scatter(x=bpa.index, y=bpa, mode="lines", line=dict(color=COLORES[1], width=2),
                                       name="BPA 12 meses"))
            fig.update_layout(title="Beneficio por acción de los últimos 12 meses", showlegend=False)
            st.plotly_chart(estilo(fig, 280), width="stretch")
            st.caption("Si el beneficio por acción sube año tras año, la empresa gana cada vez más por cada acción.")

# --- Dividendos ----------------------------------------------------------------------------------

with pestanas[4]:
    rent = fc.rentabilidad_dividendo(f.dividendos, cierre)
    por_ano = fc.dividendos_por_ano(f.dividendos)
    if rent is None or por_ano.empty:
        st.write("Esta empresa no reparte dividendos (o no hay datos).")
    else:
        cajas([
            ("Rentabilidad por dividendo", cifra(rent, "{:.2%}"), "Dividendos cobrados en el último año / precio de hoy."),
            ("Parte del beneficio que reparte", cifra(f.v("payoutRatio"), "{:.0%}"),
             "Por encima del 80-90 % puede ser difícil de mantener."),
            ("Años sin bajar el dividendo", str(fc.anos_subiendo_dividendo(por_ano)),
             "Años seguidos en que el dividendo anual se ha mantenido o ha subido."),
            ("Pagando desde", str(por_ano.index.min()), "Primer año con dividendos en los datos de Yahoo."),
        ])
        fig = go.Figure(go.Bar(x=por_ano.index.astype(str), y=por_ano, marker_color=COLORES[0]))
        fig.update_layout(title=f"Dividendo pagado por acción cada año ({f.moneda})", xaxis_type="category")
        st.plotly_chart(estilo(fig, 300), width="stretch")

# --- Accionistas y analistas ----------------------------------------------------------------

with pestanas[5]:
    st.markdown("<div class='titulo-bloque'>Quién tiene las acciones</div>", unsafe_allow_html=True)
    cajas([
        ("Directivos y fundadores", cifra(f.v("heldPercentInsiders"), "{:.1%}"),
         "Un porcentaje alto indica que los dueños siguen dentro."),
        ("Fondos e instituciones", cifra(f.v("heldPercentInstitutions"), "{:.1%}"), "Bancos, fondos de inversión y pensiones."),
        ("Apuestas a la baja", cifra(f.v("shortPercentOfFloat"), "{:.1%}"),
         "Parte de las acciones vendidas «en corto». Por encima del 10 % es mucho."),
    ])
    st.markdown("<div class='titulo-bloque' style='margin-top:1rem'>Qué dicen los analistas</div>",
                unsafe_allow_html=True)
    n_analistas = f.v("numberOfAnalystOpinions")
    if not n_analistas:
        st.write("No hay analistas que sigan a esta empresa en Yahoo.")
    else:
        rec = f.recomendaciones
        if isinstance(rec, pd.DataFrame) and not rec.empty and "strongBuy" in rec:
            fila = rec.iloc[0]
            partes = [("Compra fuerte", "strongBuy", "#047857"), ("Compra", "buy", "#34D399"), ("Mantener", "hold", "#9CA3AF"),
                      ("Venta", "sell", "#F87171"), ("Venta fuerte", "strongSell", "#B91C1C")]
            total = sum(int(fila.get(k, 0)) for _, k, _ in partes) or 1
            barra = "".join(f"<div title='{n}: {int(fila.get(k, 0))}' style='width:{int(fila.get(k, 0)) / total:.0%};"
                            f"background:{c};height:100%'></div>" for n, k, c in partes if fila.get(k, 0))
            leyenda = " · ".join(f"{n}: {int(fila.get(k, 0))}" for n, k, _ in partes)
            st.markdown(f"<div style='display:flex;height:14px;border-radius:7px;overflow:hidden'>{barra}</div>"
                        f"<div class='suave' style='margin-top:.3rem'>{leyenda}</div>", unsafe_allow_html=True)
        objetivo, bajo_o, alto_o = f.v("targetMeanPrice"), f.v("targetLowPrice"), f.v("targetHighPrice")
        cajas([
            ("Opinión media", fc.RECOMENDACIONES.get(f.info.get("recommendationKey"), "—"), f"De {int(n_analistas)} analistas."),
            ("Precio objetivo medio", f"{cifra(objetivo)} {f.moneda}",
             f"{(objetivo / cierre - 1):+.0%} respecto a hoy." if objetivo else ""),
            ("Rango de sus objetivos", f"{cifra(bajo_o)} – {cifra(alto_o)}", "El más bajo y el más alto de los analistas."),
        ])
        st.caption("Datos de terceros recogidos por Yahoo Finance, no una predicción de esta web. Los precios objetivo "
                   "de los analistas se equivocan a menudo y suelen ser optimistas.")

# --- Análisis técnico ----------------------------------------------------------------------------

with pestanas[6]:
    indice_t, indice_n = INDICE_DEL_PAIS.get(pais_de(ticker), (MERCADO, "S&P 500"))
    referencia = cargar((indice_t,), INICIO, hoy).get(indice_t)
    lecturas = tc.analizar(precios, referencia["Close"] if referencia is not None else None, f"el {indice_n}")
    alc, baj, neu = tc.resumen(lecturas)
    st.markdown(f"<div class='nota'><b>Resumen técnico:</b> {alc} lecturas alcistas, {baj} bajistas y {neu} neutrales. "
                "Describen la situación de hoy con reglas fijas; no predicen el futuro (en «Experimentos» se ve que estas "
                "señales, por sí solas, apenas aciertan la dirección mejor que el azar).</div>", unsafe_allow_html=True)
    periodo = st.segmented_control("Periodo", ["6 meses", "1 año", "3 años", "10 años"], default="1 año",
                                   key="periodo_tecnico", label_visibility="collapsed") or "1 año"
    meses = {"6 meses": 6, "1 año": 12, "3 años": 36, "10 años": 120}[periodo]
    d = precios.loc[precios.index[-1] - pd.DateOffset(months=meses):]
    mm50 = precios["Close"].rolling(50).mean().loc[d.index]
    mm200 = precios["Close"].rolling(200).mean().loc[d.index]
    from analizador.indicators import rsi as rsi_f
    rsi_s = rsi_f(precios["Close"], 14).loc[d.index]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.18, 0.2], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Precio",
                                 increasing_line_color=VERDE, decreasing_line_color=ROJO), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=mm50, name="Media 50", line=dict(color="#F59E0B", width=1.4)), 1, 1)
    fig.add_trace(go.Scatter(x=d.index, y=mm200, name="Media 200", line=dict(color=COLORES[0], width=1.4)), 1, 1)
    for nv in tc.niveles(precios):
        fig.add_hline(y=nv.precio, line=dict(color=VERDE if nv.tipo == "soporte" else ROJO, width=1, dash="dash"),
                      annotation_text=f"{'S' if nv.tipo == 'soporte' else 'R'} {numero(nv.precio)}",
                      annotation_position="right", row=1, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d["Volume"], name="Volumen", marker_color="#CBD5E1"), 2, 1)
    fig.add_trace(go.Scatter(x=d.index, y=rsi_s, name="RSI", line=dict(color="#8B5CF6", width=1.3)), 3, 1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(color="#9CA3AF", width=1, dash="dot"), row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.04), hovermode="x unified")
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    st.plotly_chart(estilo(fig, 620).update_layout(margin=dict(r=90)), width="stretch")
    st.caption("Velas verdes: cerró más alto que abrió; rojas, más bajo. Líneas discontinuas: soportes (S, verde) y "
               "resistencias (R, rojo) donde el precio se ha girado en el último año. Abajo, volumen y RSI.")

    tabla = pd.DataFrame([{"Grupo": l.grupo, "Indicador": l.indicador, "Valor": l.valor,
                           "Lectura": l.lectura, "Tono": {"alcista": "▲ alcista", "bajista": "▼ bajista",
                                                          "neutral": "· neutral"}[l.tono]} for l in lecturas])
    st.dataframe(tabla, hide_index=True, width="stretch", height=min(38 * len(tabla) + 40, 680))
    niveles_tab = tc.niveles(precios)
    if niveles_tab:
        st.markdown("<div class='titulo-bloque'>Soportes y resistencias del último año</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([{"Tipo": n.tipo.capitalize(), "Precio": numero(n.precio),
                                    "Distancia": f"{n.distancia:+.1%}", "Veces que se giró ahí": n.toques,
                                    "Última vez": n.ultimo.strftime("%d/%m/%Y")} for n in niveles_tab]),
                     hide_index=True, width="stretch")

# --- Resultados y noticias ---------------------------------------------------------------------

with pestanas[7]:
    publicaciones, estudio, noticias = datos_resultados(ticker, hoy)
    prox = proxima_publicacion(publicaciones)
    r = resumen_publicaciones(estudio) if not estudio.empty else {"n": 0}
    cajas([
        ("Próximos resultados", f"{pd.Timestamp(prox['fecha']):%d/%m/%Y}" if prox else "—",
         f"Beneficio por acción esperado: {cifra(prox['BPA estimado'])}" if prox else "Sin fecha anunciada."),
        ("Supera lo esperado", cifra(r.get("% superan estimación"), "{:.0%}"),
         f"De sus últimas {r['n']} presentaciones." if r.get("n") else ""),
        ("Suele moverse ese día", cifra(r.get("Reacción absoluta media"), "±{:.1%}"), "Movimiento medio del precio."),
    ])
    if not estudio.empty:
        hechas = estudio.dropna(subset=["BPA publicado"]).sort_index(ascending=False).head(12)
        st.dataframe(pd.DataFrame({
            "Fecha": [f"{pd.Timestamp(i):%d/%m/%Y}" for i in hechas.index],
            "BPA esperado": hechas["BPA estimado"].map(cifra),
            "BPA real": hechas["BPA publicado"].map(cifra),
            "Sorpresa": hechas["Sorpresa %"].map(lambda v: cifra(v, "{:+.1f} %")),
            "Reacción del precio": hechas["reaccion"].map(lambda v: cifra(v, "{:+.1%}")),
        }), hide_index=True, width="stretch")
    if noticias is not None and not noticias.empty:
        st.markdown("<div class='titulo-bloque' style='margin-top:1rem'>Noticias recientes</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Fecha": [f"{i.tz_convert('Europe/Madrid'):%d/%m %H:%M}" for i in noticias.index],
            "Titular": noticias["titular"], "Fuente": noticias["fuente"],
            "Abrir": noticias["enlace"],
        }), hide_index=True, width="stretch",
            column_config={"Abrir": st.column_config.LinkColumn("Abrir", display_text="Leer →")})
