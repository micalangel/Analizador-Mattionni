"""Página «Analizar una empresa»: toda la información útil de una empresa y su análisis técnico objetivo."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analizador import analistas as an
from analizador import ficha as fc
from analizador import noticias as nt
from analizador import per_historico as ph
from analizador import tecnico as tc
from analizador.catalogo import ACTIVOS, buscar, descripcion, nombre, opciones_ordenadas
from analizador.catalogo import pais as pais_de
from analizador.catalogo import sector as sector_de
from analizador.indices import INDICES, MERCADO
from analizador.resultados import (estudio_publicaciones, margenes_y_crecimiento, obtener_resultados, proxima_publicacion,
                                   resumen_publicaciones)
from analizador.salud import comparar_per, ultimo_anio
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


@st.cache_data(show_spinner="Revisando el historial de los analistas…", ttl=60 * 60 * 12)
def datos_analistas(ticker: str):
    vacio = pd.DataFrame()
    try:
        precios_s, cambios, splits = an.descargar(ticker)
        objetivos = an.normalizar(cambios, precios_s, splits)
        if objetivos.empty:
            return precios_s, objetivos, vacio, vacio
        consenso_m = an.consenso(objetivos, an.fin_de_mes(precios_s.loc[objetivos["fecha"].min():]))
        return precios_s, objetivos, consenso_m, an.historial(precios_s, consenso_m)
    except Exception:
        return pd.Series(dtype=float), vacio, vacio, vacio


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

pestanas = st.tabs(["Resumen", "Precio objetivo", "Valoración", "Rentabilidad y deuda", "Crecimiento", "Dividendos",
                    "Accionistas", "Análisis técnico", "Resultados y noticias"])

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

# --- Precio objetivo de los analistas ------------------------------------------------------------

with pestanas[1]:
    st.markdown("<div class='nota'>En Yahoo o Investing ves el precio objetivo de los analistas. Aquí ves además "
                "<b>si los analistas han acertado antes con esta empresa</b>, para que sepas cuánto fiarte de lo que "
                "dicen ahora.</div>", unsafe_allow_html=True)
    n_analistas = f.v("numberOfAnalystOpinions")
    objetivo, bajo_o, alto_o = f.v("targetMeanPrice"), f.v("targetLowPrice"), f.v("targetHighPrice")
    p_sin_div, objetivos, consenso_m, hist = datos_analistas(ticker)
    fiab = an.fiabilidad(hist)
    if not n_analistas or not objetivo:
        st.write("Ningún analista publica precio objetivo de esta empresa en Yahoo Finance.")
    else:
        st.markdown("<div class='titulo-bloque' style='margin-top:1rem'>Cuánto creen que valdrá dentro de un año</div>",
                    unsafe_allow_html=True)
        potencial = objetivo / cierre - 1
        with tarjeta("objetivo", "verde" if potencial > 0 else "ambar"):
            c1, c2 = st.columns([1, 1.4])
            with c1:
                st.markdown(f"<div class='etiqueta'>Precio objetivo medio a 12 meses</div>"
                            f"<div class='cifra' style='font-size:2.2rem;white-space:nowrap'>{cifra(objetivo)} {f.moneda}</div>"
                            f"<div style='font-size:1.1rem;font-weight:600;color:{VERDE if potencial > 0 else ROJO}'>"
                            f"{cifra(potencial * 100, '{:+.1f}')} % respecto a hoy ({cifra(cierre)})</div>"
                            f"<div class='suave' style='margin-top:.3rem'>Media de {int(n_analistas)} analistas. Opinión "
                            f"media: <b>{fc.RECOMENDACIONES.get(f.info.get('recommendationKey'), '—')}</b>.</div>",
                            unsafe_allow_html=True)
            with c2:
                if bajo_o and alto_o:
                    lo, hi = min(bajo_o, cierre), max(alto_o, cierre)
                    margen = (hi - lo) * 0.05 or 1
                    lo, hi = lo - margen, hi + margen

                    def x(v):
                        return f"{(v - lo) / (hi - lo):.1%}"

                    st.markdown(
                        f"<div class='etiqueta'>Del objetivo más bajo al más alto</div>"
                        f"<div style='position:relative;height:64px;margin-top:.4rem'>"
                        f"<div style='position:absolute;top:26px;left:{x(bajo_o)};width:calc({x(alto_o)} - {x(bajo_o)});"
                        f"height:10px;background:#D1FAE5;border-radius:5px'></div>"
                        f"<div style='position:absolute;top:18px;left:calc({x(objetivo)} - 2px);width:4px;height:26px;"
                        f"background:{VERDE};border-radius:2px'></div>"
                        f"<div style='position:absolute;top:18px;left:calc({x(cierre)} - 2px);width:4px;height:26px;"
                        f"background:#111;border-radius:2px'></div>"
                        f"<div style='position:absolute;top:0;left:{x(cierre)};transform:translateX(-50%);font-size:.75rem;"
                        f"white-space:nowrap'>Hoy</div>"
                        f"<div style='position:absolute;top:46px;left:{x(objetivo)};transform:translateX(-50%);font-size:.75rem;"
                        f"color:{VERDE};white-space:nowrap'>Media</div></div>"
                        f"<div style='display:flex;justify-content:space-between;font-size:.85rem;color:{GRIS_T}'>"
                        f"<span>Más bajo: {cifra(bajo_o)} ({cifra((bajo_o / cierre - 1) * 100, '{:+.0f}')} %)</span>"
                        f"<span>Más alto: {cifra(alto_o)} ({cifra((alto_o / cierre - 1) * 100, '{:+.0f}')} %)</span></div>",
                        unsafe_allow_html=True)
            rec = f.recomendaciones
            if isinstance(rec, pd.DataFrame) and not rec.empty and "strongBuy" in rec:
                fila = rec.iloc[0]
                partes = [("Compra fuerte", "strongBuy", "#047857"), ("Compra", "buy", "#34D399"),
                          ("Mantener", "hold", "#9CA3AF"), ("Venta", "sell", "#F87171"), ("Venta fuerte", "strongSell", "#B91C1C")]
                total = sum(int(fila.get(k, 0)) for _, k, _ in partes) or 1
                barra = "".join(f"<div title='{n}: {int(fila.get(k, 0))}' style='width:{int(fila.get(k, 0)) / total:.0%};"
                                f"background:{c};height:100%'></div>" for n, k, c in partes if fila.get(k, 0))
                leyenda = " · ".join(f"{n}: {int(fila.get(k, 0))}" for n, k, _ in partes)
                st.markdown(f"<div class='etiqueta' style='margin-top:.8rem'>Qué aconsejan</div>"
                            f"<div style='display:flex;height:14px;border-radius:7px;overflow:hidden'>{barra}</div>"
                            f"<div class='suave' style='margin-top:.3rem'>{leyenda}</div>", unsafe_allow_html=True)

        rev = an.revisiones(objetivos, pd.Timestamp(hoy))
        if rev["suben"] or rev["bajan"]:
            tono = "verde" if rev["suben"] > 2 * rev["bajan"] else "ambar" if rev["bajan"] > rev["suben"] else ""
            tendencia = ("cada vez más optimistas" if tono == "verde" else "menos optimistas que antes" if tono == "ambar"
                         else "sin una tendencia clara")
            nota(f"<b>Últimos 3 meses:</b> {rev['suben']} analistas subieron su precio objetivo y {rev['bajan']} lo "
                 f"bajaron; {rev['mejoran']} mejoraron su opinión y {rev['empeoran']} la empeoraron. Están {tendencia}.",
                 tono)

    # --- ¿Aciertan con esta empresa? ---
    st.markdown("<div class='titulo-bloque' style='margin-top:1.2rem'>¿Han acertado antes con esta empresa?</div>",
                unsafe_allow_html=True)
    if fiab is None:
        st.write("Yahoo no guarda el historial de precios objetivo de esta empresa (suele tenerlo solo para empresas "
                 "de EE. UU.), así que no se puede comprobar cómo han acertado. Toma su objetivo con más cautela.")
    else:
        diferencia = fiab.real_medio - fiab.prometido_medio
        st.markdown(f"Hemos tomado el precio objetivo medio de cada mes de los últimos {fiab.anios:.0f} años y lo hemos "
                    f"comparado con lo que valía la acción un año después.")
        cajas([
            ("Acertaron si subiría o bajaría", f"{fiab.acierto_direccion:.0%}",
             "Veces que el precio fue en la dirección que marcaba su objetivo."),
            ("Llegó a su objetivo en un año", f"{fiab.llego_al_ano:.0%}",
             f"Al cabo de un año estaba en su objetivo o por encima. En algún momento del año lo tocó el {fiab.lo_toco:.0%}."),
            ("Error típico", f"±{fiab.error_medio * 100:.0f} puntos",
             "Distancia media entre la subida que anunciaban y la que hubo de verdad."),
            ("Anunciaban / pasó", f"{fiab.prometido_medio * 100:+.0f} % / {fiab.real_medio * 100:+.0f} %",
             "Subida media a un año que anunciaban y subida media real."),
        ])
        if diferencia > 0.05:
            texto = (f"Con esta empresa los analistas se han quedado <b>cortos</b>: de media, la acción acabó "
                     f"{diferencia * 100:.0f} puntos por encima de lo que anunciaban. Ojo: suele pasar en empresas que "
                     f"han subido muchísimo, algo que nadie preveía y que no tiene por qué repetirse.")
        elif diferencia < -0.05:
            texto = (f"Con esta empresa los analistas han sido <b>demasiado optimistas</b>: de media, la acción acabó "
                     f"{-diferencia * 100:.0f} puntos por debajo de lo que anunciaban.")
        else:
            texto = "Con esta empresa, de media, los analistas han acertado bastante bien la subida a un año."
        nota(texto + f" Su error típico (±{fiab.error_medio * 100:.0f} puntos) indica cuánto puede desviarse el resultado"
             f" del objetivo actual.", "verde" if abs(diferencia) <= 0.05 else "ambar")

        hace_un_ano = consenso_m["objetivo"].copy()
        hace_un_ano.index = hace_un_ano.index + pd.Timedelta(days=365)
        desde = consenso_m.index[0]
        fig = go.Figure()
        fig.add_scatter(x=p_sin_div.loc[desde:].index, y=p_sin_div.loc[desde:], name="Precio real",
                        line=dict(color=COLORES[1], width=2))
        fig.add_scatter(x=hace_un_ano.index, y=hace_un_ano, name="Lo que decían un año antes que valdría",
                        line=dict(color=VERDE, width=2, dash="dot"), line_shape="hv")
        fig.update_layout(title="Precio real frente al objetivo que daban los analistas un año antes",
                          legend=dict(orientation="h", y=-0.15), hovermode="x unified")
        st.plotly_chart(estilo(fig, 400), width="stretch")
        st.caption("Si la línea verde va por encima del precio, se pasaron de optimistas; si va por debajo, se quedaron "
                   "cortos. Precio sin ajustar por dividendos, como los objetivos de los analistas. Las comparaciones "
                   "mensuales se solapan, así que con pocos años de datos las cifras orientan pero no son concluyentes.")

    # --- Cómo usar esta información ---
    st.markdown("<div class='titulo-bloque' style='margin-top:1.2rem'>Cómo usar esta información</div>",
                unsafe_allow_html=True)
    st.markdown("""
1. **El objetivo es a un año y es una media de opiniones.** No dice qué pasará mañana ni es una cifra segura.
2. **Mira su error típico con esta empresa.** Si prometen +10 % y suelen equivocarse en ±30 puntos, el objetivo dice
   poco por sí solo. Cuanto menor sea el error típico, más útil es.
3. **Mira si están de acuerdo.** Si el objetivo más bajo y el más alto están muy separados, hay mucha incertidumbre.
4. **Mira hacia dónde van las revisiones.** Muchos analistas subiendo su objetivo indica que la empresa va mejor de lo
   esperado. Los estudios encuentran que esto tiene algo de valor a corto plazo, pero el efecto es pequeño.
5. **Contrasta con lo demás.** Compara con la [probabilidad de subir de la IA](/proyeccion?activo={t}), el PER frente
   a su media (pestaña *Valoración*) y la salud de las cuentas. Si todo apunta en la misma dirección, la idea es más
   sólida; si se contradicen, cuidado.
6. **Decide tú cuánto arriesgas.** Ninguna fuente acierta siempre: invierte solo lo que puedas mantener si baja,
   piensa en el plazo y no lo pongas todo en una sola empresa.
""".replace("{t}", ticker))
    st.caption("Precios objetivo y opiniones: datos de terceros recogidos por Yahoo Finance, no una predicción de esta web.")

# --- Valoración ----------------------------------------------------------------------------------

with pestanas[2]:
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
    bpa = f.v("trailingEps")
    pierde = per_actual is None and isinstance(bpa, float) and bpa < 0
    cajas([
        ("PER actual", "No tiene" if pierde else cifra(per_actual, "{:.1f}"),
         f"Pierde dinero (beneficio por acción de {cifra(bpa)} {f.moneda} en 12 meses), así que no hay PER. Mira el "
         f"PER esperado." if pierde else "Precio / beneficio de los últimos 12 meses."),
        ("Su PER medio (10 años)", cifra(per.mediana if per else None, "{:.1f}"),
         "Con qué PER suele cotizar esta empresa." if per else
         (f"Cotiza en bolsa desde el {precios.index[0]:%d/%m/%Y}: todavía no tiene historia suficiente."
          if (precios.index[-1] - precios.index[0]).days < 730 else "No hay datos fiables para calcularlo.")),
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

with pestanas[3]:
    fcf_anual, nota_fcf = f.v("freeCashflow"), ""
    if fcf_anual is None and not f.trimestral.empty and "Flujo de caja libre" in f.trimestral:
        fcf_anual = ultimo_anio(f.trimestral["Flujo de caja libre"])
        nota_fcf = " Suma de los 4 últimos trimestres." if pd.notna(fcf_anual) else ""
    if (fcf_anual is None or pd.isna(fcf_anual)) and not f.anual.empty and "Flujo de caja libre" in f.anual:
        serie_fcf = f.anual["Flujo de caja libre"].dropna()
        if len(serie_fcf):
            fcf_anual, nota_fcf = float(serie_fcf.iloc[-1]), f" Dato del año {serie_fcf.index[-1].year}."
    if fcf_anual is not None and pd.notna(fcf_anual) and fcf_anual < 0:
        nota_fcf += " Negativo: quema caja."
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
        ("Flujo de caja libre anual", grande(fcf_anual) + f" {f.moneda}",
         "Dinero que genera de verdad cada año después de invertir." + nota_fcf),
    ])

# --- Crecimiento --------------------------------------------------------------------------------

with pestanas[4]:
    crec_ventas, crec_beneficio = f.v("revenueGrowth"), f.v("earningsGrowth")
    if not f.trimestral.empty and "Ingresos" in f.trimestral:
        tabla_crec = margenes_y_crecimiento(f.trimestral.dropna(subset=["Ingresos"]), 4)

        def ultimo(columna):
            serie = tabla_crec[columna].dropna()
            return float(serie.iloc[-1]) if len(serie) else None

        crec_ventas = crec_ventas if crec_ventas is not None else ultimo("Crec. ingresos (interanual)")
        crec_beneficio = crec_beneficio if crec_beneficio is not None else ultimo("Crec. beneficio neto (interanual)")
    cajas([
        ("Crecimiento de ventas", cifra(crec_ventas, "{:+.1%}"), "Último trimestre frente al mismo del año anterior."),
        ("Crecimiento del beneficio", cifra(crec_beneficio, "{:+.1%}"),
         "Último trimestre frente al mismo del año anterior." + (" Sin dato si el año anterior tuvo pérdidas."
                                                                 if crec_beneficio is None else "")),
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

with pestanas[5]:
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

# --- Accionistas ----------------------------------------------------------------------------

with pestanas[6]:
    st.markdown("<div class='titulo-bloque'>Quién tiene las acciones</div>", unsafe_allow_html=True)
    cajas([
        ("Directivos y fundadores", cifra(f.v("heldPercentInsiders"), "{:.1%}"),
         "Un porcentaje alto indica que los dueños siguen dentro."),
        ("Fondos e instituciones", cifra(f.v("heldPercentInstitutions"), "{:.1%}"), "Bancos, fondos de inversión y pensiones."),
        ("Apuestas a la baja", cifra(f.v("shortPercentOfFloat"), "{:.1%}"),
         "Parte de las acciones vendidas «en corto». Por encima del 10 % es mucho."),
    ])

# --- Análisis técnico ----------------------------------------------------------------------------

with pestanas[7]:
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

with pestanas[8]:
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
        tabla_res = pd.DataFrame({
            "Fecha": [f"{pd.Timestamp(i):%d/%m/%Y}" for i in hechas.index],
            "BPA esperado": hechas["BPA estimado"].map(cifra),
            "BPA real": hechas["BPA publicado"].map(cifra),
            "Sorpresa": hechas["Sorpresa %"].map(lambda v: cifra(v, "{:+.1f} %")),
            "Reacción del precio": hechas["reaccion"].map(lambda v: cifra(v, "{:+.1%}")),
        }).reset_index(drop=True)
        sorpresas = hechas["Sorpresa %"].to_numpy()
        reacciones = hechas["reaccion"].to_numpy()

        def colorear(fila):
            """Fila verde clara si superó lo esperado y roja clara si no; la reacción, según si subió o bajó."""
            sorpresa, reaccion = sorpresas[fila.name], reacciones[fila.name]
            fondo = "#ECFDF5" if sorpresa > 0 else "#FEF2F2" if sorpresa < 0 else "#F9FAFB"
            estilos = [f"background-color: {fondo}"] * len(fila)
            if pd.notna(reaccion):
                color = "#047857" if reaccion > 0 else "#B91C1C"
                estilos[-1] += f"; color: {color}; font-weight: 600"
            return estilos

        st.dataframe(tabla_res.style.apply(colorear, axis=1), hide_index=True, width="stretch")
        st.caption("Verde: el beneficio superó lo que esperaban los analistas; rojo: se quedó por debajo. La reacción del "
                   "precio va en verde si subió y en rojo si bajó: a veces la bolsa reacciona al revés de lo esperado.")
    if noticias is not None and not noticias.empty:
        st.markdown("<div class='titulo-bloque' style='margin-top:1rem'>Noticias recientes</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Fecha": [f"{i.tz_convert('Europe/Madrid'):%d/%m %H:%M}" for i in noticias.index],
            "Titular": noticias["titular"], "Fuente": noticias["fuente"],
            "Abrir": noticias["enlace"],
        }), hide_index=True, width="stretch",
            column_config={"Abrir": st.column_config.LinkColumn("Abrir", display_text="Leer →")})
