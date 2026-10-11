"""Página principal: ¿más arriba o más abajo? y ¿hasta dónde?, a 5 días, 10 días, 1 mes, 6 meses y 1 año."""

from __future__ import annotations

import zlib
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analizador import direccion as dr
from analizador import entrenamiento as ent
from analizador import noticias as nt
from analizador import salud as sl
from analizador import situaciones as sit
from analizador import volumen as vol
from analizador.catalogo import sector as sector_de, tipo as tipo_de
from analizador.fundamentals import obtener_fundamentales
from analizador.resultados import margenes_y_crecimiento, obtener_resultados
from analizador.catalogo import ACTIVOS, buscar, buscar_en_yahoo, descripcion, nombre, opciones_ordenadas
from analizador.forecast import HORIZONTES
from analizador.indices import INDICES, MERCADO
from analizador.proyeccion import DESCRIPCIONES, ESTADOS, NOMBRES_CORTOS, camino_probable, proyectar
from vistas.comun import (COLORES, cargar, cargar_macro, estilo, lista_cargar, lista_guardar, nota, numero,
                          per_referencia, seccion, tarjeta)

REFERENCIA_NUEVOS = "^IXIC"  # para activos sin historia propia suficiente

INICIO_HISTORIA = date(1990, 1, 1)
CONFIANZAS = {"50 %": 0.5, "80 %": 0.8, "95 %": 0.95}
# Periodo inicial del gráfico según el plazo, y opciones para ver más o menos historia.
HISTORIA_GRAFICO = {5: "3 meses", 10: "3 meses", 21: "6 meses", 126: "1 año", 252: "2 años"}
PERIODOS_GRAFICO = {"3 meses": pd.DateOffset(months=3), "6 meses": pd.DateOffset(months=6),
                    "1 año": pd.DateOffset(years=1), "2 años": pd.DateOffset(years=2),
                    "5 años": pd.DateOffset(years=5), "10 años": pd.DateOffset(years=10), "Todo": None}
VERDE, ROJO = "#0E9F6E", "#DC2626"
POPULARES = ["^GSPC", "^IBEX", "AAPL", "NVDA", "SAN.MC", "ITX.MC", "GC=F", "BTC-USD", "EURUSD=X", "SPCX"]


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def buscar_yahoo(texto: str) -> list[tuple[str, str]]:
    return buscar_en_yahoo(texto)


@st.cache_resource(show_spinner="Cargando la IA…")
def cargar_modelos() -> tuple[dict, dict]:
    rangos = {h: c for h in HORIZONTES.values() if (c := ent.cargar(h)) is not None}
    direcciones = {h: {v: dr.cargar(h, ent.DIRECTORIO, v) for v in dr.VARIANTES} for h in HORIZONTES.values()}
    return rangos, direcciones


@st.cache_data(show_spinner="Calculando y comprobando con todo su histórico… La primera vez que alguien consulta un "
                             "activo puede tardar hasta un minuto; después sale al momento.", ttl=60 * 60 * 6)
def calcular(ticker: str, confianza: float, hoy: date):
    datos = cargar((ticker,), INICIO_HISTORIA, hoy).get(ticker)
    if datos is None or datos.empty:
        return None, None
    mercado = None
    if ticker != MERCADO:
        m = cargar((MERCADO,), INICIO_HISTORIA, hoy).get(MERCADO)
        mercado = m["Close"] if m is not None else None
    rangos, direcciones = cargar_modelos()
    return datos, proyectar(datos, confianza, rangos, mercado, direcciones=direcciones, macro=cargar_macro(hoy))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def situaciones(ticker: str, horizonte: int, hoy: date) -> tuple[dict, dict]:
    datos = cargar((ticker,), INICIO_HISTORIA, hoy).get(ticker)
    macro = cargar_macro(hoy)
    precios = datos["Close"]
    hoy_estado = sit.situacion_hoy(precios, macro)
    historiales = {r: sit.historial(precios, r, horizonte, macro) for r in sit.REGLAS}
    return hoy_estado, historiales


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def frecuencia_referencia(horizonte: int, hoy: date) -> float | None:
    ref = cargar((REFERENCIA_NUEVOS,), INICIO_HISTORIA, hoy).get(REFERENCIA_NUEVOS)
    if ref is None:
        return None
    f = dr.frecuencia_historica(ref["Close"], horizonte).dropna()
    return float(f.iloc[-1]) if len(f) else None


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def nombre_yahoo(ticker: str) -> str | None:
    """Nombre de un ticker que no está en el catálogo (p. ej. «Tesla, Inc.» para TSLA)."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        return info.get("shortName") or info.get("longName")
    except Exception:
        return None


@st.cache_data(show_spinner="Consultando las cuentas de la empresa…", ttl=60 * 60 * 12)
def ficha_salud(ticker: str):
    fund = obtener_fundamentales([ticker]).iloc[0]
    res = obtener_resultados(ticker)
    trimestral = res.trimestral.dropna(subset=["Ingresos"])
    crecimiento = margenes_y_crecimiento(trimestral, 4) if not trimestral.empty else trimestral
    fund = sl.completar(fund, trimestral, res.anual)
    return sl.indicadores(fund, trimestral, crecimiento, sector_de(ticker)), trimestral


@st.cache_data(show_spinner="Buscando noticias…", ttl=60 * 30)
def ficha_noticias(ticker: str, hoy: date) -> pd.DataFrame:
    datos = cargar((ticker,), INICIO_HISTORIA, hoy).get(ticker)
    return nt.estudio_noticias(datos["Close"], nt.obtener_noticias(ticker, 12))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def experimento_volumen() -> pd.DataFrame | None:
    ruta = ent.DIRECTORIO / "estudio_volumen.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


COLOR_ESTADO = {"bien": ("#ECFDF5", "#047857", "●"), "regular": ("#FFFBEB", "#B45309", "●"),
                "mal": ("#FEF2F2", "#B91C1C", "●"), "info": ("#F3F4F6", "#4B5563", "○")}


def caja_indicador(ind) -> str:
    fondo, color, punto = COLOR_ESTADO[ind.estado]
    return (f"<div class='caja' style='background:{fondo};border-color:transparent'>"
            f"<div class='etiqueta'><span style='color:{color}'>{punto}</span> {ind.nombre}</div>"
            f"<div class='cifra-media' style='font-size:1.5rem'>{ind.valor}</div>"
            f"<p style='font-size:.85rem;margin-top:.35rem'>{ind.explicacion}</p></div>")


def resolver(texto: str) -> tuple[str, str | None]:
    """Convierte lo elegido o escrito en un ticker; busca por nombre, país o ticker en el catálogo.

    Un «=» delante fuerza el ticker de Yahoo tal cual (p. ej. «=IBEX» para la acción con ese ticker).
    """
    if texto in ACTIVOS:
        return texto, None
    bruto = str(texto).strip()
    if bruto.startswith("="):
        return bruto[1:].strip().upper(), None
    encontrados = buscar(bruto)
    if encontrados:
        t = encontrados[0]
        aviso_txt = None
        if bruto.upper() != t and len(encontrados) > 1:
            otros = ", ".join(nombre(x) for x in encontrados[1:4])
            aviso_txt = f"«{bruto}»: se muestra {nombre(t)}. También encaja: {otros}."
        return t, aviso_txt
    en_yahoo = buscar_yahoo(bruto)
    if en_yahoo:
        t, n = en_yahoo[0]
        otros = ", ".join(f"{nn} ({tt})" for tt, nn in en_yahoo[1:4])
        return t, f"«{bruto}» no está en la lista; Yahoo Finance lo identifica como {n} ({t})." + (
            f" Otras coincidencias: {otros}." if otros else "")
    return bruto.upper(), None


def fecha_objetivo(ultima: pd.Timestamp, sesiones: int) -> pd.Timestamp:
    return pd.bdate_range(ultima, periods=sesiones + 1)[-1]


def inclinacion(p: float) -> tuple[str, str]:
    """Frase corta y color según la probabilidad de estar más arriba."""
    if p >= 0.65:
        return "Bastante más probable que esté más arriba", VERDE
    if p >= 0.55:
        return "Algo más probable que esté más arriba", VERDE
    if p > 0.45:
        return "Prácticamente a cara o cruz", "#6B6B6B"
    if p > 0.35:
        return "Algo más probable que esté más abajo", ROJO
    return "Bastante más probable que esté más abajo", ROJO


def barra_probabilidad(p: float) -> str:
    arriba = round(p * 100)
    return f"""
<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;margin:14px 0 6px 0;background:#E5E7EB">
  <div style="width:{100 - arriba}%;background:#E5E7EB"></div>
  <div style="width:{arriba}%;background:#2563EB"></div>
</div>
<div style="display:flex;justify-content:space-between;font-size:13px;color:#6B6B6B">
  <span>{100 - arriba} % más abajo</span><span>{arriba} % más arriba</span>
</div>"""


# --- Entrada -------------------------------------------------------------------------------

st.title("¿Subirá o bajará?")
st.markdown("<div class='subtitulo'>Elige una empresa, índice, divisa o materia prima y un plazo. Verás la "
            "probabilidad de que esté más arriba que hoy y entre qué precios se moverá probablemente.</div>",
            unsafe_allow_html=True)
st.write("")

lista = opciones_ordenadas()
if "activo" not in st.session_state:  # primera visita: enlace ?activo=… o el S&P 500
    desde_enlace = st.query_params.get("activo")
    st.session_state["activo"] = desde_enlace if desde_enlace else MERCADO


def _elegir_popular():
    if st.session_state.get("popular"):
        st.session_state["activo"] = st.session_state["popular"]


with seccion("elige", "suave"):
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("<div class='titulo-bloque'>Qué quieres mirar</div>", unsafe_allow_html=True)
        eleccion = st.selectbox("Qué quieres mirar", lista, format_func=descripcion, key="activo",
                                accept_new_options=True, label_visibility="collapsed",
                                placeholder="Escribe una empresa, índice, divisa…",
                                help="Busca por nombre (Amazon, Zara, Santander, oro, euro, Japón…). Si no está en "
                                     "la lista, se busca en Yahoo Finance.")
    with c2:
        st.markdown("<div class='titulo-bloque'>Plazo</div>", unsafe_allow_html=True)
        plazo = st.segmented_control("Plazo", list(HORIZONTES), default="1 mes", key="plazo",
                                     label_visibility="collapsed")
    st.pills("Atajos", POPULARES, format_func=lambda t: nombre(t).split(" (")[0], key="popular",
             on_change=_elegir_popular, label_visibility="collapsed")
    with st.popover("Opciones", icon=":material/tune:"):
        seguridad = st.radio("Seguridad del rango de precios", list(CONFIANZAS), index=1, key="seg",
                             help="Con 80 %, el precio real debería quedar dentro del rango unas 8 de cada 10 "
                                  "veces. Más seguridad = rango más ancho.")

plazo = plazo or "1 mes"
confianza = CONFIANZAS[seguridad or "80 %"]
ticker, aviso_busqueda = resolver(eleccion)
if aviso_busqueda:
    st.caption(f"ℹ️ {aviso_busqueda}")
st.query_params["activo"] = ticker  # la dirección de la página sirve para compartir este activo

datos, proyecciones = calcular(ticker, confianza, date.today())
if datos is None:
    st.error(f"No se ha encontrado «{eleccion}». Puede que no cotice en bolsa (por ejemplo, Mercadona o Mango) o "
             "que tenga otro nombre: prueba con el nombre de la empresa matriz o con su ticker de Yahoo Finance.")
    st.stop()

nombre_activo = nombre(ticker, nombre_yahoo(ticker) if ticker not in ACTIVOS else None)
p = proyecciones[plazo]
r, f, d = p.rango, p.fiabilidad, p.direccion
ultima = datos.index[-1]
destino = fecha_objetivo(ultima, p.horizonte)
precio_hoy = float(datos["Close"].iloc[-1])
estado_hoy, historiales = (situaciones(ticker, p.horizonte, date.today()) if len(datos) >= 260 else ({}, {}))
ref = frecuencia_referencia(p.horizonte, date.today()) if d is None else None
obj = int(round(confianza * 100))

# --- Resultado ---------------------------------------------------------------------------------

en_lista = ticker in lista_cargar()
c_nombre, c_estrella = st.columns([4, 1])
with c_estrella:
    if st.button("★ En mi lista" if en_lista else "☆ Añadir a mi lista", key="estrella", width="stretch",
                 help="Quitar de «Mi lista»" if en_lista else "Seguir este activo en «Mi lista»"):
        actual = lista_cargar()
        lista_guardar([t for t in actual if t != ticker] if en_lista else actual + [ticker])
        st.rerun()

with tarjeta("resultado", "azul"):
    st.markdown(f"<div class='etiqueta'>{nombre_activo} · dentro de {plazo} (hacia el {destino:%d/%m/%Y}) · hoy "
                f"{numero(precio_hoy)}</div>", unsafe_allow_html=True)
    izq, der = st.columns(2, gap="large")
    with izq:
        if d is not None:
            frase, color = inclinacion(d.probabilidad)
            st.markdown(f"<div class='cifra'>{d.probabilidad:.0%}</div>"
                        f"<div class='suave'>probabilidad de estar más arriba que hoy</div>"
                        f"<div style='color:{color};font-weight:500;margin-top:.35rem'>{frase}</div>",
                        unsafe_allow_html=True)
            st.markdown(barra_probabilidad(d.probabilidad), unsafe_allow_html=True)
        else:
            st.markdown("<div class='cifra' style='color:#9CA3AF'>—</div>"
                        "<div class='suave'>sin historia suficiente para estimar si subirá o bajará</div>",
                        unsafe_allow_html=True)
            if ref is not None:
                st.caption(f"Como referencia, el Nasdaq ha estado más arriba a {plazo} el {ref:.0%} de las veces "
                           "desde 1990.")
    with der:
        if r is not None:
            st.markdown(f"<div class='etiqueta'>Rango probable ({obj} % de seguridad)</div>"
                        f"<div class='cifra-media'>{numero(r['inferior'])} – {numero(r['superior'])}</div>"
                        f"<div class='suave'>{r['inferior'] / precio_hoy - 1:+.0%} / {r['superior'] / precio_hoy - 1:+.0%} "
                        f"respecto a hoy</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='suave'>Sin datos para calcular el rango.</div>", unsafe_allow_html=True)
        st.write("")
        if d is not None and d.fuente != "frecuencia" and pd.notna(d.acierto_validacion):
            dir_txt = f"acertó si subía o bajaba <b>{d.acierto_validacion:.0%}</b> de las veces"
        elif d is not None:
            dir_txt = f"ha estado más arriba el <b>{d.frecuencia:.0%}</b> de las veces a este plazo"
        else:
            dir_txt = "sin historia suficiente"
        rango_txt = (f"el precio quedó dentro <b>{f['aciertos_de_100']} de cada 100</b> veces (se prometían {obj})"
                     if f.get("aciertos_de_100") is not None else "sin historia suficiente para comprobarlo")
        st.markdown(f"<div class='etiqueta'>Fiabilidad en años recientes</div>"
                    f"<div style='font-size:.95rem;line-height:1.6'>Dirección: {dir_txt}.<br>"
                    f"Rango: {rango_txt}.</div>", unsafe_allow_html=True)

if estado_hoy and (estado_hoy["panico"] or estado_hoy["caida_fuerte"]):
    activas = [nm for nm, on in (("pánico en los mercados", estado_hoy["panico"]),
                                 ("caída fuerte (más de un 20 % por debajo de su máximo del año)",
                                  estado_hoy["caida_fuerte"])) if on]
    nota(f"<b>Situación especial hoy:</b> {' y '.join(activas)}. Históricamente, estos momentos han ido seguidos de "
         "subidas más a menudo que un día normal, aunque con pocos casos. Detalle en «Para saber más».", "ambar")

# --- Gráfico -----------------------------------------------------------------------------

if r is not None:
    with seccion("grafico", "blanco"):
        st.markdown(f"<div class='titulo-bloque'>Precio hasta hoy y camino probable hasta {plazo}</div>",
                    unsafe_allow_html=True)
        c_per, c_ej = st.columns([3, 1.3], vertical_alignment="center")
        periodo = c_per.segmented_control("Periodo del gráfico", list(PERIODOS_GRAFICO),
                                          default=HISTORIA_GRAFICO.get(p.horizonte, "1 año"),
                                          key=f"periodo_grafico_{p.horizonte}", label_visibility="collapsed")
        ver_ejemplos = c_ej.toggle("Caminos de ejemplo", value=True,
                                   help="Recorridos simulados para ver cómo suele moverse de verdad un precio.")
        desplazamiento = PERIODOS_GRAFICO.get(periodo or "1 año")
        historia = datos["Close"] if desplazamiento is None else datos["Close"].loc[ultima - desplazamiento:]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historia.index, y=historia, name="Precio real", line=dict(color=COLORES[1], width=2),
                                 hovertemplate="%{x|%d/%m/%Y}: %{y:,.2f}<extra></extra>"))
        plazos_camino = [(ph.horizonte, ph.rango["inferior"], ph.rango["superior"],
                          ph.direccion.probabilidad if ph.direccion else None)
                         for ph in proyecciones.values() if ph.rango and ph.horizonte <= p.horizonte]
        camino = camino_probable(r["precio_ultimo"], plazos_camino, confianza,
                                 ejemplos=3 if ver_ejemplos else 0, semilla=zlib.crc32(ticker.encode()))
        fechas = pd.bdate_range(ultima, periods=len(camino))
        fig.add_trace(go.Scatter(x=fechas, y=camino["superior"], mode="lines", line=dict(width=1, color=COLORES[0]),
                                 showlegend=False, hovertemplate="%{x|%d/%m/%Y}: máx. %{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=fechas, y=camino["inferior"], mode="lines", line=dict(width=1, color=COLORES[0]),
                                 fill="tonexty", fillcolor="rgba(37,99,235,0.10)", name="Rango probable",
                                 hovertemplate="%{x|%d/%m/%Y}: mín. %{y:,.2f}<extra></extra>"))
        for k, col in enumerate([c for c in camino.columns if c.startswith("ejemplo_")]):
            fig.add_trace(go.Scatter(x=fechas, y=camino[col], mode="lines", line=dict(width=1, color="#9CA3AF"),
                                     opacity=0.8, name="Caminos de ejemplo (simulados)", legendgroup="ejemplos",
                                     showlegend=k == 0, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fechas, y=camino["central"], mode="lines", name="Camino central",
                                 line=dict(color=COLORES[0], width=2.5, dash="dash"),
                                 hovertemplate="%{x|%d/%m/%Y}: camino central %{y:,.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[destino], y=[camino["central"].iloc[-1]], mode="markers+text", showlegend=False,
                                 marker=dict(size=8, color=COLORES[0]), text=[numero(camino["central"].iloc[-1])],
                                 textposition="middle right", cliponaxis=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[destino, destino], y=[r["inferior"], r["superior"]], mode="lines+markers+text",
                                 line=dict(color=COLORES[0], width=3), marker=dict(size=8), showlegend=False,
                                 text=[numero(r["inferior"]), numero(r["superior"])], textposition="middle right",
                                 cliponaxis=False, hoverinfo="skip"))
        fig.add_hline(y=r["precio_ultimo"], line=dict(color="#9CA3AF", width=1, dash="dot"))
        fig.update_layout(hovermode="closest", showlegend=True, legend=dict(orientation="h", y=-0.12))
        margen = min((destino - historia.index[0]) * 0.10, pd.Timedelta(days=150))
        fig.update_xaxes(range=[historia.index[0], destino + margen])
        st.plotly_chart(estilo(fig, 380).update_layout(margin=dict(r=80)), width="stretch")
        st.caption("Camino central (discontinua): a cada fecha, el precio con la mitad de probabilidad por encima y la "
                   "mitad por debajo, coherente con la probabilidad de subir. No es el precio que va a tener: el precio "
                   "real se mueve en zigzag, como los caminos de ejemplo en gris (simulados), y lo normal es que acabe en "
                   "cualquier punto de la zona azul. Línea de puntos = precio de hoy. Puedes ampliar arrastrando con el "
                   "ratón y volver con doble clic.")

# --- Todos los plazos ------------------------------------------------------------------

with seccion("plazos", "azul"):
    st.markdown("<div class='titulo-bloque'>Todos los plazos</div>", unsafe_allow_html=True)
    filas = []
    for etq, ph in proyecciones.items():
        rr, ff, dd = ph.rango, ph.fiabilidad, ph.direccion
        filas.append({
            "Plazo": etq,
            "Prob. más arriba": round(dd.probabilidad * 100) if dd else None,
            "Rango probable": f"{numero(rr['inferior'])} – {numero(rr['superior'])}" if rr else "—",
            "Variación": (f"{rr['inferior'] / rr['precio_ultimo'] - 1:+.0%} / "
                          f"{rr['superior'] / rr['precio_ultimo'] - 1:+.0%}" if rr else "—"),
            "Rango acertó": f"{ff['aciertos_de_100']} de 100" if "aciertos_de_100" in ff else "—",
            "Fiabilidad del rango": ESTADOS[ph.estado],
        })
    st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch",
                 column_config={"Prob. más arriba": st.column_config.ProgressColumn(
                     "Prob. más arriba", format="%d %%", min_value=0, max_value=100)})

# --- Ficha de la empresa ---------------------------------------------------------------

if tipo_de(ticker) in {"Acción", "Otro"} and "Volume" in datos and (datos["Volume"] > 0).mean() > 0.8:
    with seccion("ficha", "blanco"):
        st.markdown(f"<div class='titulo-bloque'>Ficha de {nombre_activo}</div>"
                    f"<a href='/analizar?activo={ticker}' target='_self'>Analizar a fondo (valoración, dividendos, "
                    "analistas, análisis técnico) →</a>", unsafe_allow_html=True)
        t_salud, t_volumen, t_noticias = st.tabs(["Salud de la empresa", "Volumen", "Noticias"])

        with t_salud:
            try:
                indicadores, trimestral = ficha_salud(ticker)
            except Exception:
                indicadores, trimestral = [], pd.DataFrame()
            if not indicadores:
                st.write("No hay datos de las cuentas de esta empresa.")
            else:
                frase_salud, estado_salud = sl.resumen(indicadores)
                fondo, color, _ = COLOR_ESTADO[estado_salud]
                st.markdown(f"<div style='background:{fondo};color:{color};border-radius:10px;padding:.6rem .9rem;"
                            f"font-weight:500;margin-bottom:.8rem'>{frase_salud}</div>", unsafe_allow_html=True)
                st.markdown("<div class='rejilla'>" + "".join(caja_indicador(i) for i in indicadores) + "</div>",
                            unsafe_allow_html=True)
                medianas, mediana_total = per_referencia()
                per_emp = next((i for i in indicadores if i.nombre.startswith("PER")), None)
                sector_emp = sector_de(ticker)
                ref_sector = medianas.get(sector_emp)
                try:
                    per_valor = float(per_emp.valor.split()[0].replace(",", ".")) if per_emp and per_emp.valor != "—" else None
                except ValueError:
                    per_valor = None
                if per_valor and (ref_sector or mediana_total):
                    etiqueta, dif = sl.comparar_per(per_valor, ref_sector or mediana_total)
                    if ref_sector is None:
                        etiqueta = etiqueta.replace("su sector", "las empresas de la web")
                    partes = [f"<b>¿Es alto su PER?</b> {nombre_activo}: <b>{per_valor:.1f}</b>"]
                    if ref_sector:
                        partes.append(f"mediana de su sector ({sector_emp.lower()}): <b>{ref_sector:.1f}</b>")
                    if mediana_total:
                        partes.append(f"mediana de todas las empresas de la web: <b>{mediana_total:.1f}</b>")
                    texto_dif = f" ({dif:+.0%})" if dif is not None else ""
                    nota(" · ".join(partes) + f".<br><b>{etiqueta}{texto_dif}.</b> Un PER más alto que el de su "
                         "sector significa que el mercado paga más por cada euro de beneficio, normalmente porque espera "
                         "que crezca más; uno más bajo puede indicar que está barata o que el mercado desconfía.")
                if not trimestral.empty:
                    ult = trimestral.iloc[-6:]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=ult.index.strftime("%m/%Y"), y=ult["Ingresos"] / 1e6, name="Ventas",
                                         marker_color="#BFDBFE"))
                    fig.add_trace(go.Bar(x=ult.index.strftime("%m/%Y"), y=ult["Beneficio neto"] / 1e6,
                                         name="Beneficio neto", marker_color=COLORES[0]))
                    fig.update_layout(title="Ventas y beneficio por trimestre (millones)", barmode="group",
                                      xaxis_type="category", hovermode="x unified")
                    st.plotly_chart(estilo(fig, 300), width="stretch")
                st.caption("Semáforos orientativos con umbrales generales, no recomendaciones. Datos de Yahoo Finance.")

        with t_volumen:
            z = vol.z_volumen(datos)
            rel = vol.volumen_relativo(datos)
            z_hoy, rel_hoy = z.iloc[-1], rel.iloc[-1]
            if pd.notna(z_hoy) and z_hoy >= vol.UMBRAL_Z:
                estado_vol = "inusualmente alto"
            elif pd.notna(z_hoy) and z_hoy <= -vol.UMBRAL_Z:
                estado_vol = "inusualmente bajo"
            else:
                estado_vol = "normal"
            k1, k2 = st.columns(2)
            k1.metric("Volumen del último día frente a su media", f"{rel_hoy:.1f} veces" if pd.notna(rel_hoy) else "—")
            k2.metric("Hoy el volumen es", estado_vol)
            inus = vol.dias_inusuales(datos, horizonte=21)
            st.markdown("<div class='titulo-bloque' style='margin-top:.6rem'>Días de volumen inusual en los últimos "
                        "dos años</div>", unsafe_allow_html=True)
            if inus.empty:
                st.write("No ha habido días de volumen inusual.")
            else:
                st.dataframe(pd.DataFrame({
                    "Día": inus.index.strftime("%d/%m/%Y"),
                    "Volumen": inus["volumen_relativo"].map(lambda v: f"{v:.1f} veces lo normal"),
                    "Ese día": inus["cambio_dia"].map(lambda v: f"{v:+.1%}"),
                    "Un mes después": inus["despues"].map(lambda v: "pendiente" if pd.isna(v) else f"{v:+.1%}"),
                }), hide_index=True, width="stretch", height=min(38 * len(inus) + 40, 320))
            exp = experimento_volumen()
            n_empresas = sum(v[1] == "Acción" for v in ACTIVOS.values())
            if exp is not None:
                fila = exp[(exp["señal"] == "Volumen inusual en un día de bajada") & (exp["plazo"] == "3 meses")]
                if not fila.empty:
                    f0 = fila.iloc[0]
                    nota(f"<b>¿Anticipa subidas el volumen inusual?</b> Probado en {n_empresas} empresas desde 1990: ninguna "
                         f"regla de volumen supera al azar de forma fiable. La que más se acerca: tras un día de "
                         f"volumen inusual con caída, la acción estaba más arriba 3 meses después el "
                         f"{f0['acierto_comp']:.0%} de las veces (un día cualquiera: {f0['base_comp']:.0%}). "
                         "Diferencia pequeña y no concluyente. Detalle en «Experimentos».")

        with t_noticias:
            try:
                noticias = ficha_noticias(ticker, date.today())
            except Exception:
                noticias = pd.DataFrame()
            if noticias.empty:
                st.write("Yahoo Finance no ha devuelto noticias recientes.")
            else:
                st.dataframe(pd.DataFrame({
                    "Fecha": [f"{i.tz_convert('Europe/Madrid'):%d/%m %H:%M}" for i in noticias.index],
                    "Titular": noticias["titular"],
                    "Fuente": noticias["fuente"],
                    "Precio en su sesión": noticias["reaccion"].map(
                        lambda v: "pendiente" if pd.isna(v) else f"{v:+.1%}"),
                    "Abrir": noticias["enlace"],
                }), hide_index=True, width="stretch",
                    column_config={"Abrir": st.column_config.LinkColumn("Abrir", display_text="Leer →")})
                st.caption("Titulares de terceros, normalmente en inglés. Que el precio se mueva el día de una "
                           "noticia no significa que la noticia sea la causa.")

# --- Para saber más ---------------------------------------------------------------------

with seccion("saber_mas", "suave"):
    st.markdown("<div class='titulo-bloque'>Para saber más</div>", unsafe_allow_html=True)
    with st.expander("¿Cómo se lee esto?", icon=":material/help:"):
        st.markdown(
            f"""
- **La probabilidad** responde a: *si estuviéramos 100 veces en una situación como la de hoy, ¿en cuántas el precio
  acabaría más arriba dentro de {plazo}?* No dice cuánto subirá.
- **El rango** es la horquilla donde probablemente estará el precio: con un {obj} % de seguridad, el precio real
  debería quedar dentro unas {obj // 10} de cada 10 veces.
- **La fiabilidad** sale de comprobarlo con años pasados que no se usaron para calcularlo.
- A plazos cortos (días) la dirección es casi una moneda al aire; a 6 meses o 1 año se acierta bastante más.
"""
        )
    with st.expander("¿Cómo se calcula?", icon=":material/calculate:"):
        st.markdown(
            f"""
**Dirección.** Se compara la frecuencia histórica de subidas del propio activo con dos IA entrenadas con
{len(INDICES)} índices mundiales desde 1990 (una de ellas usa además el miedo del mercado y los tipos de interés).
Con el 70 % más antiguo de la historia se elige la que mejor funcionó; con el 30 % más reciente se mide su acierto.

**Rango.** Para cada día del pasado se calculó el rango que cada método (EWMA, GARCH, volatilidad histórica,
histórico de movimientos o IA) habría dado ese día, sin usar datos posteriores. Se elige igual: 70 % para elegir,
30 % para medir.
"""
        )
    if historiales:
        with st.expander("Situaciones especiales: qué pasó otras veces", icon=":material/history:"):
            for regla, hist in historiales.items():
                ep = sit.resumen(hist)
                st.markdown(f"**{sit.REGLAS[regla].nombre}** — {sit.REGLAS[regla].descripcion} "
                            + (f"Después, {nombre_activo} estaba más arriba a {plazo} vista en **{ep['episodios_arriba']} "
                               f"de {ep['n']} episodios** (antes de 2010: {ep['arriba_antes']} de {ep['n_antes']}; desde "
                               f"2010: {ep['arriba_despues']} de {ep['n_despues']})." if ep["n"] else "Sin episodios completos."))
                if not hist.empty:
                    vista = pd.DataFrame({
                        "Empieza": hist["inicio"].dt.strftime("%d/%m/%Y"),
                        "Termina": hist["fin"].dt.strftime("%d/%m/%Y"),
                        ("VIX máximo" if regla == "panico" else "Peor caída"):
                            hist["vix_max"].map(lambda v: numero(v, 0)) if regla == "panico"
                            else hist["caida_max"].map("{:.0%}".format),
                        f"Días que acabaron más arriba a {plazo}": hist["prop_mas_arriba"].map(
                            lambda v: "pendiente" if pd.isna(v) else f"{v:.0%}"),
                        f"Desde el 1er día, a {plazo}": hist["retorno_desde_inicio"].map(
                            lambda v: "pendiente" if pd.isna(v) else f"{v:+.0%}"),
                    })
                    st.dataframe(vista.iloc[::-1], hide_index=True, width="stretch")
            if estado_hoy:
                donde = ("está en máximos del año" if estado_hoy["caida"] > -0.01
                         else f"está a {estado_hoy['caida']:.0%} de su máximo del año")
                st.caption(f"Hoy {nombre_activo} {donde}; el índice del miedo (VIX) está en "
                           f"{numero(estado_hoy['vix'], 1)} (pánico = 30 o más).")
    with st.expander(f"Detalle de los métodos de rango para {plazo}", icon=":material/table:"):
        if p.comparativa.empty:
            st.write("Sin datos suficientes para comparar métodos.")
        else:
            comp = p.comparativa.copy()
            comp.index = [DESCRIPCIONES.get(m, m) for m in comp.index]
            st.dataframe(pd.DataFrame({
                "Aciertos (selección)": comp["cobertura selección"].map("{:.1%}".format),
                "Error (selección)": comp["error selección"].map("{:.2%}".format),
                "Calibrado": comp["calibrado en selección"].map({True: "Sí", False: "No"}),
                "Aciertos (validación)": comp["cobertura validación"].map("{:.1%}".format),
                "Ancho medio": comp["ancho medio"].map("{:.1%}".format),
                "Elegido": comp["elegido"].map({True: "✔", False: ""}),
            }, index=comp.index), width="stretch")
            st.caption(f"Método elegido para el rango: {NOMBRES_CORTOS.get(p.metodo, p.metodo)}.")
    if d is not None and not d.calibracion.empty:
        with st.expander("¿Se puede confiar en la probabilidad? (comprobación)", icon=":material/fact_check:"):
            cal = d.calibracion
            st.dataframe(pd.DataFrame({
                "Cuando dijo": [str(i).replace("(", "").replace("]", "").replace(", ", " – ") for i in cal.index],
                "Probabilidad media dicha": cal["dijo"].map("{:.0%}".format),
                "Subió de verdad": cal["subio"].map("{:.0%}".format),
                "Veces": cal["veces"],
            }), hide_index=True, width="stretch")
            st.caption(f"Comprobado entre {d.desde:%Y} y {d.hasta:%Y}. Si las dos columnas se parecen, la probabilidad "
                       "es de fiar.")
