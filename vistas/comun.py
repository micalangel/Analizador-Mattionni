"""Elementos compartidos por las páginas de la web: datos en caché, estilo y aviso."""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analizador import backtest as bt
from analizador import ia
from analizador import noticias as nt
from analizador import resultados as rs
from analizador.calibration import informe_calibracion
from analizador.data import descargar_varios
from analizador.forecast import METODOS
from analizador.fundamentals import obtener_fundamentales

# Estilo «moderno y limpio»: blanco, texto casi negro y un único azul de acento.
AZUL, NEGRO, GRIS_TEXTO = "#2563EB", "#111111", "#6B6B6B"
# Series de los gráficos (orden fijo): azul, negro y después tonos sobrios bien distinguibles.
COLORES = [AZUL, NEGRO, "#0E9F6E", "#9333EA", "#F59E0B", "#0891B2", "#DB2777", "#DC2626"]
GRIS = "#9CA3AF"
CSS_ESTILO = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, .stMarkdown p, .stMarkdown li, label, input, textarea, button p, td, th,
h1, h2, h3, h4, h1 *, h2 *, h3 *, h4 * { font-family: 'Inter', system-ui, sans-serif; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
.block-container { padding-top: 2.5rem; max-width: 1100px; }
h1 { font-size: 2.1rem !important; font-weight: 600 !important; letter-spacing: -0.02em; color: #111111; }
h2, h3 { font-weight: 600 !important; letter-spacing: -0.01em; }
h4 { font-weight: 500 !important; }
[data-testid="stMetricValue"] { font-weight: 500; letter-spacing: -0.02em; }
[data-testid="stMetricLabel"] p { color: #6B6B6B; font-size: .8rem !important; }
[data-testid="stCaptionContainer"] { color: #6B6B6B; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
[data-testid="stSidebarNav"] a span { font-size: .92rem; }
.etiqueta { font-size: .8rem; color: #6B6B6B; margin-bottom: .1rem; }
.cifra { font-size: 3.6rem; font-weight: 500; letter-spacing: -0.03em; line-height: 1.05; color: #111111; }
.cifra-media { font-size: 1.9rem; font-weight: 500; letter-spacing: -0.02em; line-height: 1.15; color: #111111; }
.suave { font-size: .9rem; color: #6B6B6B; }
.subtitulo { font-size: 1.15rem; color: #4B4B4B; line-height: 1.55; max-width: 46rem; }
.titulo-bloque { font-size: 1.05rem; font-weight: 600; color: #111111; margin: 0 0 .2rem 0; }
/* Apartados con fondo propio (como las bandas de la web de HM) */
div[class*="st-key-sec_"] { border-radius: 16px; padding: 1.3rem 1.5rem 1.4rem; margin: .35rem 0 .9rem; }
div[class*="st-key-sec_suave_"]  { background: #F7F6F2; border: 1px solid #E9E7E1; }
div[class*="st-key-sec_azul_"]   { background: #F0F5FF; border: 1px solid #DCE6FB; }
div[class*="st-key-sec_verde_"]  { background: #EFF8F3; border: 1px solid #D5EDE1; }
div[class*="st-key-sec_ambar_"]  { background: #FFF8EB; border: 1px solid #F3E2BF; }
div[class*="st-key-sec_blanco_"] { background: #FFFFFF; border: 1px solid #ECECEF; }
/* Tarjetas blancas con franja de color arriba */
div[class*="st-key-tar_"] { background: #FFFFFF; border-radius: 16px; padding: 1.3rem 1.5rem 1.4rem;
    margin: .35rem 0 .9rem; border: 1px solid #ECECEF;
    box-shadow: 0 1px 2px rgba(0,0,0,.04), 0 10px 28px rgba(0,0,0,.06); }
div[class*="st-key-tar_azul_"]  { border-top: 5px solid #2563EB; }
div[class*="st-key-tar_verde_"] { border-top: 5px solid #0E9F6E; }
div[class*="st-key-tar_ambar_"] { border-top: 5px solid #F59E0B; }
div[class*="st-key-tar_negro_"] { border-top: 5px solid #111111; }
/* Notas tipo «¿Sabías que…?» */
.nota { background: #F0F5FF; border-left: 4px solid #2563EB; border-radius: 0 12px 12px 0; padding: .8rem 1rem;
        color: #1F2937; font-size: .95rem; line-height: 1.5; }
.nota.ambar { background: #FFF8EB; border-left-color: #F59E0B; }
.nota.verde { background: #EFF8F3; border-left-color: #0E9F6E; }
/* Portada y bloques en HTML */
.hero { background: #F0F5FF; border: 1px solid #DCE6FB; border-radius: 20px; padding: 2.4rem 2.4rem 2.2rem;
        margin-bottom: 1.2rem; }
.hero h1 { font-size: 2.6rem !important; margin: 0 0 .5rem 0 !important; padding: 0 !important; }
.hero .subtitulo { font-size: 1.2rem; margin-bottom: 1.4rem; }
.boton { display: inline-block; background: #2563EB; color: #FFFFFF !important; text-decoration: none !important;
         font-weight: 500; padding: .75rem 1.4rem; border-radius: 10px; }
.boton:hover { background: #1D4ED8; }
.banda { border-radius: 20px; padding: 1.8rem 2rem 2rem; margin: 0 0 1.2rem 0; }
.banda.blanca { background: #FFFFFF; border: 1px solid #ECECEF; }
.banda.gris   { background: #F7F6F2; border: 1px solid #E9E7E1; }
.banda.azul   { background: #F0F5FF; border: 1px solid #DCE6FB; }
.banda.ambar  { background: #FFF8EB; border: 1px solid #F3E2BF; }
.sobretitulo { font-size: .75rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
               color: #2563EB; margin-bottom: .3rem; }
.banda h2 { font-size: 1.55rem !important; margin: 0 0 1.1rem 0 !important; padding: 0 !important; }
.rejilla { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }
.caja { background: #FFFFFF; border: 1px solid #ECECEF; border-radius: 14px; padding: 1.2rem 1.3rem;
        height: 100%; box-sizing: border-box; }
a.caja { display: block; text-decoration: none !important; color: inherit !important; }
a.caja:hover { border-color: #2563EB; }
.caja h3 { font-size: 1.1rem !important; margin: .6rem 0 .35rem 0 !important; padding: 0 !important; }
.caja p { font-size: .95rem; color: #4B4B4B; margin: 0; line-height: 1.5; }
.circulo { display: inline-flex; align-items: center; justify-content: center; width: 2rem; height: 2rem;
           border-radius: 50%; font-weight: 600; font-size: .95rem; }
.circulo.c1 { background: #DBEAFE; color: #1D4ED8; }
.circulo.c2 { background: #D1FAE5; color: #047857; }
.circulo.c3 { background: #FEF3C7; color: #B45309; }
.cita { background: #FFFFFF; border-left: 4px solid #2563EB; border-radius: 0 12px 12px 0; padding: 1rem 1.2rem;
        font-size: 1.1rem; line-height: 1.55; margin-bottom: 1rem; }
.banda ul { margin: 0; padding-left: 1.2rem; }
.banda li { margin-bottom: .45rem; line-height: 1.55; }
</style>
"""


def seccion(nombre: str, tono: str = "suave"):
    """Apartado con fondo propio: «suave» (gris cálido), «azul», «verde», «ambar» o «blanco»."""
    return st.container(key=f"sec_{tono}_{nombre}")


def tarjeta(nombre: str, color: str = "azul"):
    """Tarjeta blanca con franja de color arriba: «azul», «verde», «ambar» o «negro»."""
    return st.container(key=f"tar_{color}_{nombre}")


def nota(texto: str, tono: str = "") -> None:
    """Recuadro de nota con línea de color a la izquierda (texto con HTML sencillo)."""
    st.markdown(f"<div class='nota {tono}'>{texto}</div>", unsafe_allow_html=True)


def estilo_global() -> None:
    """Tipografía y detalles del estilo. Se aplica una vez por ejecución desde app.py."""
    st.markdown(CSS_ESTILO, unsafe_allow_html=True)

# --- Caché -------------------------------------------------------------------------------

@st.cache_data(show_spinner="Descargando datos…", ttl=60 * 60 * 6)
def cargar(tickers: tuple[str, ...], inicio: date, fin: date) -> dict[str, pd.DataFrame]:
    return descargar_varios(list(tickers), inicio, fin)


@st.cache_data(show_spinner="Calculando backtest…")
def calcular_backtest(datos: pd.DataFrame, estrategia: str, coste: float, desliz: float,
                      nivel: float, historia: int):
    res = bt.comparar_con_referencias(datos, bt.ESTRATEGIAS[estrategia](), nombre=estrategia,
                                      nivel_ingenuo=nivel, coste=coste, deslizamiento=desliz,
                                      historia_minima=historia)
    return ({k: r.capital for k, r in res.items()},
            {k: r.posiciones for k, r in res.items()},
            bt.tabla_metricas(res))


@st.cache_data(show_spinner="Evaluando la calibración…")
def calcular_calibracion(cierre: pd.Series, horizonte: int, confianza: float, ventana: int,
                         tabla_ia: pd.DataFrame | None = None):
    extra = {ia.NOMBRE_METODO: (ia.DESCRIPCION, tabla_ia)} if tabla_ia is not None else None
    return informe_calibracion(cierre, horizonte, confianza, list(METODOS), ventana, extra=extra)


@st.cache_data(show_spinner="Entrenando la IA paso a paso con el histórico (puede tardar ~15 s)…")
def entrenar_ia(ticker: str, datos: pd.DataFrame, horizonte: int, confianza: float,
                publicaciones: pd.DataFrame, mercado: pd.Series | None):
    tabla, modelos = ia.serie_intervalos_ia(datos, horizonte, confianza, publicaciones, mercado)
    X = ia.construir_caracteristicas(datos, publicaciones, mercado)
    final = ia.entrenar(X, ia.objetivo_escalado(datos["Close"], horizonte), horizonte, confianza)
    if final is not None:
        try:
            ia.guardar_modelo(final, ticker)
        except OSError:
            pass  # p. ej. sistema de archivos de solo lectura
    historial = pd.DataFrame({
        "Entrenado hasta": [m.entrenado_hasta for m in modelos],
        "Muestras de ajuste": [m.n_muestras for m in modelos],
        "Muestras de calibración": [m.n_calibracion for m in modelos],
        "Ajuste conformal (z)": [m.ajuste_conformal for m in modelos],
    })
    return tabla, final, historial


@st.cache_data(show_spinner="Consultando fundamentales…", ttl=60 * 60 * 24)
def cargar_fundamentales(tickers: tuple[str, ...]) -> pd.DataFrame:
    return obtener_fundamentales(list(tickers))


@st.cache_data(show_spinner="Consultando resultados financieros…", ttl=60 * 60 * 6)
def cargar_resultados(ticker: str):
    return rs.obtener_resultados(ticker)


@st.cache_data(show_spinner="Buscando noticias…", ttl=60 * 30)
def cargar_noticias(ticker: str, cantidad: int) -> pd.DataFrame:
    return nt.obtener_noticias(ticker, cantidad)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def cargar_macro(hoy: date) -> pd.DataFrame | None:
    """VIX y tipos de interés de EE. UU. desde 1990 (para la IA y las situaciones especiales)."""
    from analizador.ia import TICKERS_MACRO, datos_macro

    return datos_macro(descargar_varios(list(TICKERS_MACRO.values()), date(1990, 1, 1), hoy))


def estilo(fig: go.Figure, alto: int = 420) -> go.Figure:
    con_titulo = bool(fig.layout.title.text)
    fig.update_layout(height=alto, margin=dict(l=10, r=10, t=90 if con_titulo else 40, b=10),
                      hovermode="x unified", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=NEGRO),
                      title_font=dict(family="Inter, sans-serif", size=16, color=NEGRO),
                      title=dict(y=0.98, yanchor="top", x=0, xanchor="left"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, gridcolor="rgba(17,17,17,0.06)")
    return fig


def numero(x: float, decimales: int = 2) -> str:
    """Formato español: 11.234,56."""
    if x is None or pd.isna(x):
        return "—"
    return f"{x:,.{decimales}f}".replace(",", "X").replace(".", ",").replace("X", ".")


@st.cache_data(ttl=60 * 10, show_spinner=False)
def cargar_estudios() -> pd.DataFrame:
    """Estudios precalculados de empresas, activos e índices, con las mismas columnas."""
    from analizador.entrenamiento import DIRECTORIO

    partes = []
    ruta = DIRECTORIO / "estudio_activos.csv"
    if ruta.exists():
        partes.append(pd.read_csv(ruta).rename(columns={"estado_rango": "estado"}))
    ruta = DIRECTORIO / "estudio_indices.csv"
    if ruta.exists():
        ind = pd.read_csv(ruta)
        ind["nombre"] = ind["índice"] + " (" + ind["país"] + ")"
        ind["tipo"], ind["sector"] = "Índice", "—"
        partes.append(ind)
    if not partes:
        return pd.DataFrame()
    tabla = pd.concat(partes, ignore_index=True)
    ruta = DIRECTORIO / "estudio_salud.csv"
    if ruta.exists():
        tabla = tabla.merge(pd.read_csv(ruta), on="ticker", how="left")
    return tabla


@st.cache_data(ttl=60 * 10, show_spinner=False)
def per_referencia() -> tuple[dict, float | None]:
    """Mediana del PER por sector y de todas las empresas de la web (del estudio de salud)."""
    from analizador.catalogo import ACTIVOS
    from analizador.entrenamiento import DIRECTORIO
    from analizador.salud import per_de_referencia

    ruta = DIRECTORIO / "estudio_salud.csv"
    if not ruta.exists():
        return {}, None
    return per_de_referencia(pd.read_csv(ruta), {t: v[3] for t, v in ACTIVOS.items()})


def lista_cargar() -> list[str]:
    """«Mi lista»: en el ordenador propio se guarda en un archivo; en la nube, solo en la sesión del visitante."""
    from analizador import mi_lista as ml

    if ml.en_la_nube():
        return list(st.session_state.get("mi_lista", []))
    return ml.cargar()


def lista_guardar(tickers: list[str]) -> None:
    from analizador import mi_lista as ml

    if ml.en_la_nube():
        st.session_state["mi_lista"] = list(dict.fromkeys(tickers))
    else:
        ml.guardar(tickers)
