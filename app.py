"""Interfaz web de Mattionni Analytics (proyecto Analizador-Mattionni).

Ejecutar con:  streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Permite ejecutar la app sin instalar el paquete (p. ej. en Streamlit Community Cloud).
RAIZ = Path(__file__).parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(RAIZ))

from analizador import AVISO_LEGAL  # noqa: E402
from vistas.comun import estilo_global  # noqa: E402

st.set_page_config(page_title="Mattionni Analytics", page_icon=":material/insights:", layout="wide")
estilo_global()

paginas = st.navigation([
    st.Page("vistas/inicio.py", title="Inicio", icon=":material/home:", default=True),
    st.Page("vistas/proyeccion.py", title="¿Subirá o bajará?", icon=":material/query_stats:", url_path="proyeccion"),
    st.Page("vistas/analizar.py", title="Analizar una empresa", icon=":material/monitoring:", url_path="analizar"),
    st.Page("vistas/mi_lista.py", title="Mi lista", icon=":material/star:", url_path="mi_lista"),
    st.Page("vistas/comparar.py", title="Comparar", icon=":material/compare_arrows:", url_path="comparar"),
    st.Page("vistas/calendario.py", title="Calendario de resultados", icon=":material/event:", url_path="calendario"),
    st.Page("vistas/empresas.py", title="Todas las empresas", icon=":material/apartment:", url_path="empresas"),
    st.Page("vistas/indices.py", title="Bolsas del mundo", icon=":material/public:", url_path="indices"),
    st.Page("vistas/la_ia.py", title="¿Acierta la IA?", icon=":material/psychology:", url_path="la_ia"),
    st.Page("vistas/simulador.py", title="Simulador de estrategias", icon=":material/candlestick_chart:",
            url_path="simulador"),
    st.Page("vistas/laboratorio.py", title="Experimentos", icon=":material/science:", url_path="laboratorio"),
    st.Page("vistas/analisis.py", title="Avanzado", icon=":material/tune:", url_path="analisis"),
])
paginas.run()

st.divider()
st.caption(f"Mattionni Analytics · {AVISO_LEGAL} Datos de Yahoo Finance: pueden estar retrasados o incompletos.")
