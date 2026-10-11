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


def _recargar_si_hay_codigo_nuevo() -> None:
    """Streamlit Cloud baja los cambios de GitHub pero sigue usando los módulos de ``src/`` ya cargados en
    memoria. Si algún archivo es más nuevo que la última carga (o es la primera vez en este proceso), se
    descartan esos módulos y las cachés para que todo se importe de nuevo con el código actual."""
    archivos = list((RAIZ / "src" / "analizador").glob("*.py")) + list((RAIZ / "vistas").glob("*.py"))
    ultimo = max(f.stat().st_mtime for f in archivos)
    previo = getattr(sys, "_mattionni_codigo", None)
    if previo is None or ultimo > previo:
        viejos = [m for m in sys.modules if m in ("analizador", "vistas") or m.startswith(("analizador.", "vistas."))]
        for nombre in viejos:
            del sys.modules[nombre]
        if viejos:
            st.cache_data.clear()
            st.cache_resource.clear()
    sys._mattionni_codigo = ultimo


_recargar_si_hay_codigo_nuevo()

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
    st.Page("vistas/estrategias.py", title="Estrategias que aprende", icon=":material/auto_graph:",
            url_path="estrategias"),
    st.Page("vistas/laboratorio.py", title="Experimentos", icon=":material/science:", url_path="laboratorio"),
    st.Page("vistas/analisis.py", title="Avanzado", icon=":material/tune:", url_path="analisis"),
])
paginas.run()

st.divider()
st.caption(f"Mattionni Analytics · {AVISO_LEGAL} Datos de Yahoo Finance: pueden estar retrasados o incompletos.")
