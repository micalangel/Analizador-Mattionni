"""Página de inicio: qué es la web y por dónde empezar."""

from __future__ import annotations

import streamlit as st


def html(texto: str) -> None:
    st.markdown(texto, unsafe_allow_html=True)


html("""
<div class="hero">
  <h1>Mattionni Analytics</h1>
  <div class="subtitulo">Probabilidades y rangos de precio para la bolsa, las empresas, las divisas y el oro,
  con cuánto se ha acertado otras veces.</div>
  <a class="boton" href="/proyeccion" target="_self">Empezar: ¿subirá o bajará? →</a>
</div>
""")

html("""
<div class="banda blanca">
  <div class="sobretitulo">Cómo funciona</div>
  <h2>Tres pasos</h2>
  <div class="rejilla">
    <div class="caja"><span class="circulo c1">1</span><h3>Eliges algo</h3>
      <p>Una empresa (Apple, Santander, Inditex…), un índice (IBEX 35, S&amp;P 500…), el oro, el euro o el bitcoin.</p></div>
    <div class="caja"><span class="circulo c2">2</span><h3>Eliges un plazo</h3>
      <p>5 días, 10 días, 1 mes, 6 meses o 1 año.</p></div>
    <div class="caja"><span class="circulo c3">3</span><h3>Ves el resultado</h3>
      <p>La probabilidad de que esté más arriba que hoy, entre qué precios se moverá y cuánto acertó en el pasado.</p></div>
  </div>
</div>
""")

html("""
<div class="banda azul">
  <div class="sobretitulo">Ejemplo</div>
  <h2>Cómo se lee un resultado</h2>
  <div class="cita">Dentro de 1 año: <b>72 %</b> de probabilidades de que Apple esté más arriba que hoy.
  Rango probable: <b>234 – 485</b>.</div>
  <ul>
    <li><b>72 %</b>: de cada 100 situaciones parecidas en el pasado, unas 72 acabaron más arriba al año y 28 más abajo.
        Es más probable que suba, pero no es seguro.</li>
    <li><b>234 – 485</b>: la horquilla donde suele acabar el precio. Unas 8 de cada 10 veces queda dentro.</li>
    <li>Junto al resultado verás <b>cuánto acertó</b> en años que no se usaron para calcularlo. Si acertó poco, desconfía.</li>
  </ul>
</div>
""")

secciones = [
    ("/proyeccion", "c1", "¿Subirá o bajará?", "Una empresa o índice, a un plazo. La página principal."),
    ("/mi_lista", "c2", "Mi lista", "Tus favoritas, con sus alertas de hoy."),
    ("/comparar", "c3", "Comparar", "De 2 a 4 activos lado a lado."),
    ("/calendario", "c1", "Calendario de resultados", "Cuándo presentan cuentas y cuánto suelen moverse."),
    ("/empresas", "c2", "Todas las empresas", "Más de 360, con filtros por país, sector y salud."),
    ("/indices", "c3", "Bolsas del mundo", "Los índices de 30 países comparados."),
    ("/simulador", "c3", "Simulador de estrategias", "Dónde habría comprado y vendido una regla, con su resultado."),
    ("/la_ia", "c1", "¿Acierta la IA?", "Su diario: predicciones apuntadas hoy y comprobadas cuando vencen."),
    ("/laboratorio", "c2", "Experimentos", "Qué reglas famosas funcionan y cuáles son un mito."),
    ("/analisis", "c3", "Avanzado", "Gráficos técnicos, resultados de empresas, noticias y simulaciones."),
]
cajas = "".join(
    f'<a class="caja" href="{url}" target="_self"><span class="circulo {c}">→</span><h3>{titulo}</h3><p>{texto}</p></a>'
    for url, c, titulo, texto in secciones)
html(f"""
<div class="banda gris">
  <div class="sobretitulo">Secciones</div>
  <h2>Qué hay en la web</h2>
  <div class="rejilla">{cajas}</div>
</div>
""")

html("""
<div class="banda ambar">
  <div class="sobretitulo">Importante</div>
  <h2>Lo que conviene saber</h2>
  <ul>
    <li><b>Nadie sabe el futuro.</b> Son probabilidades, no certezas: incluso con un 80 %, 2 de cada 10 veces sale al revés.</li>
    <li><b>A corto plazo es casi lanzar una moneda.</b> A 5 o 10 días se acierta poco más del 55 %; a 1 año, alrededor del 70 %.</li>
    <li><b>No es una recomendación</b> de comprar ni vender: sirve para entender riesgos y probabilidades.</li>
  </ul>
</div>
""")
