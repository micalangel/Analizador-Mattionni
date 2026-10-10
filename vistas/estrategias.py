"""Página «Estrategias que aprende»: decenas de estrategias de cartera, un aprendiz que elige cada año
cuál usar y la comprobación honesta de cómo le fue en los años que no había visto."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analizador import estrategias as es
from analizador.entrenamiento import DIRECTORIO
from vistas.comun import COLORES, estilo, nota, seccion, tarjeta

VERDE, ROJO = "#0E9F6E", "#DC2626"
REFERENCIA = "Comprar todo (igual peso)"
CORTOS = {"bolsas": "Bolsas del mundo", "empresas": "Empresas", "otros": "Materias primas, divisas y cripto",
          "todo": "Todo junto", "analistas": "EE. UU. con analistas"}


@st.cache_data(ttl=60 * 10, show_spinner=False)
def cargar(universo: str):
    return es.cargar(universo, DIRECTORIO)


def pct(x, decimales=1, signo=False) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:{'+' if signo else ''}.{decimales}f} %".replace(".", ",")


def euros(x) -> str:
    return f"{x:,.0f} €".replace(",", ".")


st.title("Estrategias que aprende")
st.markdown("<div class='subtitulo'>El laboratorio prueba decenas de estrategias que eligen cada mes entre muchos "
            "activos. Un «aprendiz» decide cada 1 de enero cuál usar, solo con lo que se sabía hasta ese día, y aquí "
            "ves cómo le fue de verdad en los años siguientes, que no había visto. Todo simulado.</div>",
            unsafe_allow_html=True)
st.write("")

with seccion("elegir", "suave"):
    st.markdown("<div class='titulo-bloque'>¿Entre qué activos elige?</div>", unsafe_allow_html=True)
    universo = st.segmented_control("Universo", list(es.UNIVERSOS), format_func=lambda k: CORTOS.get(k, k),
                                    default="bolsas", key="universo_estr", label_visibility="collapsed") or "bolsas"

e = cargar(universo)
if e is None:
    st.info("Todavía no se ha calculado este laboratorio. Ejecuta `python -m analizador.estrategias`.")
    st.stop()
r = e.resumen
desde = pd.Timestamp(r["desde_aprendiz"])
curvas = e.curvas.loc[desde:]
anios = r.get("aprendiz_anios", len(curvas) / 12)

CONCLUSIONES = {
    "bolsas": "En las bolsas del mundo destaca el <b>momentum con filtro de mercado</b>: invertir en los índices que "
              "más han subido en el último año, pero solo cuando el conjunto del mercado está en tendencia alcista. "
              "Ha ganado más que comprar todo con caídas mucho menores y aguanta las pruebas de resistencia.",
    "empresas": "Con empresas gana el <b>momentum</b> (lo que más ha subido en 6–12 meses), sobre todo con filtro de "
                "mercado. Las cifras están infladas porque la lista solo tiene empresas que sobrevivieron.",
    "otros": "En materias primas, divisas y cripto <b>ninguna estrategia gana de forma clara a repartir entre todos</b>: "
             "el aprendiz se equivoca a menudo porque lo que funciona cambia mucho de una década a otra.",
    "todo": "Mezclar bolsas, materias primas, divisas y cripto ayuda: el <b>momentum con filtro de mercado</b> elige "
            "entre más tipos de activos y reduce las caídas frente a comprar todo.",
    "analistas": "Novedad: aquí se combinan precios con el historial de <b>analistas</b> y <b>resultados</b>. Conclusión "
                 "honesta: las revisiones de los analistas no ganan a comprar todo, la sorpresa en resultados apenas "
                 "aporta, y comprar donde los analistas ven más potencial ha sido <b>lo peor</b>. Lo que mejor "
                 "funciona sigue siendo el momentum (inflado por el sesgo de supervivencia).",
}
nota(f"<b>Qué ha descubierto el laboratorio:</b> {CONCLUSIONES.get(universo, '')}", "verde")

if universo in ("empresas", "analistas"):
    nota("<b>Ojo con las empresas:</b> la lista es la de hoy, así que solo incluye las que sobrevivieron y "
         "crecieron. Eso infla todas las cifras (también la de comprar todo). Lo útil es comparar las estrategias "
         "entre sí, no las cifras absolutas.", "ambar")

# --- Resultado principal -------------------------------------------------------------------------

st.markdown(f"<div class='titulo-bloque' style='margin-top:.6rem'>Si hubieras puesto 10.000 € el "
            f"{desde:%d/%m/%Y}</div>", unsafe_allow_html=True)
columnas = st.columns(3, gap="medium")
fichas = [("Aprendiz prudente", "aprendiz", "verde",
           "Elige la estrategia con mejor rentabilidad por unidad de riesgo."),
          ("Aprendiz ambicioso", "ambicioso", "azul", "Elige la estrategia que más ha ganado, aunque arriesgue más."),
          ("Comprar todo", "referencia", "negro", "Reparte a partes iguales entre todos y no toca nada.")]
for col, (titulo, clave, color, texto) in zip(columnas, fichas):
    rent = r.get(f"{clave}_rent_anual", np.nan)
    caida = r.get(f"{clave}_max_caida", np.nan)
    final = 10_000 * (1 + rent) ** anios if pd.notna(rent) else np.nan
    with col, tarjeta(f"res_{clave}", color):
        st.markdown(f"<div class='etiqueta'>{titulo}</div>"
                    f"<div class='cifra' style='font-size:2rem;white-space:nowrap'>{euros(final)}</div>"
                    f"<div style='font-weight:600'>{pct(rent)} al año</div>"
                    f"<div class='suave'>Peor caída: <b style='color:{ROJO}'>{pct(caida, 0)}</b></div>"
                    f"<p style='font-size:.85rem;margin-top:.4rem'>{texto}</p>", unsafe_allow_html=True)

capital = (1 + curvas[["Aprendiz", "Aprendiz ambicioso", REFERENCIA]].fillna(0)).cumprod() * 10_000
fig = go.Figure()
for nombre_serie, color, ancho in (("Aprendiz", VERDE, 2.5), ("Aprendiz ambicioso", COLORES[0], 2),
                                   (REFERENCIA, COLORES[1], 1.5)):
    etiqueta = {"Aprendiz": "Aprendiz prudente", REFERENCIA: "Comprar todo"}.get(nombre_serie, nombre_serie)
    fig.add_scatter(x=capital.index, y=capital[nombre_serie], name=etiqueta, line=dict(color=color, width=ancho),
                    hovertemplate="%{y:,.0f} €<extra>" + etiqueta + "</extra>")
fig.update_layout(title="Evolución de 10.000 € (escala logarítmica)", yaxis_type="log",
                  legend=dict(orientation="h", y=-0.15), hovermode="x unified")
st.plotly_chart(estilo(fig, 420), width="stretch")

# --- Qué ha aprendido ------------------------------------------------------------------------------

with seccion("aprendido", "suave"):
    st.markdown("<div class='titulo-bloque'>Qué ha ido eligiendo cada año</div>", unsafe_allow_html=True)
    el = e.elecciones
    prudente = el[el["criterio"] == "sharpe"].set_index("anio")
    ambicioso = el[el["criterio"] == "rentabilidad"].set_index("anio")
    ref_anual = (1 + curvas[REFERENCIA]).groupby(curvas.index.year).prod() - 1
    tabla_anios = pd.DataFrame({
        "Año": prudente.index.astype(str),
        "Prudente eligió": prudente["variante"].to_numpy(),
        "Prudente": prudente["rent_anio"].to_numpy() * 100,
        "Ambicioso eligió": ambicioso["variante"].reindex(prudente.index).to_numpy(),
        "Ambicioso": ambicioso["rent_anio"].reindex(prudente.index).to_numpy() * 100,
        "Comprar todo": ref_anual.reindex(prudente.index).to_numpy() * 100,
    }).iloc[::-1]

    def color_anio(fila):
        estilos = [""] * len(fila)
        for i, col in enumerate(fila.index):
            if col in ("Prudente", "Ambicioso") and pd.notna(fila[col]) and pd.notna(fila["Comprar todo"]):
                estilos[i] = f"color: {VERDE if fila[col] >= fila['Comprar todo'] else ROJO}; font-weight: 600"
        return estilos

    st.dataframe(tabla_anios.style.apply(color_anio, axis=1).format(
        {"Prudente": "{:+.1f} %", "Ambicioso": "{:+.1f} %", "Comprar todo": "{:+.1f} %"}, na_rep="—"),
        hide_index=True, width="stretch", height=380)
    batio = (tabla_anios["Prudente"] >= tabla_anios["Comprar todo"]).mean()
    st.caption(f"Verde: ese año ganó más que comprar todo; rojo: menos. El aprendiz prudente ganó a comprar todo "
               f"el {batio:.0%} de los años: incluso una buena estrategia pasa muchos años por detrás.")

# --- Todas las estrategias ---------------------------------------------------------------------

st.markdown("<div class='titulo-bloque' style='margin-top:1rem'>Todas las estrategias probadas</div>",
            unsafe_allow_html=True)
t = e.tabla
tabla = pd.DataFrame({
    "Estrategia": t["variante"],
    "Rentab. anual": t["rent_anual"] * 100,
    f"Desde {desde.year}": t["rent_anual_desde_aprendiz"] * 100,
    "Volatilidad": t["volatilidad"] * 100,
    "Peor caída": t["max_caida"] * 100,
    "Sharpe": t["sharpe"],
    "Mejor que el azar": t["vs_azar"] * 100,
    "Años elegida": t["veces_elegida"],
})
st.dataframe(tabla, hide_index=True, width="stretch", height=420, column_config={
    "Rentab. anual": st.column_config.NumberColumn(format="%+.1f %%", help=f"Desde {r['desde']} (toda la historia)."),
    f"Desde {desde.year}": st.column_config.NumberColumn(format="%+.1f %%", help="En el mismo periodo que el aprendiz."),
    "Volatilidad": st.column_config.NumberColumn(format="%.0f %%"),
    "Peor caída": st.column_config.NumberColumn(format="%.0f %%"),
    "Sharpe": st.column_config.NumberColumn(format="%.2f", help="Rentabilidad por unidad de riesgo: más alto es mejor."),
    "Mejor que el azar": st.column_config.ProgressColumn(
        format="%d %%", min_value=0, max_value=100,
        help="% de carteras con los mismos activos elegidos al azar a las que supera. Por encima del 95 % es difícil "
             "que sea suerte."),
})
with st.expander("Qué hace cada familia de estrategias"):
    for familia, texto in es.FAMILIAS.items():
        st.markdown(f"- {texto}")

# --- Pruebas de resistencia -------------------------------------------------------------------

rob = t.dropna(subset=["rob_normal"])
corte = pd.Timestamp(r.get("corte_mitades", "2012-12-31"))
if not rob.empty:
    with seccion("resistencia", "suave"):
        st.markdown("<div class='titulo-bloque'>Pruebas de resistencia de las mejores</div>", unsafe_allow_html=True)
        st.markdown("Una estrategia de verdad debe aguantar si las cosas salen peor que en la simulación.")
        tabla_rob = pd.DataFrame({
            "Estrategia": rob["variante"],
            "Normal": rob["rob_normal"] * 100,
            "Comisiones dobles": rob["rob_comision_doble"] * 100,
            "Operando un mes tarde": rob["rob_un_mes_tarde"] * 100,
            f"{desde.year}–{corte.year}": rob["rob_primera_mitad"] * 100,
            f"{corte.year + 1}–hoy": rob["rob_segunda_mitad"] * 100,
        })
        formato = {c: "{:+.1f} %" for c in tabla_rob.columns if c != "Estrategia"}
        ref = tabla_rob[tabla_rob["Estrategia"] == REFERENCIA].iloc[0]

        def color_rob(fila):
            return [f"color: {VERDE if fila[c] > ref[c] else ROJO}" if c in formato and fila["Estrategia"] != REFERENCIA
                    else "" for c in fila.index]

        st.dataframe(tabla_rob.style.apply(color_rob, axis=1).format(formato), hide_index=True, width="stretch")
        st.caption(f"Rentabilidad anual desde {desde.year}. Verde: supera a comprar todo en esa misma prueba.")

# --- Cartera de hoy ------------------------------------------------------------------------------

st.markdown("<div class='titulo-bloque' style='margin-top:1rem'>Qué tendría hoy la estrategia simulada</div>",
            unsafe_allow_html=True)
with tarjeta("hoy", "azul"):
    st.markdown(f"Con todos los datos hasta hoy, el aprendiz prudente usaría <b>{r['elegida_hoy']}</b>. Su cartera "
                f"simulada con los últimos precios (se fija con el cierre de fin de mes):", unsafe_allow_html=True)
    actual = e.actual.copy()
    if actual.empty:
        st.write("Ahora mismo estaría **todo en liquidez**: la estrategia no ve ninguna tendencia favorable.")
    else:
        actual["Peso"] = actual["peso"] * 100
        st.dataframe(actual[["nombre", "Peso"]].rename(columns={"nombre": "Activo"}), hide_index=True,
                     width="stretch", column_config={"Peso": st.column_config.ProgressColumn(
                         format="%.1f %%", min_value=0, max_value=float(actual["Peso"].max()))})
        if r.get("liquidez_hoy", 0) > 0.005:
            st.caption(f"El {pct(r['liquidez_hoy'], 0)} restante estaría en liquidez.")
    st.caption("Se recalcula con los cierres de cada fin de mes. Es el resultado de una simulación educativa, no una "
               "recomendación: la estrategia puede pasar años por detrás del mercado.")

# --- Lo que hay que saber -----------------------------------------------------------------------

st.markdown("<div class='titulo-bloque' style='margin-top:1rem'>Lo que hay que saber</div>", unsafe_allow_html=True)
st.markdown(f"""
- **Cómo aprende:** cada 1 de enero mira cómo le fue a cada estrategia hasta ese día y se queda con la mejor para
  todo el año. El resultado de arriba se mide solo con años que no había visto al elegir.
- **Por qué funciona a veces el momentum con filtro:** lo que sube suele seguir subiendo unos meses, y el filtro
  saca el dinero cuando todo el mercado cae (2001, 2008, 2022), que es cuando el momentum sufre más.
- **Sesgo del investigador:** algunas familias (como el filtro de mercado) se conocen por estudios publicados
  después de 2000. Elegir qué probar sabiendo eso favorece un poco al pasado.
- **Índices sin dividendos:** muchos índices no incluyen dividendos, y la liquidez sí cobra intereses. Eso favorece
  algo a las estrategias que salen al mercado (unos 2 puntos al año cuando están fuera).
- **Lo que no aparece:** impuestos, cambio de divisa y el esfuerzo de seguir la estrategia durante años malos.
  Con comisiones de {es.COSTE:.1%} por cambio; en la prueba de resistencia, el doble.
""".replace("0.2%", "0,2 %"))
st.caption(f"Calculado con datos hasta el {pd.Timestamp(r['hasta']):%d/%m/%Y} · {int(r['activos'])} activos · "
           f"{int(r['variantes'])} estrategias.")
