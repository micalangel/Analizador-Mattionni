"""Página «Empresas»: todas las empresas, divisas, materias primas y criptomonedas de un vistazo."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from analizador.entrenamiento import DIRECTORIO
from analizador.proyeccion import ESTADOS
from analizador.salud import comparar_per
from vistas.comun import numero, per_referencia, seccion, tarjeta

TIPOS = {"Acción": "Empresas", "Divisa": "Divisas", "Materia prima": "Materias primas",
         "Criptomoneda": "Criptomonedas", "ETF": "Fondos (ETF)"}
FUENTES = {"ia": "IA", "ia_macro": "IA + miedo y tipos", "frecuencia": "historia"}


SALUD = {"bien": "Sólida", "regular": "Mixta", "mal": "Débil", "info": "Sin datos"}
ICONO_SALUD = {"bien": "🟢", "regular": "🟡", "mal": "🔴", "info": "⚪"}


@st.cache_data(ttl=60 * 10)
def cargar_salud() -> pd.DataFrame | None:
    ruta = DIRECTORIO / "estudio_salud.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


@st.cache_data(ttl=60 * 10)
def cargar_estudio() -> pd.DataFrame | None:
    ruta = DIRECTORIO / "estudio_activos.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


st.title("Todas las empresas")
st.markdown("<div class='subtitulo'>Empresas, divisas, materias primas y criptomonedas con su probabilidad de estar "
            "más arriba, hasta dónde podrían moverse y si hoy están en una caída fuerte.</div>", unsafe_allow_html=True)
st.write("")

estudio = cargar_estudio()
if estudio is None:
    st.info("Aún no hay estudio de activos. Se genera con `python -m analizador.estudio`.")
    st.stop()

# --- Filtros -----------------------------------------------------------------------------

with seccion("filtros", "suave"):
    st.markdown("<div class='titulo-bloque'>Filtrar</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.6, 1.4, 1.4])
    with c1:
        plazo = st.segmented_control("Plazo", ["1 mes", "6 meses", "1 año"], default="1 año", key="plazo_emp") or "1 año"
    with c2:
        tipos = [t for t in TIPOS if t in set(estudio["tipo"])]
        tipo = st.selectbox("Qué mostrar", tipos, format_func=TIPOS.get)
    with c3:
        texto = st.text_input("Buscar por nombre", placeholder="p. ej. banco, Tesla…")

    tabla = estudio[(estudio["plazo"] == plazo) & (estudio["tipo"] == tipo)].copy()
    salud = cargar_salud()
    vista_salud = False
    if tipo == "Acción" and salud is not None:
        tabla = tabla.merge(salud, on="ticker", how="left")
        tabla["salud"] = tabla["salud"].fillna("info")
        for col in ["per", "margen", "deuda_patrimonio", "crecimiento_ventas", "rentabilidad_fcf", "beneficio_anual"]:
            tabla[col] = pd.to_numeric(tabla[col], errors="coerce")
        medianas_per, _ = per_referencia()
        tabla["per_sector"] = tabla["sector"].map(medianas_per)
        tabla["per_vs_sector"] = [comparar_per(p, r)[0] for p, r in zip(tabla["per"], tabla["per_sector"])]
    if tipo == "Acción":
        if salud is not None:
            vista_salud = st.segmented_control(
                "Ver", ["Probabilidades", "Salud financiera"], default="Probabilidades", key="vista_emp",
                help="«Salud financiera» muestra PER, margen, deuda, crecimiento y beneficio de cada empresa."
            ) == "Salud financiera"
        c4, c5, c6 = st.columns([1.6, 1.6, 1.2])
        with c4:
            paises = st.multiselect("País", sorted(tabla["país"].unique()), placeholder="Todos")
        with c5:
            sectores = st.multiselect("Sector", sorted(tabla["sector"].unique()), placeholder="Todos")
        with c6:
            st.write("")
            solo_caida = st.toggle("Solo en caída fuerte", help="Un 20 % o más por debajo de su máximo del último año.")
        if paises:
            tabla = tabla[tabla["país"].isin(paises)]
        if sectores:
            tabla = tabla[tabla["sector"].isin(sectores)]
        if solo_caida:
            tabla = tabla[tabla["caida_fuerte"] == True]  # noqa: E712 - columna con posibles vacíos
        if salud is not None:
            c7, c8, c9 = st.columns([1.6, 1.6, 1.2])
            with c7:
                notas = st.multiselect("Salud financiera", ["bien", "regular", "mal", "info"], format_func=SALUD.get,
                                       placeholder="Todas")
            with c8:
                per_max = st.slider("PER máximo", 5, 100, 100, step=5,
                                    help="Deja fuera las empresas con un PER mayor. Las que no tienen PER (pierden "
                                         "dinero) se quedan fuera si mueves el control.")
            with c9:
                st.write("")
                solo_beneficio = st.toggle("Solo las que ganan dinero", help="Beneficio de los 4 últimos trimestres > 0.")
            per_rel = st.multiselect("PER frente a su sector", ["Por debajo de su sector", "En la media de su sector",
                                                                 "Por encima de su sector"], placeholder="Todos")
            if notas:
                tabla = tabla[tabla["salud"].isin(notas)]
            if per_rel:
                tabla = tabla[tabla["per_vs_sector"].isin(per_rel)]
            if per_max < 100:
                tabla = tabla[tabla["per"].notna() & (tabla["per"] <= per_max)]
            if solo_beneficio:
                tabla = tabla[tabla["beneficio_anual"] > 0]
    if texto:
        clave = texto.strip().lower()
        tabla = tabla[tabla["nombre"].str.lower().str.contains(clave, regex=False)
                      | tabla["sector"].str.lower().str.contains(clave, regex=False)
                      | tabla["ticker"].str.lower().str.contains(clave, regex=False)]

# --- Resumen -----------------------------------------------------------------------------

with seccion("resumen", "azul"):
    con_prob = tabla["prob_sube"].dropna()
    k = st.columns(4)
    k[0].metric("Mostrando", len(tabla))
    k[1].metric(f"Prob. media más arriba ({plazo})", f"{con_prob.mean():.0%}" if len(con_prob) else "—")
    if vista_salud:
        k[1].metric("Con cuentas sólidas", int((tabla["salud"] == "bien").sum()))
    k[2].metric("En caída fuerte", int((tabla["caida_fuerte"] == True).sum()))  # noqa: E712
    k[3].metric("Datos al", pd.to_datetime(estudio["fecha"]).max().strftime("%d/%m/%Y"))

# --- Las más probables -------------------------------------------------------------------

candidatas = tabla.dropna(subset=["prob_sube"]).copy()
candidatas["caida_posible"] = candidatas["inferior"] / candidatas["cierre"] - 1
candidatas["subida_posible"] = candidatas["superior"] / candidatas["cierre"] - 1
# Más probabilidad primero; a igualdad, la que menos podría caer.
top = candidatas.sort_values(["prob_sube", "caida_posible"], ascending=[False, False]).head(3)
if not top.empty:
    with tarjeta("destacadas", "verde"):
        st.markdown(f"<div class='titulo-bloque'>Más probable que estén más arriba dentro de {plazo}</div>"
                    "<div class='suave' style='margin-bottom:.8rem'>Con los filtros que has elegido.</div>",
                    unsafe_allow_html=True)
        fuentes = {"ia": "IA", "ia_macro": "IA + miedo y tipos", "frecuencia": "su historia"}
        cajas = []
        for pos, (_, f) in enumerate(top.iterrows(), 1):
            cajas.append(
                f"<a class='caja' href='/proyeccion?activo={f['ticker']}' target='_self'>"
                f"<div class='etiqueta'>{pos}.º · {f['sector'] if tipo == 'Acción' else TIPOS.get(tipo, '')}</div>"
                f"<h3>{f['nombre']}</h3>"
                f"<div class='cifra-media' style='color:#047857'>{f['prob_sube']:.0%}</div>"
                f"<p>probabilidad de estar más arriba</p>"
                f"<p style='margin-top:.5rem'>Podría caer hasta <b>{f['caida_posible']:+.0%}</b> o subir hasta "
                f"<b>{f['subida_posible']:+.0%}</b></p>"
                f"<p style='font-size:.8rem;margin-top:.4rem'>Según {fuentes.get(f['fuente_direccion'], '—')} · "
                f"ver ficha →</p></a>")
        st.markdown("<div class='rejilla'>" + "".join(cajas) + "</div>", unsafe_allow_html=True)
        diferencia = (top["prob_sube"].iloc[0] - top["prob_sube"].iloc[-1]) * 100
        if len(top) > 1 and diferencia < 5:
            st.caption("Entre estas tres hay menos de 5 puntos de diferencia: en la práctica tienen una probabilidad "
                       "parecida. Fíjate también en cuánto podrían caer.")

# --- Tabla -------------------------------------------------------------------------------

with seccion("tabla", "blanco"):
    tabla = tabla.sort_values(["prob_sube", "nombre"], ascending=[False, True])  # más probables arriba
    var_min = tabla["inferior"] / tabla["cierre"] - 1
    var_max = tabla["superior"] / tabla["cierre"] - 1
    vista = pd.DataFrame({
        "Nombre": tabla["nombre"],
        "Sector": tabla["sector"],
        "País": tabla["país"],
        "Prob. más arriba": (tabla["prob_sube"] * 100).round(),
        "Según": tabla["fuente_direccion"].map(FUENTES).fillna("sin historia"),
        "Caída posible": var_min * 100,
        "Rango al 80 %": [f"{a:+.0%} / {b:+.0%}" if pd.notna(a) else "—" for a, b in zip(var_min, var_max)],
        "Fiabilidad del rango": tabla["estado_rango"].map(ESTADOS),
        "Desde su máximo": tabla["caida_desde_maximo"],
        "Último cierre": tabla["cierre"].map(numero),
        "Ver": "/proyeccion?activo=" + tabla["ticker"],
    })
    if tipo != "Acción":
        vista = vista.drop(columns=["Sector", "País"])
    if vista_salud:
        vista = pd.DataFrame({
            "Nombre": tabla["nombre"],
            "Sector": tabla["sector"],
            "País": tabla["país"],
            "Salud": [f"{ICONO_SALUD[e]} {SALUD[e]}" for e in tabla["salud"]],
            "PER": tabla["per"],
            "PER del sector": tabla["per_sector"],
            "PER frente al sector": tabla["per_vs_sector"].str.replace(" de su sector", "", regex=False),
            "Margen": tabla["margen"] * 100,
            "Deuda / patrimonio": tabla["deuda_patrimonio"],
            "Crec. ventas": tabla["crecimiento_ventas"] * 100,
            "Rent. caja": tabla["rentabilidad_fcf"] * 100,
            "¿Gana dinero?": tabla["beneficio_anual"].map(lambda v: "—" if pd.isna(v) else ("Sí" if v > 0 else "No")),
            "Prob. más arriba": (tabla["prob_sube"] * 100).round(),
            "Caída posible": var_min * 100,
            "Ver": "/proyeccion?activo=" + tabla["ticker"],
        })
    st.dataframe(
        vista, hide_index=True, width="stretch", height=min(38 * len(vista) + 40, 700),
        column_config={
            "Prob. más arriba": st.column_config.ProgressColumn(
                "Prob. más arriba", format="%d %%", min_value=0, max_value=100,
                help=f"Probabilidad estimada de que dentro de {plazo} esté más arriba que hoy."),
            "Desde su máximo": st.column_config.NumberColumn(
                "Desde su máximo", format="percent", help="Distancia al máximo del último año."),
            "Ver": st.column_config.LinkColumn("Ver", display_text="Ver →", help="Abre su proyección detallada"),
            "PER": st.column_config.NumberColumn("PER", format="%.1f", help="Precio / beneficio anual."),
            "PER del sector": st.column_config.NumberColumn(
                "PER del sector", format="%.1f", help="Mediana del PER de las empresas de su sector en esta web."),
            "Caída posible": st.column_config.NumberColumn(
                "Caída posible", format="%+.0f %%",
                help="Lo que podría caer en el plazo en un caso malo (límite inferior del rango al 80 %)."),
            "Margen": st.column_config.NumberColumn("Margen", format="%.1f %%",
                                                    help="Beneficio por cada 100 € de ventas."),
            "Deuda / patrimonio": st.column_config.NumberColumn(
                "Deuda / patrimonio", format="%.2f", help="En bancos y aseguradoras suele ser alta por su negocio."),
            "Crec. ventas": st.column_config.NumberColumn("Crec. ventas", format="%+.1f %%",
                                                          help="Último trimestre frente al mismo del año anterior."),
            "Rent. caja": st.column_config.NumberColumn("Rent. caja", format="%.1f %%",
                                                        help="Flujo de caja libre anual / valor en bolsa."),
        },
    )
    st.caption("Ordenada de más a menos probabilidad de estar más arriba (pulsa el título de cualquier columna para "
               "ordenar por ella). Una probabilidad alta no dice cuánto podría caer: mira «Caída posible». «Sin historia» = cotiza desde hace "
               "menos de un año. Los datos se calculan con `python -m analizador.estudio` y pueden tener unas horas.")
