import numpy as np
import pandas as pd
import pytest

from analizador import registro as rg


def estudio(fecha="2026-01-05", prob=0.7, cierre=100.0):
    return pd.DataFrame({"ticker": ["AAA", "AAA"], "nombre": ["Empresa A"] * 2, "fecha": [fecha] * 2,
                         "plazo": ["5 días", "1 mes"], "sesiones_plazo": [5, 21], "cierre": [cierre] * 2,
                         "prob_sube": [prob] * 2, "inferior": [95.0, 90.0], "superior": [105.0, 110.0],
                         "confianza": [0.8] * 2, "fuente_direccion": ["ia"] * 2})


def test_registrar_no_duplica_ni_reescribe(tmp_path):
    ruta = tmp_path / "pred.csv"
    assert rg.registrar_predicciones(estudio(), ruta) == 2
    # Mismo día y activo con otra probabilidad: no se reescribe lo ya apuntado.
    assert rg.registrar_predicciones(estudio(prob=0.2), ruta) == 0
    assert rg.registrar_predicciones(estudio(fecha="2026-01-06"), ruta) == 2
    tabla = rg.cargar_predicciones(ruta)
    assert len(tabla) == 4 and (tabla.loc[tabla["fecha"] == "2026-01-05", "prob_sube"] == 0.7).all()


def test_estudio_de_indices_tambien_se_registra(tmp_path):
    indices = pd.DataFrame({"ticker": ["^IBEX"], "índice": ["IBEX 35"], "país": ["España"], "fecha": ["2026-01-05"],
                            "plazo": ["5 días"], "sesiones": [5], "cierre": [100.0], "prob_sube": [0.6],
                            "inferior": [95.0], "superior": [105.0], "confianza": [0.8], "fuente_direccion": ["ia"]})
    ruta = tmp_path / "pred.csv"
    rg.registrar_predicciones(indices, ruta)
    fila = rg.cargar_predicciones(ruta).iloc[0]
    assert fila["nombre"] == "IBEX 35 (España)" and fila["sesiones_plazo"] == 5


def test_comprobar_usa_las_sesiones_del_activo():
    idx = pd.bdate_range("2026-01-05", periods=30)
    precios = pd.Series(np.linspace(100, 129, 30), index=idx)  # sube 1 por sesión
    pred = estudio()
    pred["fecha"] = pd.to_datetime(pred["fecha"])
    res = rg.comprobar(pred, {"AAA": precios})
    cinco = res[res["plazo"] == "5 días"].iloc[0]
    assert cinco["comprobada"] and cinco["precio_real"] == precios.iloc[5]
    assert cinco["acierto_lado"] and cinco["dentro_rango"]  # 105 ≤ 105
    mes = res[res["plazo"] == "1 mes"].iloc[0]
    assert mes["precio_real"] == precios.iloc[21] and not mes["dentro_rango"]  # 121 > 110


def test_pendientes_no_cuentan():
    idx = pd.bdate_range("2026-01-05", periods=10)
    pred = estudio()
    pred["fecha"] = pd.to_datetime(pred["fecha"])
    res = rg.comprobar(pred, {"AAA": pd.Series(100.0, index=idx)})
    assert res.loc[res["plazo"] == "1 mes", "comprobada"].iloc[0] == False  # noqa: E712
    r = rg.resumen(res)
    assert list(r.index) == ["5 días"]


def test_historial_de_entrenamientos(tmp_path):
    ruta = tmp_path / "ent.csv"
    rg.registrar_entrenamiento([{"fecha": "2026-10-10", "modelo": "rangos", "valor": 0.79}], ruta)
    rg.registrar_entrenamiento([{"fecha": "2026-11-10", "modelo": "rangos", "valor": 0.80}], ruta)
    assert list(rg.cargar_entrenamientos(ruta)["valor"]) == pytest.approx([0.79, 0.80])
