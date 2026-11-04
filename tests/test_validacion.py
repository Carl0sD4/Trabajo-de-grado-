"""Validación del dataset con los 22 indicadores (src/generador/validacion.py) y disponibilidad cruzada.

Diseño: docs/Diccionario_de_datos.docx, sección 3, y docs/Validacion_dataset.docx.
"""
import numpy as np
import pandas as pd
import pytest

from src.generador import disponibilidad, duraciones, generar, registro, validacion

SEMANAS = 4


@pytest.fixture(scope="module")
def corrida():
    registros, eventos, _ = generar.generar_corrida("base", 2026, "v0.2", semanas=SEMANAS)
    return registros, eventos


@pytest.fixture(scope="module")
def v(corrida):
    return validacion.validar(*corrida, "v0.2", semanas=SEMANAS)


def test_resumen_de_los_22_indicadores(v):
    resumen = v["resumen"]
    assert len(resumen) == 22 and resumen["codigo"].is_unique
    assert set(resumen["veredicto"]) <= {"cumple", "no cumple", "se reporta", "sin datos suficientes"}
    assert resumen["valor"].str.len().gt(10).all()
    con_criterio = ["T1", "T2", "T3", "T4", "T5", "H1", "H2", "H3", "M1", "M2", "C1", "C2", "V1", "V3", "V4"]
    ver = resumen.set_index("codigo")["veredicto"]
    assert (ver[con_criterio] == "cumple").all(), ver[con_criterio][ver[con_criterio] != "cumple"]


def test_las_tablas_de_detalle(v, tmp_path):
    assert v["T5"]["tablas"]["tareas"]["z"].abs().max() <= validacion.Z_MEDIA
    assert v["M4"]["veredicto"] == "se reporta"                       # cuatro semanas: menos de un trimestre
    assert (v["H3"]["tablas"]["horas_por_analista"]["horas"] > 0).all()
    assert v["V4"]["tablas"]["nulos"].loc[["escenario", "duracion_simulada"], "vacios"].eq(0).all()
    rutas = validacion.guardar(v, "prueba", tmp_path)
    assert all(r.exists() for r in rutas)
    assert len(pd.read_csv(rutas[0])) == 22
    assert "Resumen" in pd.ExcelFile(rutas[1]).sheet_names


def test_fuera_de_rango_marca_solo_los_valores_malos(corrida):
    registros = corrida[0].head(200).copy()
    assert not validacion.fuera_de_rango(registros).to_numpy().any()
    malos = {"prioridad": 9, "recurso_faltante": "nadie", "equipo": "HPLC-G99", "n_soluciones": 7,
             "t_corrida": -1.0, "id_muestra": "PT-12"}
    for k, (campo, valor) in enumerate(malos.items()):
        registros.iloc[k, registros.columns.get_loc(campo)] = valor
    fuera = validacion.fuera_de_rango(registros)
    for k, campo in enumerate(malos):
        assert fuera.iloc[k][campo], campo
    assert fuera.to_numpy().sum() == len(malos)


def test_un_vacio_cuenta_como_fuera_de_rango(corrida):
    registros = corrida[0].head(50).copy()
    registros.loc[registros.index[0], "analista"] = np.nan
    primero = registros["intento"].eq(1)
    fuera = validacion.fuera_de_rango(registros)
    assert fuera.iloc[0]["analista"]
    assert not fuera.loc[primero, ["id_actividad_origen", "causa_repeticion"]].to_numpy().any()


def test_ajuste_betapert_detecta_otra_forma(corrida):
    registros = corrida[0]
    bien = validacion.ajuste_betapert(registros, "hoja")
    assert (bien["p_bonferroni"] >= validacion.ALFA_V1).all()
    mal = validacion.ajuste_betapert(registros, 4.0)       # los datos son λ = 1: la PERT clásica no les ajusta
    assert (mal["p_bonferroni"] < validacion.ALFA_V1).any()


def test_trazabilidad():
    tabla = validacion.trazabilidad()
    assert tabla.at["Parámetros", "filas"] == 79 and tabla.at["Supuestos", "filas"] == 30
    assert tabla.at["Total", "con_fuente"] == 78
    assert tabla.at["Total", "porcentaje"] == pytest.approx(78 / 109)


def test_tiempos_de_la_fuente():
    t5 = validacion.tiempo_en_sistema_fuente()        # Costigliola et al. (2017, Tabla 5, p. 9018)
    assert t5.to_dict() == {"HPLC": 31.0, "GC": 10.0, "PSA": 4.0, "KF": 3.0, "DSC": 5.0, "XRPD": 6.0}
    t6 = duraciones.tiempos_promedio_fuente()          # Tabla 6, p. 9019
    assert t6.loc["HPLC", ["t_prep_sistema", "t_idoneidad", "t_preparacion", "t_corrida", "t_procesamiento"]].tolist() == [
        207, 227, 96, 85, 59]
    assert np.isnan(t6.at["PSA", "t_prep_sistema"])
    sis, ido = duraciones.tres_puntos_alistamiento("GC")
    assert sis == pytest.approx((67 * 0.8, 67, 67 * 1.2)) and ido == pytest.approx((152 * 0.8, 152, 152 * 1.2))


def test_indicadores_de_disponibilidad_cruzada(corrida):
    registros = corrida[0]
    ind = disponibilidad.indicadores(registros, desde=14 * 24 * 60)
    assert ind["c2"]["porcentaje"] == 100
    tabla = ind["c3_c4"]
    assert np.allclose(tabla[list(disponibilidad.FALTANTES)].sum(axis=1), 100)
    assert np.allclose(tabla["restringidas"], 100 - tabla["ninguno"])
    assert set(ind["c1"].index) == set(registro.METODOS)
    assert disponibilidad.combinaciones_factibles("M08") and len(disponibilidad.combinaciones_factibles("M08")) == 6
