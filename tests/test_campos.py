"""Campos del registro: la hoja «Diccionario», el código y el ejemplo coinciden, y cada regla detecta su error.

Diseño: docs/Diccionario_de_datos.docx, sección 2; el registro completo de ejemplo es su Tabla 13.
"""
import pandas as pd
import pytest

from src.generador import parametros, registro
from src.generador.escenarios import ESCENARIOS

ROLES = {"Identificador", "Estrato", "Predictor", "Objetivo", "O3 y validación"}
MOMENTOS = ["Antes de la corrida", "Al llegar", "Al asignar", "Al terminar"]
DIA = 24 * 60


@pytest.fixture(scope="module")
def diccionario() -> pd.DataFrame:
    return parametros.cargar_hoja("Diccionario")


@pytest.fixture(scope="module")
def ejemplo() -> pd.DataFrame:
    return registro.ejemplo_del_diccionario()


def test_el_diccionario_coincide_con_el_codigo(diccionario, ejemplo):
    assert list(diccionario["Campo"]) == registro.CAMPOS_REGISTRO
    assert diccionario[["Descripción", "Tipo", "Unidad", "Valores o rango", "Origen"]].notna().all().all()
    assert set(diccionario["Se conoce"]) <= set(MOMENTOS)
    assert set(diccionario["Rol en el O2"]) <= ROLES
    assert list(diccionario.loc[diccionario["Rol en el O2"].eq("Objetivo"), "Campo"]) == ["duracion_simulada"]
    assert set(diccionario["Regla de validación"]) <= set(registro.chequear_registro(ejemplo).columns)


def test_los_predictores_se_conocen_antes_de_ejecutar(diccionario):
    predictores = diccionario[diccionario["Rol en el O2"].eq("Predictor")]
    assert set(predictores["Se conoce"]) <= set(MOMENTOS[:3])


def test_el_ejemplo_cumple_todas_las_reglas(ejemplo):
    reglas = registro.chequear_registro(ejemplo)
    assert reglas.all().all(), reglas.columns[~reglas.all()].tolist()


def test_tres_puntos_de_sustancias_relacionadas():
    # Costigliola et al. (2017, Tabla 3, p. 9016), HPLC: preparación por solución T(5, 40, 100),
    # montaje T(25, 50, 65) y procesamiento T(2, 5, 10); la corrida de M02 sale del catálogo
    corrida = registro.corrida_tres_puntos("M02")
    esperado = [4 * p + mo + c + pr for p, mo, c, pr in zip((5, 40, 100), (25, 50, 65), corrida, (2, 5, 10))]
    assert registro.tres_puntos("M02", 4) == pytest.approx(esperado)


def test_corrida_en_tramos():
    catalogo = parametros.cargar_hoja("Catálogo").set_index("ID")
    assert registro.corrida_tres_puntos("M07")[1] == pytest.approx(5)    # KF T(2, 5, 56): la moda cae en el tramo
    assert registro.corrida_tres_puntos("M08")[1] == pytest.approx(catalogo.at["M08", "Corrida mín (min)"])
    lo, hi = catalogo.loc["M09", ["Corrida mín (min)", "Corrida máx (min)"]]
    assert registro.corrida_tres_puntos("M09")[1] == pytest.approx((lo + hi) / 2)   # DSC uniforme: punto medio


def test_plazo_en_dias_habiles():
    martes_810, viernes_10 = 1 * DIA + 8 * 60 + 10, 4 * DIA + 10 * 60
    assert registro.plazo_minutos("PT", False, martes_810) == 2 * DIA    # vence el jueves a las 8:10
    assert registro.plazo_minutos("GR", False, viernes_10) == 3 * DIA    # un día hábil: vence el lunes
    assert registro.plazo_minutos("ES", False, viernes_10) == 14 * DIA   # diez días hábiles: dos semanas
    assert registro.plazo_minutos("PT", True, martes_810) == 240         # urgente: plazo del control en proceso


def test_escenarios_coinciden_con_la_hoja_demanda():
    demanda = parametros.cargar_hoja("Demanda")
    primera, segunda = demanda.columns[:2]
    filas = demanda[demanda[primera].isin(["Baja", "Base", "Alta"])]
    hoja = dict(zip(filas[primera].str.lower(), filas[segunda].astype(float)))
    assert hoja == {nombre: e["multiplicador_carga"] for nombre, e in ESCENARIOS.items()}


CORRUPCIONES = [
    ("n_soluciones", 0, 6, "ensayo_coherente", [0]),                 # M02 pide 3 o 4 soluciones
    ("analista", 0, "A10", "asignacion_coherente", [0]),             # júnior, no calificado en M02
    ("equipo", 0, "HPLC-G16", "asignacion_coherente", [0]),          # solo hay 15 HPLC con gradiente
    ("t_optimista", 1, 60.0, "tres_puntos_ordenados", [1]),          # mayor que el más probable
    ("duracion_simulada", 1, 50.0, "duracion_coherente", [1]),       # no es la suma de las tareas
    ("espera", 0, 10.0, "tiempos_coherentes", [0]),                  # inicio − llegada son 30 min
    ("bandera_falla", 4, False, "banderas_coherentes", [4]),         # repite una ejecución interrumpida
    ("analistas_libres", 1, 0, "combinaciones_coherentes", [1]),     # recurso_faltante sigue en «ninguno»
    ("carga", 2, 1.5, "corrida_coherente", [2]),                     # el escenario es el base
    ("urgente", 3, True, "muestra_y_plazo", [3, 4]),                 # el control en proceso no llega urgente
    ("prioridad", 0, None, "campos_completos", [0]),                 # un campo obligatorio vacío
]


@pytest.mark.parametrize("campo, fila, valor, regla, fallan", CORRUPCIONES)
def test_cada_regla_detecta_su_error(ejemplo, campo, fila, valor, regla, fallan):
    df = ejemplo.copy()
    df[campo] = df[campo].astype(object)
    df.loc[fila, campo] = valor
    reglas = registro.chequear_campos(df)
    assert not reglas.loc[fallan, regla].any()
    assert reglas.drop(index=fallan)[regla].all()
