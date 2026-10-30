"""Amarre del registro con el sistema de variables: cada campo alimenta un indicador y cada indicador sabe con qué campos se calcula.

Diseño: docs/Diccionario_de_datos.docx, sección 3 (anteproyecto, §8.5).
"""
import pandas as pd
import pytest

from src.generador import parametros, registro

# Anteproyecto, §8.5: las cinco dimensiones de la variable «datos sintéticos» y sus indicadores, en su orden
SISTEMA_DE_VARIABLES = {
    "Tiempos de ejecución": [
        "Tiempo optimista por actividad", "Tiempo más probable", "Tiempo pesimista",
        "Duración simulada de la actividad", "Media y dispersión de las duraciones generadas"],
    "Recursos humanos": [
        "Número de analistas disponibles", "Calificación o elegibilidad del analista por actividad",
        "Disponibilidad temporal del analista", "Carga asignada al analista"],
    "Recursos materiales": [
        "Número de equipos disponibles", "Compatibilidad equipo–actividad",
        "Disponibilidad temporal del equipo", "Condición de disponibilidad o indisponibilidad del equipo"],
    "Disponibilidad cruzada de recursos": [
        "Número de combinaciones factibles analista–equipo por actividad",
        "Porcentaje de actividades con al menos una combinación factible",
        "Coincidencia temporal entre analista calificado y equipo compatible",
        "Porcentaje de actividades restringidas por falta de alguno de los recursos requeridos"],
    "Variabilidad y consistencia estructural": [
        "Distribución de los tiempos simulados", "Correlaciones entre variables definidas",
        "Porcentaje de registros que cumplen las restricciones establecidas",
        "Porcentaje de valores fuera de los rangos definidos",
        "Porcentaje de parámetros con trazabilidad bibliográfica"],
}
TODOS_LOS_CAMPOS = ("V3", "V4")   # cada campo tiene una regla y un rango
SIN_CAMPOS = ("V5",)              # se calcula con las hojas Parámetros y Supuestos


@pytest.fixture(scope="module")
def indicadores() -> pd.DataFrame:
    return parametros.cargar_hoja("Indicadores").set_index("Código")


@pytest.fixture(scope="module")
def diccionario() -> pd.DataFrame:
    return parametros.cargar_hoja("Diccionario")


def codigos(texto) -> list:
    return [c.strip() for c in str(texto).split(",") if c.strip()]


def test_estan_los_22_indicadores_del_anteproyecto(indicadores):
    esperado = [(dimension, indicador) for dimension, lista in SISTEMA_DE_VARIABLES.items() for indicador in lista]
    assert list(zip(indicadores["Dimensión"], indicadores["Indicador (anteproyecto, §8.5)"])) == esperado
    assert indicadores[["Qué se calcula", "Campos del registro", "Nivel", "Referencia o criterio"]].notna().all().all()


def test_cada_campo_alimenta_al_menos_un_indicador(diccionario, indicadores):
    assert list(diccionario["Campo"]) == registro.CAMPOS_REGISTRO
    for campo, texto in zip(diccionario["Campo"], diccionario["Indicadores"]):
        assert codigos(texto), campo
        assert set(codigos(texto)) <= set(indicadores.index), campo


def test_cada_indicador_dice_con_que_campos_se_calcula(diccionario, indicadores):
    por_indicador = {codigo: [] for codigo in indicadores.index}
    for campo, texto in zip(diccionario["Campo"], diccionario["Indicadores"]):
        for codigo in codigos(texto):
            por_indicador[codigo].append(campo)
    for codigo, campos in indicadores["Campos del registro"].items():
        if codigo in TODOS_LOS_CAMPOS:
            assert campos.startswith("Todos"), codigo
        elif codigo in SIN_CAMPOS:
            assert campos == "Ninguno" and not por_indicador[codigo], codigo
        else:
            assert por_indicador[codigo], codigo
            assert codigos(campos) == por_indicador[codigo], codigo

