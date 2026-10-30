"""Equipos y analistas: totales de la fuente, compatibilidad y matriz de calificación coherentes.

Diseño: docs/Diseno_laboratorio_referencia.docx, sección 3.
"""
import pandas as pd
import pytest

from src.generador import parametros

METODOS = [f"M{i:02d}" for i in range(1, 13)]
RANGO_NIVEL = {"júnior": 1, "intermedio": 2, "sénior": 3}
RANGO_COMPLEJIDAD = {"baja": 1, "media": 2, "alta": 3}   # nivel mínimo que exige cada complejidad (S12)
# Costigliola et al. (2017): 70 equipos y 20 analistas (p. 9015); turnos de la Tabla 1; equipos numerados hasta HPLC22 (Fig. 5)
EQUIPOS_FUENTE, ANALISTAS_FUENTE, HPLC_MINIMO = 70, 20, 22
ANALISTAS_POR_TURNO = {"T1": 6, "T2": 4, "T3": 10}


@pytest.fixture(scope="module")
def equipos() -> pd.DataFrame:
    return parametros.cargar_hoja("Equipos").set_index("Código")


@pytest.fixture(scope="module")
def analistas() -> pd.DataFrame:
    return parametros.cargar_hoja("Analistas").set_index("ID")


@pytest.fixture(scope="module")
def catalogo() -> pd.DataFrame:
    return parametros.cargar_hoja("Catálogo").set_index("ID")


def test_totales_de_equipos(equipos):
    assert equipos["Equipos"].sum() == EQUIPOS_FUENTE
    assert equipos.loc[equipos["Técnica"] == "HPLC", "Equipos"].sum() >= HPLC_MINIMO
    assert (equipos["Equipos"] >= 2).all()


def test_cada_metodo_tiene_equipo_compatible(equipos):
    compatibles = equipos[METODOS].eq("X").mul(equipos["Equipos"], axis=0).sum()
    assert (compatibles >= 1).all(), compatibles[compatibles < 1]


def test_plantilla_y_turnos(analistas):
    assert len(analistas) == ANALISTAS_FUENTE
    assert analistas["Turno"].value_counts().to_dict() == ANALISTAS_POR_TURNO
    assert analistas["Nivel"].isin(RANGO_NIVEL).all()


def test_nadie_esta_calificado_por_encima_de_su_nivel(analistas, catalogo):
    minimo = catalogo.loc[METODOS, "Complejidad"].map(RANGO_COMPLEJIDAD)
    nivel = analistas["Nivel"].map(RANGO_NIVEL)
    matriz = analistas[METODOS]
    assert matriz.isin([0, 1]).all().all()
    for metodo in METODOS:
        calificados = matriz.index[matriz[metodo] == 1]
        assert (nivel[calificados] >= minimo[metodo]).all(), metodo


def test_cobertura_de_cada_metodo(analistas, equipos):
    matriz = analistas[METODOS]
    assert (matriz.sum() >= 2).all(), matriz.sum()[matriz.sum() < 2]
    compatibles = equipos[METODOS].eq("X").mul(equipos["Equipos"], axis=0).sum()
    assert ((matriz.sum() * compatibles) > 0).all()


def test_turno_1_cubre_el_control_en_proceso(analistas, catalogo):
    metodos_in = [m for m in METODOS if catalogo.loc[m, "IN"] == "X"]
    turno_1 = analistas[analistas["Turno"] == "T1"]
    assert (turno_1[metodos_in] == 1).all().all()
