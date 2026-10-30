"""Los tipos de muestra se leen bien y no cambian la mezcla de métodos del catálogo.

Diseño: docs/Diseno_laboratorio_referencia.docx, sección 2.
"""
import pandas as pd
import pytest

from src.generador import parametros

TIPOS = ["MP", "IN", "GR", "PT", "ES"]
METODOS = [f"M{i:02d}" for i in range(1, 13)]
ANALISIS_AL_ANO = 15323        # Costigliola et al. (2017), Tabla 4, p. 9018
ENSAYOS_POR_MUESTRA = 1.20     # Cunha et al. (2019), p. 1425: 85 % con 1, 10 % con 2 y 5 % con 3


@pytest.fixture(scope="module")
def tipos() -> pd.DataFrame:
    return parametros.cargar_hoja("Tipos de muestra").set_index("Código")


@pytest.fixture(scope="module")
def catalogo() -> pd.DataFrame:
    return parametros.cargar_hoja("Catálogo").set_index("ID")


def test_cinco_tipos_que_suman_uno(tipos):
    assert list(tipos.index) == TIPOS
    assert (tipos["Proporción de muestras"] > 0).all()
    assert tipos["Proporción de muestras"].sum() == pytest.approx(1.0)


def test_probabilidades_por_tipo(tipos, catalogo):
    probabilidades = tipos[METODOS]
    assert probabilidades.sum(axis=1).to_numpy() == pytest.approx([1.0] * 5)
    for tipo in TIPOS:
        for metodo in METODOS:
            if catalogo.loc[metodo, tipo] != "X":
                assert probabilidades.loc[tipo, metodo] == 0, (tipo, metodo)


def test_la_mezcla_de_metodos_se_conserva(tipos, catalogo):
    mezcla = tipos[METODOS].mul(tipos["Proporción de muestras"], axis=0).sum()
    esperado = catalogo.loc[METODOS, "Participación en la demanda"].astype(float)
    assert mezcla.to_numpy() == pytest.approx(esperado.to_numpy(), abs=1e-6)


def test_llegadas_plazos_y_urgentes(tipos):
    assert set(tipos["Días de llegada por semana"]) <= {5, 7}
    assert tipos.loc["IN", "Días de llegada por semana"] == 7
    assert (tipos["Llegada desde (h)"] <= tipos["Llegada hasta (h)"]).all()
    assert tipos["Llegada hasta (h)"].max() <= 24
    assert tipos["Unidad del plazo"].isin(["h", "días hábiles"]).all()
    assert (tipos["Plazo"] > 0).all()
    assert tipos.loc["IN", "Prioridad"] == 1
    assert tipos.loc["ES", "Puede ser urgente"] == "No"


def test_volumen_anual(tipos):
    muestras = tipos["Muestras al año"].sum()
    assert muestras * ENSAYOS_POR_MUESTRA == pytest.approx(ANALISIS_AL_ANO, rel=1e-6)
