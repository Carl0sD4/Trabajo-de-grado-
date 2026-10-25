"""Prueba mínima: el proyecto está bien armado y la hoja de parámetros se puede leer."""
from src.generador import parametros
from src.generador.escenarios import ESCENARIOS


def test_hoja_de_parametros_existe():
    assert parametros.RUTA_PARAMETROS.exists()


def test_hoja_de_parametros_se_lee():
    df = parametros.cargar_hoja("Parámetros")
    assert "Parámetro" in df.columns
    assert len(df.dropna(subset=["Parámetro"])) >= 50


def test_escenarios_definidos():
    assert set(ESCENARIOS) == {"baja", "base", "alta"}
