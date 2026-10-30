"""El catálogo de métodos se lee bien y conserva los promedios de Costigliola et al. (2017).

Si cambian una complejidad o un supuesto en la hoja Catálogo o en la hoja
Calibración del catálogo, estas pruebas avisan si se rompe la calibración.
Diseño: docs/Diseno_laboratorio_referencia.docx, sección 1.
"""
import pandas as pd
import pytest

from src.generador import parametros

TECNICAS = ["HPLC", "GC", "PSA", "KF", "DSC", "XRPD"]
TIPOS_DE_MUESTRA = ["MP", "IN", "GR", "PT", "ES"]

# Datos de la fuente (Costigliola et al., 2017), copiados aquí para verificar la hoja por otro camino.
PREP_ANALISIS_TABLA6 = {"HPLC": 96, "GC": 48, "PSA": 31, "KF": 45, "DSC": 47, "XRPD": 46}  # p. 9019
PREP_SOLUCION_TABLA3 = {"HPLC": (5, 40, 100), "GC": (5, 10, 35), "PSA": (5, 7.5, 21),      # p. 9016
                        "KF": (1, 6, 25), "DSC": (5, 10, 30), "XRPD": (5, 10, 20)}
CORRIDA_TABLA6 = {"HPLC": 85, "GC": 62}                                                     # p. 9019
CORRIDA_TABLA3_MEDIA = {"PSA": (5 + 10 + 30) / 3, "KF": (2 + 5 + 56) / 3,                   # p. 9016
                        "DSC": (12 + 180) / 2, "XRPD": (11 + 25 + 203) / 3}


@pytest.fixture(scope="module")
def catalogo() -> pd.DataFrame:
    return parametros.cargar_hoja("Catálogo")


def test_doce_metodos_dos_por_tecnica(catalogo):
    assert len(catalogo) == 12
    assert catalogo["ID"].is_unique
    assert sorted(catalogo["Técnica"].unique()) == sorted(TECNICAS)
    assert (catalogo.groupby("Técnica").size() == 2).all()


def test_complejidad_y_soluciones(catalogo):
    assert catalogo["Complejidad"].isin(["baja", "media", "alta"]).all()
    assert (catalogo["Soluciones mín"] <= catalogo["Soluciones máx"]).all()


def test_participaciones(catalogo):
    participacion = pd.to_numeric(catalogo["Participación en la técnica"], errors="coerce")
    assert participacion.between(0, 1).all(), "Revisen la columna Chequeo de la hoja Calibración del catálogo"
    assert catalogo.groupby("Técnica")["Participación en la técnica"].sum().to_numpy() == pytest.approx([1.0] * 6)
    assert catalogo["Participación en la demanda"].sum() == pytest.approx(1.0)


def test_soluciones_promedio_reproducen_costigliola(catalogo):
    media = (catalogo["Soluciones mín"] + catalogo["Soluciones máx"]) / 2
    promedio = (media * catalogo["Participación en la técnica"]).groupby(catalogo["Técnica"]).sum()
    for tecnica, (a, m, b) in PREP_SOLUCION_TABLA3.items():
        implicito = PREP_ANALISIS_TABLA6[tecnica] / ((a + m + b) / 3)
        assert promedio[tecnica] == pytest.approx(implicito), tecnica


def test_corrida_promedio_por_tecnica(catalogo):
    corrida = (catalogo["Corrida promedio (min)"] * catalogo["Participación en la técnica"]).groupby(catalogo["Técnica"]).sum()
    for tecnica, esperado in {**CORRIDA_TABLA6, **CORRIDA_TABLA3_MEDIA}.items():
        assert corrida[tecnica] == pytest.approx(esperado), tecnica


def test_reanalisis_promedio_igual_a_s01(catalogo):
    supuestos = parametros.cargar_hoja("Supuestos").set_index("ID")
    s01 = float(supuestos.loc["S01", "Propuesta inicial (supuesto)"])
    promedio = (catalogo["Prob. de reanálisis"] * catalogo["Participación en la demanda"]).sum()
    assert promedio == pytest.approx(s01)


def test_cada_metodo_aplica_a_algun_tipo_de_muestra(catalogo):
    marcas = catalogo[TIPOS_DE_MUESTRA].eq("X")
    assert marcas.any(axis=1).all()
    assert marcas.any(axis=0).all()
