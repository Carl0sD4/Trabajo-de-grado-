"""Llegadas de muestras, ensayos de cada muestra, plazos y urgentes (src/generador/llegadas.py).

Diseño: docs/Generador_de_datos.docx, sección 5; supuestos S03, S06, S10, S17 y S18.
"""
import numpy as np
import pandas as pd
import pytest

from src.generador import llegadas, parametros, registro
from src.generador.recursos import MIN_POR_DIA

DIAS = 364
Z = 5   # tolerancia en errores estándar, el mismo margen de las otras pruebas del generador


@pytest.fixture(scope="module")
def tipos() -> pd.DataFrame:
    return llegadas.tipos_de_muestra()


@pytest.fixture(scope="module")
def muestras() -> pd.DataFrame:
    return llegadas.generar_muestras(DIAS, 1.0, np.random.default_rng(1), np.random.default_rng(2))


@pytest.fixture(scope="module")
def ensayos(muestras) -> pd.DataFrame:
    return llegadas.generar_ensayos(muestras, np.random.default_rng(3))


def test_reproducible_con_la_semilla():
    a = llegadas.generar_muestras(14, 1.0, 7, 8)
    b = llegadas.generar_muestras(14, 1.0, 7, 8)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(llegadas.generar_ensayos(a, 9), llegadas.generar_ensayos(b, 9))


def test_muestras_por_dia_de_cada_tipo(muestras, tipos):
    dias_habiles = sum(1 for d in range(DIAS) if d % 7 < 5)
    for tipo, f in tipos.iterrows():
        dias = dias_habiles if int(f["Días de llegada por semana"]) == 5 else DIAS
        esperado = float(f["Muestras por día de llegada"]) * dias
        observado = (muestras["tipo_muestra"] == tipo).sum()
        assert abs(observado - esperado) <= Z * np.sqrt(esperado), tipo


def test_la_carga_escala_las_llegadas():
    base = llegadas.generar_llegadas(DIAS, 1.0, 11)
    alta = llegadas.generar_llegadas(DIAS, 1.5, 11)
    assert len(alta) / len(base) == pytest.approx(1.5, rel=0.03)


def test_horas_de_llegada(muestras, tipos):
    dia = (muestras["llegada_muestra"] // MIN_POR_DIA).astype(int)
    hora = (muestras["llegada_muestra"] % MIN_POR_DIA) / 60
    for tipo, f in tipos.iterrows():
        sel = muestras["tipo_muestra"] == tipo
        desde, hasta = float(f["Llegada desde (h)"]), float(f["Llegada hasta (h)"])
        if int(f["Días de llegada por semana"]) == 5:
            assert (dia[sel] % 7 < 5).all(), tipo
        if hasta > desde:
            assert hora[sel].between(desde, hasta).all(), tipo
        else:
            assert (hora[sel] == desde).all(), tipo     # estabilidad: lote de las 8:00
    assert muestras["llegada_muestra"].is_monotonic_increasing
    assert hora[muestras["tipo_muestra"] == "IN"].lt(8).any()   # control en proceso también de noche


def test_identificador_consecutivo_por_tipo(muestras):
    assert muestras["id_muestra"].str.fullmatch(r"(MP|IN|GR|PT|ES)-\d{6}").all()
    assert muestras["id_muestra"].is_unique
    for tipo, g in muestras.groupby("tipo_muestra"):
        numeros = g["id_muestra"].str[-6:].astype(int)
        assert numeros.tolist() == list(range(1, len(g) + 1)), tipo


def test_urgentes_prioridad_y_plazo(muestras, tipos):
    puede = tipos.index[tipos["Puede ser urgente"].eq("Sí")]
    assert not muestras.loc[~muestras["tipo_muestra"].isin(puede), "urgente"].any()
    candidatas = muestras["tipo_muestra"].isin(puede)
    s03 = float(parametros.supuesto("S03"))
    n = int(candidatas.sum())
    assert abs(muestras.loc[candidatas, "urgente"].mean() - s03) <= Z * np.sqrt(s03 * (1 - s03) / n)
    for t, u, ll, p, plazo, venc in muestras[["tipo_muestra", "urgente", "llegada_muestra", "prioridad", "plazo_min",
                                              "vencimiento"]].itertuples(index=False):
        assert p == registro.prioridad(t, u)
        assert plazo == pytest.approx(registro.plazo_minutos(t, u, ll))
        assert venc == pytest.approx(ll + plazo)
    assert (muestras.loc[muestras["urgente"], "plazo_min"] == 240).all()
    assert (muestras.loc[muestras["urgente"], "prioridad"] == 1).all()


def test_ensayos_por_muestra(muestras, ensayos):
    k = ensayos.groupby("id_muestra").size().reindex(muestras["id_muestra"], fill_value=0)
    assert k.between(*registro.ENSAYOS_POR_MUESTRA).all()
    esperado = llegadas.ensayos_por_muestra()        # Cunha et al. (2019, p. 1425): 85 %, 10 % y 5 %
    assert esperado.to_dict() == pytest.approx({1: 0.85, 2: 0.10, 3: 0.05})
    observado = k.value_counts(normalize=True)
    for n, p in esperado.items():
        assert abs(observado.get(n, 0) - p) <= Z * np.sqrt(p * (1 - p) / len(k)), n


def test_metodos_distintos_y_permitidos_por_el_tipo(muestras, ensayos, tipos):
    assert not ensayos.duplicated(["id_muestra", "metodo"]).any()     # S18
    metodos = [c for c in tipos.columns if str(c).startswith("M") and str(c)[1:].isdigit()]
    tipo = ensayos["id_muestra"].str[:2]
    for t, g in ensayos.groupby(tipo):
        permitidos = {m for m in metodos if float(tipos.at[t, m]) > 0}
        assert set(g["metodo"]) <= permitidos, t


def test_mezcla_de_metodos_dentro_del_tipo(muestras, ensayos, tipos):
    # con un solo ensayo por muestra, la frecuencia de cada método es la de la hoja; con más, cambia muy poco
    metodos = [c for c in tipos.columns if str(c).startswith("M") and str(c)[1:].isdigit()]
    un_ensayo = ensayos.groupby("id_muestra").filter(lambda g: len(g) == 1)
    tipo = un_ensayo["id_muestra"].str[:2]
    for t, g in un_ensayo.groupby(tipo):
        p = tipos.loc[t, metodos].astype(float)
        observado = g["metodo"].value_counts(normalize=True).reindex(metodos, fill_value=0)
        assert (observado - p).abs().max() <= Z * np.sqrt(0.25 / len(g)), t


def test_soluciones_en_el_rango_del_metodo(ensayos):
    catalogo = parametros.cargar_hoja("Catálogo").set_index("ID")
    lo = ensayos["metodo"].map(catalogo["Soluciones mín"])
    hi = ensayos["metodo"].map(catalogo["Soluciones máx"])
    assert ensayos["n_soluciones"].between(lo, hi).all()
    # uniforme en el rango (S08): los dos valores de cada método aparecen
    assert (ensayos.groupby("metodo")["n_soluciones"].nunique() == 2).all()
