"""Perturbaciones del generador (src/generador/perturbaciones.py): urgentes, reanálisis y fallas de equipos.

Diseño: docs/Generador_de_datos.docx, sección 6; supuestos S01, S02, S03 y S13.
"""
import numpy as np
import pandas as pd
import pytest

from src.generador import parametros, perturbaciones
from src.generador.recursos import MIN_POR_DIA, Laboratorio

Z = 5


def test_tasas_de_la_hoja_supuestos():
    tasas = perturbaciones.tasas_base()
    assert tasas == {"reanalisis": float(parametros.supuesto("S01")), "fallas": float(parametros.supuesto("S02")),
                     "urgentes": float(parametros.supuesto("S03"))}


def test_urgentes_solo_en_los_tipos_que_pueden_serlo():
    tipos = np.array(["MP", "IN", "GR", "PT", "ES"] * 20_000)
    urgente = perturbaciones.marcar_urgentes(tipos, 1)
    assert not urgente[np.isin(tipos, ["IN", "ES"])].any()
    puede = np.isin(tipos, ["MP", "GR", "PT"])
    s03 = perturbaciones.tasas_base()["urgentes"]
    assert abs(urgente[puede].mean() - s03) <= Z * np.sqrt(s03 * (1 - s03) / puede.sum())
    assert not perturbaciones.marcar_urgentes(tipos, 1, tasa=0.0).any()


def test_probabilidad_de_reanalisis():
    catalogo = parametros.cargar_hoja("Catálogo").set_index("ID")
    s01 = perturbaciones.tasas_base()["reanalisis"]
    for m in catalogo.index:
        p = perturbaciones.probabilidad_reanalisis(m)
        assert p == pytest.approx(float(catalogo.at[m, "Prob. de reanálisis"]))
        assert perturbaciones.probabilidad_reanalisis(m, 2 * s01) == pytest.approx(2 * p)
        assert perturbaciones.probabilidad_reanalisis(m, 0.0) == 0.0
    # el multiplicador de la complejidad (S13): alta > media > baja
    por_complejidad = catalogo.groupby("Complejidad")["Prob. de reanálisis"].mean()
    assert por_complejidad["baja"] < por_complejidad["media"] < por_complejidad["alta"]


def test_tiempo_medio_entre_fallas():
    a, m, b = perturbaciones.reparacion()
    assert (a, m, b) == (8.0, 48.0, 168.0)
    assert perturbaciones.tiempo_medio_entre_fallas() == pytest.approx(1418.67, abs=0.01)
    assert perturbaciones.tiempo_medio_entre_fallas(0.0) == float("inf")


def test_fallas_dan_la_fraccion_de_s02():
    equipos = list(Laboratorio.desde_excel().equipos.values())
    minutos = 10 * 364 * MIN_POR_DIA
    fallas = perturbaciones.generar_fallas(equipos, minutos, 7)
    assert set(fallas.columns) == {"equipo", "configuracion", "evento", "inicio", "fin"}
    assert (fallas["evento"] == "falla").all()
    horas = (fallas["fin"] - fallas["inicio"]) / 60
    assert horas.between(8, 168).all()
    assert horas.mean() == pytest.approx(224 / 3, rel=0.02)          # media de la triangular (8, 48, 168)
    fraccion = (fallas["fin"].clip(upper=minutos) - fallas["inicio"]).sum() / (len(equipos) * minutos)
    assert fraccion == pytest.approx(perturbaciones.tasas_base()["fallas"], abs=0.003)
    for _, g in fallas.groupby("equipo"):                            # una unidad no falla mientras está en reparación
        g = g.sort_values("inicio")
        assert (g["inicio"].iloc[1:].to_numpy() >= g["fin"].iloc[:-1].to_numpy()).all()
    assert perturbaciones.generar_fallas(equipos, minutos, 7, fraccion=0.0).empty


def test_fallas_reproducibles():
    equipos = list(Laboratorio.desde_excel().equipos.values())[:5]
    a = perturbaciones.generar_fallas(equipos, 364 * MIN_POR_DIA, 3)
    pd.testing.assert_frame_equal(a, perturbaciones.generar_fallas(equipos, 364 * MIN_POR_DIA, 3))
