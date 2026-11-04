"""Modelo de recursos del laboratorio (src/generador/recursos.py): calendarios de analistas y de equipos.

Es el entorno que comparten el generador del O1 y el simulador del O3.
Diseño: docs/Diseno_laboratorio_referencia.docx, secciones 3 y 4, y docs/Generador_de_datos.docx, sección 3.
"""
import pandas as pd
import pytest

from src.generador import disponibilidad, parametros
from src.generador.recursos import MIN_POR_DIA, Laboratorio, dia_y_hora, es_dia_habil

CICLO = 42
HORAS_SEMANA = {"T1": 56.0, "T2": 37.5, "T3": 40.0}   # diseño del laboratorio, Tabla 15
UNIDADES = 70                                         # Costigliola et al. (2017, p. 9015)


@pytest.fixture(scope="module")
def lab() -> Laboratorio:
    return Laboratorio.desde_excel()


def test_recursos_de_las_hojas(lab):
    assert len(lab.analistas) == 20 and len(lab.equipos) == UNIDADES
    assert lab.configuraciones["unidades"].sum() == UNIDADES
    assert (lab.configuraciones["unidades"] >= 2).all()
    assert set(lab.metodos) == {f"M{i:02d}" for i in range(1, 13)}
    for m in lab.metodos:
        assert all(m in lab.analistas[a].metodos for a in lab.calificados[m])
        assert all(m in lab.equipos[e].metodos for e in lab.compatibles[m])


def test_combinaciones_de_la_tabla_13(lab):
    cobertura = lab.cobertura()
    for m in lab.metodos:
        pares = lab.combinaciones_factibles(m)
        assert len(pares) == cobertura.at[m, "combinaciones_factibles"]
        assert len(pares) == len(lab.calificados[m]) * len(lab.compatibles[m])
    assert cobertura.loc[["M08", "M12"], "combinaciones_factibles"].tolist() == [6, 6]
    assert cobertura.at["M01", "combinaciones_factibles"] == 615


def test_configuracion_principal(lab):
    # la que corre menos métodos: M01 se cuenta en el HPLC isocrático (diseño, Tabla 11)
    assert lab.principal["M01"] == "HPLC-I"
    for m, c in lab.principal.items():
        assert any(lab.equipos[e].configuracion == c for e in lab.compatibles[m])


def test_horas_por_semana_de_cada_analista(lab):
    cal = lab.calendario_analistas(CICLO)
    horas = cal.assign(h=(cal["fin"] - cal["inicio"]) / 60).groupby("analista")["h"].sum() / (CICLO / 7)
    for a, h in horas.items():
        assert h == pytest.approx(HORAS_SEMANA[lab.analistas[a].turno], abs=0.5), a
    assert horas.sum() == pytest.approx(886, abs=2)


def test_tramos_sin_traslapes_y_dentro_de_la_corrida(lab):
    cal = lab.calendario_analistas(CICLO)
    assert (cal["inicio"] >= 0).all() and (cal["fin"] <= CICLO * MIN_POR_DIA).all()
    assert (cal["fin"] > cal["inicio"]).all()
    for _, g in cal.groupby("analista"):
        assert (g["inicio"].iloc[1:].to_numpy() >= g["fin"].iloc[:-1].to_numpy()).all()


def test_analistas_en_turno(lab):
    turno = lab.analistas_en_turno(CICLO, 60)
    habil = turno["inicio"].map(es_dia_habil)
    assert turno["analistas"].min() == 2                   # la pareja del turno 1, a toda hora
    assert turno.loc[habil, "analistas"].max() == 14
    assert turno.loc[~habil, "analistas"].max() == 2


def test_turno_de_noche_del_domingo_anterior(lab):
    # el lunes a las 3:00 de la primera semana ya hay una pareja en turno (empezó el domingo a las 20:00)
    turno = lab.analistas_en_turno(1, 60)
    assert turno.loc[turno["inicio"] == 3 * 60, "analistas"].item() == 2


def test_ventanas_de_mantenimiento(lab):
    dias = 364
    v = lab.ventanas(dias)
    assert len(v) == 4 * UNIDADES                           # una por trimestre y unidad (S26)
    por_unidad = v.groupby("equipo")["evento"].value_counts().unstack(fill_value=0)
    assert (por_unidad["mantenimiento"] == 1).all() and (por_unidad["verificación"] == 3).all()
    assert v["inicio"].map(es_dia_habil).all()
    assert ((v["inicio"] % MIN_POR_DIA) == 8 * 60).all()
    mant = parametros.cargar_hoja("Mantenimiento").set_index("Código")
    for c, g in v.groupby("configuracion"):
        horas = (g["fin"] - g["inicio"]) / 60
        assert set(horas[g["evento"].eq("mantenimiento")]) == {mant.at[c, "Duración del mantenimiento (h)"]}
        g = g.sort_values("inicio")
        assert (g["inicio"].iloc[1:].to_numpy() >= g["fin"].iloc[:-1].to_numpy()).all(), c


def test_cobertura_en_el_tiempo(lab):
    c = disponibilidad.cobertura_en_el_tiempo(CICLO, laboratorio=lab)
    assert (c["con_equipo"] == 1).all()
    for m in lab.metodos:
        if m in ("M08", "M12"):
            assert c.at[m, "con_analista"] == pytest.approx(125 / 168, abs=0.01)
        else:
            assert c.at[m, "con_analista"] == pytest.approx(1.0)


def test_cobertura_con_un_equipo_fuera(lab):
    eventos = pd.DataFrame({"equipo": ["KF-C01", "KF-C02"], "inicio": [0.0, 0.0], "fin": [7 * MIN_POR_DIA] * 2})
    c = disponibilidad.cobertura_en_el_tiempo(14, eventos, lab)
    assert c.at["M08", "con_equipo"] == pytest.approx(0.5)
    assert c.at["M08", "con_ambos"] <= c.at["M08", "con_analista"]


def test_dia_y_hora():
    assert dia_y_hora(0) == "lun. 0:00 (día 1)"
    assert dia_y_hora(5 * MIN_POR_DIA + 3 * 60 + 10) == "sáb. 3:10 (día 6)"
