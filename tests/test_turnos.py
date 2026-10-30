"""Turnos, rotación y ventanas de mantenimiento: reproducen la Tabla 1 y cubren cada método en el tiempo.

Diseño: docs/Diseno_laboratorio_referencia.docx, sección 4.
"""
import pandas as pd
import pytest

from src.generador import parametros

METODOS = [f"M{i:02d}" for i in range(1, 13)]
CICLO = 42
FRANJAS = [(0, 8), (8, 12), (12, 13), (13, 14), (14, 17), (17, 20), (20, 24)]
# Costigliola et al. (2017), Tabla 1, p. 9015: 2 analistas en turno en los turnos 1 y 2; turno 3 con 10 analistas
HORAS_SEMANA = {"T1": 56.0, "T2": 37.5, "T3": 40.0}   # con 1 h de almuerzo en los turnos de 8:00 a 17:00 (S24)
BRECHA_MAXIMA_H = 12   # ningún método espera más de un turno de 12 horas a un analista calificado (sección 4.3)


@pytest.fixture(scope="module")
def turnos() -> pd.DataFrame:
    return parametros.cargar_hoja("Turnos").set_index("Código")


@pytest.fixture(scope="module")
def rotacion() -> pd.DataFrame:
    return parametros.cargar_hoja("Rotación").set_index("Día")


@pytest.fixture(scope="module")
def analistas() -> pd.DataFrame:
    return parametros.cargar_hoja("Analistas").set_index("ID")


def cubre(turnos: pd.DataFrame, codigo: str, inicio: int, fin: int) -> bool:
    """True si el código de turno cubre la franja [inicio, fin) del mismo día, sin contar la pausa."""
    if codigo == "L":
        return False
    t = turnos.loc[codigo]
    a, b = t["Inicio (h)"], t["Fin (h)"]
    dentro = (a <= inicio and fin <= b) if b > a else (inicio >= a)   # el turno de noche cubre hasta las 24 h
    en_pausa = t["Pausa hasta (h)"] > t["Pausa desde (h)"] and inicio >= t["Pausa desde (h)"] and fin <= t["Pausa hasta (h)"]
    return bool(dentro and not en_pausa)


def en_turno(turnos, rotacion, dia: int, franja: tuple) -> list:
    inicio, fin = franja
    anterior = CICLO if dia == 1 else dia - 1
    presentes = []
    for analista in rotacion.columns.drop("Día de la semana"):
        if fin <= 8:   # antes de las 8:00 trabaja quien empezó turno de noche el día anterior
            if rotacion.loc[anterior, analista] == "N":
                presentes.append(analista)
        elif cubre(turnos, rotacion.loc[dia, analista], inicio, fin):
            presentes.append(analista)
    return presentes


def test_horas_de_cada_codigo(turnos):
    esperado = {"D": 12, "N": 12, "M2": 8, "T2": 7, "M3a": 8, "M3b": 8, "L": 0}
    for codigo, horas in esperado.items():
        assert turnos.loc[codigo, "Horas efectivas"] == pytest.approx(horas), codigo


def test_rotacion_reproduce_la_tabla_1(rotacion):
    assert len(rotacion) == CICLO
    codigos = rotacion.drop(columns="Día de la semana")
    for dia, fila in codigos.iterrows():
        conteo = fila.value_counts()
        assert conteo.get("D", 0) == 2 and conteo.get("N", 0) == 2, dia
        habil = (dia - 1) % 7 < 5
        assert conteo.get("M2", 0) == (2 if habil else 0), dia
        assert conteo.get("T2", 0) == (2 if habil else 0), dia
        assert conteo.get("M3a", 0) + conteo.get("M3b", 0) == (10 if habil else 0), dia


def test_horas_por_semana(rotacion, turnos, analistas):
    horas = turnos["Horas efectivas"]
    for analista, turno in analistas["Turno"].items():
        semana = rotacion[analista].map(horas).sum() / (CICLO / 7)
        assert semana == pytest.approx(HORAS_SEMANA[turno]), analista


def test_cobertura_en_el_tiempo(turnos, rotacion, analistas):
    catalogo = parametros.cargar_hoja("Catálogo").set_index("ID")
    cubiertas = {m: 0.0 for m in METODOS}
    for dia in range(1, CICLO + 1):
        for franja in FRANJAS:
            presentes = en_turno(turnos, rotacion, dia, franja)
            for m in METODOS:
                if presentes and analistas.loc[presentes, m].sum() > 0:
                    cubiertas[m] += franja[1] - franja[0]
    fraccion = {m: h / (CICLO * 24) for m, h in cubiertas.items()}
    for m in METODOS:
        if catalogo.loc[m, "IN"] == "X":          # control en proceso: a toda hora
            assert fraccion[m] == pytest.approx(1.0), m
        assert fraccion[m] >= 0.70, (m, fraccion[m])



def test_brecha_maxima_sin_analista(turnos, rotacion, analistas):
    horas = [(dia, (h, h + 1)) for dia in range(1, CICLO + 1) for h in range(24)]
    presentes = [en_turno(turnos, rotacion, dia, franja) for dia, franja in horas]
    for m in METODOS:
        cubierto = [bool(p) and analistas.loc[p, m].sum() > 0 for p in presentes]
        mayor = actual = 0
        for c in cubierto + cubierto:   # dos vueltas para contar la brecha que cruza el final del ciclo
            actual = 0 if c else actual + 1
            mayor = max(mayor, actual)
        assert mayor <= BRECHA_MAXIMA_H, (m, mayor)


def test_ventanas_de_mantenimiento():
    mant = parametros.cargar_hoja("Mantenimiento").set_index("Código")
    assert (mant["Fracción del año (plan)"] < 0.01).all()
    assert (mant["Separación entre equipos (días hábiles)"] >= 1).all()
    assert (mant["Mantenimientos preventivos al año"] >= 1).all()
