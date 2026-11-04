"""Contraste estructural con las instancias de Martins et al. (2023) y del TLSP (src/generador/contraste.py).

Las instancias de terceros no están en el repositorio: estas pruebas usan una instancia mínima con el formato de
Martins et al. y corridas cortas del generador. La última prueba revisa las 54 instancias reales si están bajadas
(python datos/descargar_instancias.py).
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.generador import contraste, generar, registro

SEMANAS = 1


def _operacion(p, analistas, maquinas):
    """Una operación con las tres intervenciones del analista de Martins et al. (2023, p. 191)."""
    pasos = [("setup", 0.0, 0.05 * p), ("intermediate", 0.3 * p, 0.4 * p), ("teardown", 0.9 * p, p)]
    return {"test": 1, "machine_time": p,
            "analyst_times": [{"step": s, "start": a, "end": b} for s, a, b in pasos],
            "suitable_analysts": analistas, "suitable_machines": maquinas}


@pytest.fixture
def carpeta_martins(tmp_path):
    instancia = {"n_jobs": 2, "n_analysts": 2, "n_machines": 3, "jobs": [
        {"job": 1, "product": 0, "release_time": 0, "due_date": 0,
         "tests": [_operacion(2, [1], [1, 2]), _operacion(4, [1, 2], [3])]},
        {"job": 2, "product": 1, "release_time": 0, "due_date": 0,
         "tests": [_operacion(1, [2], [1, 2, 3])]},
    ]}
    (tmp_path / "ex_j2_m3_w2_s2_f0.5_r1.json").write_text(json.dumps(instancia), encoding="utf-8")
    (tmp_path / "LEEME.txt").write_text("no es una instancia", encoding="utf-8")
    return tmp_path


@pytest.fixture(scope="module")
def version_corta():
    """Una corrida corta de cada escenario, como las nueve del dataset pero de una semana."""
    corridas = [generar.generar_corrida(e, 2026, "v0.2", semanas=SEMANAS) for e in ("baja", "base", "alta")]
    registros = pd.concat([r for r, _, _ in corridas], ignore_index=True)
    return registros, [res for _, _, res in corridas]


def test_leer_martins(carpeta_martins):
    instancias, operaciones = contraste.leer_martins(carpeta_martins)
    assert instancias.to_dict("records") == [{
        "instancia": "ex_j2_m3_w2_s2_f0.5_r1", "trabajos": 2, "analistas": 2, "maquinas": 3, "tipos_de_trabajo": 2,
        "flexibilidad": 0.5, "replica": 1, "operaciones": 3}]
    assert operaciones["posicion"].tolist() == [1, 2, 1]
    # el analista está en las tres intervenciones: 5 % + 10 % + 10 % del tiempo de la máquina
    assert np.allclose(operaciones["fraccion_analista"], 0.25)
    assert operaciones["combinaciones"].tolist() == [2, 2, 3]
    assert np.allclose(operaciones["densidad_analistas"], [0.5, 1.0, 0.5])
    assert np.allclose(operaciones["densidad_maquinas"], [2 / 3, 1 / 3, 1.0])
    assert np.allclose(operaciones["densidad_combinaciones"], [2 / 6, 2 / 6, 3 / 6])


def test_leer_martins_sin_instancias(tmp_path):
    with pytest.raises(FileNotFoundError, match="descargar_instancias"):
        contraste.leer_martins(tmp_path)


def test_resumir_martins(carpeta_martins):
    m = contraste.resumir_martins(*contraste.leer_martins(carpeta_martins))
    assert m["tareas_por_trabajo_media"] == 1.5
    assert (m["tareas_por_trabajo_min"], m["tareas_por_trabajo_max"]) == (1, 2)
    assert m["tipos_iguales"]          # un solo trabajo por tipo
    assert m["liberacion_max"] == m["vencimiento_max"] == 0
    assert m["densidad_analistas_f0.5"] == pytest.approx(2 / 3)


def test_tipos_iguales_detecta_trabajos_distintos_del_mismo_tipo(carpeta_martins):
    _, operaciones = contraste.leer_martins(carpeta_martins)
    assert contraste._tipos_iguales(operaciones)
    operaciones.loc[operaciones["trabajo"] == 2, "tipo"] = 0   # ahora el tipo 0 tiene dos trabajos distintos
    assert not contraste._tipos_iguales(operaciones)


def test_medir_dataset(version_corta):
    registros, resumenes = version_corta
    t = contraste.medir_dataset(registros, resumenes)
    lab = contraste.laboratorio()
    assert (lab["analistas"], lab["equipos"], lab["configuraciones"], lab["tecnicas"], lab["turnos"]) == (20, 70, 10, 6, 3)
    assert len(t["corridas"]) == 3 and (t["corridas"]["dias"] == 7).all()
    assert (t["pedidos"]["intento"] == 1).all()
    assert len(t["pedidos"]) == registros["intento"].eq(1).sum()
    assert not t["terminadas"]["resultado"].eq("interrumpido").any()
    # la fracción del analista es la suma de sus tareas sobre la duración, y no pasa de 1
    fraccion = t["terminadas"]["fraccion_analista"]
    assert ((fraccion > 0) & (fraccion <= 1 + registro.TOL)).all()
    # la holgura solo se mide en el primer intento
    assert t["terminadas"].loc[t["terminadas"]["intento"] > 1, "holgura"].isna().all()
    p = t["pedidos"]
    assert np.allclose(p["densidad_analistas"], p["analistas_calificados"] / 20)
    assert np.allclose(p["densidad_combinaciones"], p["combinaciones_factibles"] / 1400)
    assert (p.loc[p["equipo_requerido"] == "HPLC-G|HPLC-I", "configuraciones_compatibles"] == 2).all()


def test_contrastar_y_guardar(version_corta, carpeta_martins, tmp_path):
    registros, resumenes = version_corta
    c = contraste.contrastar(registros, resumenes, carpeta_martins)
    tabla = c["tabla"]
    assert list(tabla.columns) == ["Dimensión", "Medida", *contraste.FUENTES]
    assert {"Trabajos y tareas", "Tipos de recurso", "Densidad de elegibilidad", "Duraciones"} <= set(tabla["Dimensión"])
    assert tabla[list(contraste.FUENTES)].notna().all().all()
    texto = " ".join(tabla.astype(str).to_numpy().ravel())
    assert "—" not in texto                    # sin rayas, como en los documentos
    assert "nan" not in texto.lower().split()
    rutas = contraste.guardar(c, tmp_path)
    assert [r.name for r in rutas] == ["contraste_estructural.csv", "contraste_estructural_medidas.csv",
                                       "contraste_estructural_instancias.csv"]
    assert pd.read_csv(rutas[0]).shape == tabla.shape


def test_tlsp_segun_el_articulo():
    t = contraste.resumir_tlsp_articulo()
    assert (t["proyectos_min"], t["proyectos_max"], t["franjas_min"], t["franjas_max"]) == (5, 90, 88, 782)
    # Tabla 4: 5,49 de 22, 5,33 de 24 y 5,70 de 19 empleados; 3,06 a 3,48 de 17 mesas
    assert t["reales_densidad_empleados_min"] == pytest.approx(5.33 / 24)
    assert t["reales_densidad_empleados_max"] == pytest.approx(5.70 / 19)
    assert t["reales_densidad_mesas_min"] == pytest.approx(3.06 / 17)


def test_rango():
    assert contraste._rango(1, 3) == "1 a 3"
    assert contraste._rango(2, 3) == "2 o 3"
    assert contraste._rango(5, 5) == "5"
    assert contraste._rango(5.33, 5.70, 1) == "5,3 a 5,7"


@pytest.mark.skipif(not any(contraste.CARPETA_MARTINS.glob("ex_*.json")),
                    reason="Faltan las instancias de Martins et al. (python datos/descargar_instancias.py)")
def test_instancias_de_martins():
    """Las 54 instancias publicadas: 7 máquinas, sin liberación ni vencimiento y un tipo de trabajo con operaciones
    fijas (pp. 191 y 192)."""
    instancias, operaciones = contraste.leer_martins()
    assert len(instancias) == 54
    assert set(instancias["maquinas"]) == {7} and set(instancias["analistas"]) == {2, 3, 7}
    assert set(instancias["trabajos"]) == {5, 10, 70} and set(instancias["flexibilidad"]) == {0.3, 0.6}
    assert (operaciones[["liberacion", "vencimiento"]] == 0).all().all()
    assert contraste._tipos_iguales(operaciones)
    assert operaciones["duracion"].between(1, 5).all()
    assert (operaciones[["analistas_elegibles", "maquinas_elegibles"]] >= 1).all().all()
