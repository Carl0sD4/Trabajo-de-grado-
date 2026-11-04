"""Escenarios de carga, semillas y configuración de cada versión del dataset.

Anteproyecto: Alcance II (varias condiciones de carga) y III (mismas condiciones de simulación para
todas las estrategias).
Diseño: docs/Generador_de_datos.docx, sección 7.
Datos: bloque «Escenarios de carga» de la hoja «Demanda» (Costigliola et al., 2017, Tabla 4, p. 9018,
y p. 9019) y supuestos S04, S29 y S30.

Una corrida es un escenario de carga con una semilla. Con la misma semilla salen las mismas
muestras, duraciones, fallas y resultados, así que cualquier corrida se puede repetir. La semilla
alimenta seis generadores independientes (llegadas, urgentes, ensayos, duraciones, resultados y
fallas): al cambiar la regla de despacho en el O3 cambian las asignaciones, pero no las muestras ni
las fallas, y las políticas se comparan en las mismas condiciones.
"""
import numpy as np

SEMILLA_BASE = 2026

ESCENARIOS = {
    "baja": {"multiplicador_carga": 0.7},  # supuesto (hoja Demanda): −30 % respecto a la base
    "base": {"multiplicador_carga": 1.0},  # volumen 2015 de Costigliola et al. (2017), Tabla 4
    "alta": {"multiplicador_carga": 1.5},  # +50 %, Costigliola et al. (2017), p. 9019
}

# Corridas de cada versión del dataset: 3 réplicas de 52 semanas por escenario. Un año cubre más de
# ocho ciclos de la rotación de 42 días y las cuatro ventanas trimestrales de cada equipo, y deja en
# cada corrida más de las 3 500 ejecuciones que necesita el indicador V2 con S04 = 0,3 (generador,
# sección 2.5). Las tres réplicas dan la variación entre corridas de cada indicador.
SEMANAS = 52
REPLICAS = 3
CALENTAMIENTO_DIAS = 14   # el laboratorio arranca vacío: la validación de colas y esperas descarta dos semanas

# Configuración de cada versión. forma: λ de las tareas («hoja» = columna «Forma λ» de la hoja
# Duraciones). rho: ρ entre las tareas del analista (None = S04 de la hoja Supuestos).
# alistamiento: «ninguno» (sin preparación ni idoneidad del sistema), «24h» (la idoneidad vale 24 h
# mientras el equipo siga con el mismo método; Costigliola et al., 2017, p. 9017, supuesto 2, y S29)
# o «siempre» (cada ensayo de HPLC y GC la necesita).
VERSIONES = {
    "v0.1": {"forma": 4.0, "rho": 0.5, "alistamiento": "ninguno",
             "descripcion": "Primera corrida completa con los valores del 29 de octubre: PERT clásica (λ = 4), "
                            "ρ = 0,5 y sin preparación ni idoneidad del sistema, cuyos tres puntos estaban pendientes."},
    "v0.2": {"forma": "hoja", "rho": None, "alistamiento": "24h",
             "descripcion": "Versión validada: λ = 1 (0 en las tareas uniformes), ρ de S04 = 0,3, preparación e "
                            "idoneidad del sistema con la regla de 24 horas (S29) y la regla de despacho S30, que "
                            "espera a una unidad apta mientras haya menos de cinco ensayos en cola por unidad apta."},
}


def semillas(replicas: int = REPLICAS) -> list:
    """Semillas de las réplicas: SEMILLA_BASE, SEMILLA_BASE + 1, ... (las mismas en los tres escenarios)."""
    return [SEMILLA_BASE + k for k in range(int(replicas))]


def corridas(replicas: int = REPLICAS) -> list:
    """Parejas (escenario, semilla) de una versión del dataset."""
    return [(e, s) for e in ESCENARIOS for s in semillas(replicas)]


def generadores(semilla: int) -> dict:
    """Seis generadores de números aleatorios independientes que salen de la misma semilla."""
    nombres = ("llegadas", "urgentes", "ensayos", "duraciones", "resultados", "fallas")
    hijos = np.random.SeedSequence(int(semilla)).spawn(len(nombres))
    return {n: np.random.default_rng(h) for n, h in zip(nombres, hijos)}
