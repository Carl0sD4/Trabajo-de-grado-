"""Perturbaciones del Alcance II: reanálisis por OOS, fallas de equipos y urgentes no planificadas.

Anteproyecto: Alcance y limitaciones II (incorporar perturbaciones para reducir el riesgo de
circularidad).
Diseño: docs/Generador_de_datos.docx, sección 6.
Datos: hoja «Supuestos», S01 (reanálisis), S02 (fallas), S03 (urgentes) y S13 (reanálisis según la
complejidad); columna «Prob. de reanálisis» de la hoja «Catálogo»; bloque D de la hoja
«Calibración de turnos» (reparación de las fallas).

Ninguna de las cinco fuentes da tasas: Coito et al. (2021, p. 74) confirman que hay fallas y
repeticiones de tareas, y que llegan muestras urgentes con prioridad máxima (p. 65); la OMS pide
sacar de servicio el equipo que falla hasta repararlo y recalificarlo (WHO, 2024, s. 5.40, p. 176).
Por eso cada tasa es un supuesto con tres escenarios de sensibilidad (hoja «Supuestos»).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.generador import parametros

MIN_POR_HORA = 60.0


@lru_cache(maxsize=1)
def tasas_base() -> dict:
    """Propuestas iniciales de S01, S02 y S03 (hoja «Supuestos»)."""
    return {"reanalisis": float(parametros.supuesto("S01")), "fallas": float(parametros.supuesto("S02")),
            "urgentes": float(parametros.supuesto("S03"))}


@lru_cache(maxsize=1)
def _catalogo() -> pd.DataFrame:
    return parametros.cargar_hoja("Catálogo").set_index("ID")


@lru_cache(maxsize=1)
def _puede_ser_urgente() -> dict:
    tipos = parametros.cargar_hoja("Tipos de muestra").set_index("Código")
    return {t: str(v).strip() == "Sí" for t, v in tipos["Puede ser urgente"].items()}


# ===================================================================== urgentes (S03)
def marcar_urgentes(tipos_muestra, rng, tasa=None) -> np.ndarray:
    """True en las muestras que llegan como urgentes: la fracción `tasa` (S03) de los tipos que pueden
    serlo (materia prima, granel y producto terminado; columna «Puede ser urgente»)."""
    rng = np.random.default_rng(rng)
    tasa = tasas_base()["urgentes"] if tasa is None else float(tasa)
    puede = np.array([_puede_ser_urgente().get(t, False) for t in tipos_muestra], dtype=bool)
    return puede & (rng.random(len(puede)) < tasa)


# ===================================================================== reanálisis (S01 y S13)
def probabilidad_reanalisis(metodo: str, tasa=None) -> float:
    """Probabilidad de que una ejecución salga fuera de especificación.

    Es la columna «Prob. de reanálisis» del catálogo: la tasa base de S01 por el multiplicador de la
    complejidad del método (0,5, 1 o 2; S13), normalizado para que el promedio del laboratorio sea S01.
    Con otra tasa (los escenarios 0 % y 5 % de S01), la probabilidad se escala en la misma proporción.
    """
    p = float(_catalogo().at[metodo, "Prob. de reanálisis"])
    if tasa is None:
        return p
    return p * float(tasa) / tasas_base()["reanalisis"]


# ===================================================================== fallas de equipos (S02)
@lru_cache(maxsize=1)
def reparacion() -> tuple:
    """Mínimo, moda y máximo de la reparación de una falla, en horas (S02; hoja «Calibración de turnos»)."""
    return tuple(float(parametros.valor("Calibración de turnos", f"Reparación: {x} (h)")) for x in ("mínimo", "moda", "máximo"))


def tiempo_medio_entre_fallas(fraccion=None) -> float:
    """Horas entre fallas para que el equipo pase la fracción `fraccion` del tiempo en reparación (S02).

    Con reparación media r, la fracción fuera de servicio es r / (MTBF + r), así que MTBF = r (1 − f) / f.
    Con S02 = 5 % y la reparación triangular (8, 48, 168) h da 1 418,7 h: unas 5,9 fallas por equipo al año.
    """
    f = tasas_base()["fallas"] if fraccion is None else float(fraccion)
    if f <= 0:
        return float("inf")
    r = sum(reparacion()) / 3
    return r * (1 - f) / f


def generar_fallas(equipos, minutos: float, rng, fraccion=None) -> pd.DataFrame:
    """Fallas aleatorias de cada unidad en [0, minutos): una fila por falla (equipo, configuración,
    evento = «falla», inicio y fin en minutos).

    Cada unidad alterna tiempos en servicio exponenciales (media tiempo_medio_entre_fallas) y
    reparaciones triangulares (S02), de modo que la fracción esperada del tiempo en falla es S02. Las
    fallas no dependen de que el equipo esté en uso; si una ocurre durante un ensayo, el motor lo
    interrumpe (src/simulacion/entorno.py).
    """
    rng = np.random.default_rng(rng)
    mtbf = tiempo_medio_entre_fallas(fraccion) * MIN_POR_HORA
    a, m, b = (x * MIN_POR_HORA for x in reparacion())
    filas = []
    if np.isfinite(mtbf):
        for e in equipos:
            t = rng.exponential(mtbf)
            while t < minutos:
                fin = t + rng.triangular(a, m, b)
                filas.append({"equipo": e.id, "configuracion": e.configuracion, "evento": "falla",
                              "inicio": float(t), "fin": float(fin)})
                t = fin + rng.exponential(mtbf)
    return pd.DataFrame(filas, columns=["equipo", "configuracion", "evento", "inicio", "fin"])
