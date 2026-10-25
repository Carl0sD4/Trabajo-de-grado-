"""Analistas, equipos y calendarios de disponibilidad.

Cuándo: miércoles 28 de octubre, 11:00 a. m.–1:00 p. m.
Anteproyecto: dimensiones «Recursos humanos» y «Recursos materiales».
Datos: hoja «Parámetros», bloques «Laboratorio y recursos» y «Elegibilidad y competencias»
(70 equipos de 6 técnicas, 20 analistas, 3 turnos).
Diseñarlo reutilizable: el simulador del O3 (src/simulacion) usa este mismo modelo de recursos.
"""
from dataclasses import dataclass, field

TECNICAS = ("HPLC", "GC", "PSA", "KF", "DSC", "XRPD")


@dataclass
class Analista:
    id: str
    turno: str
    calificaciones: set = field(default_factory=set)  # técnicas o métodos que puede ejecutar


@dataclass
class Equipo:
    id: str
    tecnica: str  # una de TECNICAS


def crear_laboratorio(escenario: dict, rng=None):
    """Crea los analistas y equipos del laboratorio de referencia para un escenario."""
    raise NotImplementedError("Pendiente (miércoles 28 oct)")
