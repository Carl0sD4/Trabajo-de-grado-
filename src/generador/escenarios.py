"""Escenarios de carga y semillas para que todo sea reproducible.

Cuándo: viernes 30 de octubre, 11:00 a. m.–1:00 p. m.
Datos: hoja «Demanda», escenarios de carga.
"""

SEMILLA_BASE = 2026

ESCENARIOS = {
    "baja": {"multiplicador_carga": 0.7},  # supuesto (hoja Demanda)
    "base": {"multiplicador_carga": 1.0},  # volumen 2015 de Costigliola et al. (2017), Tabla 4
    "alta": {"multiplicador_carga": 1.5},  # +50 %, Costigliola et al. (2017), p. 9019
}
