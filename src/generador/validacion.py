"""Validación del dataset con los indicadores del sistema de variables.

Cuándo: sábado 31 de octubre.
Dimensiones: tiempos de ejecución, recursos humanos, recursos materiales, disponibilidad cruzada
y variabilidad y consistencia estructural (% de registros que cumplen las restricciones,
% de valores fuera de rango y % de parámetros con trazabilidad bibliográfica).
Chequeos externos: el 90 % de los ensayos dura entre 458 y 786 min en total
(Coito et al., 2021, p. 74); un día normal tiene 30–40 muestras (Cunha et al., 2019, p. 1424).
Qué calcular: los 22 indicadores de la hoja «Indicadores» del Excel de parámetros, cada uno con sus
campos, su nivel (registro, corrida, analista o equipo) y su referencia (docs/Diccionario_de_datos.docx,
sección 3). V3 y V4 salen de src/generador/registro.py (chequear_registro) y de la hoja «Diccionario»;
H1, H3, H4, M1, M3 y M4 necesitan además el calendario de turnos y la tabla de eventos de los equipos.
"""


def indicadores(dataset):
    """Calcula los indicadores de las cinco dimensiones del sistema de variables."""
    raise NotImplementedError("Pendiente (sábado 31 oct)")
