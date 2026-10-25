"""Factores de correlación entre variables (complejidad, n.º de soluciones y duración).

Cuándo: martes 27 de octubre, 1:30–4:00 p. m.
Anteproyecto: §1.4 y §5.2 (no asumir independencia por conveniencia);
indicador «correlaciones entre variables definidas».
Datos: hoja «Supuestos», S04, y relación estructural de Costigliola et al. (2017, p. 9019):
preparación = tiempo por muestra × n.º de soluciones.
"""


def inducir_correlacion(muestras, matriz_objetivo, rng=None):
    """Reordena las columnas de `muestras` para aproximar la matriz de correlación objetivo
    (cópula gaussiana o método de Iman-Conover)."""
    raise NotImplementedError("Pendiente (martes 27 oct)")
