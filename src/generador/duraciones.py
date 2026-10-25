"""Duraciones de las tareas con BetaPERT (tres puntos: mínimo a, moda m, máximo b).

Cuándo: martes 27 de octubre, 11:00 a. m.–1:00 p. m.
Anteproyecto: Monte Carlo con estimaciones de tres puntos (Hassi et al., 2025);
dimensión «Tiempos de ejecución».
Datos: hoja «Duraciones» (Costigliola et al., 2017, Tabla 3, p. 9016).
"""


def muestrear_betapert(a: float, m: float, b: float, n: int, rng=None):
    """Devuelve n duraciones BetaPERT entre a y b con moda m.

    Pistas:
        alfa = 1 + 4 (m - a) / (b - a)
        beta = 1 + 4 (b - m) / (b - a)
        x = a + (b - a) * Beta(alfa, beta)
        media esperada = (a + 4m + b) / 6
    """
    raise NotImplementedError("Pendiente (martes 27 oct)")
