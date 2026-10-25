"""Políticas de asignación a comparar.

Cuándo: 13 al 15 de noviembre.
- FIFO y asignación aleatoria (referencias usadas en la literatura).
- Regla experta: maximum lateness first + preferir el equipo ya apto para el método
  (Costigliola et al., 2017, pp. 9017 y 9019).
- Regla informada: prioriza y asigna con las duraciones que estima el modelo del O2.
"""


def fifo(cola, recursos_libres):
    raise NotImplementedError("Pendiente (14 nov)")


def aleatoria(cola, recursos_libres, rng=None):
    raise NotImplementedError("Pendiente (14 nov)")


def regla_experta(cola, recursos_libres):
    raise NotImplementedError("Pendiente (14 nov)")


def regla_informada(cola, recursos_libres, modelo):
    raise NotImplementedError("Pendiente (15 nov)")
