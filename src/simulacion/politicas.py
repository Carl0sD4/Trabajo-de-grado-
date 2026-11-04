"""Reglas de despacho: en qué orden se atienden los ensayos en cola y qué analista y qué equipo los hacen.

Una regla es un objeto con tres métodos que usa el motor (src/simulacion/entorno.py):
- clave(ejecucion): orden de la cola; se atiende primero la clave menor.
- elegir_analista(candidatos, ejecucion, t): uno de los analistas calificados, en turno y libres.
- elegir_equipo(candidatos, ejecucion, t): una de las unidades compatibles, en servicio y libres.

El motor aplica a todas las reglas la de cierre de turno y revisa, en cada despacho, los primeros
ensayos de cada cola (entorno.CABEZAS). El generador del O1 usa ReglaGenerador. Las del O3 se
comparan con el mismo motor, en las mismas corridas (13 al 15 de noviembre):
- FIFO y asignación aleatoria (referencias usadas en la literatura; Martins et al., 2023, pp. 181–182).
- Regla experta: maximum lateness first con las duraciones que estima el supervisor + preferir el
  equipo ya apto para el método (Costigliola et al., 2017, pp. 9017 y 9019; Coito et al., 2021, p. 74).
- Regla informada: prioriza y asigna con las duraciones que estima el modelo del O2.
"""
import numpy as np

from src.generador.recursos import RANGO_NIVEL


class ReglaGenerador:
    """Regla con que el generador del O1 despacha los ensayos (supuesto S30).

    Cola: primero el vencimiento más cercano y, a igualdad, la llegada más temprana; con tiempos de
    proceso parecidos es la regla maximum lateness first del modelo de Costigliola et al. (2017,
    p. 9017). Las muestras urgentes y de control en proceso, con plazo de 4 horas, pasan adelante de
    las que vencen después, pero no de las que ya vencieron o vencen antes.
    Analista: el de menor nivel que cumple la calificación exigida, para no ocupar a los sénior en
    lo que puede hacer otro; a igualdad, el que lleva más tiempo libre. Equipo: el que ya está apto
    para el método, porque su idoneidad sigue vigente (Costigliola et al., 2017, p. 9019); si no hay,
    una que no esté apta para otro método, de la configuración principal y, a igualdad, la que lleva
    más tiempo libre. Si ninguna unidad libre está apta, los ensayos esperan a una apta mientras haya
    menos de cinco en cola por unidad apta (esperar_unidad_apta), como la heurística de esa fuente.
    El cierre de turno lo aplica el motor (entorno.Simulador._cabe_en_turno).
    """

    nombre = "generador"
    COLA_POR_UNIDAD_APTA = 5   # Costigliola et al. (2017, p. 9019)

    def clave(self, ejecucion):
        return (ejecucion.vencimiento, ejecucion.llegada, ejecucion.secuencia)

    def elegir_analista(self, candidatos, ejecucion, t):
        return min(candidatos, key=lambda a: (RANGO_NIVEL[a.nivel], a.libre_desde, a.id))

    def elegir_equipo(self, candidatos, ejecucion, t):
        return min(candidatos, key=lambda e: (not e.apto_para(ejecucion.metodo, t),
                                              e.apto_metodo is not None and e.apto_para(e.apto_metodo, t),
                                              e.configuracion != ejecucion.principal, e.libre_desde, e.id))

    def esperar_unidad_apta(self, aptas, en_cola):
        """Si ninguna unidad libre está apta para el método, sus ensayos esperan a que se libere una de las aptas
        mientras haya menos de cinco en cola por cada unidad apta; con cinco o más, se alista otra unidad. Es la
        heurística de Costigliola et al. (2017, p. 9019): el equipo ya apto, salvo que tenga cinco o más muestras en
        cola."""
        return aptas > 0 and en_cola < self.COLA_POR_UNIDAD_APTA * aptas


class FIFO:
    """Primero en llegar, primero en ser atendido; el analista y el equipo que llevan más tiempo libres."""

    nombre = "FIFO"

    def clave(self, ejecucion):
        return (ejecucion.llegada, ejecucion.secuencia)

    def elegir_analista(self, candidatos, ejecucion, t):
        return min(candidatos, key=lambda a: (a.libre_desde, a.id))

    def elegir_equipo(self, candidatos, ejecucion, t):
        return min(candidatos, key=lambda e: (e.libre_desde, e.id))


class Aleatoria:
    """Orden de llegada y analista y equipo al azar entre los factibles."""

    nombre = "aleatoria"

    def __init__(self, semilla=None):
        self.rng = np.random.default_rng(semilla)

    def clave(self, ejecucion):
        return (ejecucion.llegada, ejecucion.secuencia)

    def elegir_analista(self, candidatos, ejecucion, t):
        candidatos = sorted(candidatos, key=lambda a: a.id)
        return candidatos[int(self.rng.integers(len(candidatos)))]

    def elegir_equipo(self, candidatos, ejecucion, t):
        candidatos = sorted(candidatos, key=lambda e: e.id)
        return candidatos[int(self.rng.integers(len(candidatos)))]


def regla_experta(*args, **kwargs):
    raise NotImplementedError("Pendiente (O3, 14 nov): necesita las duraciones que estima el supervisor.")


def regla_informada(*args, **kwargs):
    raise NotImplementedError("Pendiente (O3, 15 nov): necesita el modelo seleccionado en el O2.")
