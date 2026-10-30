"""Duraciones con BetaPERT (tres puntos: mínimo a, moda m, máximo b), de cada tarea y de un ensayo.

Anteproyecto: Monte Carlo con estimaciones de tres puntos (Hassi et al., 2025); dimensión
«Tiempos de ejecución» (indicadores T1 a T5 y V1 de la hoja «Indicadores»).
Datos: hoja «Duraciones» (Costigliola et al., 2017, Tabla 3, p. 9016).
Diseño: docs/Generador_de_datos.docx, sección 1.

La BetaPERT es una beta reescalada al intervalo [a, b], con un parámetro de forma λ:

    alfa = 1 + λ (m − a) / (b − a)        beta = 1 + λ (b − m) / (b − a)
    x = a + (b − a) · Beta(alfa, beta)
    media = (a + λ·m + b) / (λ + 2)       varianza = (media − a)(b − media) / (λ + 3)

Con λ = 4 es la PERT clásica del anteproyecto: media = (a + 4m + b) / 6 y varianza =
(media − a)(b − media) / 7. Su moda es m para cualquier λ > 0. Otros dos valores sirven para
comparar con la fuente: con λ = 1 la media es la de la triangular de Costigliola et al.,
(a + m + b) / 3 (hoja «Léeme», ojo 4), y con λ = 0 la distribución es la uniforme U(a, b), la que
la fuente da para el montaje de PSA y la corrida de DSC.

La duración de un ensayo (muestrear_ensayo) suma sus tareas como registro.duracion: la idoneidad
del sistema corre en paralelo con la preparación del análisis. Las tareas pueden salir
correlacionadas entre sí (bloque de factores de correlación, src/generador/correlacion.py).
"""
import numpy as np
import pandas as pd
from scipy import stats

from src.generador import registro
from src.generador.correlacion import como_matriz, uniformes_correlacionadas

FORMA_PERT = 4.0   # λ de la PERT clásica


def _tres_puntos(a, m, b, forma):
    """Convierte los tres puntos y la forma en arreglos de la misma forma y verifica que tengan sentido."""
    a, m, b, forma = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (a, m, b, forma)))
    if not (np.isfinite(a).all() and np.isfinite(m).all() and np.isfinite(b).all()):
        raise ValueError("Los tres puntos deben ser números finitos.")
    if (a > m).any() or (m > b).any():
        raise ValueError("Los tres puntos deben cumplir a ≤ m ≤ b.")
    if not (np.isfinite(forma) & (forma >= 0)).all():
        raise ValueError("La forma λ debe ser un número mayor o igual que 0.")
    return a, m, b, forma


def _como_salida(x):
    """Un número si el resultado es un solo valor; un arreglo si son varios."""
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def parametros_beta(a, m, b, forma=FORMA_PERT):
    """alfa y beta de la beta que, reescalada a [a, b], tiene moda m."""
    a, m, b, forma = _tres_puntos(a, m, b, forma)
    rango = np.where(b > a, b - a, 1.0)   # si a = b la duración es fija y cualquier beta sirve
    alfa, beta = 1 + forma * (m - a) / rango, 1 + forma * (b - m) / rango
    return _como_salida(alfa), _como_salida(beta)


def media_betapert(a, m, b, forma=FORMA_PERT):
    """Media de la BetaPERT: (a + λ·m + b) / (λ + 2); con λ = 4, (a + 4m + b) / 6."""
    a, m, b, forma = _tres_puntos(a, m, b, forma)
    return _como_salida((a + forma * m + b) / (forma + 2))


def varianza_betapert(a, m, b, forma=FORMA_PERT):
    """Varianza exacta de la BetaPERT: (media − a)(b − media) / (λ + 3).

    La regla «desviación = (b − a) / 6» de los libros de PERT es solo una aproximación.
    """
    a, m, b, forma = _tres_puntos(a, m, b, forma)
    media = (a + forma * m + b) / (forma + 2)
    return _como_salida((media - a) * (b - media) / (forma + 3))


def muestrear_betapert(a, m, b, n=None, rng=None, forma=FORMA_PERT):
    """Duraciones BetaPERT entre a y b con moda m.

    a, m, b: números, o arreglos con la misma forma para muestrear muchos casos a la vez
        (por ejemplo, la preparación de todos los ensayos de una corrida).
    n: cuántos valores por caso. Si se omite, sale uno por caso: un número si a, m y b son
        números, o un arreglo con su forma. Si se da, sale un arreglo de forma (n, *forma de a).
    rng: generador de números aleatorios de numpy, semilla entera o None. Con la misma semilla
        salen los mismos valores, lo que hace reproducible cada corrida.
    forma: λ; 4 es la PERT clásica, 1 da la media de la triangular y 0 da la uniforme. Puede ser
        un arreglo, para dar a cada caso su propia forma.
    """
    alfa, beta = parametros_beta(a, m, b, forma)
    a, m, b, _ = _tres_puntos(a, m, b, forma)
    rng = np.random.default_rng(rng)
    tamano = a.shape if n is None else (int(n),) + a.shape
    return _como_salida(a + (b - a) * rng.beta(alfa, beta, size=tamano))


def cuantil_betapert(u, a, m, b, forma=FORMA_PERT):
    """Duración que deja la probabilidad u por debajo (inversa de la función de distribución).

    Sirve para muestrear a partir de uniformes ya correlacionadas, como las de una cópula
    gaussiana o del método de Iman-Conover (bloque de factores de correlación).
    """
    u = np.asarray(u, dtype=float)
    if not ((u >= 0) & (u <= 1)).all():
        raise ValueError("Las probabilidades u deben estar entre 0 y 1.")
    alfa, beta = parametros_beta(a, m, b, forma)
    a, m, b, _ = _tres_puntos(a, m, b, forma)
    return _como_salida(a + (b - a) * stats.beta.ppf(u, alfa, beta))


def muestrear_ensayo(metodo, n_soluciones, n=1, rng=None, forma=FORMA_PERT, nivel="intermedio",
                     prep_sistema=(0.0, 0.0, 0.0), idoneidad=(0.0, 0.0, 0.0), correlacion=None) -> pd.DataFrame:
    """Duración y tareas de n ejecuciones de un ensayo, una por fila, con los nombres de campo del
    diccionario: duracion_simulada y t_prep_sistema a t_procesamiento.

    Cada tarea sale de su BetaPERT con los tres puntos de registro.tres_puntos_tareas, los mismos de
    t_optimista, t_mas_probable y t_pesimista, y la duración las suma con registro.duracion.

    metodo, n_soluciones: el ensayo (M01 a M12) y cuántas soluciones se preparan, dentro del rango
        de su complejidad (S08). La preparación es un solo valor del tiempo por solución
        multiplicado por n_soluciones (diccionario, campo t_preparacion; S04).
    n: cuántas ejecuciones.
    rng: como en muestrear_betapert.
    forma: λ de todas las tareas, o un diccionario {campo: λ} para las tareas que lleven otra
        forma; las que falten usan FORMA_PERT. Por ejemplo, {"t_corrida": 0} deja uniforme la
        corrida de DSC, como en la fuente.
    nivel: nivel del analista. Su factor de duración (S07) multiplica las tareas del analista; la
        idoneidad y la corrida son del equipo y no cambian.
    prep_sistema, idoneidad: tres puntos de la preparación y la idoneidad del sistema, solo si la
        ejecución las necesita (HPLC y GC); por omisión, cero.
    correlacion: correlación de rangos (ρ de Spearman) entre las tareas. Con None salen
        independientes, como se comparan con la fuente, que da una distribución por tarea y no dice
        nada de su dependencia. Con un número ρ, las cuatro tareas del
        analista quedan con ese ρ entre sí y los demás pares, como en la hoja «Correlaciones» (hoy,
        independientes; S04: 0,5, con sensibilidad de 0,3 y 0,7). También recibe una matriz de
        6 × 6 en el orden de registro.TAREAS, por ejemplo correlacion.matriz_objetivo(), que es la
        que debe pasar el generador. Las tareas salen entonces de una cópula gaussiana y de
        cuantil_betapert, con las mismas distribuciones de cada tarea.
    """
    puntos = registro.tres_puntos_tareas(metodo, n_soluciones, prep_sistema, idoneidad)
    factor = registro.factor_duracion(nivel)
    if isinstance(forma, dict):
        if set(forma) - set(registro.TAREAS):
            raise ValueError(f"Tareas desconocidas en la forma: {sorted(set(forma) - set(registro.TAREAS))}.")
        forma = [forma.get(t, FORMA_PERT) for t in registro.TAREAS]
    a, m, b = (np.array([puntos[t][k] for t in registro.TAREAS]) for k in range(3))
    if correlacion is None:
        x = muestrear_betapert(a, m, b, n=n, rng=rng, forma=forma)
    else:
        u = uniformes_correlacionadas(n, como_matriz(correlacion), rng=rng)
        x = cuantil_betapert(u, a, m, b, forma=forma)
    tareas = pd.DataFrame(x, columns=registro.TAREAS)
    tareas[list(registro.TAREAS_ANALISTA)] *= factor
    tareas.insert(0, "duracion_simulada", registro.duracion(*(tareas[t] for t in registro.TAREAS)))
    return tareas
