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
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from src.generador import parametros, registro
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
        analista quedan con ese ρ entre sí y los demás pares, independientes, como en la hoja
        «Correlaciones» (S04: 0,3, con sensibilidad de 0,5 y 0,7). También recibe una matriz de
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


# ===================================================================== ensayos del generador (secciones 3 a 7)
TAREA_HOJA = {"t_preparacion": "Preparación de muestra", "t_montaje": "Montaje del equipo",
              "t_corrida": "Corrida", "t_procesamiento": "Procesamiento de datos"}
TAREAS_ALISTAMIENTO = ("t_prep_sistema", "t_idoneidad")


@lru_cache(maxsize=None)
def formas_por_tarea(tecnica: str, forma="hoja") -> tuple:
    """λ de cada tarea de un ensayo de la técnica, en el orden de registro.TAREAS.

    forma: un número, que se usa en las seis tareas (4 es la PERT clásica), o «hoja», que toma la
    columna «Forma λ» de la hoja «Duraciones» (la propuesta del bloque de duraciones: λ = 1, y λ = 0
    en las tareas uniformes de la fuente). La preparación y la idoneidad del sistema usan la λ de la
    corrida de su técnica, porque sus tres puntos siguen la misma regla de ±20 % (S29 y S05).
    """
    if forma != "hoja":
        return (float(forma),) * len(registro.TAREAS)
    d = parametros.cargar_hoja("Duraciones")
    if "Forma λ" not in d.columns:
        raise KeyError("La hoja «Duraciones» no tiene la columna «Forma λ».")
    d = d[d["Tarea"].isin(TAREA_HOJA.values())].set_index(["Técnica", "Tarea"])
    lam = {t: float(d.at[(tecnica, nombre), "Forma λ"]) for t, nombre in TAREA_HOJA.items()}
    lam["t_prep_sistema"] = lam["t_idoneidad"] = lam["t_corrida"]
    return tuple(lam[t] for t in registro.TAREAS)


@lru_cache(maxsize=1)
def tiempos_promedio_fuente() -> pd.DataFrame:
    """Tiempos promedio por tarea del modelo de Costigliola et al. (2017, Tabla 6, p. 9019), en minutos.

    Una fila por técnica y una columna por tarea, con los nombres de registro.TAREAS salvo el montaje,
    que la tabla no separa: t_prep_sistema, t_idoneidad, t_preparacion, t_corrida y t_procesamiento.
    Vacío donde la fuente marca n.a. Está en la hoja «Duraciones», debajo de la tabla principal.
    """
    hoja = parametros._hoja_cruda("Duraciones")   # noqa: SLF001
    fila_enc = hoja.index[hoja[0].astype(str).eq("Técnica") & hoja[1].astype(str).eq("Prep. del sistema")]
    if len(fila_enc) != 1:
        raise KeyError("No encontré la tabla de tiempos promedio de Costigliola et al. (2017, Tabla 6) en la hoja «Duraciones».")
    k = fila_enc[0]
    tabla = hoja.iloc[k + 1:k + 7, :6].set_index(0)

    def promedio(texto):
        texto = str(texto).split("±")[0].strip().replace(",", ".")
        return float(texto) if texto.replace(".", "").isdigit() else np.nan

    tabla = tabla.apply(lambda col: col.map(promedio))
    tabla.columns = ["t_prep_sistema", "t_idoneidad", "t_preparacion", "t_corrida", "t_procesamiento"]
    tabla.index.name = "tecnica"
    return tabla.astype(float)


@lru_cache(maxsize=None)
def tres_puntos_alistamiento(tecnica: str, margen=None) -> tuple:
    """Tres puntos de la preparación y de la idoneidad del sistema de una técnica (S29).

    La moda es el tiempo promedio del modelo de Costigliola et al. (2017, Tabla 6, p. 9019), que está
    en la hoja «Duraciones», y el mínimo y el máximo quedan a ±margen de la moda (S29 propone 20 %,
    la misma regla de la corrida de HPLC y GC, S05). Solo HPLC y GC tienen alistamiento; en las demás
    técnicas devuelve ceros.
    """
    if tecnica not in registro.TECNICAS_CON_IDONEIDAD:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    margen = float(parametros.supuesto("S29")) if margen is None else float(margen)
    fuente = tiempos_promedio_fuente()
    sis, ido = float(fuente.at[tecnica, "t_prep_sistema"]), float(fuente.at[tecnica, "t_idoneidad"])
    return ((sis * (1 - margen), sis, sis * (1 + margen)), (ido * (1 - margen), ido, ido * (1 + margen)))


def medias_tareas(puntos: dict, formas) -> dict:
    """Media BetaPERT de cada tarea, para estimar la duración antes de ejecutarla (regla de despacho)."""
    return {t: float(media_betapert(*puntos[t], forma=f)) for t, f in zip(registro.TAREAS, formas)}


def tareas_desde_uniformes(u, puntos: dict, formas, factor: float) -> dict:
    """Las seis tareas de una ejecución a partir de seis uniformes ya correlacionadas (cópula gaussiana).

    Cada tarea es el cuantil de su BetaPERT (cuantil_betapert), como en muestrear_ensayo con
    correlación, y las tareas del analista se multiplican por el factor de su nivel (S07).
    """
    a, m, b = (np.array([puntos[t][k] for t in registro.TAREAS]) for k in range(3))
    x = np.asarray(cuantil_betapert(np.asarray(u, dtype=float), a, m, b, forma=np.asarray(formas, dtype=float)))
    tareas = dict(zip(registro.TAREAS, (float(v) for v in x)))
    for t in registro.TAREAS_ANALISTA:
        tareas[t] *= factor
    return tareas
