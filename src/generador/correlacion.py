"""Factores de correlación entre las tareas de un ensayo y su verificación.

Anteproyecto: §1.4 (Monte Carlo con tres puntos y factores de correlación) y §5.2 (incorporar la
correlación entre variables en lugar de asumir independencia por conveniencia); indicador V2,
«correlaciones entre variables definidas», de la hoja «Indicadores».
Datos: hoja «Correlaciones» del Excel de parámetros, que toma el ρ del supuesto S04.
Diseño: docs/Generador_de_datos.docx, sección 2.

Qué se correlaciona. La dependencia principal entre complejidad, número de soluciones y duración
es estructural y no se induce: el número de soluciones sale del rango de la complejidad (S08) y la
preparación es el tiempo por solución por el número de soluciones (Costigliola et al., 2017,
p. 9019). La correlación se induce solo en el ruido, entre las tareas de una misma ejecución: las
cuatro del analista con ρ = 0,5 (S04, con sensibilidad de 0,3 y 0,7) y la idoneidad y la corrida,
que son del equipo, independientes. Todos los ρ son correlaciones de rangos (Spearman).

Dos maneras de inducirla:

- Cópula gaussiana (uniformes_correlacionadas): normales con la matriz de correlación
  r = 2 sen(πρ/6), que da exactamente la ρ de Spearman pedida (Clemen y Reilly, 1999, p. 211),
  llevadas a uniformes con la función de distribución normal; duraciones.cuantil_betapert las
  convierte en minutos. Es la que usa duraciones.muestrear_ensayo.
- Iman y Conover (1982) (inducir_correlacion): reordena columnas ya generadas, sin cambiar sus
  valores, para que sus rangos sigan a los de unos puntajes normales con esa misma matriz r.

Verificación (indicador V2): la correlación de rangos dentro de grupos de registros comparables,
frente a la que se espera con la matriz objetivo y el tamaño de cada grupo, con una tolerancia de
cinco errores estándar (verificar_por_grupos, veredicto_v2, indicador_v2).
"""
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from src.generador import parametros, registro

HOJA = "Correlaciones"
GRUPOS_V2 = ("metodo", "n_soluciones", "nivel_analista", "idoneidad_ejecutada")   # indicador V2
ESTRUCTURA_V2 = ("complejidad", "n_soluciones", "t_preparacion", "duracion_simulada")
ORDEN_COMPLEJIDAD = {"baja": 1, "media": 2, "alta": 3}
MINIMO_POR_GRUPO = 10   # registros por grupo
Z_V2 = 5          # tolerancia de V2 en errores estándar, el mismo margen de las pruebas de medias
Z_POTENCIA = 2    # con Z_V2 + 2 errores estándar, un par detecta la correlación que debe ver el 97,7 % de las veces
RHO_DETECTABLE = 0.3   # la menor correlación que V2 debe poder ver si la objetivo es cero: el escenario más bajo de S04
TOL = 1e-9


# ===================================================================== conversión entre medidas
def rho_normal(rho_spearman):
    """Correlación de la normal que da la correlación de rangos ρ: r = 2 sen(πρ/6)."""
    return 2 * np.sin(np.pi * np.asarray(rho_spearman, dtype=float) / 6)


def rho_spearman(r):
    """Correlación de rangos de dos normales con correlación r: ρ = (6/π) arcsen(r/2)."""
    return 6 / np.pi * np.arcsin(np.asarray(r, dtype=float) / 2)


def rho_spearman_esperada(rho, n):
    """Valor esperado de la correlación de rangos de n pares de una cópula gaussiana con correlación de
    rangos ρ: E(r_s) = 6/(π(n + 1)) · [arcsen(r) + (n − 2) arcsen(r/2)], con r = 2 sen(πρ/6) (Moran, 1948).

    Con pocos pares sale más cerca de cero que ρ (con ρ = 0,5 y 10 pares, 0,458); con muchos tiende a ρ.
    """
    r = rho_normal(rho)
    n = np.asarray(n, dtype=float)
    return 6 / (np.pi * (n + 1)) * (np.arcsin(r) + (n - 2) * np.arcsin(r / 2))


def varianza_rho_spearman(n):
    """Varianza de la correlación de rangos de n pares independientes: 1/(n − 1), la exacta (Xu et al.,
    2010, ec. 20, con ρ = 0).

    Con correlación la varianza es menor (tests/test_correlacion.py lo comprueba por simulación), así
    que sirve de cota: con ella, el error estándar de V2 es exacto en los pares que deben ser
    independientes y mayor que el real en los correlacionados (entre un 20 y un 30 % con ρ = 0,5, y
    entre un 60 y un 80 % con ρ = 0,7).
    """
    return 1 / (np.asarray(n, dtype=float) - 1)


# ===================================================================== matriz objetivo
def validar_matriz(matriz) -> np.ndarray:
    """La matriz como arreglo, si sirve como matriz de correlación de rangos para la cópula.

    Debe ser cuadrada, simétrica, con unos en la diagonal, valores entre −1 y 1, y la matriz de la
    normal que le corresponde (rho_normal) debe ser definida positiva.
    """
    m = np.asarray(matriz, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] == 0:
        raise ValueError("La matriz de correlación debe ser cuadrada.")
    if not np.isfinite(m).all():
        raise ValueError("La matriz de correlación tiene valores vacíos o infinitos.")
    if not np.allclose(m, m.T, atol=TOL):
        raise ValueError("La matriz de correlación debe ser simétrica.")
    if not np.allclose(np.diag(m), 1, atol=TOL):
        raise ValueError("La diagonal de la matriz de correlación debe ser 1.")
    if (np.abs(m) > 1 + TOL).any():
        raise ValueError("Las correlaciones deben estar entre −1 y 1.")
    _cholesky(rho_normal(m))
    return m


def _cholesky(r, mensaje="La matriz de correlación no es definida positiva: esas correlaciones no pueden darse a la vez."):
    try:
        return np.linalg.cholesky(r)
    except np.linalg.LinAlgError:
        raise ValueError(mensaje) from None


@lru_cache(maxsize=1)
def _hoja() -> pd.DataFrame:
    hoja = parametros.cargar_hoja(HOJA)
    return hoja[hoja["Tarea"].isin(registro.TAREAS)].set_index("Tarea")


def matriz_objetivo(rho=None) -> pd.DataFrame:
    """Matriz objetivo de correlación de rangos entre las tareas (hoja «Correlaciones»).

    Filas y columnas en el orden de registro.TAREAS. Con `rho`, los pares de tareas del analista
    toman ese valor en lugar del de S04 (por ejemplo, 0,3 o 0,7 para la sensibilidad); los demás
    quedan como en la hoja.
    """
    m = _hoja().loc[registro.TAREAS, registro.TAREAS].astype(float)
    if rho is not None:
        for a in registro.TAREAS_ANALISTA:
            for b in registro.TAREAS_ANALISTA:
                if a != b:
                    m.loc[a, b] = float(rho)
    m.index.name = m.columns.name = None
    validar_matriz(m)
    return m


def como_matriz(correlacion) -> pd.DataFrame:
    """La matriz objetivo de las tareas a partir de un número (el ρ entre las tareas del analista; los
    demás pares, como en la hoja) o de una matriz de 6 × 6, con nombres de tarea o en el orden de
    registro.TAREAS."""
    if isinstance(correlacion, bool):
        raise TypeError("La correlación debe ser un número o una matriz, no True o False.")
    if np.ndim(correlacion) == 0:
        return matriz_objetivo(rho=float(correlacion))
    if isinstance(correlacion, pd.DataFrame):
        faltan = sorted((set(registro.TAREAS) - set(correlacion.index)) | (set(registro.TAREAS) - set(correlacion.columns)))
        if faltan:
            raise ValueError(f"Faltan tareas en la matriz de correlación: {faltan}.")
        m = correlacion.loc[registro.TAREAS, registro.TAREAS].astype(float)
    else:
        arreglo = np.asarray(correlacion, dtype=float)
        if arreglo.shape != (len(registro.TAREAS),) * 2:
            raise ValueError(f"La matriz de correlación debe ser de {len(registro.TAREAS)} × "
                             f"{len(registro.TAREAS)}, en el orden de registro.TAREAS.")
        m = pd.DataFrame(arreglo, index=registro.TAREAS, columns=registro.TAREAS)
    validar_matriz(m)
    return m


# ===================================================================== inducir la correlación
def uniformes_correlacionadas(n, matriz, rng=None) -> np.ndarray:
    """n filas de uniformes entre 0 y 1 cuya correlación de rangos es la matriz (cópula gaussiana).

    Cada columna sigue siendo uniforme, así que cualquier distribución se obtiene con su cuantil
    (por ejemplo, duraciones.cuantil_betapert) sin cambiar la correlación de rangos.
    """
    raiz = _cholesky(rho_normal(validar_matriz(matriz)))
    rng = np.random.default_rng(rng)
    z = rng.standard_normal((int(n), raiz.shape[0])) @ raiz.T
    return stats.norm.cdf(z)


def inducir_correlacion(muestras, matriz_objetivo, rng=None):
    """Reordena las columnas de `muestras` para que su correlación de rangos se acerque a la matriz
    objetivo, con el método de Iman y Conover (1982).

    Cada columna conserva exactamente sus valores; solo cambia el orden en que se emparejan.
    Puntajes de van der Waerden, Φ⁻¹(i/(n + 1)), permutados al azar por columna; se les quita la
    correlación que traen por azar, se les da una correlación y cada columna de `muestras` toma los
    rangos de su columna de puntajes. Única diferencia con el artículo: la correlación que se da a
    los puntajes es la de la normal que corresponde a la objetivo, r = 2 sen(πρ/6), y no la objetivo
    misma; así la correlación de rangos llega a ρ (sin ese paso, ρ = 0,5 queda en 0,48).

    muestras: arreglo de n × k o DataFrame, sin valores vacíos; si es un DataFrame, la matriz puede
        traer los nombres de sus columnas. Sale un objeto del mismo tipo y con el mismo índice, para
        poder asignarlo de vuelta, aunque las filas sean combinaciones nuevas.
    matriz_objetivo: correlación de rangos de k × k.
    rng: generador de números aleatorios de numpy, semilla entera o None.
    """
    es_tabla = isinstance(muestras, pd.DataFrame)
    x = np.asarray(muestras, dtype=float)
    if x.ndim != 2:
        raise ValueError("Las muestras deben ser una tabla de n filas por k columnas.")
    if not np.isfinite(x).all():
        raise ValueError("Las muestras tienen valores vacíos o infinitos.")
    n, k = x.shape
    if es_tabla and isinstance(matriz_objetivo, pd.DataFrame):
        matriz_objetivo = matriz_objetivo.loc[muestras.columns, muestras.columns]
    objetivo = validar_matriz(matriz_objetivo)
    if objetivo.shape[0] != k:
        raise ValueError(f"La matriz es de {objetivo.shape[0]} × {objetivo.shape[0]} y hay {k} columnas.")
    if n <= k:
        raise ValueError("Hacen falta más filas que columnas para inducir la correlación.")
    rng = np.random.default_rng(rng)
    puntajes = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))
    s = np.column_stack([rng.permutation(puntajes) for _ in range(k)])
    azar = _cholesky(np.corrcoef(s, rowvar=False),
                     "Los puntajes al azar salieron linealmente dependientes, lo que pasa con muy pocas filas: "
                     "use más filas u otra semilla.")
    t = s @ np.linalg.inv(azar).T @ _cholesky(rho_normal(objetivo)).T
    rangos = t.argsort(axis=0).argsort(axis=0)
    salida = np.sort(x, axis=0)[rangos, np.arange(k)]
    return pd.DataFrame(salida, index=muestras.index, columns=muestras.columns) if es_tabla else salida


# ===================================================================== verificar
def correlacion_rangos(datos) -> pd.DataFrame:
    """Matriz de correlación de rangos (Spearman) entre las columnas; NaN en las columnas fijas."""
    return pd.DataFrame(datos).rank().corr()


def comparar(obtenida, objetivo) -> pd.DataFrame:
    """Una fila por par de columnas con la correlación objetivo, la obtenida y la diferencia.

    Los pares con una columna fija (correlación obtenida vacía) quedan fuera.
    """
    obtenida = pd.DataFrame(obtenida)
    objetivo = pd.DataFrame(objetivo).loc[obtenida.index, obtenida.columns]
    filas = []
    columnas = list(obtenida.columns)
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            if pd.notna(obtenida.at[a, b]):
                filas.append({"variable_1": a, "variable_2": b, "objetivo": float(objetivo.at[a, b]),
                              "obtenida": float(obtenida.at[a, b]),
                              "diferencia": float(obtenida.at[a, b] - objetivo.at[a, b])})
    return pd.DataFrame(filas, columns=["variable_1", "variable_2", "objetivo", "obtenida", "diferencia"])


def error_maximo(obtenida, objetivo) -> float:
    """La mayor diferencia, en valor absoluto, entre la correlación obtenida y la objetivo."""
    return float(comparar(obtenida, objetivo)["diferencia"].abs().max())


def _grupos(registros, columnas, por, minimo):
    """Tamaño y matriz de correlación de rangos de cada grupo de al menos `minimo` registros."""
    for _, grupo in registros.groupby(list(por), dropna=False, sort=False):
        if len(grupo) >= minimo:
            yield len(grupo), correlacion_rangos(grupo[columnas]).to_numpy()


def correlacion_por_grupos(registros, columnas=None, por=GRUPOS_V2, minimo=MINIMO_POR_GRUPO) -> pd.DataFrame:
    """Correlación de rangos dentro de grupos de registros, promediada con el número de registros.

    Dentro de un grupo del mismo método, número de soluciones, nivel del analista e idoneidad
    ejecutada, las tareas tienen los mismos tres puntos y el mismo factor, así que su correlación de
    rangos es la que indujo la cópula. Mezclar grupos la cambia (las tareas de un método largo salen
    largas a la vez). En cada par se omiten los grupos de menos de `minimo` registros y aquellos en
    que alguna de las dos columnas es fija.
    """
    columnas = list(registro.TAREAS if columnas is None else columnas)
    k = len(columnas)
    suma, peso = np.zeros((k, k)), np.zeros((k, k))
    for n, r in _grupos(registros, columnas, por, minimo):
        hay = ~np.isnan(r)
        suma[hay] += n * r[hay]
        peso[hay] += n
    with np.errstate(invalid="ignore", divide="ignore"):
        promedio = np.where(peso > 0, suma / np.where(peso > 0, peso, 1), np.nan)
    return pd.DataFrame(promedio, index=columnas, columns=columnas)


def verificar_por_grupos(registros, objetivo, columnas=None, por=GRUPOS_V2, minimo=MINIMO_POR_GRUPO,
                         z=Z_V2) -> pd.DataFrame:
    """Compara, par por par, la correlación de rangos dentro de los grupos con la que da la objetivo.

    obtenida: la de correlacion_por_grupos. esperada: el mismo promedio de rho_spearman_esperada con
    el tamaño de cada grupo, porque con grupos chicos la correlación sale más cerca de cero que la
    objetivo. error_estandar: el de la diferencia promedio, con la varianza de cada grupo de
    varianza_rho_spearman, así que no depende de cuántos grupos haya ni de cuán distintos sean sus
    tamaños. tolerancia: z errores estándar; el par cumple si la diferencia queda dentro de ella.

    Hay una fila por cada par de columnas que varían juntas en algún grupo. Si ninguno de esos grupos
    tiene `minimo` registros, el par queda sin datos suficientes: «cumple» vacío (veredicto_v2). También
    queda así si cabe en su tolerancia pero no tiene registros para ver la correlación que debe ver: la
    mayor de la objetivo (hoy, el ρ de S04), la misma para todos los pares, o RHO_DETECTABLE si es menor.
    El par la ve si minima_detectable,
    Z_V2 + Z_POTENCIA errores estándar, no pasa de la correlación esperada con ese ρ y estos grupos;
    entonces una correlación de ese tamaño se sale de la tolerancia el 97,7 % de las veces. Un par que se
    sale de su tolerancia no cumple, aunque tenga pocos registros. Un par con una columna fija en todos los
    grupos, como una tarea que vale cero, no tiene ruido que correlacionar y no aparece.
    """
    columnas = list(registro.TAREAS if columnas is None else columnas)
    objetivo = pd.DataFrame(objetivo).loc[columnas, columnas].to_numpy(dtype=float)
    k = len(columnas)
    rho_visible = max(float(np.abs(objetivo[np.triu_indices(k, 1)]).max()) if k > 1 else 0.0, RHO_DETECTABLE)
    varian = np.zeros((k, k), dtype=bool)
    pares = {(i, j): [] for i in range(k) for j in range(i + 1, k)}
    for _, grupo in registros.groupby(list(por), dropna=False, sort=False):
        datos = grupo[columnas]
        cambia = (datos.nunique() > 1).to_numpy()
        varian |= np.outer(cambia, cambia)
        if len(datos) < minimo:
            continue
        r = correlacion_rangos(datos).to_numpy()
        for (i, j), lista in pares.items():
            if not np.isnan(r[i, j]):
                lista.append((len(datos), r[i, j], float(rho_spearman_esperada(objetivo[i, j], len(datos)))))
    filas = []
    for (i, j), lista in pares.items():
        if not varian[i, j]:
            continue
        fila = {"variable_1": columnas[i], "variable_2": columnas[j], "objetivo": objetivo[i, j],
                "grupos": len(lista), "registros": 0, "cumple": pd.NA}
        if lista:
            n, obtenida, esperada = (np.array(c, dtype=float) for c in zip(*lista))
            w = n / n.sum()
            diferencia = float(w @ (obtenida - esperada))
            se = float(np.sqrt(w ** 2 @ varianza_rho_spearman(n)))
            cumple = abs(diferencia) <= z * se
            ve = (z + Z_POTENCIA) * se <= float(w @ rho_spearman_esperada(rho_visible, n))
            fila.update(esperada=float(w @ esperada), obtenida=float(w @ obtenida), diferencia=diferencia,
                        error_estandar=se, tolerancia=z * se, minima_detectable=(z + Z_POTENCIA) * se,
                        registros=int(n.sum()), cumple=pd.NA if cumple and not ve else bool(cumple))
        filas.append(fila)
    tabla = pd.DataFrame(filas, columns=["variable_1", "variable_2", "objetivo", "esperada", "obtenida", "diferencia",
                                         "error_estandar", "tolerancia", "minima_detectable", "grupos", "registros",
                                         "cumple"])
    return tabla.astype({"esperada": float, "obtenida": float, "diferencia": float, "error_estandar": float,
                         "tolerancia": float, "minima_detectable": float, "grupos": int, "registros": int,
                         "cumple": "boolean"})


def veredicto_v2(tareas) -> str:
    """Resumen de verificar_por_grupos: «no cumple» si algún par se sale de su tolerancia; si no,
    «sin datos suficientes» si algún par no se pudo decidir (o si no hay ningún par), y «cumple» si
    cumplen todos."""
    cumple = pd.Series(tareas["cumple"], dtype="boolean")
    if cumple.eq(False).any():
        return "no cumple"
    if cumple.empty or cumple.isna().any():
        return "sin datos suficientes"
    return "cumple"


def indicador_v2(registros, objetivo=None, z=Z_V2) -> dict:
    """Indicador V2, «correlaciones entre variables definidas», sobre registros del dataset.

    Usa solo las ejecuciones terminadas: las interrumpidas por una falla (resultado «interrumpido»)
    tienen tareas parciales. Las duraciones deben ir sin redondear, como números reales (diccionario):
    con empates, la correlación de rangos esperada ya no es la de rho_spearman_esperada. Devuelve:
    - «tareas»: verificar_por_grupos con la matriz objetivo (por omisión, la de la hoja
      «Correlaciones»; también recibe un ρ o una matriz, como muestrear_ensayo);
    - «veredicto»: veredicto_v2 de esa tabla;
    - «estructura»: correlación de rangos entre complejidad (baja, media y alta como 1, 2 y 3), número
      de soluciones, preparación y duración, en todo el dataset y dentro de cada técnica, sin criterio
      de aceptación.
    """
    objetivo = matriz_objetivo() if objetivo is None else como_matriz(objetivo)
    if "resultado" in registros:
        registros = registros[registros["resultado"].ne("interrumpido")]
    tareas = verificar_por_grupos(registros, objetivo, z=z)
    datos = registros.copy()
    if not pd.api.types.is_numeric_dtype(datos["complejidad"]):
        datos["complejidad"] = datos["complejidad"].map(ORDEN_COMPLEJIDAD)
    columnas = list(ESTRUCTURA_V2)
    todo = correlacion_rangos(datos[columnas])
    tecnica = correlacion_por_grupos(datos, columnas=columnas, por=("tecnica",))
    estructura = pd.DataFrame([{"variable_1": a, "variable_2": b, "todo_el_dataset": todo.at[a, b],
                                "dentro_de_cada_tecnica": tecnica.at[a, b]}
                               for i, a in enumerate(columnas) for b in columnas[i + 1:]])
    return {"tareas": tareas, "veredicto": veredicto_v2(tareas), "estructura": estructura}
