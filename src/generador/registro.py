"""El registro del dataset: unidad de análisis (sección 1) y campos (sección 2).

Diseño: docs/Diccionario_de_datos.docx (anteproyecto, §1.4, §8.4 y §8.5) y hoja «Diccionario»
del Excel de parámetros, que lista los 51 campos con su tipo, unidad, valores, momento en que se
conocen, papel en el O2 y regla que los verifica.

Un registro es una ejecución de un ensayo sobre una muestra: la pareja muestra/análisis de
Costigliola et al. (2017, p. 9017), con un método del catálogo, un equipo y un analista, dentro de
una corrida (escenario de carga y semilla). Las repeticiones (reanálisis por resultado fuera de
especificación y repetición por falla del equipo) son registros nuevos enlazados a la ejecución
anterior (supuesto S27). Los ensayos de una muestra no tienen precedencia (S28).

`chequear_registro` dice, para cada registro, si cumple cada regla. El sábado 31 de octubre
alimenta el indicador «% de registros que cumplen las restricciones establecidas»
(src/generador/validacion.py).
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.generador import parametros
from src.generador.escenarios import ESCENARIOS

# ===================================================================== sección 1: unidad de análisis
TIPOS_MUESTRA = ("MP", "IN", "GR", "PT", "ES")
METODOS = tuple(f"M{i:02d}" for i in range(1, 13))
RESULTADOS = ("cumple", "fuera de especificación", "interrumpido")
# causa de una repetición según el resultado de la ejecución anterior (S27)
CAUSA_SEGUN_RESULTADO = {"fuera de especificación": "reanálisis", "interrumpido": "falla de equipo"}
CORRIDA = ["escenario", "semilla"]
CLAVE = CORRIDA + ["id_muestra", "metodo", "intento"]
CAMPOS = CLAVE + ["id_actividad", "tipo_muestra", "resultado", "id_actividad_origen", "causa_repeticion"]
ENSAYOS_POR_MUESTRA = (1, 3)   # Cunha et al. (2019, p. 1425), siempre métodos distintos (S18)
REANALISIS_POR_ENSAYO = 1      # el resultado del reanálisis es definitivo (S27)


def id_actividad(id_muestra: str, metodo: str, intento: int) -> str:
    """Identificador legible de un registro, por ejemplo PT-000123-M02-1."""
    return f"{id_muestra}-{metodo}-{int(intento)}"


def chequear_unidad(registros: pd.DataFrame) -> pd.DataFrame:
    """Reglas de la unidad de análisis: una columna booleana por regla y una fila por registro."""
    faltan = [c for c in CAMPOS if c not in registros.columns]
    if faltan:
        raise KeyError(f"Faltan columnas de la unidad de análisis: {faltan}")
    df = registros.reset_index(drop=True)
    intento = pd.to_numeric(df["intento"], errors="coerce")
    muestra = [df[c] for c in CORRIDA + ["id_muestra"]]
    ensayo = muestra + [df["metodo"]]
    r = pd.DataFrame(index=df.index)

    r["valores_validos"] = (df["tipo_muestra"].isin(TIPOS_MUESTRA) & df["metodo"].isin(METODOS)
                            & df["resultado"].isin(RESULTADOS) & intento.ge(1) & intento.mod(1).eq(0))
    r["clave_unica"] = ~df.duplicated(CLAVE, keep=False)
    propio = pd.Series([id_actividad(m, t, i) if pd.notna(i) else None
                        for m, t, i in zip(df["id_muestra"], df["metodo"], intento)], index=df.index)
    r["id_coherente"] = df["id_actividad"].eq(propio) & ~df.duplicated(CORRIDA + ["id_actividad"], keep=False)
    prefijo = df["id_muestra"].astype(str).str.split("-").str[0]
    r["muestra_coherente"] = (df.groupby(muestra)["tipo_muestra"].transform("nunique").eq(1)
                              & prefijo.eq(df["tipo_muestra"]))
    r["ensayos_por_muestra"] = df.groupby(muestra)["metodo"].transform("nunique").between(*ENSAYOS_POR_MUESTRA)

    # una repetición apunta a la ejecución anterior del mismo ensayo, y su causa sale del resultado de esa ejecución
    anterior = (df[CORRIDA + ["id_actividad", "resultado"]].drop_duplicates(CORRIDA + ["id_actividad"])
                .rename(columns={"id_actividad": "id_actividad_origen", "resultado": "resultado_origen"}))
    enlace = df[CORRIDA + ["id_actividad_origen"]].merge(anterior, on=CORRIDA + ["id_actividad_origen"], how="left")
    previo = pd.Series([id_actividad(m, t, i - 1) if pd.notna(i) and i > 1 else None
                        for m, t, i in zip(df["id_muestra"], df["metodo"], intento)], index=df.index)
    original = intento.eq(1) & df["id_actividad_origen"].isna() & df["causa_repeticion"].isna()
    repeticion = (intento.gt(1) & df["id_actividad_origen"].eq(previo)
                  & enlace["resultado_origen"].map(CAUSA_SEGUN_RESULTADO).eq(df["causa_repeticion"]))
    r["repeticion_enlazada"] = original | repeticion
    reanalisis = df["causa_repeticion"].eq("reanálisis").astype(int)
    r["reanalisis_por_ensayo"] = reanalisis.groupby(ensayo).transform("sum").le(REANALISIS_POR_ENSAYO)

    r.index = registros.index
    return r


def cumple_unidad(registros: pd.DataFrame) -> pd.Series:
    """True en los registros que cumplen todas las reglas de la unidad de análisis."""
    return chequear_unidad(registros).all(axis=1)


# ===================================================================== sección 2: campos del registro
MIN_POR_DIA = 24 * 60          # los momentos van en minutos desde el lunes a las 0:00 en que empieza la corrida
TOL = 1e-6
NIVELES = ("júnior", "intermedio", "sénior")
TECNICAS_CON_IDONEIDAD = ("HPLC", "GC")   # Costigliola et al. (2017, Tabla 6, p. 9019)
TAREAS = ["t_prep_sistema", "t_idoneidad", "t_preparacion", "t_montaje", "t_corrida", "t_procesamiento"]
TAREAS_ANALISTA = ("t_prep_sistema", "t_preparacion", "t_montaje", "t_procesamiento")   # la idoneidad y la corrida son del equipo
CAMPOS_REGISTRO = [
    "escenario", "carga", "semilla",
    "id_actividad", "id_muestra", "metodo", "intento", "id_actividad_origen", "causa_repeticion", "resultado",
    "tipo_muestra", "prioridad", "llegada_muestra", "plazo_min", "vencimiento",
    "tecnica", "complejidad", "n_soluciones", "calificacion_exigida", "equipo_requerido",
    "t_optimista", "t_mas_probable", "t_pesimista",
    "analista", "nivel_analista", "equipo", "configuracion", "idoneidad_ejecutada",
    "duracion_simulada", *TAREAS,
    "llegada", "inicio", "fin", "espera", "pausas", "tiempo_en_sistema", "tardanza",
    "urgente", "bandera_reanalisis", "bandera_falla",
    "analistas_calificados", "equipos_compatibles", "combinaciones_factibles",
    "analistas_libres", "equipos_libres", "recurso_faltante",
]
OPCIONALES = ("id_actividad_origen", "causa_repeticion")   # vacíos cuando el intento es 1


@lru_cache(maxsize=1)
def _hojas():
    """Hojas del Excel de parámetros que usan las reglas; se leen una sola vez."""
    catalogo = parametros.cargar_hoja("Catálogo").set_index("ID")
    analistas = parametros.cargar_hoja("Analistas").set_index("ID")
    equipos = parametros.cargar_hoja("Equipos").set_index("Código")
    tipos = parametros.cargar_hoja("Tipos de muestra").set_index("Código")
    duraciones = parametros.cargar_hoja("Duraciones")
    tareas = ["Preparación de muestra", "Montaje del equipo", "Corrida", "Procesamiento de datos"]
    duraciones = duraciones[duraciones["Tarea"].isin(tareas)].set_index(["Técnica", "Tarea"])
    return catalogo, analistas, equipos, tipos, duraciones


def nivel_minimo(calificacion) -> str:
    """Nivel de la columna «Calificación mínima» del catálogo: «Intermedio o superior» -> «intermedio»."""
    texto = str(calificacion).lower()
    return next((n for n in reversed(NIVELES) if n in texto), None)


def equipo_requerido(metodo: str) -> str:
    """Configuraciones compatibles con el método, separadas por «|» (hoja Equipos, S22)."""
    equipos = _hojas()[2]
    return "|".join(equipos.index[equipos[metodo].eq("X")])


def combinaciones(metodo: str) -> tuple:
    """Analistas calificados, equipos compatibles y combinaciones factibles del método."""
    _, analistas, equipos, _, _ = _hojas()
    n_analistas = int(analistas[metodo].sum())
    n_equipos = int(equipos.loc[equipos[metodo].eq("X"), "Equipos"].sum())
    return n_analistas, n_equipos, n_analistas * n_equipos


def sumar_dias_habiles(minuto: float, dias: int) -> float:
    """El mismo momento del día, `dias` días hábiles después (lunes a viernes, sin festivos: S23)."""
    dia, resto = divmod(minuto, MIN_POR_DIA)
    dia, contados = int(dia), 0
    while contados < dias:
        dia += 1
        contados += dia % 7 < 5
    return dia * MIN_POR_DIA + resto


def plazo_minutos(tipo_muestra: str, urgente: bool, llegada_muestra: float) -> float:
    """Plazo de la muestra en minutos (S06); una urgente toma el del control en proceso (S03)."""
    fila = _hojas()[3].loc["IN" if urgente else tipo_muestra]
    if fila["Unidad del plazo"] == "h":
        return float(fila["Plazo"]) * 60
    return sumar_dias_habiles(llegada_muestra, int(fila["Plazo"])) - llegada_muestra


def prioridad(tipo_muestra: str, urgente: bool) -> int:
    """Prioridad del tipo de muestra; la urgente tiene la máxima (Coito et al., 2021, p. 65)."""
    return 1 if urgente else int(_hojas()[3].at[tipo_muestra, "Prioridad"])


def duracion(t_prep_sistema, t_idoneidad, t_preparacion, t_montaje, t_corrida, t_procesamiento):
    """Tiempo de ejecución: la idoneidad corre en paralelo con la preparación del análisis
    (Costigliola et al., 2017, p. 9017, Fig. 3)."""
    return t_prep_sistema + t_montaje + np.maximum(t_idoneidad, t_preparacion) + t_corrida + t_procesamiento


def corrida_tres_puntos(metodo: str) -> tuple:
    """Mínimo, moda y máximo de la corrida del método.

    HPLC y GC: moda del método ±20 % (S05 y S14). PSA, KF, DSC y XRPD: el tramo del método (S15),
    con la moda en el punto de mayor densidad de la distribución de la técnica dentro del tramo;
    en DSC, cuya distribución es uniforme, el punto medio del tramo.
    """
    catalogo, _, _, _, duraciones = _hojas()
    fila = catalogo.loc[metodo]
    lo, hi = float(fila["Corrida mín (min)"]), float(fila["Corrida máx (min)"])
    if pd.notna(fila["Corrida moda (min)"]):
        return lo, float(fila["Corrida moda (min)"]), hi
    tecnica = duraciones.loc[(fila["Técnica"], "Corrida")]
    if str(tecnica["Distribución original"]).startswith("U"):
        return lo, (lo + hi) / 2, hi
    return lo, min(max(float(tecnica["m (moda)"]), lo), hi), hi


def tres_puntos_tareas(metodo: str, n_soluciones: int, prep_sistema=(0.0, 0.0, 0.0), idoneidad=(0.0, 0.0, 0.0)) -> dict:
    """Mínimo, moda y máximo de cada tarea del registro, con las claves de TAREAS.

    Salen de la hoja «Duraciones» (Costigliola et al., 2017, Tabla 3, p. 9016) y de la corrida del
    método; la preparación del análisis es el tiempo por solución por el número de soluciones
    (p. 9019). La preparación y la idoneidad del sistema se pasan solo si el registro las necesita;
    sus tres puntos siguen pendientes (docs/Generador_de_datos.docx, sección 1.10).
    """
    catalogo, _, _, _, duraciones = _hojas()
    tecnica = catalogo.at[metodo, "Técnica"]
    lo, hi = int(catalogo.at[metodo, "Soluciones mín"]), int(catalogo.at[metodo, "Soluciones máx"])
    if not (float(n_soluciones).is_integer() and lo <= n_soluciones <= hi):
        raise ValueError(f"{metodo} prepara entre {lo} y {hi} soluciones (S08), no {n_soluciones}.")

    def tarea(nombre):
        f = duraciones.loc[(tecnica, nombre)]
        return float(f["a (mín)"]), float(f["m (moda)"]), float(f["b (máx)"])

    return {"t_prep_sistema": tuple(float(x) for x in prep_sistema), "t_idoneidad": tuple(float(x) for x in idoneidad),
            "t_preparacion": tuple(x * n_soluciones for x in tarea("Preparación de muestra")),
            "t_montaje": tarea("Montaje del equipo"), "t_corrida": corrida_tres_puntos(metodo),
            "t_procesamiento": tarea("Procesamiento de datos")}


def tres_puntos(metodo: str, n_soluciones: int, prep_sistema=(0.0, 0.0, 0.0), idoneidad=(0.0, 0.0, 0.0)) -> tuple:
    """Tiempos optimista, más probable y pesimista del registro, para un analista intermedio.

    Es la misma suma de la duración simulada con el mínimo, la moda o el máximo de cada tarea
    (tres_puntos_tareas).
    """
    t = tres_puntos_tareas(metodo, n_soluciones, prep_sistema, idoneidad)
    return tuple(float(duracion(*(t[c][k] for c in TAREAS))) for k in range(3))


def factor_duracion(nivel: str) -> float:
    """Factor de duración del nivel del analista (hoja «Analistas», S07)."""
    analistas = _hojas()[1]
    factores = analistas.loc[analistas["Nivel"].eq(nivel), "Factor de duración"].unique()
    if len(factores) != 1:
        raise ValueError(f"Nivel de analista desconocido: {nivel!r}; los niveles son {NIVELES}.")
    return float(factores[0])


def _unidad_valida(equipo, configuracion, equipos) -> bool:
    """«HPLC-G04» es válida si existe la configuración HPLC-G y tiene al menos 4 unidades."""
    if not (isinstance(equipo, str) and configuracion in equipos.index and equipo.startswith(configuracion)):
        return False
    numero = equipo[len(configuracion):]
    return len(numero) == 2 and numero.isdigit() and 1 <= int(numero) <= int(equipos.at[configuracion, "Equipos"])


def chequear_campos(registros: pd.DataFrame) -> pd.DataFrame:
    """Reglas de los campos (sección 2): una columna booleana por regla y una fila por registro."""
    faltan = [c for c in CAMPOS_REGISTRO if c not in registros.columns]
    if faltan:
        raise KeyError(f"Faltan campos del registro: {faltan}")
    catalogo, analistas, equipos, tipos, _ = _hojas()
    df = registros.reset_index(drop=True)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")   # noqa: E731
    iguales = lambda a, b: (a - b).abs().lt(TOL)              # noqa: E731
    r = pd.DataFrame(index=df.index)

    r["campos_completos"] = df[[c for c in CAMPOS_REGISTRO if c not in OPCIONALES]].notna().all(axis=1)
    carga = df["escenario"].map({k: v["multiplicador_carga"] for k, v in ESCENARIOS.items()})
    r["corrida_coherente"] = iguales(num("carga"), carga)

    # muestra: atributos iguales en todos sus registros; prioridad y plazo según el tipo (S03 y S06)
    urgente = df["urgente"].eq(True)
    tipo = df["tipo_muestra"].where(df["tipo_muestra"].isin(tipos.index))
    llegada_m = num("llegada_muestra")
    prio = pd.Series([prioridad(t, u) if pd.notna(t) else np.nan for t, u in zip(tipo, urgente)], index=df.index)
    plazo = pd.Series([plazo_minutos(t, u, ll) if pd.notna(t) and pd.notna(ll) else np.nan
                       for t, u, ll in zip(tipo, urgente, llegada_m)], index=df.index)
    muestra = [df[c] for c in CORRIDA + ["id_muestra"]]
    atributos = ["llegada_muestra", "plazo_min", "vencimiento", "urgente", "prioridad"]
    constantes = df.groupby(muestra)[atributos].transform("nunique").eq(1).all(axis=1)
    r["muestra_y_plazo"] = ((~urgente | tipo.map(tipos["Puede ser urgente"]).eq("Sí")) & iguales(num("prioridad"), prio)
                            & iguales(num("plazo_min"), plazo) & iguales(num("vencimiento"), llegada_m + num("plazo_min"))
                            & constantes)

    # ensayo: atributos del catálogo y soluciones en el rango de la complejidad (S08)
    metodo = df["metodo"].where(df["metodo"].isin(catalogo.index))
    cat = catalogo.reindex(metodo).set_axis(df.index)
    requerido = metodo.map({m: equipo_requerido(m) for m in catalogo.index})
    n = num("n_soluciones")
    r["ensayo_coherente"] = (df["tecnica"].eq(cat["Técnica"]) & df["complejidad"].eq(cat["Complejidad"])
                             & df["calificacion_exigida"].eq(cat["Calificación mínima"].map(nivel_minimo))
                             & df["equipo_requerido"].eq(requerido)
                             & n.between(cat["Soluciones mín"], cat["Soluciones máx"]) & n.mod(1).eq(0))

    # asignación: analista calificado del nivel exigido y equipo de una configuración compatible (S12 y S22)
    an = analistas.reindex(df["analista"].where(df["analista"].isin(analistas.index))).set_axis(df.index)
    calificado = pd.Series([a in analistas.index and m in METODOS and analistas.at[a, m] == 1
                            for a, m in zip(df["analista"], df["metodo"])], index=df.index)
    rango = {nv: k for k, nv in enumerate(NIVELES)}
    nivel_ok = (df["nivel_analista"].eq(an["Nivel"])
                & (df["nivel_analista"].map(rango) >= df["calificacion_exigida"].map(rango)))
    config_ok = pd.Series([isinstance(q, str) and c in q.split("|")
                           for c, q in zip(df["configuracion"], df["equipo_requerido"])], index=df.index)
    unidad_ok = pd.Series([_unidad_valida(e, c, equipos) for e, c in zip(df["equipo"], df["configuracion"])],
                          index=df.index)
    idoneidad_ok = ~df["idoneidad_ejecutada"].eq(True) | df["tecnica"].isin(TECNICAS_CON_IDONEIDAD)
    r["asignacion_coherente"] = calificado & nivel_ok & config_ok & unidad_ok & idoneidad_ok

    a, m, b = num("t_optimista"), num("t_mas_probable"), num("t_pesimista")
    r["tres_puntos_ordenados"] = a.gt(0) & (a <= m + TOL) & (m <= b + TOL)

    # duración: suma de las tareas y, si la ejecución terminó, dentro de los tres puntos corregidos por el factor (S07)
    t = df[TAREAS].apply(pd.to_numeric, errors="coerce")
    d = num("duracion_simulada")
    con_idoneidad = df["idoneidad_ejecutada"].eq(True)
    terminada = df["resultado"].ne("interrumpido")
    sin_sistema = t["t_prep_sistema"].eq(0) & t["t_idoneidad"].eq(0)
    con_sistema = t["t_prep_sistema"].gt(0) & t["t_idoneidad"].gt(0)
    f = pd.to_numeric(an["Factor de duración"], errors="coerce")
    dentro = d.between(np.minimum(f, 1) * a - TOL, np.maximum(f, 1) * b + TOL)
    r["duracion_coherente"] = (t.ge(0).all(axis=1) & (con_idoneidad | sin_sistema)
                               & (~(con_idoneidad & terminada) | con_sistema)
                               & iguales(d, duracion(*(t[c] for c in TAREAS))) & (~terminada | dentro))

    # tiempos de la corrida (sección 1, Tabla 2): la repetición llega cuando termina la ejecución que repite
    llegada, inicio, fin = num("llegada"), num("inicio"), num("fin")
    fines = (df[CORRIDA + ["id_actividad", "fin"]].drop_duplicates(CORRIDA + ["id_actividad"])
             .rename(columns={"id_actividad": "id_actividad_origen", "fin": "fin_origen"}))
    fin_origen = pd.to_numeric(df[CORRIDA + ["id_actividad_origen"]].merge(
        fines, on=CORRIDA + ["id_actividad_origen"], how="left")["fin_origen"], errors="coerce").set_axis(df.index)
    llegada_ok = iguales(llegada, llegada_m.where(df["intento"].eq(1), fin_origen))
    r["tiempos_coherentes"] = (llegada_ok & (inicio >= llegada - TOL) & (fin >= inicio + d - TOL)
                               & iguales(num("espera"), inicio - llegada) & iguales(num("pausas"), fin - inicio - d)
                               & iguales(num("tiempo_en_sistema"), fin - llegada)
                               & iguales(num("tardanza"), (fin - num("vencimiento")).clip(lower=0)))

    reanalisis = df["causa_repeticion"].eq("reanálisis") | df["resultado"].eq("fuera de especificación")
    falla = df["causa_repeticion"].eq("falla de equipo") | df["resultado"].eq("interrumpido")
    r["banderas_coherentes"] = df["bandera_reanalisis"].eq(reanalisis) & df["bandera_falla"].eq(falla)

    # combinaciones: las factibles salen de las matrices; lo que faltaba, de los recursos libres al llegar
    totales = {mt: combinaciones(mt) for mt in catalogo.index}
    n_an, n_eq, n_comb = (metodo.map({mt: v[k] for mt, v in totales.items()}) for k in range(3))
    libres_an, libres_eq = num("analistas_libres"), num("equipos_libres")
    faltante = np.select([libres_an.gt(0) & libres_eq.gt(0), libres_an.eq(0) & libres_eq.gt(0),
                          libres_an.gt(0) & libres_eq.eq(0)], ["ninguno", "analista", "equipo"], default="ambos")
    r["combinaciones_coherentes"] = (iguales(num("analistas_calificados"), n_an) & iguales(num("equipos_compatibles"), n_eq)
                                     & iguales(num("combinaciones_factibles"), n_comb)
                                     & libres_an.between(0, n_an) & libres_eq.between(0, n_eq)
                                     & df["recurso_faltante"].eq(pd.Series(faltante, index=df.index)))
    r.index = registros.index
    return r


def chequear_registro(registros: pd.DataFrame) -> pd.DataFrame:
    """Todas las reglas de las secciones 1 y 2: una columna por regla y una fila por registro."""
    return pd.concat([chequear_unidad(registros), chequear_campos(registros)], axis=1)


def ejemplo_del_diccionario() -> pd.DataFrame:
    """Los cinco registros de la Tabla 4 del diccionario, con sus 51 campos.

    Una muestra de producto terminado que llega el martes a las 8:10 con dos ensayos, uno de ellos
    fuera de especificación y luego reanalizado, y una de control en proceso que llega el sábado a
    las 3:10 y cuyo único ensayo se interrumpe por una falla del equipo. Lo usan las pruebas y las
    tablas del documento.
    """
    catalogo, analistas, _, _, _ = _hojas()
    muestras = {"PT-000123": ("PT", 1 * MIN_POR_DIA + 8 * 60 + 10), "IN-000457": ("IN", 5 * MIN_POR_DIA + 3 * 60 + 10)}
    # muestra, método, intento, causa, resultado, soluciones, analista, equipo, espera (min),
    # tareas (preparación, montaje, corrida, procesamiento) y recursos libres al llegar (analistas, equipos)
    ejecuciones = [
        ("PT-000123", "M02", 1, None, "fuera de especificación", 4, "A03", "HPLC-G04", 30.0, (148.0, 47.0, 158.2, 6.0), 0, 12),
        ("PT-000123", "M07", 1, None, "cumple", 3, "A18", "KF-V01", 0.0, (21.0, 9.0, 7.4, 18.5), 3, 2),
        ("PT-000123", "M02", 2, "reanálisis", "cumple", 4, "A01", "HPLC-G04", 0.0, (118.8, 40.5, 150.0, 4.5), 2, 13),
        ("IN-000457", "M04", 1, None, "interrumpido", 3, "A05", "GC-H07", 0.0, (33.0, 6.0, 25.0, 0.0), 2, 11),
        ("IN-000457", "M04", 2, "falla de equipo", "cumple", 3, "A05", "GC-H02", 0.0, (28.0, 4.0, 70.1, 22.0), 2, 10),
    ]
    redondeo = lambda x: round(float(x), 6) + 0.0   # noqa: E731  (sin ruido de coma flotante ni −0,0)
    filas, fin_de = [], {}
    for (id_m, mt, intento, causa, resultado, n, analista, equipo, espera, (prep, montaje, corrida, proc),
         libres_an, libres_eq) in ejecuciones:
        tipo, llegada_m = muestras[id_m]
        ida = id_actividad(id_m, mt, intento)
        origen = id_actividad(id_m, mt, intento - 1) if intento > 1 else None
        llegada = llegada_m if intento == 1 else fin_de[origen]
        d = redondeo(duracion(0.0, 0.0, prep, montaje, corrida, proc))
        inicio = redondeo(llegada + espera)
        fin = fin_de[ida] = redondeo(inicio + d)
        plazo = plazo_minutos(tipo, False, llegada_m)
        a, m, b = tres_puntos(mt, n)
        n_an, n_eq, n_comb = combinaciones(mt)
        faltante = {(True, True): "ninguno", (False, True): "analista", (True, False): "equipo"}.get(
            (libres_an > 0, libres_eq > 0), "ambos")
        filas.append({
            "escenario": "base", "carga": ESCENARIOS["base"]["multiplicador_carga"], "semilla": 2026,
            "id_actividad": ida, "id_muestra": id_m, "metodo": mt, "intento": intento,
            "id_actividad_origen": origen, "causa_repeticion": causa, "resultado": resultado,
            "tipo_muestra": tipo, "prioridad": prioridad(tipo, False), "llegada_muestra": float(llegada_m),
            "plazo_min": plazo, "vencimiento": llegada_m + plazo,
            "tecnica": catalogo.at[mt, "Técnica"], "complejidad": catalogo.at[mt, "Complejidad"], "n_soluciones": n,
            "calificacion_exigida": nivel_minimo(catalogo.at[mt, "Calificación mínima"]),
            "equipo_requerido": equipo_requerido(mt),
            "t_optimista": a, "t_mas_probable": m, "t_pesimista": b,
            "analista": analista, "nivel_analista": analistas.at[analista, "Nivel"], "equipo": equipo,
            "configuracion": equipo[:-2], "idoneidad_ejecutada": False,
            "duracion_simulada": d, "t_prep_sistema": 0.0, "t_idoneidad": 0.0, "t_preparacion": prep,
            "t_montaje": montaje, "t_corrida": corrida, "t_procesamiento": proc,
            "llegada": float(llegada), "inicio": inicio, "fin": fin, "espera": espera, "pausas": redondeo(fin - inicio - d),
            "tiempo_en_sistema": redondeo(fin - llegada), "tardanza": max(0.0, redondeo(fin - (llegada_m + plazo))),
            "urgente": False, "bandera_reanalisis": causa == "reanálisis" or resultado == "fuera de especificación",
            "bandera_falla": causa == "falla de equipo" or resultado == "interrumpido",
            "analistas_calificados": n_an, "equipos_compatibles": n_eq, "combinaciones_factibles": n_comb,
            "analistas_libres": libres_an, "equipos_libres": libres_eq, "recurso_faltante": faltante,
        })
    return pd.DataFrame(filas, columns=CAMPOS_REGISTRO)
