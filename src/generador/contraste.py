"""Contraste estructural del dataset con las instancias públicas de Martins et al. (2023) y del Test Laboratory
Scheduling Problem (TLSP; Mischek y Musliu, 2021).

Bloque del 1 de noviembre del plan (O1). Compara estructura y no magnitudes, porque cada fuente mide el tiempo en
otra unidad (minutos aquí, unidades de tiempo sin nombre en Martins et al. y franjas de medio día en el TLSP): trabajos
y tareas por trabajo, tipos de recurso, densidad de elegibilidad y rangos de duración, y como contexto el horizonte,
las llegadas, los plazos, el orden entre tareas, el alistamiento y las perturbaciones.

Las unidades se emparejan como en el diccionario de datos (sección 1.5, Tabla 5):

    este trabajo      muestra    ensayo (cada ejecución es un registro)
    Martins et al.    trabajo    operación
    TLSP              proyecto   tarea (las tareas de un proyecto se agrupan en trabajos)

Las instancias no se suben al repositorio: se bajan con datos/descargar_instancias.py (ver los LEEME de
datos/instancias_martins y datos/instancias_tlsp). Si las del TLSP no están, su columna sale de lo que reporta el
artículo (TLSP_ARTICULO y TLSP_REALES) y las celdas que solo se pueden medir en los archivos quedan «n. d.».

Uso:
    from src.generador import contraste, generar
    registros, _, resumenes = generar.cargar_version("v0.2")
    c = contraste.contrastar(registros, resumenes)
    c["tabla"]             # una fila por medida y una columna por fuente
    c["medidas"]           # las mismas medidas en números, una fila por fuente, conjunto y medida
    contraste.guardar(c)   # resultados/tablas/contraste_estructural.csv, _medidas.csv e _instancias.csv
"""
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.generador import registro
from src.generador.recursos import Laboratorio
from src.generador.validacion import _n, _pct

RAIZ = Path(__file__).resolve().parents[2]
CARPETA_MARTINS = RAIZ / "datos" / "instancias_martins"
CARPETA_TLSP = RAIZ / "datos" / "instancias_tlsp"
CARPETA_TABLAS = RAIZ / "resultados" / "tablas"

ESTE = "Este trabajo (dataset v0.2)"
MARTINS = "Martins et al. (2023)"
TLSP = "TLSP (Mischek y Musliu, 2021)"
FUENTES = (ESTE, MARTINS, TLSP)
ESCENARIO = "base"   # las medidas por tarea de este trabajo se calculan en el escenario base (tres semillas)
TECNICAS_CON_ALISTAMIENTO = ("HPLC", "GC")

# Lo que reporta el artículo del TLSP. Vale para los conjuntos General y LabStructure, de 60 instancias cada uno, de
# los que el conjunto de evaluación toma 15 y 15 (p. 550).
TLSP_ARTICULO = {
    "proyectos": (5, 90),                      # p. 550
    "franjas": (88, 782),                      # p. 550; cada franja es medio día de trabajo (p. 538)
    "horas_por_franja": 4,                     # «a four-hour timeslot» (p. 540)
    "trabajos_por_proyecto": 4,                # «close to 4» (p. 550)
    "empleados_calificados_por_trabajo": 6,    # en promedio (p. 550)
    "mesas_por_trabajo": 5,                    # en promedio (p. 550)
    "modos": {"Single": (1.0, 1), "Shift": (0.6, 2), "External": (1.0, 0)},   # factor y empleados (Tabla 2, p. 548)
}
# Las tres instancias reales del socio industrial (Tabla 4, p. 551): no están en el conjunto de evaluación.
TLSP_REALES = pd.DataFrame(
    [("2019-04", 74, 297, 606, 22, 17, 1, 5.49, 3.06),
     ("2019-07", 59, 251, 700, 24, 17, 1, 5.33, 3.17),
     ("2019-10", 59, 223, 572, 19, 17, 1, 5.70, 3.48)],
    columns=["instancia", "proyectos", "trabajos", "franjas", "empleados", "mesas", "grupos_de_equipos",
             "empleados_calificados_por_trabajo", "mesas_por_trabajo"])
TLSP_REALES["densidad_empleados"] = TLSP_REALES["empleados_calificados_por_trabajo"] / TLSP_REALES["empleados"]
TLSP_REALES["densidad_mesas"] = TLSP_REALES["mesas_por_trabajo"] / TLSP_REALES["mesas"]

PATRON_MARTINS = re.compile(r"ex_j(\d+)_m(\d+)_w(\d+)_s(\d+)_f([0-9.]+)_r(\d+)\.json$")


# ===================================================================== Martins et al. (2023)
def leer_martins(carpeta: Path | str = CARPETA_MARTINS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lee las instancias JSON de Martins et al. (2023). Devuelve una tabla con una fila por instancia y otra con una
    fila por operación.

    En cada operación, la máquina está ocupada todo el tiempo de proceso y el analista solo en sus intervenciones
    (pp. 182 y 183); `ocupacion_analista` es la suma de esas intervenciones. La flexibilidad y la réplica salen del
    nombre del archivo (LEEME de datos/instancias_martins)."""
    instancias, operaciones = [], []
    for ruta in sorted(Path(carpeta).glob("ex_*.json")):
        nombre = PATRON_MARTINS.search(ruta.name)
        if not nombre:
            continue
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        flexibilidad, replica = float(nombre.group(5)), int(nombre.group(6))
        w, m = int(datos["n_analysts"]), int(datos["n_machines"])
        for trabajo in datos["jobs"]:
            for posicion, op in enumerate(trabajo["tests"], start=1):
                operaciones.append({
                    "instancia": ruta.stem, "flexibilidad": flexibilidad, "analistas": w, "maquinas": m,
                    "trabajo": int(trabajo["job"]), "tipo": int(trabajo["product"]), "posicion": posicion,
                    "duracion": float(op["machine_time"]),
                    "ocupacion_analista": float(sum(t["end"] - t["start"] for t in op["analyst_times"])),
                    "intervenciones": len(op["analyst_times"]),
                    "analistas_elegibles": len(op["suitable_analysts"]),
                    "maquinas_elegibles": len(op["suitable_machines"]),
                    "liberacion": float(trabajo["release_time"]), "vencimiento": float(trabajo["due_date"]),
                })
        instancias.append({"instancia": ruta.stem, "trabajos": int(datos["n_jobs"]), "analistas": w, "maquinas": m,
                           "tipos_de_trabajo": len({t["product"] for t in datos["jobs"]}),
                           "flexibilidad": flexibilidad, "replica": replica})
    if not instancias:
        raise FileNotFoundError(f"No hay instancias de Martins et al. (2023) en {carpeta}. "
                                "Bájelas con: python datos/descargar_instancias.py")
    operaciones = pd.DataFrame(operaciones)
    instancias = pd.DataFrame(instancias)
    por_instancia = operaciones.groupby("instancia")
    instancias = instancias.merge(
        por_instancia.size().rename("operaciones").reset_index(), on="instancia")
    operaciones["fraccion_analista"] = operaciones["ocupacion_analista"] / operaciones["duracion"]
    operaciones["densidad_analistas"] = operaciones["analistas_elegibles"] / operaciones["analistas"]
    operaciones["densidad_maquinas"] = operaciones["maquinas_elegibles"] / operaciones["maquinas"]
    operaciones["combinaciones"] = operaciones["analistas_elegibles"] * operaciones["maquinas_elegibles"]
    operaciones["densidad_combinaciones"] = operaciones["combinaciones"] / (operaciones["analistas"] * operaciones["maquinas"])
    return instancias, operaciones


# ===================================================================== este trabajo
def medir_dataset(registros: pd.DataFrame, resumenes: list) -> dict:
    """Tablas de este trabajo para el contraste: `corridas` (una fila por corrida, con los conteos del resumen),
    `pedidos` (el primer intento de cada ensayo pedido, para la elegibilidad y los plazos) y `terminadas` (las
    ejecuciones que no se interrumpieron, para las duraciones, como en el entrenamiento del O2)."""
    columnas = ("escenario", "carga", "semilla", "semanas", "muestras", "ensayos_pedidos", "registros",
                "reanalisis", "interrumpidos", "fallas", "ventanas")
    corridas = pd.DataFrame([{c: r[c] for c in columnas} for r in resumenes])
    corridas["dias"] = corridas["semanas"] * 7
    corridas["muestras_por_dia"] = corridas["muestras"] / corridas["dias"]
    corridas["ensayos_por_dia"] = corridas["ensayos_pedidos"] / corridas["dias"]
    corridas["ensayos_por_muestra"] = corridas["ensayos_pedidos"] / corridas["muestras"]

    lab = laboratorio()
    pedidos = registros[registros["intento"] == 1].copy()
    pedidos["densidad_analistas"] = pedidos["analistas_calificados"] / lab["analistas"]
    pedidos["densidad_equipos"] = pedidos["equipos_compatibles"] / lab["equipos"]
    pedidos["densidad_combinaciones"] = pedidos["combinaciones_factibles"] / (lab["analistas"] * lab["equipos"])
    # Martins et al. (2023, p. 191) dicen que sus 7 máquinas representan tipos de equipo: el par más cercano aquí es
    # la configuración, no la unidad
    pedidos["configuraciones_compatibles"] = pedidos["equipo_requerido"].str.count(r"\|") + 1
    pedidos["densidad_configuraciones"] = pedidos["configuraciones_compatibles"] / lab["configuraciones"]

    # Como en los documentos del O1 (tools/cifras.py): todos los registros de HPLC y GC, también los interrumpidos
    con_idoneidad = registros[registros["tecnica"].isin(TECNICAS_CON_ALISTAMIENTO)]
    alistamiento = con_idoneidad.groupby(["escenario", "tecnica"])["idoneidad_ejecutada"].mean()

    terminadas = registros[registros["resultado"] != "interrumpido"].copy()
    terminadas["fraccion_analista"] = (terminadas[list(registro.TAREAS_ANALISTA)].sum(axis=1)
                                       / terminadas["duracion_simulada"])
    primeras = terminadas[terminadas["intento"] == 1]
    terminadas["holgura"] = np.nan
    terminadas.loc[primeras.index, "holgura"] = primeras["plazo_min"] / primeras["duracion_simulada"]
    return {"corridas": corridas, "pedidos": pedidos, "terminadas": terminadas, "alistamiento": alistamiento}


@lru_cache(maxsize=1)
def laboratorio() -> dict:
    """Tamaño del laboratorio de referencia, leído del Excel de parámetros (diseño del laboratorio, secciones 3 y 4)."""
    lab = Laboratorio.desde_excel()
    return {"analistas": len(lab.analistas), "equipos": len(lab.equipos),
            "configuraciones": len(lab.configuraciones),
            "tecnicas": lab.configuraciones["tecnica"].nunique(),
            "turnos": len({a.turno for a in lab.analistas.values()}),
            "metodos": len(lab.metodos)}


# ===================================================================== resúmenes
def _estadisticos(serie: pd.Series, prefijo: str) -> dict:
    s = pd.Series(serie, dtype=float).dropna()
    p10, p50, p90 = s.quantile([0.1, 0.5, 0.9])
    return {f"{prefijo}_min": s.min(), f"{prefijo}_p10": p10, f"{prefijo}_p50": p50, f"{prefijo}_p90": p90,
            f"{prefijo}_max": s.max(), f"{prefijo}_media": s.mean(), f"{prefijo}_cv": s.std() / s.mean(),
            f"{prefijo}_p90_p10": p90 / p10}


def resumir_dataset(tablas: dict, escenario: str = ESCENARIO) -> dict:
    """Medidas de este trabajo. El tamaño se da por escenario (promedio de las semillas); lo demás, en el escenario
    base, con las tres semillas juntas."""
    lab = laboratorio()
    corridas = tablas["corridas"]
    por_escenario = corridas.groupby("escenario")[["muestras", "ensayos_pedidos", "registros", "muestras_por_dia",
                                                   "ensayos_por_dia", "ensayos_por_muestra"]].mean()
    pedidos = tablas["pedidos"][tablas["pedidos"]["escenario"] == escenario]
    terminadas = tablas["terminadas"][tablas["terminadas"]["escenario"] == escenario]
    ensayos_por_muestra = pedidos.groupby(["semilla", "id_muestra"])["metodo"].nunique()
    reparto = ensayos_por_muestra.value_counts(normalize=True).sort_index()
    m = {
        "corridas": len(corridas), "semanas": int(corridas["semanas"].iloc[0]),
        **{f"muestras_{e}": v for e, v in por_escenario["muestras"].items()},
        **{f"ensayos_{e}": v for e, v in por_escenario["ensayos_pedidos"].items()},
        **{f"registros_{e}": v for e, v in por_escenario["registros"].items()},
        **{f"muestras_por_dia_{e}": v for e, v in por_escenario["muestras_por_dia"].items()},
        **{f"ensayos_por_dia_{e}": v for e, v in por_escenario["ensayos_por_dia"].items()},
        "tareas_por_trabajo_media": por_escenario.loc[escenario, "ensayos_por_muestra"],
        "tareas_por_trabajo_min": ensayos_por_muestra.min(), "tareas_por_trabajo_max": ensayos_por_muestra.max(),
        **{f"tareas_por_trabajo_{k}": v for k, v in reparto.items()},
        "tipos_de_trabajo": pedidos["tipo_muestra"].nunique(), **lab,
        **_estadisticos(pedidos["analistas_calificados"], "analistas_elegibles"),
        **_estadisticos(pedidos["equipos_compatibles"], "equipos_elegibles"),
        **_estadisticos(pedidos["combinaciones_factibles"], "combinaciones"),
        "densidad_analistas": pedidos["densidad_analistas"].mean(),
        "densidad_equipos": pedidos["densidad_equipos"].mean(),
        "densidad_combinaciones": pedidos["densidad_combinaciones"].mean(),
        **_estadisticos(terminadas["duracion_simulada"], "duracion"),
        **_estadisticos(terminadas["fraccion_analista"], "fraccion_analista"),
        **_estadisticos(terminadas["holgura"], "holgura"),
    }
    # La misma densidad sin pesar por la demanda: un valor por método
    por_metodo = pedidos.groupby("metodo")[["analistas_calificados", "equipos_compatibles"]].first()
    m["densidad_analistas_por_metodo"] = por_metodo["analistas_calificados"].mean() / lab["analistas"]
    m["densidad_equipos_por_metodo"] = por_metodo["equipos_compatibles"].mean() / lab["equipos"]
    por_tecnica = terminadas.groupby("tecnica")["duracion_simulada"]
    m.update({f"duracion_p50_{k}": v for k, v in por_tecnica.median().items()})
    m.update({f"duracion_max_{k}": v for k, v in por_tecnica.max().items()})
    m.update({f"fraccion_analista_p50_{k}": v
              for k, v in terminadas.groupby("tecnica")["fraccion_analista"].median().items()})
    m["configuraciones_elegibles_min"] = pedidos["configuraciones_compatibles"].min()
    m["configuraciones_elegibles_max"] = pedidos["configuraciones_compatibles"].max()
    m["densidad_configuraciones"] = pedidos["densidad_configuraciones"].mean()
    for tecnica in TECNICAS_CON_ALISTAMIENTO:
        m[f"alistamiento_{tecnica}"] = tablas["alistamiento"].at[(escenario, tecnica)]
    plazos = pedidos.groupby("tipo_muestra")["plazo_min"].median() / 60
    m["plazo_h_min"], m["plazo_h_max"] = plazos.min(), plazos.max()
    m["reanalisis_por_ensayo"] = corridas["reanalisis"].sum() / corridas["ensayos_pedidos"].sum()
    m["fallas_por_corrida"] = corridas["fallas"].mean()
    m["ventanas_por_corrida"] = corridas["ventanas"].mean()
    return m


def resumir_martins(instancias: pd.DataFrame, operaciones: pd.DataFrame) -> dict:
    """Medidas de las 54 instancias de Martins et al. (2023), con todas las operaciones juntas y la densidad de
    elegibilidad también por flexibilidad."""
    por_trabajo = operaciones.groupby(["instancia", "trabajo"]).size()
    reparto = por_trabajo.value_counts(normalize=True).sort_index()
    m = {
        "instancias": len(instancias), "trabajos": sorted(instancias["trabajos"].unique().tolist()),
        "operaciones_min": instancias["operaciones"].min(), "operaciones_max": instancias["operaciones"].max(),
        "analistas_por_instancia": sorted(instancias["analistas"].unique().tolist()),
        "maquinas_por_instancia": sorted(instancias["maquinas"].unique().tolist()),
        "flexibilidades": sorted(instancias["flexibilidad"].unique().tolist()),
        "tipos_de_trabajo": int(instancias["tipos_de_trabajo"].max()),
        "tareas_por_trabajo_media": por_trabajo.mean(), "tareas_por_trabajo_min": por_trabajo.min(),
        "tareas_por_trabajo_max": por_trabajo.max(), **{f"tareas_por_trabajo_{k}": v for k, v in reparto.items()},
        "intervenciones": int(operaciones["intervenciones"].max()),
        **_estadisticos(operaciones["analistas_elegibles"], "analistas_elegibles"),
        **_estadisticos(operaciones["maquinas_elegibles"], "equipos_elegibles"),
        **_estadisticos(operaciones["combinaciones"], "combinaciones"),
        "densidad_analistas": operaciones["densidad_analistas"].mean(),
        "densidad_equipos": operaciones["densidad_maquinas"].mean(),
        "densidad_combinaciones": operaciones["densidad_combinaciones"].mean(),
        **_estadisticos(operaciones["duracion"], "duracion"),
        **_estadisticos(operaciones["fraccion_analista"], "fraccion_analista"),
        "liberacion_max": operaciones["liberacion"].max(), "vencimiento_max": operaciones["vencimiento"].max(),
        "tipos_iguales": bool(_tipos_iguales(operaciones)),
    }
    for f, grupo in operaciones.groupby("flexibilidad"):
        m[f"densidad_analistas_f{f}"] = grupo["densidad_analistas"].mean()
        m[f"densidad_equipos_f{f}"] = grupo["densidad_maquinas"].mean()
    return m


def _tipos_iguales(operaciones: pd.DataFrame) -> bool:
    """True si, en cada instancia, todos los trabajos de un mismo tipo piden las mismas operaciones con los mismos
    tiempos (el «tipo de trabajo» de Martins et al., 2023, p. 191)."""
    firmas = (operaciones.sort_values("posicion").groupby(["instancia", "tipo", "trabajo"])["duracion"]
              .apply(tuple))
    return bool((firmas.groupby(["instancia", "tipo"]).nunique() == 1).all())


def resumir_tlsp_articulo() -> dict:
    """Medidas del TLSP según el artículo, para cuando no están los archivos de las instancias."""
    a, r = TLSP_ARTICULO, TLSP_REALES
    return {
        "proyectos_min": a["proyectos"][0], "proyectos_max": a["proyectos"][1],
        "franjas_min": a["franjas"][0], "franjas_max": a["franjas"][1],
        "trabajos_por_proyecto": a["trabajos_por_proyecto"],
        "empleados_calificados_por_trabajo": a["empleados_calificados_por_trabajo"],
        "mesas_por_trabajo": a["mesas_por_trabajo"],
        "reales_proyectos_min": r["proyectos"].min(), "reales_proyectos_max": r["proyectos"].max(),
        "reales_empleados_min": r["empleados"].min(), "reales_empleados_max": r["empleados"].max(),
        "reales_mesas": r["mesas"].max(), "reales_grupos": r["grupos_de_equipos"].max(),
        "reales_empleados_calificados_min": r["empleados_calificados_por_trabajo"].min(),
        "reales_empleados_calificados_max": r["empleados_calificados_por_trabajo"].max(),
        "reales_mesas_por_trabajo_min": r["mesas_por_trabajo"].min(),
        "reales_mesas_por_trabajo_max": r["mesas_por_trabajo"].max(),
        "reales_densidad_empleados_min": r["densidad_empleados"].min(),
        "reales_densidad_empleados_max": r["densidad_empleados"].max(),
        "reales_densidad_mesas_min": r["densidad_mesas"].min(),
        "reales_densidad_mesas_max": r["densidad_mesas"].max(),
    }


# ===================================================================== tabla
ND = "n. d. (falta medirlo en los archivos)"


def _rango(a, b, d=0) -> str:
    """«1 a 3»; si los extremos son enteros seguidos, «2 o 3»; si son iguales, uno solo."""
    if np.isclose(a, b):
        return _n(a, d)
    if d == 0 and float(a).is_integer() and np.isclose(b - a, 1):
        return f"{_n(a, 0)} o {_n(b, 0)}"
    return f"{_n(a, d)} a {_n(b, d)}"


def _lista(valores, d=0) -> str:
    textos = [_n(v, d) for v in valores]
    return textos[0] if len(textos) == 1 else ", ".join(textos[:-1]) + " o " + textos[-1]


def _por_flexibilidad(m: dict, recurso: str) -> str:
    """«42 % con f = 0,3 y de 61 % con f = 0,6», con las flexibilidades que haya en las instancias."""
    partes = [f"{_pct(m[f'densidad_{recurso}_f{f}'], 0)} con f = {_n(f, 1)}" for f in m["flexibilidades"]]
    return partes[0] if len(partes) == 1 else ", de ".join(partes[:-1]) + " y de " + partes[-1]


def tabla(este: dict, martins: dict, tlsp: dict) -> pd.DataFrame:
    """La tabla de contraste: una fila por medida, con su dimensión, y una columna por fuente. Las celdas son texto
    con el formato de los documentos (coma decimal, espacio de miles, sin rayas)."""
    e, m, t = este, martins, tlsp
    filas = []

    def fila(dimension, medida, a, b, c):
        filas.append({"Dimensión": dimension, "Medida": medida, ESTE: a, MARTINS: b, TLSP: c})

    # ------------------------------------------------------------------ trabajos y tareas
    d = "Trabajos y tareas"
    fila(d, "Trabajo y tarea",
         "Muestra y ensayo; cada ejecución del ensayo es un registro",
         "Trabajo y operación (pp. 182 y 183)",
         "Proyecto y tarea; las tareas de un proyecto se agrupan en trabajos (pp. 538 a 540)")
    fila(d, "Trabajos por instancia",
         f"{_n(e['muestras_base'], 0)} muestras y {_n(e['ensayos_base'], 0)} ensayos en {e['semanas']} semanas en el "
         f"escenario base, unas {_n(e['muestras_por_dia_base'], 0)} muestras por día ({_n(e['muestras_baja'], 0)} "
         f"muestras en el bajo y {_n(e['muestras_alta'], 0)} en el alto)",
         f"{_lista(m['trabajos'])} trabajos, con {_rango(m['operaciones_min'], m['operaciones_max'])} operaciones: "
         "10 trabajos son la carga de un día y 70 la de una semana (p. 191)",
         f"{_rango(t['proyectos_min'], t['proyectos_max'])} proyectos (p. 550); "
         f"{_rango(t['reales_proyectos_min'], t['reales_proyectos_max'])} en las tres instancias reales (Tabla 4)")
    fila(d, "Tareas por trabajo",
         f"{_rango(e['tareas_por_trabajo_min'], e['tareas_por_trabajo_max'])} ensayos por muestra "
         f"(media {_n(e['tareas_por_trabajo_media'], 2)}; {_pct(e.get('tareas_por_trabajo_1', 0), 0)} de las "
         "muestras pide uno)",
         f"{_rango(m['tareas_por_trabajo_min'], m['tareas_por_trabajo_max'])} operaciones por trabajo "
         f"(media {_n(m['tareas_por_trabajo_media'], 2)}); el artículo dice DU(1, 3) (Tabla 3, p. 192)",
         t.get("tareas_por_proyecto_texto", f"Cerca de {t['trabajos_por_proyecto']} trabajos por proyecto (p. 550); "
               f"tareas por proyecto: {ND}"))
    fila(d, "Tipos de trabajo",
         f"{e['tipos_de_trabajo']} tipos de muestra y {e['metodos']} métodos; cada tipo pide sus métodos con "
         "probabilidades del catálogo",
         f"{m['tipos_de_trabajo']} por instancia; todos los trabajos de un tipo piden las mismas operaciones "
         "con los mismos tiempos (p. 191)" if m["tipos_iguales"] else f"{m['tipos_de_trabajo']} por instancia",
         t.get("familias_texto", "Familias de tareas dentro de cada proyecto; solo tareas de la misma familia "
               "van en un trabajo (p. 539)"))
    fila(d, "Orden entre las tareas de un trabajo",
         "Ninguno: los ensayos de una muestra pueden ir a la vez (S28)",
         "Cadena: cada operación empieza cuando termina la anterior (restricción 5, p. 185)",
         t.get("precedencias_texto", "Opcional y escaso: la mayoría de las tareas no tiene predecesoras (p. 549)"))

    # ------------------------------------------------------------------ recursos
    d = "Tipos de recurso"
    fila(d, "Tipos de recurso",
         "Dos: analistas y equipos",
         "Dos: analistas y máquinas",
         "Tres: empleados, mesas de trabajo y dispositivos de grupos de equipos auxiliares (p. 538)")
    fila(d, "Cantidad de recursos",
         f"{e['analistas']} analistas en {e['turnos']} turnos; {e['equipos']} equipos en {e['configuraciones']} "
         f"configuraciones de {e['tecnicas']} técnicas",
         f"{_lista(m['analistas_por_instancia'])} analistas; {_lista(m['maquinas_por_instancia'])} máquinas (p. 191)",
         t.get("recursos_texto", f"En las instancias reales: {_rango(t['reales_empleados_min'], t['reales_empleados_max'])}"
               f" empleados, {t['reales_mesas']} mesas y {t['reales_grupos']} grupo de equipos (Tabla 4)"))
    fila(d, "Recursos que pide cada tarea",
         "Un analista calificado y un equipo compatible",
         "Un analista y una máquina elegibles",
         "Cero, uno o dos empleados según el modo, una mesa si la requiere y dispositivos de los grupos que pida "
         "(pp. 538 y 539)")
    fila(d, "Ocupación del analista durante la tarea",
         f"Parcial: en la mediana, {_pct(e['fraccion_analista_p50'], 0)} de la duración "
         f"({_pct(e['fraccion_analista_p10'], 0)} a {_pct(e['fraccion_analista_p90'], 0)} entre los percentiles 10 "
         "y 90); queda libre durante la idoneidad y la corrida",
         f"Parcial: {_pct(m['fraccion_analista_min'], 0)} a {_pct(m['fraccion_analista_max'], 0)} de la "
         f"operación, en {m['intervenciones']} intervenciones fijas (p. 191)",
         "Toda la duración del trabajo en los modos Single y Shift; ninguna en el External (Tabla 2, p. 548)")
    fila(d, "Disponibilidad de los recursos en el tiempo",
         f"Turnos con rotación, unas {_n(e['ventanas_por_corrida'], 0)} ventanas de mantenimiento y "
         f"{_n(e['fallas_por_corrida'], 0)} fallas de equipo al año",
         "Todos disponibles todo el tiempo",
         "Todos disponibles en todas las franjas; solo no pueden estar en dos trabajos a la vez (H8, p. 543)")

    # ------------------------------------------------------------------ elegibilidad
    d = "Densidad de elegibilidad"
    fila(d, "Analistas elegibles por tarea",
         f"{_rango(e['analistas_elegibles_min'], e['analistas_elegibles_max'])} de {e['analistas']} "
         f"(media {_n(e['analistas_elegibles_media'], 1)}): densidad de {_pct(e['densidad_analistas'], 0)} pesando "
         f"por la demanda y de {_pct(e['densidad_analistas_por_metodo'], 0)} por método",
         f"{_rango(m['analistas_elegibles_min'], m['analistas_elegibles_max'])} (media "
         f"{_n(m['analistas_elegibles_media'], 1)}): densidad de {_por_flexibilidad(m, 'analistas')}",
         t.get("empleados_texto", f"{t['empleados_calificados_por_trabajo']} empleados calificados por trabajo en "
               f"promedio (p. 550); en las reales, {_n(t['reales_empleados_calificados_min'], 1)} a "
               f"{_n(t['reales_empleados_calificados_max'], 1)}: densidad de "
               f"{_pct(t['reales_densidad_empleados_min'], 0)} a {_pct(t['reales_densidad_empleados_max'], 0)} (Tabla 4)"))
    fila(d, "Equipos elegibles por tarea",
         f"{_rango(e['equipos_elegibles_min'], e['equipos_elegibles_max'])} de {e['equipos']} unidades (media "
         f"{_n(e['equipos_elegibles_media'], 1)}): densidad de {_pct(e['densidad_equipos'], 0)} pesando por la "
         f"demanda y de {_pct(e['densidad_equipos_por_metodo'], 0)} por método; por configuración, "
         f"{_rango(e['configuraciones_elegibles_min'], e['configuraciones_elegibles_max'])} de "
         f"{e['configuraciones']} ({_pct(e['densidad_configuraciones'], 0)})",
         f"{_rango(m['equipos_elegibles_min'], m['equipos_elegibles_max'])} de {_lista(m['maquinas_por_instancia'])} (media "
         f"{_n(m['equipos_elegibles_media'], 1)}): densidad de {_por_flexibilidad(m, 'equipos')}; cada máquina "
         "representa un tipo de equipo (p. 191)",
         t.get("mesas_texto", f"{t['mesas_por_trabajo']} mesas disponibles por trabajo en promedio (p. 550); en las "
               f"reales, {_n(t['reales_mesas_por_trabajo_min'], 1)} a {_n(t['reales_mesas_por_trabajo_max'], 1)} de "
               f"{t['reales_mesas']}: densidad de {_pct(t['reales_densidad_mesas_min'], 0)} a "
               f"{_pct(t['reales_densidad_mesas_max'], 0)} (Tabla 4)"))
    fila(d, "Combinaciones analista y equipo por tarea",
         f"{_rango(e['combinaciones_min'], e['combinaciones_max'])} (mediana {_n(e['combinaciones_p50'], 0)}): "
         f"{_pct(e['densidad_combinaciones'], 0)} de los pares posibles",
         f"{_rango(m['combinaciones_min'], m['combinaciones_max'])} (mediana {_n(m['combinaciones_p50'], 0)}): "
         f"{_pct(m['densidad_combinaciones'], 0)} de los pares posibles",
         t.get("combinaciones_texto", ND))
    fila(d, "Cómo se reparte la elegibilidad",
         "Por área, técnica y nivel: un analista de cromatografía no hace fisicoquímica y cada método tiene sus "
         "configuraciones de equipo (diseño del laboratorio, secciones 3.1 y 3.3)",
         "Al azar: cada analista y cada máquina es elegible con probabilidad f (p. 191)",
         "Tomada de los datos del laboratorio del socio industrial (p. 549)")

    # ------------------------------------------------------------------ duraciones
    d = "Duraciones"
    fila(d, "Unidad", "Minutos",
         "Unidades de tiempo sin especificar, del orden de magnitud de las de Costigliola et al. (2017) (p. 190)",
         "Franjas de medio día, unas 4 horas (pp. 538 y 540), con decimales")
    fila(d, "Rango",
         f"{_n(e['duracion_min'], 0)} a {_n(e['duracion_max'], 0)} min; del percentil 10 al 90, "
         f"{_n(e['duracion_p10'], 0)} a {_n(e['duracion_p90'], 0)} (mediana {_n(e['duracion_p50'], 0)})",
         f"{_n(m['duracion_min'], 0)} a {_n(m['duracion_max'], 0)}, enteros; el artículo dice DU(1, 5) "
         "(Tabla 3, p. 192)",
         t.get("duracion_texto", ND))
    fila(d, "Dispersión relativa",
         f"Coeficiente de variación de {_n(e['duracion_cv'], 2)}; el percentil 90 es "
         f"{_n(e['duracion_p90_p10'], 1)} veces el 10",
         f"Coeficiente de variación de {_n(m['duracion_cv'], 2)}; el percentil 90 es "
         f"{_n(m['duracion_p90_p10'], 1)} veces el 10",
         t.get("dispersion_texto", ND))
    fila(d, "Variación de una misma tarea",
         "Aleatoria: BetaPERT por tarea, nivel del analista y correlación entre las tareas del analista",
         "Ninguna: la duración es un dato",
         "Ninguna: es un dato, que el modo Shift multiplica por 0,6 (Tabla 2, p. 548)")

    # ------------------------------------------------------------------ contexto
    d = "Contexto"
    fila(d, "Horizonte y llegadas",
         f"{e['semanas']} semanas, con llegadas continuas por tipo de muestra",
         "Sin horizonte: todos los trabajos están disponibles desde el comienzo"
         if m["liberacion_max"] == 0 else "Fechas de liberación",
         f"{_rango(t['franjas_min'], t['franjas_max'])} franjas ({_rango(t['franjas_min'] / 2, t['franjas_max'] / 2)} "
         "días); cada tarea tiene su fecha de liberación (pp. 539 y 550)")
    fila(d, "Plazos",
         f"Vencimiento por tipo de muestra, de {_n(e['plazo_h_min'], 0)} horas a {_n(e['plazo_h_max'] / 24, 0)} días; "
         f"en la mediana, el plazo es {_n(e['holgura_p50'], 0)} veces la duración del ensayo",
         f"No tiene: la liberación y el vencimiento valen 0 en las {m['instancias']} instancias"
         if m["vencimiento_max"] == 0 else "Vencimiento por trabajo",
         t.get("plazos_texto", "Fecha de entrega, que se puede incumplir con penalización, y plazo máximo, que no "
               "(p. 539)"))
    fila(d, "Alistamiento",
         f"Preparación e idoneidad del sistema en HPLC y GC, válida 24 horas (S29): en "
         f"{_pct(e['alistamiento_HPLC'], 0)} y {_pct(e['alistamiento_GC'], 0)} de sus ejecuciones",
         "Dentro de cada operación: la primera intervención del analista (p. 191)",
         "Un tiempo por familia, que se suma a cada trabajo; agrupar tareas lo ahorra (pp. 539 y 540)")
    fila(d, "Perturbaciones",
         f"Reanálisis ({_pct(e['reanalisis_por_ensayo'], 1)} de los ensayos), fallas de equipo y muestras urgentes",
         "Ninguna",
         "Ninguna: la instancia es estática y trae un programa base (p. 539)")
    return pd.DataFrame(filas)


# ===================================================================== todo junto
def _largo(fuente: str, medidas: dict) -> pd.DataFrame:
    filas = []
    for medida, valor in medidas.items():
        if isinstance(valor, (list, tuple)):
            valor = ", ".join(str(v) for v in valor)
        filas.append({"fuente": fuente, "medida": medida, "valor": valor})
    return pd.DataFrame(filas)


def contrastar(registros: pd.DataFrame, resumenes: list, carpeta_martins: Path | str = CARPETA_MARTINS) -> dict:
    """Calcula el contraste: devuelve la tabla (texto), las medidas (números) y el detalle por corrida e instancia."""
    tablas = medir_dataset(registros, resumenes)
    instancias_m, operaciones_m = leer_martins(carpeta_martins)
    este = resumir_dataset(tablas)
    martins = resumir_martins(instancias_m, operaciones_m)
    tlsp = resumir_tlsp_articulo()
    medidas = pd.concat([_largo(ESTE, este), _largo(MARTINS, martins), _largo(TLSP, tlsp)], ignore_index=True)
    instancias = pd.concat([
        tablas["corridas"].assign(fuente=ESTE, instancia=lambda x: x["escenario"] + "_" + x["semilla"].astype(str)),
        instancias_m.assign(fuente=MARTINS),
        TLSP_REALES.assign(fuente=TLSP + ", instancias reales (Tabla 4)"),
    ], ignore_index=True)
    columnas = ["fuente", "instancia"] + [c for c in instancias.columns if c not in ("fuente", "instancia")]
    return {"tabla": tabla(este, martins, tlsp), "medidas": medidas, "instancias": instancias[columnas],
            "este": este, "martins": martins, "tlsp": tlsp, "operaciones_martins": operaciones_m}


def guardar(contraste: dict, carpeta: Path | str | None = None) -> list:
    """Escribe la tabla, las medidas y el detalle por instancia en resultados/tablas. Devuelve las rutas."""
    carpeta = Path(carpeta) if carpeta else CARPETA_TABLAS
    carpeta.mkdir(parents=True, exist_ok=True)
    rutas = [carpeta / "contraste_estructural.csv", carpeta / "contraste_estructural_medidas.csv",
             carpeta / "contraste_estructural_instancias.csv"]
    contraste["tabla"].to_csv(rutas[0], index=False)
    contraste["medidas"].to_csv(rutas[1], index=False)
    contraste["instancias"].to_csv(rutas[2], index=False)
    return rutas
