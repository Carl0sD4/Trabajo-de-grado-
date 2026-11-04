"""Disponibilidad cruzada: combinaciones analista–equipo factibles de cada actividad y si coinciden en el tiempo.

Anteproyecto: dimensión «Disponibilidad cruzada de recursos» (§6.1.3): un analista disponible no
genera capacidad efectiva si el equipo está en uso, en mantenimiento o fuera de servicio.
Indicadores (hoja «Indicadores»): C1, número de combinaciones factibles analista–equipo por
actividad; C2, porcentaje de actividades con al menos una combinación factible; C3, coincidencia
temporal entre analista calificado y equipo compatible; C4, porcentaje de actividades restringidas
por falta de alguno de los recursos requeridos.
Diseño: docs/Diseno_laboratorio_referencia.docx, secciones 3.3 y 4.3, y docs/Generador_de_datos.docx,
sección 4.

Las combinaciones factibles salen de las matrices (hojas «Analistas» y «Equipos»); las libres, del
estado de los recursos cuando llega cada registro, que guarda el motor (src/simulacion/entorno.py)
en analistas_libres, equipos_libres y recurso_faltante.
"""
import numpy as np
import pandas as pd

from src.generador.recursos import MIN_POR_DIA, Laboratorio

FALTANTES = ("ninguno", "analista", "equipo", "ambos")


def combinaciones_factibles(metodo: str, laboratorio: Laboratorio = None) -> list:
    """Parejas (analista, equipo) que pueden ejecutar el método según las matrices de calificación y
    compatibilidad (S12, S20 y S22)."""
    return (laboratorio or Laboratorio.desde_excel()).combinaciones_factibles(metodo)


def cobertura_en_el_tiempo(dias: int, eventos: pd.DataFrame = None, laboratorio: Laboratorio = None,
                           paso: int = 60) -> pd.DataFrame:
    """Fracción del tiempo en que cada método tiene a la vez un analista calificado en turno y un equipo
    compatible en servicio (la coincidencia que permiten los calendarios, sin contar la ocupación).

    eventos: tabla de eventos de los equipos de la corrida (ventanas y fallas); sin ella se cuentan
    todos los equipos en servicio. Del lado de los analistas debe dar 100 % en los diez métodos de
    complejidad baja y media y unas 125 de las 168 horas de la semana (74 %) en M08 y M12 (diseño del
    laboratorio, sección 4.3).
    """
    lab = laboratorio or Laboratorio.desde_excel()
    t = np.arange(0, dias * MIN_POR_DIA, paso) + paso / 2
    cal = lab.calendario_analistas(dias)
    en_turno = {a: np.zeros(len(t), dtype=bool) for a in lab.analistas}
    for a, ini, fin in zip(cal["analista"], cal["inicio"], cal["fin"]):
        en_turno[a] |= (t >= ini) & (t < fin)
    en_servicio = {e: np.ones(len(t), dtype=bool) for e in lab.equipos}
    if eventos is not None:
        for e, ini, fin in zip(eventos["equipo"], eventos["inicio"], eventos["fin"]):
            en_servicio[e] &= ~((t >= ini) & (t < fin))
    filas = []
    for m in lab.metodos:
        analista = np.any([en_turno[a] for a in lab.calificados[m]], axis=0)
        equipo = np.any([en_servicio[e] for e in lab.compatibles[m]], axis=0)
        filas.append({"metodo": m, "con_analista": analista.mean(), "con_equipo": equipo.mean(),
                      "con_ambos": (analista & equipo).mean()})
    return pd.DataFrame(filas).set_index("metodo")


def indicadores(registros: pd.DataFrame, desde: float = 0.0) -> dict:
    """C1 a C4 a partir de los registros; C3 y C4 solo con los que llegaron desde `desde` (calentamiento).

    Devuelve un diccionario de tablas: c1 (combinaciones factibles por método), c2 (porcentaje con al
    menos una), c3_c4 (porcentaje de registros según lo que faltaba al llegar, por escenario) y
    libres (analistas y equipos libres promedio al llegar, por escenario y método).
    """
    r = registros
    c1 = r.groupby("metodo")["combinaciones_factibles"].agg(["min", "max"]).rename(
        columns={"min": "combinaciones_min", "max": "combinaciones_max"})
    c2 = pd.Series({"registros": len(r), "con_al_menos_una": int((r["combinaciones_factibles"] >= 1).sum()),
                    "porcentaje": 100 * float((r["combinaciones_factibles"] >= 1).mean()) if len(r) else np.nan})
    v = r[r["llegada"] >= desde]
    tabla = (pd.crosstab(v["escenario"], v["recurso_faltante"], normalize="index") * 100).reindex(columns=FALTANTES,
                                                                                                  fill_value=0.0)
    tabla["restringidas"] = 100 - tabla["ninguno"]
    libres = v.groupby(["escenario", "metodo"])[["analistas_libres", "equipos_libres"]].mean()
    return {"c1": c1, "c2": c2, "c3_c4": tabla, "libres": libres}
