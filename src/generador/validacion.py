"""Validación del dataset con los 22 indicadores del sistema de variables.

Anteproyecto: §8.5, variable «datos sintéticos» y sus cinco dimensiones: tiempos de ejecución,
recursos humanos, recursos materiales, disponibilidad cruzada y variabilidad y consistencia
estructural. La hoja «Indicadores» del Excel de parámetros dice, para cada indicador, qué se calcula,
con qué campos del registro, a qué nivel y frente a qué referencia, y el diccionario de datos
(docs/Diccionario_de_datos.docx, sección 3) explica el amarre. Los resultados están en
docs/Validacion_dataset.docx; las tablas y las figuras salen del cuaderno
notebooks/03_validacion_dataset.ipynb.

Uso:
    from src.generador import generar, validacion
    registros, eventos, _ = generar.cargar_version("v0.2")
    v = validacion.validar(registros, eventos, "v0.2")
    v["resumen"]                 # una fila por indicador: valor, referencia y veredicto
    v["T5"]["tablas"]["tareas"]  # el detalle de cada indicador
    validacion.guardar(v, "v0.2")

Las funciones reciben los registros y los eventos de los equipos de una o varias corridas; las
columnas escenario y semilla las separan. Las colas, la carga y el tiempo en el sistema (H4, C3, C4
y el de V1) se miden sin las dos primeras semanas, en que el laboratorio arranca vacío
(escenarios.CALENTAMIENTO_DIAS).

Veredictos: «cumple» o «no cumple» cuando el indicador tiene un criterio de aceptación (el 100 % de
las reglas, el 0 % fuera de rango, la tolerancia de V2, las horas del diseño...) y «se reporta»
cuando la referencia es un orden de magnitud o una tendencia que se discute en el informe.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.generador import correlacion, disponibilidad, duraciones, escenarios, parametros, perturbaciones, registro
from src.generador.recursos import MIN_POR_DIA, MIN_POR_SEMANA, RANGO_NIVEL, TECNICAS, Laboratorio

CORRIDA = registro.CORRIDA
TOL = registro.TOL
TAREAS = registro.TAREAS
TAREAS_EQUIPO = ("t_montaje", "t_idoneidad", "t_corrida")   # el equipo está en uso (indicador M3)
CUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# Criterios de aceptación
Z_MEDIA = 5                    # T5: la media de cada tarea, a menos de 5 errores estándar de su media esperada
ALFA_V1 = 0.05                 # V1: Kolmogorov-Smirnov de cada tarea frente a su BetaPERT, con Bonferroni
HORAS_POR_TURNO = {"T1": 56.0, "T2": 37.5, "T3": 40.0}   # diseño del laboratorio, Tabla 15
HORAS_ANALISTA_SEMANA = 886.0  # diseño del laboratorio, sección 4.2
TOLERANCIA_HORAS = 1.0         # H3: horas por semana de cada turno
ANALISTAS_EN_TURNO = (2, 14)   # H1: mínimo a toda hora y máximo en un día hábil (diseño, Tabla 16)
EQUIPOS_TOTAL, EQUIPOS_POR_CONFIGURACION = 70, 2   # M1: Costigliola et al. (2017, p. 9015) y S21
DISPONIBILIDAD_DISENO = (0.948, 0.949)             # M3: diseño del laboratorio, Tabla 17
TOLERANCIA_FALLA = 0.01       # M4: tiempo en falla, a menos de un punto porcentual de S02
COITO_90 = (458.0, 786.0)      # T5: el 90 % de los ensayos de HPLC y GC (Coito et al., 2021, p. 74)

CUMPLE, NO_CUMPLE, SE_REPORTA = "cumple", "no cumple", "se reporta"


# ===================================================================== utilidades
def _n(x, d=1) -> str:
    """Número con coma decimal y espacio de miles, como en los documentos: 10 950 · 0,3 · 94,8."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n. d."
    texto = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return texto


def _pct(x, d=1) -> str:
    return f"{_n(100 * x, d)} %"


def _veredicto(ok) -> str:
    return CUMPLE if bool(ok) else NO_CUMPLE


def _indicador(valor, veredicto, **tablas) -> dict:
    return {"valor": valor, "veredicto": veredicto, "tablas": tablas}


def _corridas(registros):
    return registros.groupby(CORRIDA, sort=True)


def _factor(registros) -> pd.Series:
    return registros["nivel_analista"].map({n: registro.factor_duracion(n) for n in registro.NIVELES})


def _puntos(registros) -> pd.DataFrame:
    """Mínimo, moda y máximo de cada tarea de cada registro, recalculados como en el motor: hoja
    «Duraciones», catálogo y, si el registro ejecutó la idoneidad, S29. Columnas a_<tarea>, m_<tarea>
    y b_<tarea>, con el índice de los registros."""
    claves = ["metodo", "n_soluciones", "idoneidad_ejecutada"]
    combos = registros[claves + ["tecnica"]].drop_duplicates(claves)
    filas = []
    for metodo, n, ido, tecnica in combos.itertuples(index=False):
        sis, idon = duraciones.tres_puntos_alistamiento(tecnica) if ido else ((0.0,) * 3, (0.0,) * 3)
        p = registro.tres_puntos_tareas(metodo, int(n), sis, idon)
        fila = {"metodo": metodo, "n_soluciones": n, "idoneidad_ejecutada": ido}
        for t in TAREAS:
            fila.update({f"a_{t}": p[t][0], f"m_{t}": p[t][1], f"b_{t}": p[t][2]})
        filas.append(fila)
    tabla = pd.DataFrame(filas).astype({"n_soluciones": registros["n_soluciones"].dtype,
                                        "idoneidad_ejecutada": registros["idoneidad_ejecutada"].dtype})
    return registros[claves].merge(tabla, on=claves, how="left").drop(columns=claves).set_axis(registros.index)


def _formas(registros, forma) -> pd.DataFrame:
    """λ de cada tarea de cada registro (duraciones.formas_por_tarea, según la versión)."""
    por_tecnica = {t: dict(zip(TAREAS, duraciones.formas_por_tarea(t, forma))) for t in registros["tecnica"].unique()}
    return pd.DataFrame({t: registros["tecnica"].map({k: v[t] for k, v in por_tecnica.items()}) for t in TAREAS})


def _terminados(registros) -> pd.DataFrame:
    return registros[registros["resultado"].ne("interrumpido")]


def _unir(intervalos) -> list:
    """Une los intervalos [inicio, fin) que se traslapan."""
    unidos = []
    for a, z in sorted(intervalos):
        if unidos and a <= unidos[-1][1]:
            unidos[-1][1] = max(unidos[-1][1], z)
        else:
            unidos.append([a, z])
    return unidos


def _minutos_por_semana(calendario) -> pd.Series:
    """Minutos en turno de cada analista en cada semana; los tramos que cruzan el cambio de semana se parten."""
    ini, fin = calendario["inicio"].to_numpy(float), calendario["fin"].to_numpy(float)
    semana = (ini // MIN_POR_SEMANA).astype(int)
    corte = (semana + 1) * MIN_POR_SEMANA
    partes = pd.concat([
        pd.DataFrame({"analista": calendario["analista"], "semana": semana, "minutos": np.minimum(fin, corte) - ini}),
        pd.DataFrame({"analista": calendario["analista"], "semana": semana + 1, "minutos": np.maximum(fin - corte, 0.0)}),
    ])
    return partes[partes["minutos"] > 0].groupby(["analista", "semana"])["minutos"].sum()


def tiempo_en_sistema_fuente() -> pd.Series:
    """Tiempo promedio en el sistema por técnica, en horas (Costigliola et al., 2017, Tabla 5, p. 9018; P40)."""
    hoja = parametros.cargar_hoja("Parámetros").set_index("ID")
    texto = str(hoja.at["P40", "Valor"])
    pares = re.findall(r"([A-Z]+) (\d+(?:,\d+)?) ±", texto)
    return pd.Series({t: float(v.replace(",", ".")) for t, v in pares}, name="horas").reindex(TECNICAS)


# ===================================================================== tiempos de ejecución (T1 a T5)
def tiempos(registros, forma="hoja") -> dict:
    """T1 a T5: tres puntos, duración simulada y media y dispersión de las duraciones.

    forma: la λ de la versión del dataset (escenarios.VERSIONES), para la media BetaPERT esperada.
    """
    r = registros
    puntos = _puntos(r)
    esperado = pd.DataFrame({k: registro.duracion(*(puntos[f"{p}_{t}"] for t in TAREAS))
                             for k, p in (("t_optimista", "a"), ("t_mas_probable", "m"), ("t_pesimista", "b"))})
    coincide = (r[list(esperado.columns)] - esperado).abs().lt(1e-5)
    ordenados = r["t_optimista"].gt(0) & (r["t_optimista"] <= r["t_mas_probable"] + TOL) & (
        r["t_mas_probable"] <= r["t_pesimista"] + TOL)
    por_metodo = r.groupby("metodo").agg(
        registros=("metodo", "size"), optimista_min=("t_optimista", "min"), optimista_media=("t_optimista", "mean"),
        optimista_max=("t_optimista", "max"), mas_probable_media=("t_mas_probable", "mean"),
        pesimista_media=("t_pesimista", "mean"))
    salida = {}
    for codigo, campo in (("T1", "t_optimista"), ("T2", "t_mas_probable"), ("T3", "t_pesimista")):
        ok = coincide[campo] & ordenados
        salida[codigo] = _indicador(
            f"{_pct(ok.mean())} de los registros con el valor que dan la hoja Duraciones y el catálogo, y ordenado",
            _veredicto(ok.all()), por_metodo=por_metodo)

    # T4: la duración de cada ejecución terminada, entre los tres puntos corregidos por el factor del analista
    f = _factor(r)
    terminada = r["resultado"].ne("interrumpido")
    suma = registro.duracion(*(r[t] for t in TAREAS))
    dentro = r["duracion_simulada"].between(np.minimum(f, 1) * r["t_optimista"] - TOL,
                                            np.maximum(f, 1) * r["t_pesimista"] + TOL)
    ok4 = (r["duracion_simulada"] - suma).abs().lt(1e-5) & (~terminada | dentro)
    t4 = r[terminada].groupby("metodo")["duracion_simulada"].agg(["size", "mean", "std", "min", "max"])
    t4.columns = ["registros", "media", "desviacion", "minimo", "maximo"]
    salida["T4"] = _indicador(f"{_pct(ok4.mean())} de los registros con la duración entre sus tiempos optimista y "
                              "pesimista corregidos y igual a la suma de sus tareas", _veredicto(ok4.all()), por_metodo=t4)

    # T5: media y desviación por escenario, técnica y método; media de cada tarea frente a la esperada y a la fuente.
    # La esperada es la media BetaPERT de los tres puntos de cada registro con la λ de la versión, por el factor del
    # analista en sus tareas; se acepta a menos de Z_MEDIA errores estándar (raíz de la suma de las varianzas BetaPERT
    # de los registros, entre su número), por escenario: escenarios distintos comparten semillas.
    tt = _terminados(r)
    pt, lam, ft = puntos.loc[tt.index], _formas(tt, forma), f.loc[tt.index]
    esperada = pd.DataFrame({t: duraciones.media_betapert(pt[f"a_{t}"], pt[f"m_{t}"], pt[f"b_{t}"], lam[t].to_numpy())
                             for t in TAREAS}, index=tt.index)
    varianza = pd.DataFrame({t: duraciones.varianza_betapert(pt[f"a_{t}"], pt[f"m_{t}"], pt[f"b_{t}"], lam[t].to_numpy())
                             for t in TAREAS}, index=tt.index)
    for t in registro.TAREAS_ANALISTA:
        esperada[t] *= ft
        varianza[t] *= ft ** 2
    fuente = duraciones.tiempos_promedio_fuente()
    filas = []
    for (esc, tecnica), g in tt.groupby(["escenario", "tecnica"]):
        for t in TAREAS:
            con = g.index if t not in duraciones.TAREAS_ALISTAMIENTO else g.index[g["idoneidad_ejecutada"]]
            if not len(con) or esperada.loc[con, t].eq(0).all():
                continue
            sim, esp = g.loc[con, t].mean(), esperada.loc[con, t].mean()
            se = float(np.sqrt(varianza.loc[con, t].sum()) / len(con))
            filas.append({"escenario": esc, "tecnica": tecnica, "tarea": t, "registros": len(con), "media_simulada": sim,
                          "media_esperada": esp, "diferencia_relativa": sim / esp - 1, "error_estandar": se,
                          "z": (sim - esp) / se if se > 0 else 0.0,
                          "tabla_6": fuente[t].get(tecnica, np.nan) if t in fuente else np.nan})
    tareas = pd.DataFrame(filas)
    tareas["frente_a_tabla_6"] = tareas["media_simulada"] / tareas["tabla_6"] - 1
    juntas = (tt.groupby("tecnica")[list(TAREAS)].mean().stack().rename("media_simulada").to_frame()
              .rename_axis(["tecnica", "tarea"]).reset_index())
    for t in duraciones.TAREAS_ALISTAMIENTO:   # el alistamiento, solo en los registros que lo ejecutaron
        con = tt[tt["idoneidad_ejecutada"]].groupby("tecnica")[t].mean()
        juntas.loc[juntas["tarea"].eq(t), "media_simulada"] = juntas.loc[juntas["tarea"].eq(t), "tecnica"].map(con)
    juntas["tabla_6"] = [fuente.at[tec, t] if t in fuente and tec in fuente.index else np.nan
                         for tec, t in zip(juntas["tecnica"], juntas["tarea"])]
    juntas["frente_a_tabla_6"] = juntas["media_simulada"] / juntas["tabla_6"] - 1
    referencia = juntas[juntas["tecnica"].isin(registro.TECNICAS_CON_IDONEIDAD)
                        & juntas["tarea"].isin(["t_preparacion", "t_corrida"])]
    duracion = tt.groupby(["escenario", "tecnica", "metodo"])["duracion_simulada"].agg(["size", "mean", "std"])
    duracion.columns = ["registros", "media", "desviacion"]
    por_tecnica = tt.groupby(["escenario", "tecnica"])["duracion_simulada"].agg(["size", "mean", "std"])
    por_tecnica.columns = ["registros", "media", "desviacion"]
    # contraste de orden de magnitud con Coito et al. (2021, p. 74): HPLC y GC con alistamiento, sin procesamiento
    con_al = tt[tt["tecnica"].isin(registro.TECNICAS_CON_IDONEIDAD) & tt["idoneidad_ejecutada"]]
    sin_proc = con_al["duracion_simulada"] - con_al["t_procesamiento"]
    coito = pd.DataFrame([{"registros": len(sin_proc), "p05": sin_proc.quantile(0.05) if len(sin_proc) else np.nan,
                           "p95": sin_proc.quantile(0.95) if len(sin_proc) else np.nan,
                           "coito_p05": COITO_90[0], "coito_p95": COITO_90[1]}])
    peor = float(tareas["z"].abs().max())
    texto = (f"Media de cada tarea a {_n(peor, 1)} errores estándar o menos de su media BetaPERT esperada (diferencia "
             f"de {_pct(tareas['diferencia_relativa'].abs().max())} o menos); preparación y corrida de HPLC y GC frente a "
             "la Tabla 6: " + ", ".join(f"{a} {b.replace('t_', '')} {_pct(c)}" for a, b, c in
                                       referencia[["tecnica", "tarea", "frente_a_tabla_6"]].itertuples(index=False)))
    if len(sin_proc):
        texto += f"; HPLC y GC con alistamiento, sin procesamiento: {_n(coito.at[0, 'p05'], 0)} a {_n(coito.at[0, 'p95'], 0)} min"
    salida["T5"] = _indicador(texto, _veredicto(peor <= Z_MEDIA), tareas=tareas, frente_a_la_fuente=juntas,
                              por_tecnica=por_tecnica, por_metodo=duracion, coito=coito)
    return salida


# ===================================================================== recursos humanos (H1 a H4)
def recursos_humanos(registros, laboratorio: Laboratorio, dias: int, desde: float = 0.0) -> dict:
    """H1 a H4: analistas en turno, calificación, disponibilidad temporal y carga."""
    r, lab = registros, laboratorio
    salida = {}

    # H1: analistas en turno en cada hora y analistas calificados libres al llegar cada registro
    turno = lab.analistas_en_turno(dias, 60)
    habil = (turno["inicio"] // MIN_POR_DIA) % 7 < 5
    perfil = (turno.assign(dia=np.where(habil, "hábil", "fin de semana"), hora=(turno["inicio"] % MIN_POR_DIA) // 60)
              .groupby(["dia", "hora"])["analistas"].agg(["min", "max", "mean"]))
    minimo, maximo = int(turno["analistas"].min()), int(turno["analistas"][habil].max())
    libres = r.groupby(["escenario", "metodo"])["analistas_libres"].mean().unstack("escenario")
    salida["H1"] = _indicador(
        f"Entre {minimo} y {maximo} analistas en turno (máximo en fin de semana: "
        f"{int(turno['analistas'][~habil].max())}); calificados libres al llegar, en promedio: " + ", ".join(
            f"{e} {_n(v)}" for e, v in r.groupby("escenario")["analistas_libres"].mean().items()),
        _veredicto((minimo, maximo) == ANALISTAS_EN_TURNO), perfil=perfil, libres_al_llegar=libres)

    # H2: el analista de cada registro está calificado en el método y tiene el nivel exigido (S12)
    calificado = pd.Series([a in lab.calificados[m] for a, m in zip(r["analista"], r["metodo"])], index=r.index)
    nivel = r["nivel_analista"].map(RANGO_NIVEL) >= r["calificacion_exigida"].map(RANGO_NIVEL)
    ok = calificado & nivel
    usados = r.groupby("metodo")["analista"].nunique()
    tabla = lab.cobertura()[["analistas_calificados"]].assign(analistas_que_lo_ejecutaron=usados)
    salida["H2"] = _indicador(f"{_pct(ok.mean())} de los registros con un analista calificado del nivel exigido",
                              _veredicto(ok.all()), por_metodo=tabla)

    # H3: horas en turno por semana y analista en turno al empezar cada ejecución
    cal = lab.calendario_analistas(dias)
    horas = (cal.assign(h=(cal["fin"] - cal["inicio"]) / 60).groupby("analista")["h"].sum() / (dias / 7)).rename("horas")
    por_turno = horas.groupby(pd.Series({a: x.turno for a, x in lab.analistas.items()})).mean()
    tramos = {a: (g["inicio"].to_numpy(), g["fin"].to_numpy()) for a, g in cal.groupby("analista")}
    en_turno = np.zeros(len(r), dtype=bool)
    for a, idx in r.groupby("analista").indices.items():
        ini, fin = tramos[a]
        t = r["inicio"].to_numpy()[idx]
        k = np.searchsorted(ini, t, side="right") - 1
        en_turno[idx] = (k >= 0) & (t < fin[np.maximum(k, 0)])
    pausas = r.groupby(["escenario", "tipo_muestra"])["pausas"].mean().unstack("escenario") / 60
    total = float(horas.sum())
    ok3 = all(abs(por_turno[t] - h) <= TOLERANCIA_HORAS for t, h in HORAS_POR_TURNO.items()) and en_turno.all()
    salida["H3"] = _indicador(
        "Horas por semana: " + ", ".join(f"turno {t[1]} {_n(por_turno[t])}" for t in sorted(HORAS_POR_TURNO))
        + f"; {_n(total)} horas-analista por semana (diseño: {_n(HORAS_ANALISTA_SEMANA, 0)}); "
          f"{_pct(en_turno.mean())} de las ejecuciones empiezan con su analista en turno", _veredicto(ok3),
        horas_por_analista=horas.to_frame().assign(turno=lambda d: d.index.map(lambda a: lab.analistas[a].turno)),
        pausas_horas=pausas)

    # H4: trabajo del analista por semana (sus cuatro tareas) entre sus horas en turno, sin el calentamiento
    v = r[r["inicio"] >= desde]
    trabajo = (v.assign(trabajo=v[list(registro.TAREAS_ANALISTA)].sum(axis=1),
                        semana=(v["inicio"] // MIN_POR_SEMANA).astype(int))
               .groupby(["escenario", "semilla", "analista", "semana"])["trabajo"].sum())
    en_turno_semana = _minutos_por_semana(cal)
    semanas = sorted({s for s in en_turno_semana.index.get_level_values("semana")
                      if s * MIN_POR_SEMANA >= desde and (s + 1) * MIN_POR_SEMANA <= dias * MIN_POR_DIA})
    filas = []
    for (esc, sem), _ in _corridas(r):
        for (a, s), minutos in en_turno_semana.items():
            if s in semanas:
                filas.append({"escenario": esc, "semilla": sem, "analista": a, "semana": s,
                              "turno": lab.analistas[a].turno, "carga": trabajo.get((esc, sem, a, s), 0.0) / minutos})
    carga = pd.DataFrame(filas)
    por_escenario = carga.groupby("escenario")["carga"].describe(percentiles=[0.5, 0.9])
    por_analista = carga.groupby(["escenario", "analista"])["carga"].mean().unstack("escenario")
    salida["H4"] = _indicador(
        "Trabajo asignado entre horas en turno, promedio de los analistas: " + ", ".join(
            f"{e} {_pct(x)}" for e, x in carga.groupby("escenario")["carga"].mean().items())
        + "; percentil 90: " + ", ".join(f"{e} {_pct(x)}" for e, x in
                                          carga.groupby("escenario")["carga"].quantile(0.9).items()),
        SE_REPORTA, por_escenario=por_escenario, por_analista=por_analista,
        por_turno=carga.groupby(["escenario", "turno"])["carga"].mean().unstack("escenario"))
    return salida


# ===================================================================== recursos materiales (M1 a M4)
def estado_equipos(eventos, laboratorio: Laboratorio, horizonte: float) -> pd.DataFrame:
    """Minutos de cada unidad en falla, en ventana (mantenimiento o verificación) y fuera de servicio
    (la unión de los dos), y número de ventanas, en cada corrida, dentro de [0, horizonte)."""
    filas = []
    for (esc, sem), g in eventos.groupby(CORRIDA):
        por_equipo = {e: x for e, x in g.groupby("equipo")}
        for e, eq in laboratorio.equipos.items():
            x = por_equipo.get(e)
            intervalos = {"falla": [], "ventana": []}
            n_ventanas = 0
            if x is not None:
                for evento, a, z in x[["evento", "inicio", "fin"]].itertuples(index=False):
                    a, z = max(a, 0.0), min(z, horizonte)
                    if z <= a:
                        continue
                    clase = "falla" if evento == "falla" else "ventana"
                    intervalos[clase].append((a, z))
                    n_ventanas += clase == "ventana"
            medir = lambda iv: sum(z - a for a, z in _unir(iv))   # noqa: E731
            filas.append({"escenario": esc, "semilla": sem, "equipo": e, "configuracion": eq.configuracion,
                          "falla": medir(intervalos["falla"]), "ventana": medir(intervalos["ventana"]),
                          "fuera": medir(intervalos["falla"] + intervalos["ventana"]), "ventanas": n_ventanas})
    return pd.DataFrame(filas)


def _minimo_en_servicio(eventos, laboratorio, horizonte) -> pd.DataFrame:
    """Menor número de unidades en servicio de cada configuración en cada corrida (barrido de los intervalos)."""
    filas = []
    for (esc, sem), g in eventos.groupby(CORRIDA):
        for config, unidades in laboratorio.configuraciones["unidades"].items():
            cambios = []
            for _, x in g[g["configuracion"].eq(config)].groupby("equipo"):
                for a, z in _unir(list(zip(x["inicio"].clip(0, horizonte), x["fin"].clip(0, horizonte)))):
                    if z > a:
                        cambios += [(a, 1), (z, -1)]
            fuera = peor = 0
            for _, paso in sorted(cambios, key=lambda c: (c[0], c[1])):
                fuera += paso
                peor = max(peor, fuera)
            filas.append({"escenario": esc, "semilla": sem, "configuracion": config, "unidades": int(unidades),
                          "minimo_en_servicio": int(unidades) - peor})
    return pd.DataFrame(filas)


def recursos_materiales(registros, eventos, laboratorio: Laboratorio, dias: int) -> dict:
    """M1 a M4: unidades en servicio, compatibilidad, disponibilidad temporal y condición de los equipos."""
    r, lab = registros, laboratorio
    horizonte = dias * MIN_POR_DIA
    estado = estado_equipos(eventos, lab, horizonte)
    salida = {}

    # M1: unidades por configuración, unidades en servicio a lo largo de la corrida y libres al llegar
    unidades = lab.configuraciones["unidades"]
    minimo = _minimo_en_servicio(eventos, lab, horizonte)
    en_servicio = (estado.assign(servicio=1 - estado["fuera"] / horizonte)
                   .groupby(["escenario", "semilla", "configuracion"])["servicio"].sum()
                   .groupby(["escenario", "configuracion"]).mean().unstack("escenario"))
    tabla = pd.concat([unidades.rename("unidades"), en_servicio.add_prefix("promedio_en_servicio_"),
                       minimo.groupby(["configuracion", "escenario"])["minimo_en_servicio"].min().unstack("escenario")
                       .add_prefix("minimo_en_servicio_")], axis=1)
    ok1 = int(unidades.sum()) == EQUIPOS_TOTAL and int(unidades.min()) >= EQUIPOS_POR_CONFIGURACION
    sin_unidad = int((minimo["minimo_en_servicio"] == 0).sum())
    salida["M1"] = _indicador(
        f"{int(unidades.sum())} unidades en {len(unidades)} configuraciones, de {int(unidades.min())} a "
        f"{int(unidades.max())} por configuración; {sin_unidad} de {len(minimo)} configuraciones-corrida "
        "quedaron en algún momento sin unidades en servicio; compatibles libres al llegar, en promedio: "
        + ", ".join(f"{e} {_n(v)}" for e, v in r.groupby("escenario")["equipos_libres"].mean().items()),
        _veredicto(ok1), por_configuracion=tabla,
        libres_al_llegar=r.groupby(["escenario", "metodo"])["equipos_libres"].mean().unstack("escenario"))

    # M2: configuración compatible con el método (S22)
    ok = pd.Series([isinstance(q, str) and c in q.split("|") for c, q in zip(r["configuracion"], r["equipo_requerido"])],
                   index=r.index) & pd.Series([e in lab.compatibles[m] for e, m in zip(r["equipo"], r["metodo"])],
                                              index=r.index)
    salida["M2"] = _indicador(f"{_pct(ok.mean())} de los registros en una unidad compatible con su método",
                              _veredicto(ok.all()),
                              por_configuracion=r.groupby(["metodo", "configuracion"]).size().unstack(fill_value=0))

    # M3: fracción del tiempo en servicio y en uso (montaje, idoneidad y corrida) de cada unidad
    uso = (r.assign(uso=r[list(TAREAS_EQUIPO)].sum(axis=1)).groupby(CORRIDA + ["equipo"])["uso"].sum() / horizonte)
    unidad = estado.set_index(CORRIDA + ["equipo"])
    unidad = unidad.assign(en_servicio=1 - unidad["fuera"] / horizonte,
                           en_uso=uso.reindex(unidad.index, fill_value=0.0)).reset_index()
    por_config = unidad.groupby(["escenario", "configuracion"])[["en_servicio", "en_uso"]].mean()
    disp = unidad.groupby("escenario")["en_servicio"].mean()
    usos = unidad.groupby("escenario")["en_uso"].mean()
    salida["M3"] = _indicador(
        "En servicio, promedio de las unidades: " + ", ".join(f"{e} {_pct(x)}" for e, x in disp.items())
        + f" (diseño: {_pct(DISPONIBILIDAD_DISENO[0])} a {_pct(DISPONIBILIDAD_DISENO[1])}); en uso: "
        + ", ".join(f"{e} {_pct(x)}" for e, x in usos.items()), SE_REPORTA, por_configuracion=por_config,
        por_unidad=unidad)

    # M4: tiempo en falla y en ventana, ventanas por trimestre y ejecuciones interrumpidas
    trimestres = dias / (13 * 7)
    falla = estado.groupby("escenario")["falla"].mean() / horizonte
    ventanas = estado.groupby("escenario")["ventanas"].mean() / trimestres
    interrumpidos = r.groupby("escenario")["resultado"].apply(lambda s: s.eq("interrumpido").mean())
    s02 = perturbaciones.tasas_base()["fallas"]
    ok4 = bool((falla.sub(s02).abs() <= TOLERANCIA_FALLA).all() and ventanas.between(0.95, 1.0 + TOL).all())
    veredicto4 = _veredicto(ok4) if dias >= 13 * 7 else SE_REPORTA   # con menos de un trimestre no hay criterio
    tabla4 = pd.DataFrame({"tiempo_en_falla": falla, "tiempo_en_ventana": estado.groupby("escenario")["ventana"].mean()
                           / horizonte, "ventanas_por_unidad_y_trimestre": ventanas,
                           "fallas_por_unidad_y_año": eventos[eventos["evento"].eq("falla")].groupby("escenario").size()
                           / eventos.groupby("escenario")["semilla"].nunique() / len(lab.equipos) / (dias / 364),
                           "registros_interrumpidos": interrumpidos})
    salida["M4"] = _indicador(
        "Tiempo en falla: " + ", ".join(f"{e} {_pct(x, 2)}" for e, x in falla.items())
        + f" (S02: {_pct(s02, 0)}); ventanas por unidad y trimestre: "
        + ", ".join(f"{e} {_n(x, 2)}" for e, x in ventanas.items())
        + "; registros interrumpidos por una falla: " + ", ".join(f"{e} {_pct(x, 2)}" for e, x in interrumpidos.items()),
        veredicto4, por_escenario=tabla4,
        por_unidad=estado.assign(falla=estado["falla"] / horizonte, ventana=estado["ventana"] / horizonte,
                                 fuera=estado["fuera"] / horizonte))
    return salida


# ===================================================================== disponibilidad cruzada (C1 a C4)
def disponibilidad_cruzada(registros, eventos, laboratorio: Laboratorio, dias: int, desde: float = 0.0) -> dict:
    """C1 a C4 (src/generador/disponibilidad.py), con los criterios de la hoja «Indicadores»."""
    r, lab = registros, laboratorio
    ind = disponibilidad.indicadores(r, desde)
    salida = {}

    # C1: combinaciones de cada registro frente a las del diseño (Tabla 13)
    diseno = lab.cobertura()["combinaciones_factibles"]
    c1 = ind["c1"].join(diseno.rename("diseno"))
    ok1 = bool((c1["combinaciones_min"] == c1["diseno"]).all() and (c1["combinaciones_max"] == c1["diseno"]).all())
    salida["C1"] = _indicador(f"De {int(diseno.min())} combinaciones (M08 y M12) a {int(diseno.max())} (M01), "
                              "las mismas del diseño en todos los registros", _veredicto(ok1), por_metodo=c1)

    # C2: registros con al menos una combinación factible
    salida["C2"] = _indicador(f"{_pct(ind['c2']['porcentaje'] / 100)} de {_n(ind['c2']['registros'], 0)} registros",
                              _veredicto(ind["c2"]["porcentaje"] == 100), por_corrida=ind["c2"].to_frame("valor"))

    # C3: coincidencia de analista calificado libre y equipo compatible libre al llegar, y cobertura del calendario
    tabla = ind["c3_c4"]
    coberturas = [disponibilidad.cobertura_en_el_tiempo(dias, g, lab).assign(escenario=esc)
                  for (esc, _), g in eventos.groupby(CORRIDA)]
    cobertura = pd.concat(coberturas).groupby(["escenario", "metodo"]).mean()
    v = r[r["llegada"] >= desde]
    por_metodo = v.assign(ninguno=v["recurso_faltante"].eq("ninguno")).groupby(["metodo", "escenario"])["ninguno"] \
        .mean().unstack("escenario")
    salida["C3"] = _indicador(
        "Registros con analista y equipo libres al llegar: " + ", ".join(
            f"{e} {_pct(x / 100)}" for e, x in tabla["ninguno"].items())
        + "; horas con analista calificado en turno: 100 % salvo M08 y M12, "
        + _pct(cobertura.xs("M08", level="metodo")["con_analista"].mean()), SE_REPORTA,
        por_escenario=tabla, por_metodo=por_metodo, cobertura_del_calendario=cobertura)

    # C4: registros restringidos por falta de analista, de equipo o de ambos, y su espera
    espera = (v.groupby(["escenario", "recurso_faltante"])["espera"].mean() / 60).unstack("recurso_faltante")
    orden = [e for e in escenarios.ESCENARIOS if e in tabla.index]
    restringidas = tabla.loc[orden, "restringidas"]
    if len(orden) == len(escenarios.ESCENARIOS):
        ok4 = bool(restringidas.is_monotonic_increasing and tabla.at["alta", "analista"] > tabla.at["alta", "equipo"])
        veredicto = _veredicto(ok4)
    else:
        veredicto = SE_REPORTA
    salida["C4"] = _indicador(
        "Restringidas: " + ", ".join(f"{e} {_pct(tabla.at[e, 'restringidas'] / 100)} (analista "
                                     f"{_pct(tabla.at[e, 'analista'] / 100)}, equipo {_pct(tabla.at[e, 'equipo'] / 100)}, "
                                     f"ambos {_pct(tabla.at[e, 'ambos'] / 100)})" for e in orden),
        veredicto, por_escenario=tabla, espera_horas=espera)
    return salida


# ===================================================================== variabilidad y consistencia (V1 a V5)
def ajuste_betapert(registros, forma="hoja") -> pd.DataFrame:
    """V1: cada tarea de cada ejecución terminada, llevada a su probabilidad con la función de distribución de
    su BetaPERT (dividida entre el factor del analista si es suya), debe salir uniforme entre 0 y 1.

    Prueba de Kolmogorov-Smirnov por escenario, técnica y tarea, con la corrección de Bonferroni por el número
    de pruebas; se omiten las tareas fijas (a = b). Las corridas de un escenario se juntan porque sus semillas
    son independientes; las de escenarios distintos no, porque comparten semillas (números aleatorios comunes):
    juntarlas contaría dos veces las mismas probabilidades.
    """
    tt = _terminados(registros)
    p, lam, f = _puntos(tt), _formas(tt, forma), _factor(tt)
    escenario, tecnica = tt["escenario"].to_numpy(), tt["tecnica"].to_numpy()
    filas = []
    for t in TAREAS:
        a, m, b = p[f"a_{t}"].to_numpy(), p[f"m_{t}"].to_numpy(), p[f"b_{t}"].to_numpy()
        x = tt[t].to_numpy() / (f.to_numpy() if t in registro.TAREAS_ANALISTA else 1.0)
        varia = b > a
        if not varia.any():
            continue
        alfa, beta = duraciones.parametros_beta(a[varia], m[varia], b[varia], lam[t].to_numpy()[varia])
        u = np.full(len(tt), np.nan)
        u[varia] = stats.beta.cdf((x[varia] - a[varia]) / (b[varia] - a[varia]), alfa, beta)
        for esc in [e for e in escenarios.ESCENARIOS if e in set(escenario)]:
            for tec in TECNICAS:
                sel = varia & (escenario == esc) & (tecnica == tec)
                if sel.sum() < 20:
                    continue
                prueba = stats.kstest(u[sel], "uniform")
                filas.append({"escenario": esc, "tecnica": tec, "tarea": t, "registros": int(sel.sum()),
                              "media_u": float(u[sel].mean()), "D": prueba.statistic, "p": prueba.pvalue})
    tabla = pd.DataFrame(filas)
    tabla["p_bonferroni"] = (tabla["p"] * len(tabla)).clip(upper=1.0)
    return tabla


def variabilidad(registros, version: str, horizonte: float | None = None, desde: float = 0.0) -> dict:
    """V1 a V5: distribución de los tiempos, correlaciones, reglas, rangos y trazabilidad.

    El tiempo en el sistema de V1 se mide con las muestras que llegan desde `desde` (sin el calentamiento).
    """
    config = escenarios.VERSIONES[version]
    r = registros
    salida = {}

    # V1: cuantiles por técnica, ajuste a la BetaPERT y tiempo en el sistema frente a la Tabla 5
    tt = _terminados(r)
    columnas = ["duracion_simulada", *TAREAS]
    cuantiles = tt.groupby("tecnica")[columnas].quantile(list(CUANTILES)).unstack(level=-1)
    ajuste = ajuste_betapert(r, config["forma"])
    tt_sin_calentamiento = tt[tt["llegada_muestra"] >= desde]
    tes = (tt_sin_calentamiento.groupby(["escenario", "tecnica"])["tiempo_en_sistema"].mean() / 60).unstack("escenario")
    ensayo = (tt_sin_calentamiento.sort_values("intento").groupby(CORRIDA + ["id_muestra", "metodo"])
              .agg(tecnica=("tecnica", "first"), llegada=("llegada_muestra", "first"), fin=("fin", "last")))
    tes_ensayo = ((ensayo["fin"] - ensayo["llegada"]) / 60).groupby([ensayo.index.get_level_values("escenario"),
                                                                     ensayo["tecnica"]]).mean().unstack(0)
    fuente = tiempo_en_sistema_fuente()
    tes_tabla = tes.add_prefix("registro_").join(tes_ensayo.add_prefix("ensayo_")).join(fuente.rename("tabla_5"))
    ok1 = bool((ajuste["p_bonferroni"] >= ALFA_V1).all())
    base = "base" if "base" in tes.columns else tes.columns[0]
    salida["V1"] = _indicador(
        f"Kolmogorov-Smirnov de {len(ajuste)} pruebas (escenario, técnica y tarea) frente a su BetaPERT: mayor D = "
        f"{_n(ajuste['D'].max(), 4)}, menor p corregido = {_n(ajuste['p_bonferroni'].min(), 3)}; tiempo en el sistema por "
        f"ensayo, escenario {base} (h): " + ", ".join(f"{t} {_n(tes_ensayo.at[t, base])} (Tabla 5: {_n(fuente[t])})"
                                                       for t in TECNICAS if t in tes_ensayo.index),
        _veredicto(ok1), cuantiles=cuantiles, ajuste=ajuste, tiempo_en_sistema_horas=tes_tabla)

    # V2: correlación de rangos entre las tareas, por corrida (correlacion.indicador_v2)
    filas, pares, estructura = [], [], []
    for (esc, sem), g in _corridas(r):
        v2 = correlacion.indicador_v2(g, config["rho"])
        t = v2["tareas"]
        filas.append({"escenario": esc, "semilla": sem, "veredicto": v2["veredicto"], "pares": len(t),
                      "pares_que_cumplen": int(t["cumple"].eq(True).sum()),
                      "mayor_diferencia_en_tolerancias": float((t["diferencia"].abs() / t["tolerancia"]).max()),
                      "menos_registros_por_par": int(t["registros"].min())})
        pares.append(t.assign(escenario=esc, semilla=sem))
        estructura.append(v2["estructura"].assign(escenario=esc, semilla=sem))
    por_corrida = pd.DataFrame(filas)
    veredictos = por_corrida["veredicto"]
    v2_ver = CUMPLE if veredictos.eq("cumple").all() else (NO_CUMPLE if veredictos.eq("no cumple").any()
                                                          else "sin datos suficientes")
    rho = config["rho"] if config["rho"] is not None else float(parametros.supuesto("S04"))
    salida["V2"] = _indicador(
        f"ρ objetivo = {_n(rho, 1)}: {int(veredictos.eq('cumple').sum())} de {len(veredictos)} corridas cumplen; la "
        f"mayor diferencia es {_n(por_corrida['mayor_diferencia_en_tolerancias'].max(), 2)} veces su tolerancia",
        v2_ver, por_corrida=por_corrida, pares=pd.concat(pares, ignore_index=True),
        estructura=pd.concat(estructura, ignore_index=True))

    # V3: registros que cumplen todas las reglas de chequear_registro
    reglas = pd.concat([registro.chequear_registro(g) for _, g in _corridas(r)])
    por_regla = reglas.mean().rename("cumplen")
    ok3 = reglas.all(axis=1)
    salida["V3"] = _indicador(f"{_pct(ok3.mean())} de {_n(len(ok3), 0)} registros cumplen las {reglas.shape[1]} reglas",
                              _veredicto(ok3.all()), por_regla=por_regla.to_frame(),
                              por_corrida=ok3.groupby([r.loc[ok3.index, c] for c in CORRIDA]).mean().to_frame("cumplen"))

    # V4: valores fuera de su rango (columna «Valores o rango» de la hoja Diccionario)
    fuera = fuera_de_rango(r, horizonte)
    por_campo = fuera.mean().rename("fuera_de_rango").to_frame()
    total = float(fuera.to_numpy().mean())
    salida["V4"] = _indicador(f"{_pct(total, 2)} de {_n(fuera.size, 0)} valores fuera de rango; "
                              f"{int((por_campo['fuera_de_rango'] > 0).sum())} de {fuera.shape[1]} campos con alguno",
                              _veredicto(total == 0), por_campo=por_campo, nulos=r[registro.CAMPOS_REGISTRO].isna().sum()
                              .rename("vacios").to_frame())

    # V5: filas de las hojas Parámetros y Supuestos con fuente y página
    v5 = trazabilidad()
    salida["V5"] = _indicador(f"{v5.at['Total', 'con_fuente']} de {v5.at['Total', 'filas']} "
                              f"({_pct(v5.at['Total', 'con_fuente'] / v5.at['Total', 'filas'], 0)}); el resto son "
                              "supuestos con escenarios de sensibilidad", SE_REPORTA, por_hoja=v5)
    return salida


def fuera_de_rango(registros, horizonte: float | None = None) -> pd.DataFrame:
    """V4: True en cada valor fuera de su rango, con la columna «Valores o rango» de la hoja «Diccionario».

    Una columna por campo, en el orden del diccionario, y una fila por registro. Un vacío cuenta como fuera
    de rango, salvo en id_actividad_origen y causa_repeticion cuando el intento es 1. horizonte: minutos de
    la corrida, para la llegada de la muestra (sin él solo se exige que no sea negativa).
    """
    _, analistas, equipos, tipos, _ = registro._hojas()   # noqa: SLF001
    df = registros.reset_index(drop=True)
    num = lambda c: pd.to_numeric(df[c], errors="coerce")        # noqa: E731
    entero = lambda s: s.notna() & s.mod(1).eq(0)                # noqa: E731
    serie = lambda x: pd.Series(list(x), index=df.index)         # noqa: E731
    d = pd.DataFrame(index=df.index)

    cargas = np.array([v["multiplicador_carga"] for v in escenarios.ESCENARIOS.values()])
    d["escenario"] = df["escenario"].isin(list(escenarios.ESCENARIOS))
    d["carga"] = serie(np.isclose(num("carga").to_numpy()[:, None], cargas[None, :], atol=TOL).any(axis=1))
    s = num("semilla")
    d["semilla"] = entero(s) & s.ge(0)
    d["id_actividad"] = df["id_actividad"].notna() & ~df.duplicated(CORRIDA + ["id_actividad"], keep=False)
    d["id_muestra"] = df["id_muestra"].astype(str).str.fullmatch(r"(MP|IN|GR|PT|ES)-\d{6}")
    d["metodo"] = df["metodo"].isin(registro.METODOS)
    i = num("intento")
    d["intento"] = entero(i) & i.ge(1)
    primero = i.eq(1)
    anterior = serie(registro.id_actividad(mu, me, int(k) - 1) if pd.notna(k) and k > 1 else None
                     for mu, me, k in zip(df["id_muestra"], df["metodo"], i))
    d["id_actividad_origen"] = np.where(primero, df["id_actividad_origen"].isna(), df["id_actividad_origen"].eq(anterior))
    d["causa_repeticion"] = np.where(primero, df["causa_repeticion"].isna(),
                                     df["causa_repeticion"].isin(list(registro.CAUSA_SEGUN_RESULTADO.values())))
    d["resultado"] = df["resultado"].isin(registro.RESULTADOS)
    d["tipo_muestra"] = df["tipo_muestra"].isin(registro.TIPOS_MUESTRA)
    urgente = df["urgente"].eq(True)
    p = num("prioridad")
    d["prioridad"] = entero(p) & p.between(1, 5) & (~urgente | p.eq(1))
    lm = num("llegada_muestra")
    d["llegada_muestra"] = lm.ge(0) & (lm.lt(horizonte) if horizonte else lm.notna())
    plazo = serie(registro.plazo_minutos(t, bool(u), x) if t in tipos.index and pd.notna(x) else np.nan
                  for t, u, x in zip(df["tipo_muestra"], urgente, lm))
    d["plazo_min"] = (num("plazo_min") - plazo).abs().lt(TOL)
    d["vencimiento"] = num("vencimiento").gt(lm)
    d["tecnica"] = df["tecnica"].isin(TECNICAS)
    d["complejidad"] = df["complejidad"].isin(["baja", "media", "alta"])
    n = num("n_soluciones")
    rango = {"baja": (1, 2), "media": (3, 4), "alta": (5, 6)}   # S08 y S11
    d["n_soluciones"] = entero(n) & n.ge(df["complejidad"].map({k: v[0] for k, v in rango.items()})) & n.le(
        df["complejidad"].map({k: v[1] for k, v in rango.items()}))
    d["calificacion_exigida"] = df["calificacion_exigida"].isin(registro.NIVELES)
    d["equipo_requerido"] = serie(isinstance(q, str) and all(c in equipos.index for c in q.split("|"))
                                  for q in df["equipo_requerido"])
    a, m, b = num("t_optimista"), num("t_mas_probable"), num("t_pesimista")
    d["t_optimista"] = a.gt(0)
    d["t_mas_probable"] = m.ge(a - TOL) & m.le(b + TOL)
    d["t_pesimista"] = b.ge(m - TOL)
    calificado = serie(x in analistas.index and me in registro.METODOS and analistas.at[x, me] == 1
                       for x, me in zip(df["analista"], df["metodo"]))
    d["analista"] = df["analista"].astype(str).str.fullmatch(r"A(0[1-9]|1\d|20)") & calificado
    d["nivel_analista"] = df["nivel_analista"].isin(registro.NIVELES) & (
        df["nivel_analista"].map(RANGO_NIVEL) >= df["calificacion_exigida"].map(RANGO_NIVEL))
    d["equipo"] = serie(isinstance(e, str) and registro._unidad_valida(e, e[:-2], equipos)   # noqa: SLF001
                        for e in df["equipo"])
    d["configuracion"] = serie(isinstance(q, str) and c in q.split("|")
                               for c, q in zip(df["configuracion"], df["equipo_requerido"]))
    ido = df["idoneidad_ejecutada"]
    d["idoneidad_ejecutada"] = ido.isin([True, False]) & (~ido.eq(True) | df["tecnica"].isin(registro.TECNICAS_CON_IDONEIDAD))
    f = df["nivel_analista"].map({nv: registro.factor_duracion(nv) for nv in registro.NIVELES})
    dur = num("duracion_simulada")
    terminada = df["resultado"].ne("interrumpido")
    d["duracion_simulada"] = dur.ge(0) & (~terminada | (dur.ge(np.minimum(f, 1) * a - TOL)
                                                        & dur.le(np.maximum(f, 1) * b + TOL)))
    for c in ("t_prep_sistema", "t_idoneidad"):
        d[c] = num(c).ge(0) & (ido.eq(True) | num(c).eq(0))
    for c in ("t_preparacion", "t_montaje", "t_corrida", "t_procesamiento"):
        d[c] = num(c).ge(0)
    llegada, inicio = num("llegada"), num("inicio")
    d["llegada"] = llegada.ge(lm - TOL)
    d["inicio"] = inicio.ge(llegada - TOL)
    d["fin"] = num("fin").ge(inicio + dur - TOL)
    for c in ("espera", "pausas", "tardanza"):
        d[c] = num(c).ge(0)
    d["tiempo_en_sistema"] = num("tiempo_en_sistema").ge(dur - TOL)
    puede = tipos.index[tipos["Puede ser urgente"].eq("Sí")]
    d["urgente"] = df["urgente"].isin([True, False]) & (~urgente | df["tipo_muestra"].isin(puede))
    reanalisis = df["causa_repeticion"].eq("reanálisis") | df["resultado"].eq("fuera de especificación")
    falla = df["causa_repeticion"].eq("falla de equipo") | df["resultado"].eq("interrumpido")
    d["bandera_reanalisis"] = df["bandera_reanalisis"].eq(reanalisis)
    d["bandera_falla"] = df["bandera_falla"].eq(falla)
    na, ne, nc = num("analistas_calificados"), num("equipos_compatibles"), num("combinaciones_factibles")
    d["analistas_calificados"] = entero(na) & na.ge(2)
    d["equipos_compatibles"] = entero(ne) & ne.ge(2)
    d["combinaciones_factibles"] = entero(nc) & nc.ge(4)
    la, le = num("analistas_libres"), num("equipos_libres")
    d["analistas_libres"] = entero(la) & la.ge(0) & la.le(na)
    d["equipos_libres"] = entero(le) & le.ge(0) & le.le(ne)
    d["recurso_faltante"] = df["recurso_faltante"].isin(disponibilidad.FALTANTES)
    fuera = ~d[registro.CAMPOS_REGISTRO].fillna(False).astype(bool)
    fuera.index = registros.index
    return fuera


def trazabilidad() -> pd.DataFrame:
    """V5: filas de las hojas «Parámetros» y «Supuestos»; con fuente, las de un dato de la literatura o derivado."""
    pa = parametros.cargar_hoja("Parámetros").dropna(subset=["ID"])
    su = parametros.cargar_hoja("Supuestos").dropna(subset=["ID"])
    con_fuente = pa["Tipo"].isin(["Dato de la literatura", "Derivado (cálculo propio)"])
    tabla = pd.DataFrame({"filas": [len(pa), len(su)], "con_fuente": [int(con_fuente.sum()), 0]},
                         index=["Parámetros", "Supuestos"])
    tabla.loc["Total"] = tabla.sum()
    tabla["porcentaje"] = tabla["con_fuente"] / tabla["filas"]
    return tabla.astype({"filas": int, "con_fuente": int})


# ===================================================================== todo junto
def validar(registros, eventos, version: str, semanas: int = escenarios.SEMANAS,
            laboratorio: Laboratorio | None = None) -> dict:
    """Los 22 indicadores de una versión del dataset y su resumen (una fila por indicador)."""
    config = escenarios.VERSIONES[version]
    lab = laboratorio or Laboratorio.desde_excel()
    dias = int(semanas) * 7
    horizonte = dias * MIN_POR_DIA
    desde = escenarios.CALENTAMIENTO_DIAS * MIN_POR_DIA
    v = {}
    v.update(tiempos(registros, config["forma"]))
    v.update(recursos_humanos(registros, lab, dias, desde))
    v.update(recursos_materiales(registros, eventos, lab, dias))
    v.update(disponibilidad_cruzada(registros, eventos, lab, dias, desde))
    v.update(variabilidad(registros, version, horizonte, desde))
    hoja = parametros.cargar_hoja("Indicadores").set_index("Código")
    v["resumen"] = pd.DataFrame([{"codigo": c, "dimension": hoja.at[c, "Dimensión"],
                                  "indicador": hoja.at[c, "Indicador (anteproyecto, §8.5)"], "valor": v[c]["valor"],
                                  "referencia": hoja.at[c, "Referencia o criterio"], "veredicto": v[c]["veredicto"]}
                                 for c in hoja.index])
    return v


def guardar(validacion: dict, version: str, carpeta: Path | None = None) -> list:
    """Escribe el resumen en resultados/tablas/validacion_<versión>_resumen.csv y las tablas de detalle en
    resultados/tablas/validacion_<versión>.xlsx (una hoja por tabla). Devuelve las rutas."""
    carpeta = Path(carpeta) if carpeta else Path(__file__).resolve().parents[2] / "resultados" / "tablas"
    carpeta.mkdir(parents=True, exist_ok=True)
    resumen = carpeta / f"validacion_{version}_resumen.csv"
    validacion["resumen"].to_csv(resumen, index=False)
    detalle = carpeta / f"validacion_{version}.xlsx"
    with pd.ExcelWriter(detalle) as libro:
        validacion["resumen"].to_excel(libro, sheet_name="Resumen", index=False)
        for codigo, ind in validacion.items():
            if codigo == "resumen":
                continue
            for nombre, tabla in ind["tablas"].items():
                hoja = f"{codigo} {nombre}"[:31]
                tabla.to_excel(libro, sheet_name=hoja)
    return [resumen, detalle]
