"""Llegada de muestras, ensayos de cada muestra y plazos de entrega.

Anteproyecto: §1.4 (plazo de cada actividad) y Alcance II (actividades no planificadas).
Diseño: docs/Diseno_laboratorio_referencia.docx, sección 2, y docs/Generador_de_datos.docx,
sección 5.
Datos: hoja «Tipos de muestra» (proporción, llegadas, plazo, prioridad y probabilidad de cada
método dentro del tipo), bloque «Ensayos por muestra» de la hoja «Demanda» (Cunha et al., 2019,
p. 1425) y supuestos S03, S06, S10, S17 y S18.

Cada muestra es un trabajo con varias tareas, como en los laboratorios de control de calidad que
estudian Ruiz-Torres et al. (2012): pide de uno a tres ensayos, siempre de métodos distintos, y
cada ensayo es una actividad del dataset (diccionario de datos, sección 1).
"""
import numpy as np
import pandas as pd

from src.generador import parametros, perturbaciones, registro

MIN_POR_DIA = registro.MIN_POR_DIA


def tipos_de_muestra() -> pd.DataFrame:
    """Hoja «Tipos de muestra», con el código del tipo como índice."""
    return parametros.cargar_hoja("Tipos de muestra").set_index("Código")


def ensayos_por_muestra() -> pd.Series:
    """Probabilidad de que una muestra pida 1, 2 o 3 ensayos (Cunha et al., 2019, p. 1425; S18)."""
    tabla = parametros.bloque("Demanda", "N.º de ensayos", 3)
    return pd.Series(tabla["% de muestras"].astype(float).to_numpy(), index=tabla["N.º de ensayos"].astype(int))


def generar_llegadas(dias: int, carga: float, rng) -> pd.DataFrame:
    """Momento de llegada y tipo de cada muestra del horizonte (S10), ordenadas por llegada.

    Control en proceso: Poisson a cualquier hora, todos los días. Materia prima, granel y producto
    terminado: Poisson de 8:00 a 17:00, de lunes a viernes. Estabilidad: un lote a las 8:00 de cada
    día hábil, con un número de muestras de Poisson. La tasa de cada tipo es la columna «Muestras por
    día de llegada» multiplicada por la carga del escenario.
    """
    rng = np.random.default_rng(rng)
    tipos = tipos_de_muestra()
    filas = []
    for tipo, f in tipos.iterrows():
        desde, hasta = float(f["Llegada desde (h)"]), float(f["Llegada hasta (h)"])
        tasa = float(f["Muestras por día de llegada"]) * carga
        solo_habiles = int(f["Días de llegada por semana"]) == 5
        for dia in range(dias):
            if solo_habiles and dia % 7 >= 5:
                continue
            n = rng.poisson(tasa)
            if n == 0:
                continue
            if hasta > desde:
                horas = np.sort(rng.uniform(desde, hasta, size=n))
            else:                                  # lote programado al inicio de la jornada
                horas = np.full(n, desde)
            filas += [(tipo, dia * MIN_POR_DIA + h * 60) for h in horas]
    llegadas = pd.DataFrame(filas, columns=["tipo_muestra", "llegada_muestra"])
    return llegadas.sort_values(["llegada_muestra", "tipo_muestra"], kind="stable", ignore_index=True)


def generar_muestras(dias: int, carga: float, rng_llegadas, rng_urgentes, tasa_urgentes=None) -> pd.DataFrame:
    """Muestras del horizonte con su tipo, llegada, urgencia, prioridad, plazo y vencimiento.

    id_muestra es el tipo y el número consecutivo de la muestra dentro de su tipo en la corrida,
    por ejemplo PT-000123. Las urgentes (S03) toman la prioridad máxima y el plazo del control en
    proceso; el plazo de las demás es el de su tipo (S06), en horas o hasta la misma hora del día
    hábil en que vence (registro.plazo_minutos).
    """
    m = generar_llegadas(dias, carga, rng_llegadas)
    m["urgente"] = perturbaciones.marcar_urgentes(m["tipo_muestra"], rng_urgentes, tasa_urgentes)
    numero = m.groupby("tipo_muestra").cumcount() + 1
    m.insert(0, "id_muestra", m["tipo_muestra"] + "-" + numero.map("{:06d}".format))
    m["prioridad"] = [registro.prioridad(t, u) for t, u in zip(m["tipo_muestra"], m["urgente"])]
    m["plazo_min"] = [registro.plazo_minutos(t, u, ll)
                      for t, u, ll in zip(m["tipo_muestra"], m["urgente"], m["llegada_muestra"])]
    m["vencimiento"] = m["llegada_muestra"] + m["plazo_min"]
    return m


def generar_ensayos(muestras: pd.DataFrame, rng) -> pd.DataFrame:
    """Ensayos que pide cada muestra: uno por fila, con el método y el número de soluciones.

    El número de ensayos sale de la distribución de Cunha et al. (2019, p. 1425), igual para los
    cinco tipos (S18). Los métodos se eligen sin repetir, con la probabilidad de cada método dentro
    del tipo (columnas M01 a M12 de la hoja «Tipos de muestra», ajustadas con la hoja «Calibración
    de tipos»). El número de soluciones es uniforme en el rango de la complejidad del método (S08).
    """
    rng = np.random.default_rng(rng)
    tipos = tipos_de_muestra()
    catalogo = parametros.cargar_hoja("Catálogo").set_index("ID")
    metodos = list(catalogo.index)
    prob_k = ensayos_por_muestra()
    ks = rng.choice(prob_k.index.to_numpy(), size=len(muestras), p=prob_k.to_numpy() / prob_k.sum())
    lo = catalogo["Soluciones mín"].astype(int)
    hi = catalogo["Soluciones máx"].astype(int)
    filas = []
    prob_tipo = {t: tipos.loc[t, metodos].to_numpy(dtype=float) for t in tipos.index}
    for (id_m, tipo), k in zip(zip(muestras["id_muestra"], muestras["tipo_muestra"]), ks):
        p = prob_tipo[tipo].copy()
        k = min(int(k), int((p > 0).sum()))
        elegidos = []
        for _ in range(k):                         # sin repetir: se saca uno y se renormaliza el resto
            j = int(rng.choice(len(metodos), p=p / p.sum()))
            elegidos.append(j)
            p[j] = 0.0
        for j in sorted(elegidos):
            m = metodos[j]
            filas.append((id_m, m, int(rng.integers(lo[m], hi[m] + 1))))
    return pd.DataFrame(filas, columns=["id_muestra", "metodo", "n_soluciones"])
