"""Analistas, equipos y calendarios de disponibilidad del laboratorio de referencia.

Anteproyecto: dimensiones «Recursos humanos» y «Recursos materiales» del sistema de variables.
Diseño: docs/Diseno_laboratorio_referencia.docx, secciones 3 y 4, y docs/Generador_de_datos.docx,
sección 3.
Datos: hojas «Analistas», «Equipos», «Catálogo», «Turnos», «Rotación», «Mantenimiento» y
«Calibración de turnos» del Excel de parámetros.

Es el modelo de recursos que comparten el generador del O1 y el simulador del O3
(src/simulacion/entorno.py): quién está calificado en qué método, qué unidad corre qué método, a
qué horas trabaja cada analista y cuándo sale de servicio cada equipo por mantenimiento o
verificación. Las fallas, que son aleatorias, están en src/generador/perturbaciones.py.

Los momentos van en minutos desde el lunes a las 0:00 en que empieza la corrida, que es también el
día 1 del ciclo de 42 días de la hoja «Rotación» (diccionario de datos, sección 2).
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from src.generador import parametros, registro

TECNICAS = ("HPLC", "GC", "PSA", "KF", "DSC", "XRPD")
MIN_POR_DIA = registro.MIN_POR_DIA
MIN_POR_SEMANA = 7 * MIN_POR_DIA
CICLO_ROTACION = 42            # días (hoja «Rotación», S23)
DIAS_HABILES_TRIMESTRE = 65    # 13 semanas de 5 días (hoja «Mantenimiento», S26)
HORA_VENTANA = 8               # las ventanas empiezan a las 8:00 de un día hábil (S26)
RANGO_NIVEL = {nivel: k for k, nivel in enumerate(registro.NIVELES)}   # júnior 0, intermedio 1, sénior 2


@dataclass(frozen=True)
class Analista:
    id: str
    turno: str                 # T1, T2 o T3 (Costigliola et al., 2017, Tabla 1, p. 9015)
    nivel: str                 # júnior, intermedio o sénior (S19)
    area: str                  # Todas, Cromatografía o Fisicoquímica (S20)
    factor: float              # factor de duración de sus tareas (S07)
    metodos: frozenset         # métodos en que está calificado (hoja «Analistas», S12 y S20)


@dataclass(frozen=True)
class Equipo:
    id: str                    # configuración y número, por ejemplo HPLC-G04
    configuracion: str         # código de la hoja «Equipos», por ejemplo HPLC-G
    tecnica: str
    numero: int                # 1 a n dentro de su configuración
    metodos: frozenset         # métodos que puede correr (S22)


@dataclass(frozen=True)
class Bloque:
    """Un turno de un analista: de su inicio a su fin, con los tramos en que trabaja (sin el almuerzo)."""
    analista: str
    codigo: str
    inicio: float
    fin: float
    tramos: tuple              # ((inicio, fin), ...) en minutos


class Laboratorio:
    """Recursos del laboratorio de referencia, leídos del Excel de parámetros.

    Se crea con Laboratorio.desde_excel(), que lee las hojas una sola vez. El generador del O1 y el
    simulador del O3 usan el mismo objeto: no cambia durante una corrida.
    """

    def __init__(self, analistas, equipos, catalogo, configuraciones, turnos, rotacion, mantenimiento):
        self.analistas = {a.id: a for a in analistas}
        self.equipos = {e.id: e for e in equipos}
        self.catalogo = catalogo
        self.configuraciones = configuraciones
        self.turnos = turnos
        self.rotacion = rotacion
        self.mantenimiento = mantenimiento
        self.metodos = tuple(catalogo.index)
        self.calificados = {m: tuple(a.id for a in analistas if m in a.metodos) for m in self.metodos}
        self.compatibles = {m: tuple(e.id for e in equipos if m in e.metodos) for m in self.metodos}
        # configuración principal de cada método: la que corre menos métodos (las horas de M01 se cuentan
        # en el HPLC isocrático; diseño del laboratorio, Tabla 11)
        self.principal = {}
        for m in self.metodos:
            confs = sorted({self.equipos[e].configuracion for e in self.compatibles[m]},
                           key=lambda c: (len(configuraciones.loc[c, "metodos"]), c))
            self.principal[m] = confs[0]

    # ------------------------------------------------------------------ lectura
    @classmethod
    @lru_cache(maxsize=1)
    def desde_excel(cls) -> "Laboratorio":
        hoja_an = parametros.cargar_hoja("Analistas").set_index("ID")
        hoja_eq = parametros.cargar_hoja("Equipos").set_index("Código")
        catalogo = parametros.cargar_hoja("Catálogo").set_index("ID")
        metodos = list(catalogo.index)
        analistas = [Analista(id=i, turno=f["Turno"], nivel=f["Nivel"], area=f["Área"],
                              factor=float(f["Factor de duración"]),
                              metodos=frozenset(m for m in metodos if f[m] == 1))
                     for i, f in hoja_an.iterrows()]
        equipos, configuraciones = [], {}
        for codigo, f in hoja_eq.iterrows():
            corre = frozenset(m for m in metodos if f[m] == "X")
            configuraciones[codigo] = {"tecnica": f["Técnica"], "nombre": f["Configuración"],
                                       "unidades": int(f["Equipos"]), "metodos": corre}
            equipos += [Equipo(id=f"{codigo}{k:02d}", configuracion=codigo, tecnica=f["Técnica"], numero=k,
                               metodos=corre) for k in range(1, int(f["Equipos"]) + 1)]
        configuraciones = pd.DataFrame.from_dict(configuraciones, orient="index")
        turnos = parametros.cargar_hoja("Turnos").set_index("Código")
        rotacion = parametros.cargar_hoja("Rotación").set_index("Día")
        mantenimiento = parametros.cargar_hoja("Mantenimiento").set_index("Código")
        return cls(analistas, equipos, catalogo, configuraciones, turnos, rotacion, mantenimiento)

    # ------------------------------------------------------------------ combinaciones
    def combinaciones_factibles(self, metodo: str) -> list:
        """Parejas (analista, equipo) que pueden ejecutar el método según las matrices (S12 y S22)."""
        return [(a, e) for a in self.calificados[metodo] for e in self.compatibles[metodo]]

    def cobertura(self) -> pd.DataFrame:
        """Analistas calificados, equipos compatibles y combinaciones de cada método (diseño, Tabla 13)."""
        filas = []
        for m in self.metodos:
            por_turno = pd.Series([self.analistas[a].turno for a in self.calificados[m]]).value_counts()
            filas.append({"metodo": m, "turno_1": int(por_turno.get("T1", 0)), "turno_2": int(por_turno.get("T2", 0)),
                          "turno_3": int(por_turno.get("T3", 0)), "analistas_calificados": len(self.calificados[m]),
                          "equipos_compatibles": len(self.compatibles[m]),
                          "combinaciones_factibles": len(self.calificados[m]) * len(self.compatibles[m])})
        return pd.DataFrame(filas).set_index("metodo")

    # ------------------------------------------------------------------ turnos
    def _bloque(self, analista: str, dia: int, codigo: str):
        """El turno `codigo` que el analista empieza el día `dia` (0 = lunes en que empieza la corrida)."""
        if codigo not in self.turnos.index or codigo == "L":
            return None
        t = self.turnos.loc[codigo]
        inicio_h, fin_h = float(t["Inicio (h)"]), float(t["Fin (h)"])
        if fin_h <= inicio_h:            # el turno de noche termina al día siguiente
            fin_h += 24
        base = dia * MIN_POR_DIA
        inicio, fin = base + inicio_h * 60, base + fin_h * 60
        desde, hasta = float(t["Pausa desde (h)"]), float(t["Pausa hasta (h)"])
        if hasta > desde:                # almuerzo (S24)
            tramos = ((inicio, base + desde * 60), (base + hasta * 60, fin))
        else:
            tramos = ((inicio, fin),)
        return Bloque(analista, codigo, inicio, fin, tramos)

    def bloques(self, dias: int) -> list:
        """Turnos de cada analista del día −1 al día `dias` (un Bloque por turno), con la hoja «Rotación».

        El calendario de 42 días se repite desde el lunes en que empieza la corrida (S23). Se incluye
        el turno de noche que empezó el domingo anterior, que cubre el lunes de 0:00 a 8:00.
        """
        filas = []
        for dia in range(-1, dias):
            fila = self.rotacion.loc[(dia % CICLO_ROTACION) + 1]
            for a in self.analistas:
                b = self._bloque(a, dia, fila[a])
                if b is not None and b.fin > 0 and b.inicio < dias * MIN_POR_DIA:
                    filas.append(b)
        return filas

    def calendario_analistas(self, dias: int) -> pd.DataFrame:
        """Tramos en que trabaja cada analista: una fila por tramo (analista, turno, inicio, fin), en minutos,
        recortados a [0, dias × 1 440)."""
        fin_corrida = dias * MIN_POR_DIA
        filas = [{"analista": b.analista, "codigo": b.codigo, "inicio": max(a, 0.0), "fin": min(z, fin_corrida)}
                 for b in self.bloques(dias) for a, z in b.tramos if z > 0 and a < fin_corrida]
        return pd.DataFrame(filas).sort_values(["analista", "inicio"], ignore_index=True)

    def analistas_en_turno(self, dias: int, paso: int = 60) -> pd.DataFrame:
        """Cuántos analistas trabajan en cada intervalo de `paso` minutos (indicador H1)."""
        cal = self.calendario_analistas(dias)
        t = np.arange(0, dias * MIN_POR_DIA, paso)
        conteo = np.zeros(len(t), dtype=int)
        for a, z in zip(cal["inicio"], cal["fin"]):
            conteo += (t >= a) & (t < z)
        return pd.DataFrame({"inicio": t, "analistas": conteo})

    # ------------------------------------------------------------------ ventanas de mantenimiento
    def ventanas(self, dias: int) -> pd.DataFrame:
        """Ventanas planificadas de cada unidad (S26): una por trimestre, en un día hábil desde las 8:00.

        Las unidades de una configuración se escalonan con la separación de la hoja «Mantenimiento»
        (65 días hábiles del trimestre ÷ unidades), así que dos unidades de la misma configuración no
        coinciden. La unidad 01 hace el mantenimiento preventivo con recalificación en el primer trimestre
        de cada año, la 02 en el segundo, y así (la 05 vuelve al primero), y la verificación de desempeño en
        los otros tres. Una fila por ventana:
        equipo, configuración, evento (mantenimiento o verificación), inicio y fin planificados.
        """
        filas = []
        dias_habiles = [d for d in range(dias) if d % 7 < 5]
        for e in self.equipos.values():
            m = self.mantenimiento.loc[e.configuracion]
            separacion = int(m["Separación entre equipos (días hábiles)"])
            k = e.numero - 1
            for trimestre in range(len(dias_habiles) // DIAS_HABILES_TRIMESTRE + 1):
                indice = trimestre * DIAS_HABILES_TRIMESTRE + k * separacion
                if indice >= len(dias_habiles):
                    break
                es_mant = trimestre % 4 == k % 4
                horas = float(m["Duración del mantenimiento (h)"] if es_mant else m["Duración de la verificación (h)"])
                inicio = dias_habiles[indice] * MIN_POR_DIA + HORA_VENTANA * 60
                filas.append({"equipo": e.id, "configuracion": e.configuracion,
                              "evento": "mantenimiento" if es_mant else "verificación",
                              "inicio": float(inicio), "fin": float(inicio + horas * 60)})
        return pd.DataFrame(filas, columns=["equipo", "configuracion", "evento", "inicio", "fin"])


def es_dia_habil(minuto: float) -> bool:
    """True de lunes a viernes (la corrida empieza un lunes)."""
    return (int(minuto // MIN_POR_DIA) % 7) < 5


def dia_y_hora(minuto: float) -> str:
    """Texto legible de un momento: «mar. 8:10 (día 2)»."""
    dias = ["lun.", "mar.", "mié.", "jue.", "vie.", "sáb.", "dom."]
    d, resto = divmod(float(minuto), MIN_POR_DIA)
    h, mi = divmod(int(round(resto)), 60)
    return f"{dias[int(d) % 7]} {h}:{mi:02d} (día {int(d) + 1})"
