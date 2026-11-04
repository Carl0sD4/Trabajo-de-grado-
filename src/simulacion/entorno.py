"""Motor de eventos discretos del laboratorio de referencia: ejecuta los ensayos con sus recursos.

Lo usan el generador del O1 (src/generador/generar.py), para llenar los campos de asignación y de
tiempos de cada registro, y el simulador del O3, para comparar reglas de despacho en las mismas
condiciones (anteproyecto, Alcance III). Diseño: docs/Generador_de_datos.docx, secciones 3 a 7.

Un ensayo solo empieza si coinciden un analista calificado en turno y libre y un equipo compatible
en servicio y libre (anteproyecto, §6.1.3). Las tareas siguen el flujo de Costigliola et al. (2017,
p. 9016, Fig. 1, y p. 9017, Fig. 3):

    analista: preparación del sistema y montaje → preparación del análisis → … → procesamiento de datos
    equipo:                                   idoneidad del sistema → corrida

La preparación y la idoneidad del sistema solo se hacen si el equipo no está apto para el método:
la idoneidad vale 24 horas (S29), así que un ensayo se ahorra el alistamiento solo si, con las
duraciones medias de sus tareas, su corrida empezaría antes de que se cumplan 24 horas desde la última
idoneidad del mismo método en esa unidad.
Las tareas del analista avanzan solo mientras está en turno: si un turno o el almuerzo las corta,
siguen en su turno siguiente, y esa espera queda en las pausas del registro. La idoneidad y la
corrida las hace el equipo y no se detienen. El equipo queda ocupado desde que empieza el ensayo
hasta que termina la corrida; una falla en ese lapso interrumpe el ensayo, que se repite en otra
unidad (S27). Las ventanas de mantenimiento que caen sobre un ensayo esperan a que el equipo se
libere, porque las hace personal externo que no corta una corrida (S26).

Qué ensayo empieza, con qué analista y en qué unidad lo decide una regla de despacho
(src/simulacion/politicas.py); el generador usa ReglaGenerador (S30). Si la regla tiene el método
esperar_unidad_apta, el motor le pregunta, cuando ninguna unidad libre está apta para el método, si
sus ensayos esperan a una unidad apta en vez de alistar otra. El motor aplica a cualquier regla la
de cierre de turno (_cabe_en_turno) y revisa en cada despacho los primeros CABEZAS ensayos de cada
cola.

Números aleatorios comunes: las uniformes de las duraciones de cada ensayo pedido y la que decide si
su resultado sale fuera de especificación se sacan antes de empezar, en el orden de los ensayos, y
cada repetición usa un generador propio, derivado de la semilla, del ensayo y del intento. Así, con
otra regla de despacho cambian las asignaciones, pero no las muestras, las corridas, los resultados
ni las fallas.

Los momentos van en minutos desde el lunes a las 0:00 en que empieza la corrida.
"""
import bisect
import heapq
import math
from collections import deque

import numpy as np

from src.generador import duraciones, perturbaciones, registro
from src.generador.correlacion import como_matriz, uniformes_correlacionadas
from src.generador.recursos import MIN_POR_DIA, MIN_POR_SEMANA, es_dia_habil

VALIDEZ_IDONEIDAD = 24 * 60     # la idoneidad vale 24 horas (Costigliola et al., 2017, p. 9017, supuesto 2)
FLUJO_REPETICIONES = 6          # las repeticiones derivan su generador de la semilla, después de los seis flujos
HOLGURA_PROCESAMIENTO = 24 * 60  # el procesamiento puede quedar para el turno siguiente, si empieza en menos de 24 h
CABEZAS = 25                    # ensayos de cada cola que se revisan en cada despacho
REDONDEO = 6                    # decimales de los tiempos guardados (micro-minutos)

# orden de los eventos que ocurren en el mismo instante
FALLA, VENTANA, FIN_TAREA, EQUIPO_VUELVE, TURNO, LLEGADA = range(6)


class EstadoAnalista:
    """Un analista durante la corrida: sus tramos de trabajo, sus turnos y lo que está haciendo."""

    def __init__(self, analista, tramos, bloques):
        self.id, self.nivel, self.factor = analista.id, analista.nivel, analista.factor
        self.turno, self.metodos = analista.turno, analista.metodos
        self.inicios = [a for a, _ in tramos]
        self.fines = [z for _, z in tramos]
        self.bloque_inicio = [a for a, _ in bloques]
        self.bloque_fin = [z for _, z in bloques]
        self.tarea = None          # (ejecución, nombre de la tarea, inicio) en curso
        self.pendientes = deque()  # procesamientos que esperan a que termine su tarea en curso
        self.libre_desde = 0.0

    def en_turno(self, t):
        k = bisect.bisect_right(self.inicios, t) - 1
        return k >= 0 and t < self.fines[k]

    def fin_turno(self, t):
        """Fin del turno (con su almuerzo) que contiene a t; None si no está en turno."""
        k = bisect.bisect_right(self.bloque_inicio, t) - 1
        return self.bloque_fin[k] if k >= 0 and t < self.bloque_fin[k] else None

    def avanzar(self, t, trabajo):
        """Momento en que termina `trabajo` minutos de trabajo que empiezan en t, contando solo el tiempo en turno."""
        k = bisect.bisect_right(self.inicios, t) - 1
        if k < 0 or t >= self.fines[k]:
            k += 1
            if k >= len(self.inicios):
                return math.inf
            t = self.inicios[k]
        while True:
            disponible = self.fines[k] - t
            if trabajo <= disponible:
                return t + trabajo
            trabajo -= disponible
            k += 1
            if k >= len(self.inicios):
                return math.inf
            t = self.inicios[k]

    def trabajado(self, t0, t1):
        """Minutos en turno entre t0 y t1."""
        total, k = 0.0, max(bisect.bisect_right(self.inicios, t0) - 1, 0)
        while k < len(self.inicios) and self.inicios[k] < t1:
            total += max(0.0, min(self.fines[k], t1) - max(self.inicios[k], t0))
            k += 1
        return total


class EstadoEquipo:
    """Una unidad de equipo durante la corrida."""

    def __init__(self, equipo):
        self.id, self.configuracion, self.tecnica = equipo.id, equipo.configuracion, equipo.tecnica
        self.metodos = equipo.metodos
        self.ocupado_por = None
        self.fuera_hasta = 0.0
        self.apto_metodo, self.apto_desde = None, -math.inf
        self.libre_desde = 0.0
        self.ventanas_pendientes = deque()

    def apto_para(self, metodo, t):
        return self.apto_metodo == metodo and t - self.apto_desde < VALIDEZ_IDONEIDAD

    def disponible(self, t):
        return self.ocupado_por is None and self.fuera_hasta <= t


class Ejecucion:
    """Una ejecución de un ensayo: lo que será un registro del dataset."""

    __slots__ = ("secuencia", "ensayo", "id_muestra", "metodo", "intento", "origen", "causa", "tipo", "prioridad",
                 "llegada_muestra", "plazo_min", "vencimiento", "urgente", "n_soluciones", "tecnica",
                 "complejidad", "calificacion", "equipo_requerido", "principal", "llegada", "u", "u_resultado",
                 "excluir", "reanalisis_previo", "cualquier_hora", "libres", "entrada", "analista", "equipo",
                 "alistamiento", "inicio", "tareas", "tareas_por", "puntos", "inicio_tarea", "esperando", "version",
                 "resultado", "fin", "medias")

    def __init__(self, **campos):
        for k in self.__slots__:
            setattr(self, k, None)
        for k, v in campos.items():
            setattr(self, k, v)
        self.version = 0
        self.inicio_tarea = {}
        self.tareas_por = {}     # tareas ya calculadas, por (alistamiento, factor del analista)


class Simulador:
    """Corre los ensayos de una corrida con una regla de despacho y devuelve los registros terminados.

    laboratorio: recursos.Laboratorio. politica: una regla de src/simulacion/politicas.py.
    formas: λ de las tareas (número o «hoja»; duraciones.formas_por_tarea). rho: ρ entre las tareas
    del analista (None = S04). alistamiento: «ninguno», «24h» o «siempre». tasa_reanalisis: S01 (None =
    la de la hoja). generadores: los de escenarios.generadores(semilla).
    """

    def __init__(self, laboratorio, politica, dias, generadores, forma="hoja", rho=None, alistamiento="24h",
                 tasa_reanalisis=None, escenario="base", carga=1.0, semilla=0):
        self.lab, self.politica, self.dias = laboratorio, politica, int(dias)
        self.fin_corrida = self.dias * MIN_POR_DIA
        self.rng = generadores
        self.forma, self.alistamiento = forma, alistamiento
        self.tasa_reanalisis = tasa_reanalisis
        self.escenario, self.carga, self.semilla = escenario, float(carga), int(semilla)
        self.matriz = como_matriz(rho) if rho is not None else como_matriz_hoja()
        self._uniformes, self._k = None, 0
        # calendario de los analistas con una semana de más, para las tareas que pasan del cierre
        extra = self.dias + 7
        bloques = {}
        for b in laboratorio.bloques(extra):
            bloques.setdefault(b.analista, []).append(b)
        self.analistas = {}
        for aid, a in laboratorio.analistas.items():
            bs = sorted(bloques.get(aid, []), key=lambda b: b.inicio)
            tramos = [tr for b in bs for tr in b.tramos]
            self.analistas[aid] = EstadoAnalista(a, tramos, [(b.inicio, b.fin) for b in bs])
        self.equipos = {eid: EstadoEquipo(e) for eid, e in laboratorio.equipos.items()}
        self.colas = {m: ([], []) for m in laboratorio.metodos}   # (a cualquier hora, solo días hábiles)
        self.eventos = []
        self._seq = 0
        self.ahora = 0.0
        self.registros = []
        self.eventos_equipos = []
        self.cat = laboratorio.catalogo
        self.con_alistamiento = {m: self.cat.at[m, "Técnica"] in registro.TECNICAS_CON_IDONEIDAD for m in laboratorio.metodos}

    # ------------------------------------------------------------------ utilidades
    def _evento(self, t, tipo, *datos):
        self._seq += 1
        heapq.heappush(self.eventos, (t, tipo, self._seq, datos))

    def _uniforme(self):
        if self._uniformes is None or self._k >= len(self._uniformes):
            self._uniformes, self._k = uniformes_correlacionadas(2000, self.matriz, rng=self.rng["duraciones"]), 0
        u = self._uniformes[self._k]
        self._k += 1
        return u

    def _formas(self, tecnica):
        return duraciones.formas_por_tarea(tecnica, self.forma)

    def _uniforme_repeticion(self, ensayo, intento):
        """Uniformes de las tareas de una repetición, de un generador propio del ensayo y del intento: no dependen
        del orden en que la regla de despacho produce las repeticiones."""
        semilla = np.random.SeedSequence(self.semilla, spawn_key=(FLUJO_REPETICIONES, int(ensayo), int(intento)))
        return uniformes_correlacionadas(1, self.matriz, rng=np.random.default_rng(semilla))[0]

    def _puntos(self, ej, alistamiento):
        sis, ido = duraciones.tres_puntos_alistamiento(ej.tecnica) if alistamiento else ((0.0,) * 3, (0.0,) * 3)
        return registro.tres_puntos_tareas(ej.metodo, ej.n_soluciones, sis, ido)

    def _tareas(self, ej, alistamiento, factor):
        """Duraciones de las seis tareas de la ejecución con o sin alistamiento, para un analista de ese factor."""
        clave = (bool(alistamiento), factor)
        if clave not in ej.tareas_por:
            ej.tareas_por[clave] = {k: round(v, REDONDEO) for k, v in duraciones.tareas_desde_uniformes(
                ej.u, self._puntos(ej, alistamiento), self._formas(ej.tecnica), factor).items()}
        return ej.tareas_por[clave]

    def corrida_prevista(self, ej, a, t):
        """Momento en que empezaría la corrida de un ensayo sin alistamiento que el analista a empieza en t, con las
        duraciones medias del montaje y de la preparación del análisis, que avanzan solo en sus horas de turno."""
        m = ej.medias[False]
        return a.avanzar(a.avanzar(t, m["t_montaje"] * a.factor), m["t_preparacion"] * a.factor)

    def _necesita_alistamiento(self, ej, equipo, a, t):
        """True si el ensayo necesita la preparación y la idoneidad del sistema al empezar en t con el analista a.

        Con la regla de 24 horas (S29), se las ahorra solo si la unidad hizo la idoneidad de su método y, con las
        duraciones medias de las tareas, su corrida empezaría antes de que se cumplan 24 horas desde que esa
        idoneidad terminó. Se planea con las medias, como el cierre de turno, porque al asignar no se sabe cuánto
        durarán las tareas: si duran más, la corrida puede empezar con la idoneidad algo vencida.
        """
        if ej.tecnica not in registro.TECNICAS_CON_IDONEIDAD or self.alistamiento == "ninguno":
            return False
        if self.alistamiento == "siempre" or not equipo.apto_para(ej.metodo, t):
            return True
        return not equipo.apto_para(ej.metodo, self.corrida_prevista(ej, a, t))

    # ------------------------------------------------------------------ ejecuciones
    def nueva_ejecucion(self, muestra, metodo, n_soluciones, intento=1, origen=None, causa=None, llegada=None,
                        excluir=None, reanalisis_previo=False, ensayo=0, u_resultado=None):
        """Crea una ejecución y programa su llegada a la cola.

        ensayo: posición del ensayo pedido en la corrida; u_resultado: la uniforme que decide si su resultado sale
        fuera de especificación (la misma en todas sus ejecuciones). Las ejecuciones del primer intento toman sus
        uniformes del flujo de duraciones, en el orden de los ensayos; las repeticiones, de _uniforme_repeticion.
        """
        cat = self.cat.loc[metodo]
        self._seq += 1
        u = self._uniforme() if intento == 1 else self._uniforme_repeticion(ensayo, intento)
        ej = Ejecucion(secuencia=self._seq, ensayo=ensayo, id_muestra=muestra["id_muestra"], metodo=metodo,
                       intento=intento, origen=origen, causa=causa, tipo=muestra["tipo_muestra"],
                       prioridad=muestra["prioridad"], llegada_muestra=muestra["llegada_muestra"],
                       plazo_min=muestra["plazo_min"], vencimiento=muestra["vencimiento"],
                       urgente=bool(muestra["urgente"]), n_soluciones=int(n_soluciones), tecnica=cat["Técnica"],
                       complejidad=cat["Complejidad"], calificacion=registro.nivel_minimo(cat["Calificación mínima"]),
                       equipo_requerido=registro.equipo_requerido(metodo), principal=self.lab.principal[metodo],
                       llegada=muestra["llegada_muestra"] if llegada is None else llegada, u=u,
                       u_resultado=u_resultado, excluir=excluir, reanalisis_previo=reanalisis_previo,
                       cualquier_hora=muestra["tipo_muestra"] == "IN" or bool(muestra["urgente"]))
        formas = self._formas(ej.tecnica)
        ej.medias = {False: duraciones.medias_tareas(self._puntos(ej, False), formas)}
        con = self.alistamiento != "ninguno" and ej.tecnica in registro.TECNICAS_CON_IDONEIDAD
        ej.medias[True] = duraciones.medias_tareas(self._puntos(ej, True), formas) if con else ej.medias[False]
        self._evento(max(ej.llegada, self.ahora), LLEGADA, ej)
        return ej

    def _llegada(self, t, ej):
        libres_an = sum(1 for a in self.lab.calificados[ej.metodo]
                        if self.analistas[a].tarea is None and self.analistas[a].en_turno(t))
        libres_eq = sum(1 for e in self.lab.compatibles[ej.metodo] if self.equipos[e].disponible(t))
        faltante = {(True, True): "ninguno", (False, True): "analista", (True, False): "equipo"}.get(
            (libres_an > 0, libres_eq > 0), "ambos")
        ej.libres = (libres_an, libres_eq, faltante)
        cola = self.colas[ej.metodo][0 if ej.cualquier_hora else 1]
        ej.entrada = (self.politica.clave(ej), ej.secuencia, ej)   # la clave con que entró a la cola
        bisect.insort(cola, ej.entrada)

    # ------------------------------------------------------------------ despacho
    def _cabezas(self, metodo, habil):
        cualquiera, habiles = self.colas[metodo]
        if not habil:
            return [x for x in cualquiera[:CABEZAS]]
        return sorted(cualquiera[:CABEZAS] + habiles[:CABEZAS], key=lambda x: (x[0], x[1]))[:CABEZAS]

    def _cabe_en_turno(self, a, ej, alistamiento, t):
        """Regla de cierre de turno (S30), con las duraciones medias de las tareas del ensayo: el analista
        termina dentro de su turno la parte que hace antes de la corrida, y el procesamiento queda, a lo sumo,
        para un turno que empiece en menos de 24 horas. El control en proceso y las urgentes, que tienen 4
        horas de plazo, se terminan en el mismo turno."""
        fin_turno = a.fin_turno(t)
        if fin_turno is None:
            return False
        m, f = ej.medias[alistamiento], a.factor
        t1 = a.avanzar(t, (m["t_prep_sistema"] + m["t_montaje"]) * f)
        t_prep = a.avanzar(t1, m["t_preparacion"] * f)
        if t_prep > fin_turno:
            return False
        t3 = max(t1 + m["t_idoneidad"], t_prep) + m["t_corrida"]
        holgura = 0.0 if ej.cualquier_hora else HOLGURA_PROCESAMIENTO
        return a.avanzar(t3, m["t_procesamiento"] * f) <= fin_turno + holgura

    def _espera_unidad_apta(self, metodo, libres, t, en_cola):
        """True si los ensayos del método esperan a que se libere una unidad ya apta en vez de alistar otra.

        Lo decide la regla de despacho (esperar_unidad_apta; S30), y solo cuando el método necesita alistamiento
        con la regla de 24 horas y ninguna de las unidades libres está apta. Una regla sin ese método nunca espera.
        """
        esperar = getattr(self.politica, "esperar_unidad_apta", None)
        if esperar is None or self.alistamiento != "24h" or not self.con_alistamiento[metodo]:
            return False
        if any(e.apto_para(metodo, t) for e in libres):
            return False
        aptas = sum(1 for e in self.lab.compatibles[metodo] if self.equipos[e].apto_para(metodo, t))
        return esperar(aptas, en_cola)

    def _despachar(self, t):
        habil = es_dia_habil(t)
        while True:
            analistas = [a for a in self.analistas.values() if a.tarea is None and a.en_turno(t)]
            if not analistas:
                return
            equipos = [e for e in self.equipos.values() if e.disponible(t)]
            if not equipos:
                return
            mejor = None
            for metodo, (cualquiera, habiles) in self.colas.items():
                if not cualquiera and not (habil and habiles):
                    continue
                an_m = [a for a in analistas if metodo in a.metodos]
                eq_m = [e for e in equipos if metodo in e.metodos]
                if not an_m or not eq_m:
                    continue
                if self._espera_unidad_apta(metodo, eq_m, t, len(cualquiera) + (len(habiles) if habil else 0)):
                    continue
                for clave, _, ej in self._cabezas(metodo, habil):
                    if mejor is not None and clave >= mejor[0]:
                        break
                    eq = [e for e in eq_m if e.id != ej.excluir]
                    if not eq:
                        continue
                    equipo = self.politica.elegir_equipo(eq, ej, t)
                    alist = {a: self._necesita_alistamiento(ej, equipo, a, t) for a in an_m}
                    an = [a for a in an_m if self._cabe_en_turno(a, ej, alist[a], t)]
                    if not an:
                        continue
                    a = self.politica.elegir_analista(an, ej, t)
                    mejor = (clave, ej, a, equipo, alist[a])
                    break
            if mejor is None:
                return
            self._asignar(t, *mejor[1:])

    def _asignar(self, t, ej, a, e, alistamiento):
        self.colas[ej.metodo][0 if ej.cualquier_hora else 1].remove(ej.entrada)
        ej.analista, ej.equipo, ej.alistamiento, ej.inicio = a, e, alistamiento, t
        ej.puntos = self._puntos(ej, alistamiento)
        ej.tareas = dict(self._tareas(ej, alistamiento, a.factor))
        e.ocupado_por, a.libre_desde = ej, t
        if alistamiento:
            e.apto_metodo, e.apto_desde = None, -math.inf   # la unidad queda apta cuando termine la idoneidad
        self._tarea_analista(t, ej, a, "t_prep_sistema" if alistamiento else "t_montaje")

    def _tarea_analista(self, t, ej, a, nombre):
        a.tarea = (ej, nombre, t)
        ej.inicio_tarea[nombre] = t
        self._evento(a.avanzar(t, ej.tareas[nombre]), FIN_TAREA, ej, nombre, ej.version)

    def _tarea_equipo(self, t, ej, nombre):
        ej.inicio_tarea[nombre] = t
        self._evento(t + ej.tareas[nombre], FIN_TAREA, ej, nombre, ej.version)

    def _analista_libre(self, t, a):
        a.tarea, a.libre_desde = None, t
        if a.pendientes:
            ej = a.pendientes.popleft()
            self._tarea_analista(t, ej, a, "t_procesamiento")

    def _fin_tarea(self, t, ej, nombre, version):
        if version != ej.version:
            return                                    # la ejecución se interrumpió
        a, e = ej.analista, ej.equipo
        if nombre == "t_prep_sistema":
            self._tarea_analista(t, ej, a, "t_montaje")
        elif nombre == "t_montaje":
            ej.esperando = {"t_preparacion"}
            if ej.alistamiento:
                ej.esperando.add("t_idoneidad")
                self._tarea_equipo(t, ej, "t_idoneidad")
            self._tarea_analista(t, ej, a, "t_preparacion")
        elif nombre in ("t_preparacion", "t_idoneidad"):
            ej.esperando.discard(nombre)
            if nombre == "t_idoneidad":
                e.apto_metodo, e.apto_desde = ej.metodo, t
                self._evento(t + VALIDEZ_IDONEIDAD, TURNO)   # al vencer, la cola que esperaba esta unidad se revisa
            else:
                self._analista_libre(t, a)
            if not ej.esperando:
                self._tarea_equipo(t, ej, "t_corrida")
        elif nombre == "t_corrida":
            self._liberar_equipo(t, e)
            if a.tarea is None:
                self._tarea_analista(t, ej, a, "t_procesamiento")
            else:
                a.pendientes.append(ej)
        elif nombre == "t_procesamiento":
            self._terminar(t, ej, self._resultado(ej))
            self._analista_libre(t, a)

    def _resultado(self, ej):
        if ej.reanalisis_previo or ej.causa == "reanálisis":
            return "cumple"                           # el reanálisis es definitivo (S27)
        p = perturbaciones.probabilidad_reanalisis(ej.metodo, self.tasa_reanalisis)
        return "fuera de especificación" if ej.u_resultado < p else "cumple"

    def _liberar_equipo(self, t, e):
        e.ocupado_por, e.libre_desde = None, t
        while e.ventanas_pendientes:
            evento, duracion = e.ventanas_pendientes.popleft()
            self._ventana(t, e, evento, duracion)

    # ------------------------------------------------------------------ fin de una ejecución
    def _terminar(self, t, ej, resultado):
        ej.resultado, ej.fin = resultado, t
        fila = self._registro(ej)
        self.registros.append(fila)
        muestra = {"id_muestra": ej.id_muestra, "tipo_muestra": ej.tipo, "prioridad": ej.prioridad,
                   "llegada_muestra": ej.llegada_muestra, "plazo_min": ej.plazo_min, "vencimiento": ej.vencimiento,
                   "urgente": ej.urgente}
        origen = registro.id_actividad(ej.id_muestra, ej.metodo, ej.intento)
        llegada = fila["fin"]          # la repetición llega cuando termina, en el registro, la ejecución que repite
        if resultado == "fuera de especificación":
            self.nueva_ejecucion(muestra, ej.metodo, ej.n_soluciones, ej.intento + 1, origen, "reanálisis",
                                 llegada, reanalisis_previo=True, ensayo=ej.ensayo, u_resultado=ej.u_resultado)
        elif resultado == "interrumpido":
            self.nueva_ejecucion(muestra, ej.metodo, ej.n_soluciones, ej.intento + 1, origen, "falla de equipo",
                                 llegada, excluir=ej.equipo.id,
                                 reanalisis_previo=ej.reanalisis_previo or ej.causa == "reanálisis",
                                 ensayo=ej.ensayo, u_resultado=ej.u_resultado)

    def _interrumpir(self, t, ej):
        """Una falla del equipo corta la ejecución: sus tareas quedan con lo que alcanzó a hacer."""
        a = ej.analista
        hechas = {}
        for nombre in registro.TAREAS:
            if nombre not in ej.inicio_tarea:
                hechas[nombre] = 0.0
                continue
            inicio = ej.inicio_tarea[nombre]
            if nombre in ("t_idoneidad", "t_corrida"):
                hechas[nombre] = min(ej.tareas[nombre], t - inicio)
            else:
                hechas[nombre] = min(ej.tareas[nombre], a.trabajado(inicio, t))
        ej.tareas = {k: round(max(v, 0.0), REDONDEO) for k, v in hechas.items()}
        ej.version += 1
        if a.tarea is not None and a.tarea[0] is ej:
            self._analista_libre(t, a)
        ej.equipo.ocupado_por, ej.equipo.libre_desde = None, t
        ej.equipo.ventanas_pendientes.clear()       # la reparación incluye la recalificación (WHO, 2024, s. 5.40)
        self._terminar(t, ej, "interrumpido")

    # ------------------------------------------------------------------ equipos
    def _falla(self, t, e, fin):
        self.eventos_equipos.append({"equipo": e.id, "configuracion": e.configuracion, "evento": "falla",
                                     "inicio": t, "fin": fin})
        e.fuera_hasta = max(e.fuera_hasta, fin)
        e.apto_metodo, e.apto_desde = None, -math.inf
        if e.ocupado_por is not None:
            self._interrumpir(t, e.ocupado_por)
        self._evento(fin, EQUIPO_VUELVE, e)

    def _ventana(self, t, e, evento, duracion):
        if e.ocupado_por is not None:
            e.ventanas_pendientes.append((evento, duracion))
            return
        fin = t + duracion
        self.eventos_equipos.append({"equipo": e.id, "configuracion": e.configuracion, "evento": evento,
                                     "inicio": t, "fin": fin})
        e.fuera_hasta = max(e.fuera_hasta, fin)
        e.apto_metodo, e.apto_desde = None, -math.inf  # tras el mantenimiento se recalifica (WHO, 2024, s. 5.39)
        self._evento(fin, EQUIPO_VUELVE, e)

    # ------------------------------------------------------------------ corrida
    def correr(self, muestras, ensayos, ventanas, fallas):
        """Corre la simulación y devuelve (registros, eventos de los equipos, resumen)."""
        por_muestra = muestras.set_index("id_muestra").to_dict("index")
        u_resultado = self.rng["resultados"].random(len(ensayos))   # una por ensayo pedido, en su orden
        for k, fila in enumerate(ensayos.itertuples(index=False)):
            m = dict(por_muestra[fila.id_muestra], id_muestra=fila.id_muestra)
            self.nueva_ejecucion(m, fila.metodo, fila.n_soluciones, ensayo=k, u_resultado=float(u_resultado[k]))
        for v in ventanas.itertuples(index=False):
            self._evento(v.inicio, VENTANA, self.equipos[v.equipo], v.evento, v.fin - v.inicio)
        for f in fallas.itertuples(index=False):
            self._evento(f.inicio, FALLA, self.equipos[f.equipo], f.fin)
        for a in self.analistas.values():
            for inicio in a.inicios:
                if 0 <= inicio < self.fin_corrida:
                    self._evento(inicio, TURNO, a)
        for lunes in range(0, int(self.fin_corrida), MIN_POR_SEMANA):
            self._evento(lunes, TURNO)       # los ensayos que solo empiezan en días hábiles se revisan a las 0:00
        while self.eventos and self.eventos[0][0] < self.fin_corrida:
            t = self.eventos[0][0]
            self.ahora = t
            while self.eventos and self.eventos[0][0] == t:
                _, tipo, _, datos = heapq.heappop(self.eventos)
                if tipo == LLEGADA:
                    self._llegada(t, *datos)
                elif tipo == FIN_TAREA:
                    self._fin_tarea(t, *datos)
                elif tipo == FALLA:
                    self._falla(t, *datos)
                elif tipo == VENTANA:
                    self._ventana(t, *datos)
            self._despachar(t)
        en_curso = sum(1 for e in self.equipos.values() if e.ocupado_por is not None)
        en_cola = sum(len(c) for par in self.colas.values() for c in par)
        pendientes = sum(len(a.pendientes) + (a.tarea is not None and a.tarea[1] == "t_procesamiento")
                         for a in self.analistas.values())   # corrida terminada, procesamiento sin terminar
        resumen = {"ejecuciones_en_curso_al_cierre": en_curso, "procesamientos_pendientes_al_cierre": pendientes,
                   "ejecuciones_en_cola_al_cierre": en_cola}
        return self.registros, self.eventos_equipos, resumen

    # ------------------------------------------------------------------ registro
    def _registro(self, ej):
        r = lambda x: round(float(x), REDONDEO) + 0.0   # noqa: E731
        t = ej.tareas
        d = r(registro.duracion(*(t[c] for c in registro.TAREAS)))
        llegada, inicio, fin = r(ej.llegada), r(ej.inicio), r(ej.fin)
        pausas = r(fin - inicio - d)
        if pausas < 0:                                  # ruido de coma flotante
            fin, pausas = r(inicio + d), 0.0
        sis, ido = duraciones.tres_puntos_alistamiento(ej.tecnica) if ej.alistamiento else ((0.0,) * 3, (0.0,) * 3)
        a, m, b = registro.tres_puntos(ej.metodo, ej.n_soluciones, sis, ido)
        n_an, n_eq, n_comb = registro.combinaciones(ej.metodo)
        libres_an, libres_eq, faltante = ej.libres
        return {
            "escenario": self.escenario, "carga": self.carga, "semilla": self.semilla,
            "id_actividad": registro.id_actividad(ej.id_muestra, ej.metodo, ej.intento), "id_muestra": ej.id_muestra,
            "metodo": ej.metodo, "intento": ej.intento, "id_actividad_origen": ej.origen, "causa_repeticion": ej.causa,
            "resultado": ej.resultado, "tipo_muestra": ej.tipo, "prioridad": int(ej.prioridad),
            "llegada_muestra": r(ej.llegada_muestra), "plazo_min": r(ej.plazo_min), "vencimiento": r(ej.vencimiento),
            "tecnica": ej.tecnica, "complejidad": ej.complejidad, "n_soluciones": ej.n_soluciones,
            "calificacion_exigida": ej.calificacion, "equipo_requerido": ej.equipo_requerido,
            "t_optimista": r(a), "t_mas_probable": r(m), "t_pesimista": r(b),
            "analista": ej.analista.id, "nivel_analista": ej.analista.nivel, "equipo": ej.equipo.id,
            "configuracion": ej.equipo.configuracion, "idoneidad_ejecutada": bool(ej.alistamiento),
            "duracion_simulada": d, **{c: t[c] for c in registro.TAREAS},
            "llegada": llegada, "inicio": inicio, "fin": fin, "espera": r(inicio - llegada), "pausas": pausas,
            "tiempo_en_sistema": r(fin - llegada), "tardanza": r(max(0.0, fin - r(ej.vencimiento))),
            "urgente": ej.urgente,
            "bandera_reanalisis": ej.causa == "reanálisis" or ej.resultado == "fuera de especificación",
            "bandera_falla": ej.causa == "falla de equipo" or ej.resultado == "interrumpido",
            "analistas_calificados": n_an, "equipos_compatibles": n_eq, "combinaciones_factibles": n_comb,
            "analistas_libres": libres_an, "equipos_libres": libres_eq, "recurso_faltante": faltante,
        }


def como_matriz_hoja():
    """Matriz objetivo de la hoja «Correlaciones», con el ρ de S04."""
    from src.generador.correlacion import matriz_objetivo
    return matriz_objetivo()
