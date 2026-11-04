"""Motor de simulación (src/simulacion/entorno.py), reglas de despacho, escenarios y corridas del generador.

Diseño: docs/Generador_de_datos.docx, secciones 3 a 8; supuestos S25 a S30.
"""
import numpy as np
import pandas as pd
import pytest

from src.generador import escenarios, generar, registro
from src.generador.recursos import MIN_POR_DIA, MIN_POR_SEMANA, Laboratorio
from src.simulacion import politicas
from src.simulacion.entorno import VALIDEZ_IDONEIDAD, Simulador

TOL = 1e-6
SEMANAS = 3


class Espia(Simulador):
    """El motor tal cual, más cuatro anotaciones: la idoneidad con que se planea y con que empieza cada corrida de HPLC
    o GC, los eventos que se programan antes del momento actual y los momentos en que despacha."""

    creados = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planes, self.corridas, self.pasados, self.despachos = [], [], [], []
        Espia.creados.append(self)

    def _asignar(self, t, ej, a, e, alistamiento):
        if ej.tecnica in registro.TECNICAS_CON_IDONEIDAD and not alistamiento:
            self.planes.append((e.apto_metodo == ej.metodo, self.corrida_prevista(ej, a, t) - e.apto_desde))
        super()._asignar(t, ej, a, e, alistamiento)

    def _tarea_equipo(self, t, ej, nombre):
        if nombre == "t_corrida" and ej.tecnica in registro.TECNICAS_CON_IDONEIDAD:
            e = ej.equipo
            self.corridas.append((e.apto_metodo == ej.metodo, t - e.apto_desde, bool(ej.alistamiento)))
        super()._tarea_equipo(t, ej, nombre)

    def _evento(self, t, tipo, *datos):
        if t < self.ahora:
            self.pasados.append((self.ahora, t, tipo))
        super()._evento(t, tipo, *datos)

    def _despachar(self, t):
        self.despachos.append(t)
        super()._despachar(t)


@pytest.fixture(scope="module")
def corrida():
    """Tres semanas del escenario base con la configuración del dataset v0.2."""
    return generar.generar_corrida("base", 2026, "v0.2", semanas=SEMANAS)


@pytest.fixture(scope="module")
def espia():
    """La misma corrida, con el motor anotado (Espia)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(generar, "Simulador", Espia)
        registros, _, _ = generar.generar_corrida("base", 2026, "v0.2", semanas=SEMANAS)
    return Espia.creados[-1], registros


@pytest.fixture(scope="module")
def lab():
    return Laboratorio.desde_excel()


def test_la_corrida_cumple_las_reglas(corrida):
    registros, eventos, resumen = corrida
    assert resumen["registros"] == len(registros) == resumen["registros_que_cumplen_las_reglas"]
    assert registro.chequear_registro(registros).all().all()
    assert list(registros.columns) == registro.CAMPOS_REGISTRO
    assert list(eventos.columns) == generar.COLUMNAS_EVENTOS
    assert resumen["configuracion"] == {k: v for k, v in escenarios.VERSIONES["v0.2"].items() if k != "descripcion"}
    assert registros["idoneidad_ejecutada"].any()                      # v0.2 tiene alistamiento (S29)
    assert set(registros.loc[registros["idoneidad_ejecutada"], "tecnica"]) <= set(registro.TECNICAS_CON_IDONEIDAD)


def test_reproducible_con_la_semilla():
    a, ea, _ = generar.generar_corrida("baja", 7, "v0.2", semanas=1)
    b, eb, _ = generar.generar_corrida("baja", 7, "v0.2", semanas=1)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ea, eb)
    c, _, _ = generar.generar_corrida("baja", 8, "v0.2", semanas=1)
    assert not a.equals(c)


def test_mismas_muestras_con_otra_regla():
    """Números aleatorios comunes: con la misma semilla, otra regla de despacho cambia las asignaciones, no las
    muestras, la corrida de cada ensayo, sus resultados ni las fallas (escenarios.generadores y entorno)."""
    a, _, ra = generar.generar_corrida("base", 2026, "v0.2", semanas=1)
    b, _, rb = generar.generar_corrida("base", 2026, "v0.2", semanas=1, politica=politicas.FIFO())
    assert (ra["muestras"], ra["ensayos_pedidos"], ra["fallas"]) == (rb["muestras"], rb["ensayos_pedidos"], rb["fallas"])
    clave = ["id_muestra", "metodo"]
    pa = a[a["intento"].eq(1) & a["resultado"].ne("interrumpido")].set_index(clave)
    pb = b[b["intento"].eq(1) & b["resultado"].ne("interrumpido")].set_index(clave)
    comunes = pa.index.intersection(pb.index)
    assert len(comunes) > 0.8 * len(pa)
    assert np.allclose(pa.loc[comunes, "t_corrida"], pb.loc[comunes, "t_corrida"])
    assert (pa.loc[comunes, "llegada_muestra"] == pb.loc[comunes, "llegada_muestra"]).all()
    assert (pa.loc[comunes, "resultado"] == pb.loc[comunes, "resultado"]).all()
    assert pa["resultado"].eq("fuera de especificación").any()
    # las repeticiones toman sus números de un generador propio del ensayo y del intento
    clave_r = ["id_muestra", "metodo", "intento"]
    qa = a[a["causa_repeticion"].eq("reanálisis") & a["resultado"].ne("interrumpido")].set_index(clave_r)
    qb = b[b["causa_repeticion"].eq("reanálisis") & b["resultado"].ne("interrumpido")].set_index(clave_r)
    comunes_r = qa.index.intersection(qb.index)
    assert len(comunes_r) > 0
    assert np.allclose(qa.loc[comunes_r, "t_corrida"], qb.loc[comunes_r, "t_corrida"])


def test_analista_en_turno_al_empezar(corrida, lab):
    registros = corrida[0]
    cal = lab.calendario_analistas(SEMANAS * 7)
    for a, g in registros.groupby("analista"):
        tramos = cal[cal["analista"] == a]
        ini, fin = tramos["inicio"].to_numpy(), tramos["fin"].to_numpy()
        k = np.searchsorted(ini, g["inicio"].to_numpy(), side="right") - 1
        assert (k >= 0).all() and (g["inicio"].to_numpy() < fin[k]).all(), a


def test_no_empieza_en_un_equipo_fuera_de_servicio(corrida):
    registros, eventos, _ = corrida
    for e, g in registros.groupby("equipo"):
        ev = eventos[eventos["equipo"] == e]
        for ini in g["inicio"]:
            assert not ((ev["inicio"] <= ini + TOL) & (ini < ev["fin"] - TOL)).any(), (e, ini)


def test_una_ejecucion_a_la_vez_en_cada_unidad(corrida):
    registros = corrida[0]
    for e, g in registros.sort_values("inicio").groupby("equipo"):
        hasta_corrida = (g["t_prep_sistema"] + g["t_montaje"] + np.maximum(g["t_idoneidad"], g["t_preparacion"])
                         + g["t_corrida"])
        ocupado = np.where(g["resultado"].eq("interrumpido"), g["fin"], g["inicio"] + hasta_corrida)
        assert (g["inicio"].to_numpy()[1:] >= ocupado[:-1] - TOL).all(), e


def test_corrida_con_idoneidad_vigente(espia):
    """Un ensayo de HPLC o GC se ahorra el alistamiento solo si, con las duraciones medias de sus tareas, su corrida
    empezaría antes de que se cumplan 24 horas desde la idoneidad de su método en esa unidad (S29). El que hace el
    alistamiento corre con su propia idoneidad. Si las tareas duran más de lo previsto, la corrida puede empezar con la
    idoneidad algo vencida: pasa en pocas corridas (en un año del escenario base, cerca del 4 %)."""
    sim, _ = espia
    assert len(sim.planes) > 100
    assert all(mismo and 0 <= edad < VALIDEZ_IDONEIDAD for mismo, edad in sim.planes)
    assert all(mismo for mismo, _, _ in sim.corridas)
    con = [edad for _, edad, alist in sim.corridas if alist]
    sin = [edad for _, edad, alist in sim.corridas if not alist]
    assert len(con) > 50 and all(0 <= edad < VALIDEZ_IDONEIDAD for edad in con)
    assert np.mean([edad >= VALIDEZ_IDONEIDAD for edad in sin]) < 0.08


def test_despacha_al_empezar_cada_semana(espia):
    """Los ensayos que solo empiezan en días hábiles se revisan el lunes a las 0:00, aunque no pase nada más."""
    sim, _ = espia
    despachos = set(sim.despachos)
    assert all(s * MIN_POR_SEMANA in despachos for s in range(SEMANAS))


def test_el_reloj_no_retrocede(espia, corrida):
    """Ningún evento se programa antes del momento actual, y el motor anotado da la misma corrida."""
    sim, registros = espia
    assert sim.pasados == []
    pd.testing.assert_frame_equal(registros, corrida[0])


def test_idoneidad_vigente_24_horas(corrida):
    """Un ensayo de HPLC o GC sin alistamiento corre en una unidad que hizo la idoneidad del mismo método hace menos de
    24 horas (más lo que tardó ese ensayo), sin otro método, ventana ni falla de por medio (S29).

    Se revisan los ensayos que empiezan hasta cuatro días antes del cierre: el que dejó apta la unidad puede no estar
    en el dataset si su procesamiento quedó pendiente al cierre (por ejemplo, de un viernes por la tarde al lunes).
    """
    registros, eventos, _ = corrida
    hasta = (SEMANAS * 7 - 4) * MIN_POR_DIA
    cromatografia = registros[registros["tecnica"].isin(registro.TECNICAS_CON_IDONEIDAD)].sort_values("inicio")
    revisados = 0
    for e, g in cromatografia.groupby("equipo"):
        ev = eventos[eventos["equipo"] == e]
        previos = []
        for fila in g.itertuples(index=False):
            if not fila.idoneidad_ejecutada and fila.inicio < hasta:
                revisados += 1
                con = [p for p in previos if p.idoneidad_ejecutada]
                assert con, (e, fila.id_actividad)
                ultimo = con[-1]
                assert ultimo.metodo == fila.metodo
                assert all(p.metodo == fila.metodo for p in previos[previos.index(ultimo):])
                assert fila.inicio - ultimo.inicio < 24 * 60 + ultimo.duracion_simulada + ultimo.pausas + TOL
                assert not ((ev["inicio"] > ultimo.inicio) & (ev["inicio"] < fila.inicio)).any()
            previos.append(fila)
    assert revisados > 50


def test_repeticiones(corrida):
    registros = corrida[0]
    origen = registros.set_index("id_actividad")
    rep = registros[registros["causa_repeticion"].eq("falla de equipo")]
    for fila in rep.itertuples(index=False):
        assert fila.equipo != origen.at[fila.id_actividad_origen, "equipo"]     # otra unidad (S27)
    reanalisis = registros[registros["causa_repeticion"].eq("reanálisis")]
    assert not reanalisis.duplicated(["id_muestra", "metodo"]).any()          # a lo sumo uno por ensayo
    assert reanalisis["resultado"].ne("fuera de especificación").all()        # resultado definitivo


def test_solo_control_en_proceso_y_urgentes_empiezan_en_fin_de_semana(corrida):
    registros = corrida[0]
    fin_de_semana = (registros["inicio"] // MIN_POR_DIA) % 7 >= 5
    otros = ~(registros["tipo_muestra"].eq("IN") | registros["urgente"])
    primeros = registros["intento"].eq(1)
    assert not (fin_de_semana & otros & primeros).any()                        # S25


def test_reglas_de_referencia_del_o3():
    for regla in (politicas.FIFO(), politicas.Aleatoria(1)):
        _, _, resumen = generar.generar_corrida("baja", 2026, "v0.2", semanas=1, politica=regla)
        assert resumen["regla_de_despacho"] == regla.nombre
        assert resumen["registros"] == resumen["registros_que_cumplen_las_reglas"] > 0
    with pytest.raises(NotImplementedError):
        politicas.regla_experta()
    with pytest.raises(NotImplementedError):
        politicas.regla_informada()


def test_espera_a_una_unidad_apta():
    regla = politicas.ReglaGenerador()
    assert not regla.esperar_unidad_apta(0, 1)       # ninguna apta: se alista una
    assert regla.esperar_unidad_apta(1, 4)           # Costigliola et al. (2017, p. 9019): menos de 5 en cola
    assert not regla.esperar_unidad_apta(1, 5)
    assert regla.esperar_unidad_apta(2, 9)
    assert not hasattr(politicas.FIFO(), "esperar_unidad_apta")


def test_escenarios_y_semillas():
    assert {e: v["multiplicador_carga"] for e, v in escenarios.ESCENARIOS.items()} == {"baja": 0.7, "base": 1.0, "alta": 1.5}
    assert escenarios.semillas() == [2026, 2027, 2028]
    assert len(escenarios.corridas()) == 9
    g1, g2 = escenarios.generadores(2026), escenarios.generadores(2026)
    assert list(g1) == ["llegadas", "urgentes", "ensayos", "duraciones", "resultados", "fallas"]
    primeros = {k: g.random() for k, g in g1.items()}
    assert primeros == {k: g.random() for k, g in g2.items()}
    assert len(set(primeros.values())) == 6                                  # flujos distintos
    assert set(escenarios.VERSIONES) == {"v0.1", "v0.2"}


def test_guardar_y_leer_una_corrida(corrida, tmp_path, monkeypatch):
    registros, eventos, resumen = corrida
    generar.guardar_corrida(registros, eventos, resumen, tmp_path)
    monkeypatch.setattr(generar, "carpeta", lambda version: tmp_path)
    leidos, ev, resumenes = generar.cargar_version("v0.2")
    assert len(resumenes) == 1 and resumenes[0]["registros"] == resumen["registros"]
    assert len(leidos) == len(registros) and len(ev) == len(eventos)
    assert registro.chequear_registro(leidos).all().all()                   # el CSV conserva vacíos y booleanos
    assert np.allclose(leidos["duracion_simulada"], registros["duracion_simulada"], atol=1e-6)


def test_las_corridas_de_prueba_no_reemplazan_el_dataset():
    base = generar.RAIZ / "datos" / "sinteticos"
    assert generar.carpeta("v0.2") == base / "v0.2"
    assert generar.carpeta("v0.1") == base / "borradores" / "v0.1"
    assert generar.carpeta("v0.2", 8) == base / "borradores" / "v0.2_8_semanas"
