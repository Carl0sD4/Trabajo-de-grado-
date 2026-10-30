"""Factores de correlación: la matriz objetivo de la hoja «Correlaciones», la cópula gaussiana, el
método de Iman y Conover, las duraciones correlacionadas de un ensayo y el cálculo del indicador V2.

Anteproyecto: §1.4 y §5.2 (incorporar la correlación en lugar de asumir independencia por
conveniencia); indicador V2, «correlaciones entre variables definidas».
Diseño: docs/Generador_de_datos.docx, sección 2.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.generador import correlacion as co
from src.generador import duraciones as du
from src.generador import parametros, registro

N = 200_000
TOL = 0.01        # diferencia permitida con la matriz objetivo; con N valores el error estándar es de unas 0,002
SISTEMA = {"prep_sistema": (60.0, 90.0, 150.0), "idoneidad": (100.0, 150.0, 200.0)}   # de ensayo, como en test_duraciones
ANALISTA = list(registro.TAREAS_ANALISTA)


# ---------------------------------------------------------------------------- matriz objetivo
def test_la_hoja_correlaciones_toma_el_rho_de_s04():
    supuestos = parametros.cargar_hoja("Supuestos").set_index("ID")
    rho = float(supuestos.at["S04", "Propuesta inicial (supuesto)"])
    m = co.matriz_objetivo()
    assert list(m.index) == list(m.columns) == registro.TAREAS
    for a in registro.TAREAS:
        for b in registro.TAREAS:
            esperado = 1.0 if a == b else rho if a in ANALISTA and b in ANALISTA else 0.0
            assert m.at[a, b] == esperado, (a, b)
    hoja = parametros.cargar_hoja("Correlaciones").set_index("Tarea")
    quien = ["Analista" if t in ANALISTA else "Equipo" for t in registro.TAREAS]
    assert list(hoja.loc[registro.TAREAS, "La hace"]) == quien


@pytest.mark.parametrize("rho", [0.3, 0.7])
def test_la_sensibilidad_cambia_solo_las_tareas_del_analista(rho):
    base, otra = co.matriz_objetivo(), co.matriz_objetivo(rho)
    for a in registro.TAREAS:
        for b in registro.TAREAS:
            cambia = a != b and a in ANALISTA and b in ANALISTA
            assert (otra.at[a, b] == rho) if cambia else (otra.at[a, b] == base.at[a, b]), (a, b)


def test_conversion_entre_spearman_y_la_normal():
    rho = np.linspace(-1, 1, 41)
    assert np.allclose(co.rho_spearman(co.rho_normal(rho)), rho)
    assert co.rho_normal(0) == 0 and co.rho_normal(1) == pytest.approx(1)
    assert co.rho_normal(0.5) == pytest.approx(0.5176, abs=1e-4)          # 2 sen(π/12)
    # dos normales con esa correlación tienen correlación de rangos 0,5 (Clemen y Reilly, 1999, p. 211)
    r = co.rho_normal(0.5)
    z = np.random.default_rng(2026).multivariate_normal([0, 0], [[1, r], [r, 1]], size=N)
    assert stats.spearmanr(z[:, 0], z[:, 1]).statistic == pytest.approx(0.5, abs=TOL)
    # sin la conversión, la correlación de rangos se quedaría en 0,48
    assert co.rho_spearman(0.5) == pytest.approx(0.4826, abs=1e-4)


@pytest.mark.parametrize("matriz", [
    [[1, 0.5], [0.4, 1]],                                   # no es simétrica
    [[1, 0.5], [0.5, 0.9]],                                 # la diagonal no es 1
    [[1, 1.2], [1.2, 1]],                                   # fuera de [−1, 1]
    [[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]],        # no pueden darse a la vez
    [[1, np.nan], [np.nan, 1]],                             # vacía
    [1, 0.5],                                               # no es cuadrada
])
def test_matrices_que_no_sirven(matriz):
    with pytest.raises(ValueError):
        co.validar_matriz(matriz)


# ---------------------------------------------------------------------------- cópula gaussiana
@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7])
def test_la_copula_llega_a_la_matriz_objetivo(rho):
    objetivo = co.matriz_objetivo(rho)
    u = co.uniformes_correlacionadas(N, objetivo, rng=2026)
    assert u.shape == (N, len(registro.TAREAS)) and (u > 0).all() and (u < 1).all()
    assert co.error_maximo(co.correlacion_rangos(pd.DataFrame(u, columns=registro.TAREAS)), objetivo) < TOL
    assert all(stats.kstest(u[:, j], "uniform").pvalue > 0.001 for j in range(u.shape[1]))   # siguen uniformes


# ---------------------------------------------------------------------------- Iman y Conover
def test_iman_conover_conserva_los_valores_y_llega_a_la_objetivo():
    independientes = du.muestrear_ensayo("M02", 4, n=N // 2, rng=2026, **SISTEMA)[registro.TAREAS]
    objetivo = co.matriz_objetivo()
    sin_correlacion = pd.DataFrame(np.eye(len(registro.TAREAS)), index=registro.TAREAS, columns=registro.TAREAS)
    assert co.error_maximo(co.correlacion_rangos(independientes), sin_correlacion) < TOL
    reordenadas = co.inducir_correlacion(independientes, objetivo, rng=2026)
    assert list(reordenadas.columns) == registro.TAREAS
    for t in registro.TAREAS:   # cada tarea conserva exactamente sus valores
        assert np.array_equal(np.sort(reordenadas[t].to_numpy()), np.sort(independientes[t].to_numpy())), t
    assert co.error_maximo(co.correlacion_rangos(reordenadas), objetivo) < TOL
    # también con un arreglo, otra distribución y otra matriz
    x = np.random.default_rng(1).exponential(size=(50_000, 3))
    m = np.array([[1, 0.6, -0.3], [0.6, 1, 0.0], [-0.3, 0.0, 1]])
    y = co.inducir_correlacion(x, m, rng=1)
    assert isinstance(y, np.ndarray) and y.shape == x.shape
    assert co.error_maximo(co.correlacion_rangos(y), m) < 0.015


def test_iman_conover_rechaza_datos_que_no_sirven():
    m = np.array([[1, 0.5], [0.5, 1]])
    vacio = np.ones((10, 2))
    vacio[3, 1] = np.nan
    for x in (np.arange(10.0), np.ones((10, 3)), np.ones((2, 2)), vacio):   # una columna, otro tamaño, pocas filas, vacíos
        with pytest.raises(ValueError):
            co.inducir_correlacion(x, m)
    # con muy pocas filas los puntajes al azar pueden salir dependientes; el mensaje lo dice
    mensajes = set()
    for semilla in range(40):
        try:
            co.inducir_correlacion(np.random.default_rng(semilla).normal(size=(4, 3)), np.eye(3), rng=semilla)
        except ValueError as error:
            mensajes.add(str(error))
    assert mensajes and all("linealmente dependientes" in m for m in mensajes)


def test_iman_conover_conserva_el_indice():
    tabla = pd.DataFrame(np.random.default_rng(3).normal(size=(500, 2)), columns=["a", "b"], index=range(1000, 1500))
    m = pd.DataFrame([[1, 0.5], [0.5, 1]], index=["a", "b"], columns=["a", "b"])
    reordenada = co.inducir_correlacion(tabla, m, rng=3)
    assert reordenada.index.equals(tabla.index)
    tabla[["a", "b"]] = reordenada          # se puede asignar de vuelta sin que aparezcan vacíos
    assert tabla.notna().all().all()


# ---------------------------------------------------------------------------- duración de un ensayo
@pytest.mark.parametrize("rho", [0.3, 0.5, 0.7])
def test_el_ensayo_sale_con_la_correlacion_objetivo(rho):
    # con las seis tareas variables y un júnior, cuyo factor (S07) no cambia los rangos
    t = du.muestrear_ensayo("M02", 4, n=N, rng=2026, nivel="júnior", correlacion=rho, **SISTEMA)
    assert co.error_maximo(co.correlacion_rangos(t[registro.TAREAS]), co.matriz_objetivo(rho)) < TOL


def test_la_correlacion_no_cambia_la_distribucion_de_cada_tarea():
    sin = du.muestrear_ensayo("M07", 3, n=N, rng=1)
    con = du.muestrear_ensayo("M07", 3, n=N, rng=2, correlacion=0.7)
    for tarea, (a, _, b) in registro.tres_puntos_tareas("M07", 3).items():
        assert con[tarea].between(a, b).all(), tarea
        if b > a:
            assert stats.ks_2samp(sin[tarea], con[tarea]).pvalue > 0.001, tarea
    # la media de la duración es la de siempre, porque es una suma; la dispersión crece
    optimista, mas_probable, pesimista = registro.tres_puntos("M07", 3)
    d = con["duracion_simulada"]
    assert abs(d.mean() - du.media_betapert(optimista, mas_probable, pesimista)) < 5 * d.std() / np.sqrt(N)
    assert d.std() > sin["duracion_simulada"].std()
    assert d.between(optimista, pesimista).all()   # regla duracion_coherente


def test_muestrear_ensayo_recibe_la_correlacion_de_varias_formas():
    objetivo = co.matriz_objetivo()
    numero = du.muestrear_ensayo("M02", 4, n=1000, rng=5, correlacion=0.5)
    for otra in (objetivo, objetivo.to_numpy(), objetivo.iloc[::-1, ::-1]):
        pd.testing.assert_frame_equal(numero, du.muestrear_ensayo("M02", 4, n=1000, rng=5, correlacion=otra))
    for mala in (True, np.eye(5), objetivo.drop(columns="t_corrida"), 1.5):
        with pytest.raises((TypeError, ValueError)):
            du.muestrear_ensayo("M02", 4, correlacion=mala)


def test_misma_semilla_mismos_valores_con_correlacion():
    uno = du.muestrear_ensayo("M11", 3, n=50, rng=2026, correlacion=0.5)
    pd.testing.assert_frame_equal(uno, du.muestrear_ensayo("M11", 3, n=50, rng=2026, correlacion=0.5))
    assert not uno.equals(du.muestrear_ensayo("M11", 3, n=50, rng=2027, correlacion=0.5))
    assert not uno.equals(du.muestrear_ensayo("M11", 3, n=50, rng=2026))    # sin correlación es otro muestreo


# ---------------------------------------------------------------------------- indicador V2
def test_v2_compara_la_matriz_dentro_de_grupos():
    partes, semilla = [], 0
    for metodo in ("M02", "M07"):
        for n_soluciones in (3, 4):
            for nivel in ("júnior", "sénior"):
                semilla += 1
                t = du.muestrear_ensayo(metodo, n_soluciones, n=5_000, rng=semilla, nivel=nivel, correlacion=0.5)
                partes.append(t.assign(metodo=metodo, n_soluciones=n_soluciones, nivel_analista=nivel,
                                       idoneidad_ejecutada=False))
    registros = pd.concat(partes, ignore_index=True)
    objetivo = co.matriz_objetivo()
    por_grupos = co.correlacion_por_grupos(registros)
    assert co.error_maximo(por_grupos, objetivo) < 0.02
    assert por_grupos[["t_prep_sistema", "t_idoneidad"]].isna().all().all()   # fijas en cero en estos grupos
    # sin agrupar, la mezcla de métodos y niveles cambia la correlación: por eso V2 compara por grupos
    assert co.error_maximo(co.correlacion_rangos(registros[registro.TAREAS]), objetivo) > 0.1
    assert co.correlacion_por_grupos(registros.head(co.MINIMO_POR_GRUPO - 1)).isna().all().all()
    # V2 deja fuera los pares con una tarea fija en todos los grupos: quedan los 6 de las otras cuatro
    v2 = co.verificar_por_grupos(registros, objetivo)
    assert len(v2) == 6 and co.veredicto_v2(v2) == "cumple"


def test_valor_esperado_y_varianza_con_pocos_pares():
    # con pocos pares la correlación de rangos sale más cerca de cero (Moran, 1948); con muchos, tiende a ρ
    assert co.rho_spearman_esperada(0.5, 10) == pytest.approx(0.458, abs=0.001)
    assert co.rho_spearman_esperada(0.5, 10**7) == pytest.approx(0.5, abs=1e-6)
    assert co.rho_spearman_esperada(0, 10) == 0 and co.rho_spearman_esperada(1, 10) == pytest.approx(1)
    grupos = 20_000
    for n in (co.MINIMO_POR_GRUPO, 30):
        for rho in (0.0, 0.5, 0.9):
            u = co.uniformes_correlacionadas(grupos * n, [[1, rho], [rho, 1]], rng=2026).reshape(grupos, n, 2)
            rangos = u.argsort(axis=1).argsort(axis=1) - (n - 1) / 2
            r = (rangos[..., 0] * rangos[..., 1]).sum(axis=1) / (n * (n * n - 1) / 12)
            assert abs(r.mean() - co.rho_spearman_esperada(rho, n)) < 5 * r.std() / np.sqrt(grupos)
            # la varianza es 1/(n − 1) sin correlación y menor con ella: sirve de cota para el error estándar de V2
            if rho == 0:
                assert r.var() == pytest.approx(co.varianza_rho_spearman(n), rel=0.05)
            else:
                assert r.var() < 0.9 * co.varianza_rho_spearman(n)


def _grupos_chicos(rho, grupos=400, n=12, semilla=0):
    """Registros sintéticos en muchos grupos chicos, con las seis tareas correlacionadas como la matriz."""
    u = co.uniformes_correlacionadas(grupos * n, co.matriz_objetivo(rho), rng=semilla)
    datos = pd.DataFrame(u, columns=registro.TAREAS)
    return datos.assign(grupo=np.repeat(np.arange(grupos), n))


def test_v2_con_grupos_chicos_compara_con_la_esperada():
    objetivo = co.matriz_objetivo()
    v2 = co.verificar_por_grupos(_grupos_chicos(0.5), objetivo, por=("grupo",))
    assert len(v2) == 15 and v2["cumple"].all() and co.veredicto_v2(v2) == "cumple"
    analista = v2[v2["objetivo"].eq(0.5)]
    # comparada con la objetivo, en cambio, la correlación sale baja en los pares del analista
    assert analista["obtenida"].mean() < 0.5 - 0.02
    assert np.allclose(analista["esperada"], co.rho_spearman_esperada(0.5, 12))
    # 400 grupos de 12: el error estándar es la raíz de 1/(400 · 11)
    assert np.allclose(v2["error_estandar"], np.sqrt(1 / (400 * 11))) and np.allclose(v2["tolerancia"], 5 * v2["error_estandar"])
    # detecta una correlación distinta de la objetivo
    otra = co.verificar_por_grupos(_grupos_chicos(0.2, semilla=1), objetivo, por=("grupo",))
    assert not otra.loc[otra["objetivo"].eq(0.5), "cumple"].any() and co.veredicto_v2(otra) == "no cumple"
    # el error estándar no sale de la dispersión entre grupos: también decide con pocos grupos o con uno que pesa casi todo
    pocos = co.verificar_por_grupos(_grupos_chicos(0.5, grupos=9, n=50), objetivo, por=("grupo",))
    assert co.veredicto_v2(pocos) == "cumple"
    assert co.veredicto_v2(co.verificar_por_grupos(_grupos_chicos(0.0, grupos=9, n=50, semilla=2), objetivo,
                                                   por=("grupo",))) == "no cumple"
    tamanos = np.r_[1_000, np.full(11, 12)]
    u = co.uniformes_correlacionadas(tamanos.sum(), objetivo, rng=3)
    dispares = pd.DataFrame(u, columns=registro.TAREAS).assign(grupo=np.repeat(np.arange(len(tamanos)), tamanos))
    assert co.veredicto_v2(co.verificar_por_grupos(dispares, objetivo, por=("grupo",))) == "cumple"
    # con grupos de menos de 10 registros, o tan pocos que no verían la correlación de S04, quedan sin datos suficientes
    chicos = co.verificar_por_grupos(_grupos_chicos(0.5, grupos=50, n=co.MINIMO_POR_GRUPO - 1), objetivo, por=("grupo",))
    assert len(chicos) == 15 and chicos["cumple"].isna().all() and chicos["registros"].eq(0).all()
    assert co.veredicto_v2(chicos) == "sin datos suficientes" and co.veredicto_v2(chicos.head(0)) == "sin datos suficientes"
    uno = co.verificar_por_grupos(_grupos_chicos(0.5, grupos=1, n=12), objetivo, por=("grupo",))
    assert uno["cumple"].isna().all() and co.veredicto_v2(uno) == "sin datos suficientes"
    assert np.allclose(uno["minima_detectable"], 7 * uno["error_estandar"])
    # con 12 grupos de 10, tareas independientes no pasan por correctas: o no cumplen o quedan sin datos
    for semilla in range(5):
        independientes = co.verificar_por_grupos(_grupos_chicos(0.0, grupos=12, n=10, semilla=semilla), objetivo, por=("grupo",))
        assert co.veredicto_v2(independientes) != "cumple"
    # con la objetivo en cero, debe poder ver una correlación de 0,3: con un grupo de 20 no la ve
    ceros = co.verificar_por_grupos(_grupos_chicos(0.9, grupos=1, n=20), co.matriz_objetivo(0.0), por=("grupo",))
    assert co.veredicto_v2(ceros) == "sin datos suficientes"


def test_indicador_v2_usa_solo_las_ejecuciones_terminadas():
    catalogo = parametros.cargar_hoja("Catálogo").set_index("ID")
    partes = []
    for k, metodo in enumerate(("M01", "M02", "M06", "M07")):
        n_sol = int(catalogo.at[metodo, "Soluciones máx"])
        t = du.muestrear_ensayo(metodo, n_sol, n=3_000, rng=k, correlacion=0.5)
        partes.append(t.assign(metodo=metodo, tecnica=catalogo.at[metodo, "Técnica"],
                               complejidad=catalogo.at[metodo, "Complejidad"], n_soluciones=n_sol,
                               nivel_analista="intermedio", idoneidad_ejecutada=False, resultado="cumple"))
    registros = pd.concat(partes, ignore_index=True)
    # ejecuciones interrumpidas: tareas parciales, sin relación con las demás
    cortadas = registros.sample(600, random_state=1).index
    registros.loc[cortadas, registro.TAREAS] *= np.random.default_rng(1).uniform(0, 1, size=(600, len(registro.TAREAS)))
    registros.loc[cortadas, "resultado"] = "interrumpido"
    v2 = co.indicador_v2(registros)
    tareas = v2["tareas"]
    assert tareas["registros"].max() == len(registros) - 600
    assert (tareas["obtenida"] - tareas["esperada"]).abs().max() < 0.03 and v2["veredicto"] == "cumple"
    # con las interrumpidas, las tareas del analista se separan de la objetivo
    todas = co.verificar_por_grupos(registros, co.matriz_objetivo())
    assert (todas["obtenida"] - todas["esperada"]).abs().max() > 0.05 and co.veredicto_v2(todas) == "no cumple"
    estructura = v2["estructura"].set_index(["variable_1", "variable_2"])
    assert len(estructura) == 6
    assert estructura.at[("n_soluciones", "t_preparacion"), "dentro_de_cada_tecnica"] > 0.3
    # lo mismo sin la columna resultado, con la complejidad ya en números y el ρ como número
    terminadas = registros[registros["resultado"].ne("interrumpido")].drop(columns="resultado")
    otra = co.indicador_v2(terminadas.assign(complejidad=terminadas["complejidad"].map(co.ORDEN_COMPLEJIDAD)), objetivo=0.5)
    for parte in ("tareas", "estructura"):
        pd.testing.assert_frame_equal(otra[parte], v2[parte])
    assert otra["veredicto"] == v2["veredicto"]
