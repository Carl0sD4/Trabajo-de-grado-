"""Muestreo BetaPERT: media, varianza, moda, límites, reproducibilidad, coincidencia con la hoja
«Duraciones» y duración de un ensayo completo.

Anteproyecto: Monte Carlo con estimaciones de tres puntos (BetaPERT), dimensión «Tiempos de ejecución».
Diseño: docs/Generador_de_datos.docx, sección 1.
"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.generador import duraciones as du
from src.generador import parametros, registro

# Preparación por solución en HPLC: T(5, 40, 100) en Costigliola et al. (2017, Tabla 3, p. 9016)
A, M, B = 5.0, 40.0, 100.0
MEDIA_PERT = (A + 4 * M + B) / 6
N = 200_000
TAREAS = ["Preparación de muestra", "Montaje del equipo", "Corrida", "Procesamiento de datos"]
N_TAREAS = 100_000   # valores por tarea en las pruebas de todas las tareas
Z = 5                # tolerancia de la media en errores estándar: una falsa alarma tiene probabilidad < 1 en un millón


@pytest.fixture(scope="module")
def tareas():
    """Las 22 tareas de la hoja «Duraciones» que tienen tres puntos (todas menos las corridas deterministas)."""
    hoja = parametros.cargar_hoja("Duraciones")
    filas = hoja[hoja["Tarea"].isin(TAREAS)].dropna(subset=["a (mín)", "m (moda)", "b (máx)"])
    a, m, b = (filas[c].astype(float).to_numpy() for c in ("a (mín)", "m (moda)", "b (máx)"))
    nombres = (filas["Técnica"] + " · " + filas["Tarea"]).tolist()
    uniforme = filas["Distribución original"].astype(str).str.startswith("U").to_numpy()
    return a, m, b, nombres, uniforme


def media_cerca(x, media, desviacion, z=Z):
    """True en las columnas de x cuya media está a menos de z errores estándar de `media`."""
    error_estandar = np.asarray(desviacion) / np.sqrt(len(x))
    return np.abs(x.mean(axis=0) - media) <= z * error_estandar


def test_media_y_varianza_de_la_pert_clasica():
    assert du.media_betapert(A, M, B) == pytest.approx(MEDIA_PERT)
    assert du.varianza_betapert(A, M, B) == pytest.approx((MEDIA_PERT - A) * (B - MEDIA_PERT) / 7)
    x = du.muestrear_betapert(A, M, B, n=N, rng=2026)
    assert x.mean() == pytest.approx(MEDIA_PERT, rel=0.005)
    assert x.var() == pytest.approx(du.varianza_betapert(A, M, B), rel=0.02)


def test_los_valores_quedan_entre_a_y_b():
    x = du.muestrear_betapert(A, M, B, n=N, rng=1)
    assert x.min() >= A and x.max() <= B


@pytest.mark.parametrize("forma", [4, 1, 0])
def test_todas_las_tareas_quedan_en_su_rango_y_con_su_media(tareas, forma):
    a, m, b, nombres, _ = tareas
    x = du.muestrear_betapert(a, m, b, n=N_TAREAS, rng=2026, forma=forma)
    fuera = [n for n, ok in zip(nombres, (x.min(axis=0) >= a) & (x.max(axis=0) <= b)) if not ok]
    assert not fuera, f"valores fuera de [a, b]: {fuera}"
    media = (a + forma * m + b) / (forma + 2)
    if forma == 4:
        assert media == pytest.approx((a + 4 * m + b) / 6)   # la PERT clásica
    lejos = [n for n, ok in zip(nombres, media_cerca(x, media, np.sqrt(du.varianza_betapert(a, m, b, forma)))) if not ok]
    assert not lejos, f"media lejos de la fórmula: {lejos}"


def test_la_prueba_de_la_media_detecta_un_muestreo_equivocado(tareas):
    # si se muestreara la triangular de la fuente en vez de la PERT clásica, la prueba fallaría en
    # todas las tareas asimétricas; solo pasa en las simétricas, donde las dos medias coinciden
    a, m, b, nombres, uniforme = tareas
    a, m, b = a[~uniforme], m[~uniforme], b[~uniforme]
    x = np.random.default_rng(2026).triangular(a, m, b, size=(N_TAREAS, len(a)))
    pasa = media_cerca(x, du.media_betapert(a, m, b), np.sqrt(du.varianza_betapert(a, m, b)))
    simetrica = np.isclose(m - a, b - m)
    assert (pasa == simetrica).all()


def test_la_moda_es_el_valor_mas_probable():
    alfa, beta = du.parametros_beta(A, M, B)
    grilla = np.linspace(A, B, 9501)
    densidad = stats.beta.pdf((grilla - A) / (B - A), alfa, beta)
    assert grilla[np.argmax(densidad)] == pytest.approx(M, abs=0.01)


def test_sigue_la_beta_reescalada():
    x = du.muestrear_betapert(A, M, B, n=20_000, rng=7)
    alfa, beta = du.parametros_beta(A, M, B)
    assert stats.kstest((x - A) / (B - A), "beta", args=(alfa, beta)).pvalue > 0.01


def test_misma_semilla_mismos_valores():
    primera = du.muestrear_betapert(A, M, B, n=5, rng=2026)
    assert np.array_equal(primera, du.muestrear_betapert(A, M, B, n=5, rng=2026))
    assert not np.array_equal(primera, du.muestrear_betapert(A, M, B, n=5, rng=2027))
    generador = np.random.default_rng(2026)   # también acepta el generador de la corrida
    assert np.array_equal(primera, du.muestrear_betapert(A, M, B, n=5, rng=generador))


def test_muchos_casos_a_la_vez():
    # preparación por solución, montaje y procesamiento de HPLC (Costigliola et al., 2017, Tabla 3)
    a, m, b = np.array([5.0, 25.0, 2.0]), np.array([40.0, 50.0, 5.0]), np.array([100.0, 65.0, 10.0])
    uno = du.muestrear_betapert(a, m, b, rng=3)
    assert uno.shape == (3,) and ((uno >= a) & (uno <= b)).all()
    muchos = du.muestrear_betapert(a, m, b, n=50_000, rng=3)
    assert muchos.shape == (50_000, 3)
    assert muchos.mean(axis=0) == pytest.approx(du.media_betapert(a, m, b), rel=0.01)


def test_un_solo_valor_es_un_numero():
    assert isinstance(du.muestrear_betapert(A, M, B, rng=1), float)


def test_casos_limite():
    assert du.muestrear_betapert(30.0, 30.0, 30.0, n=3, rng=1).tolist() == [30.0] * 3   # duración fija
    for m in (2.0, 56.0):                                                              # moda en un extremo
        x = du.muestrear_betapert(2.0, m, 56.0, n=10_000, rng=1)
        assert x.min() >= 2.0 and x.max() <= 56.0
        assert x.mean() == pytest.approx(du.media_betapert(2.0, m, 56.0), rel=0.02)


@pytest.mark.parametrize("a, m, b", [(10, 5, 20), (10, 30, 20), (np.nan, 5, 20), (5, 10, np.inf)])
def test_tres_puntos_invalidos(a, m, b):
    with pytest.raises(ValueError):
        du.muestrear_betapert(a, m, b, rng=1)


def test_forma_negativa_no_vale():
    with pytest.raises(ValueError):
        du.muestrear_betapert(A, M, B, rng=1, forma=-1)
    with pytest.raises(ValueError):
        du.media_betapert(A, M, B, forma=[4.0, -1.0])


def test_cada_caso_puede_tener_su_forma():
    assert du.media_betapert(A, M, B, forma=[4.0, 1.0, 0.0]) == pytest.approx([MEDIA_PERT, (A + M + B) / 3, (A + B) / 2])


def test_forma_1_da_la_media_de_la_triangular():
    assert du.media_betapert(A, M, B, forma=1) == pytest.approx((A + M + B) / 3)
    x = du.muestrear_betapert(A, M, B, n=N, rng=5, forma=1)
    assert x.mean() == pytest.approx((A + M + B) / 3, rel=0.005)


def test_forma_0_es_la_uniforme():
    # montaje de PSA: U(1, 32) en Costigliola et al. (2017, Tabla 3); la moda no importa
    x = du.muestrear_betapert(1.0, 16.5, 32.0, n=20_000, rng=2026, forma=0)
    assert du.varianza_betapert(1.0, 16.5, 32.0, forma=0) == pytest.approx((32 - 1) ** 2 / 12)
    assert stats.kstest(x, "uniform", args=(1.0, 31.0)).pvalue > 0.01


def test_cuantil():
    assert du.cuantil_betapert([0.0, 1.0], A, M, B).tolist() == pytest.approx([A, B])
    u = np.sort(np.random.default_rng(4).uniform(size=1000))
    assert (np.diff(du.cuantil_betapert(u, A, M, B)) >= 0).all()          # conserva el orden de las uniformes
    x = du.muestrear_betapert(A, M, B, n=N, rng=11)
    assert np.median(x) == pytest.approx(du.cuantil_betapert(0.5, A, M, B), rel=0.01)
    with pytest.raises(ValueError):
        du.cuantil_betapert(1.5, A, M, B)


def test_coincide_con_la_hoja_duraciones():
    hoja = parametros.cargar_hoja("Duraciones")
    filas = hoja[hoja["Tarea"].isin(["Preparación de muestra", "Montaje del equipo", "Corrida", "Procesamiento de datos"])]
    filas = filas.dropna(subset=["a (mín)", "m (moda)", "b (máx)"])
    a, m, b = (filas[c].astype(float).to_numpy() for c in ("a (mín)", "m (moda)", "b (máx)"))
    assert len(filas) == 22   # todas las tareas menos las corridas deterministas de HPLC y GC
    assert du.media_betapert(a, m, b) == pytest.approx(filas["Media PERT (a+4m+b)/6"].astype(float).to_numpy())
    exacta = filas["Desv. BetaPERT exacta (λ = 4)"].astype(float).to_numpy()
    assert np.sqrt(du.varianza_betapert(a, m, b)) == pytest.approx(exacta)


# ---------------------------------------------------------------------------- duración de un ensayo
@pytest.fixture(scope="module")
def soluciones():
    """Rango de soluciones de cada método (S08)."""
    catalogo = parametros.cargar_hoja("Catálogo").set_index("ID")
    return {mt: (int(catalogo.at[mt, "Soluciones mín"]), int(catalogo.at[mt, "Soluciones máx"])) for mt in registro.METODOS}


def test_cada_ensayo_queda_entre_sus_tiempos_optimista_y_pesimista(soluciones):
    for metodo, rango in soluciones.items():
        for n_soluciones in rango:
            t = du.muestrear_ensayo(metodo, n_soluciones, n=20_000, rng=2026)
            assert list(t.columns) == ["duracion_simulada", *registro.TAREAS]
            for tarea, (a, _, b) in registro.tres_puntos_tareas(metodo, n_soluciones).items():
                assert t[tarea].between(a, b).all(), (metodo, n_soluciones, tarea)
            optimista, _, pesimista = registro.tres_puntos(metodo, n_soluciones)
            assert t["duracion_simulada"].between(optimista, pesimista).all(), (metodo, n_soluciones)
            assert np.allclose(t["duracion_simulada"], t[registro.TAREAS].sum(axis=1))   # sin idoneidad, una suma


@pytest.mark.parametrize("forma", [4, 1])
def test_la_media_del_ensayo_sale_de_sus_tres_puntos(soluciones, forma):
    # sin idoneidad y con un analista intermedio la duración es una suma de BetaPERT independientes: su
    # media es la fórmula de la media aplicada a t_optimista, t_mas_probable y t_pesimista; con λ = 4, (a + 4m + b)/6
    lejos = []
    for metodo, (_, n_soluciones) in soluciones.items():
        d = du.muestrear_ensayo(metodo, n_soluciones, n=N_TAREAS, rng=2026, forma=forma)["duracion_simulada"]
        tareas = registro.tres_puntos_tareas(metodo, n_soluciones).values()
        desviacion = np.sqrt(sum(du.varianza_betapert(*p, forma=forma) for p in tareas))
        if not media_cerca(d.to_numpy(), du.media_betapert(*registro.tres_puntos(metodo, n_soluciones), forma=forma), desviacion):
            lejos.append(metodo)
    assert not lejos, f"media lejos de la fórmula: {lejos}"


def test_el_nivel_del_analista_solo_cambia_sus_tareas():
    # con la misma semilla, el factor (S07) multiplica las tareas del analista; la corrida es del equipo
    base = du.muestrear_ensayo("M02", 4, n=1000, rng=7)
    analista = list(registro.TAREAS_ANALISTA)
    optimista, _, pesimista = registro.tres_puntos("M02", 4)
    for nivel in registro.NIVELES:
        f = registro.factor_duracion(nivel)
        otro = du.muestrear_ensayo("M02", 4, n=1000, rng=7, nivel=nivel)
        assert np.allclose(otro[analista], f * base[analista])
        assert np.array_equal(otro["t_corrida"], base["t_corrida"])
        assert otro["duracion_simulada"].between(min(f, 1) * optimista, max(f, 1) * pesimista).all()   # duracion_coherente
    assert registro.factor_duracion("júnior") > registro.factor_duracion("intermedio") == 1 > registro.factor_duracion("sénior")
    with pytest.raises(ValueError):
        du.muestrear_ensayo("M02", 4, nivel="experto")


def test_la_idoneidad_corre_en_paralelo_con_la_preparacion():
    # tres puntos de ensayo para la preparación y la idoneidad del sistema; los definitivos se fijan después
    prep_sistema, idoneidad = (60.0, 90.0, 150.0), (100.0, 150.0, 200.0)
    t = du.muestrear_ensayo("M02", 3, n=20_000, rng=3, prep_sistema=prep_sistema, idoneidad=idoneidad)
    esperado = (t["t_prep_sistema"] + t["t_montaje"] + np.maximum(t["t_idoneidad"], t["t_preparacion"])
                + t["t_corrida"] + t["t_procesamiento"])   # Costigliola et al. (2017, Fig. 3)
    assert np.allclose(t["duracion_simulada"], esperado)
    assert (t["t_idoneidad"] > t["t_preparacion"]).any() and (t["t_idoneidad"] < t["t_preparacion"]).any()
    optimista, _, pesimista = registro.tres_puntos("M02", 3, prep_sistema, idoneidad)
    assert t["duracion_simulada"].between(optimista, pesimista).all()


def test_cada_tarea_puede_tener_su_forma():
    # propuesta de la sección 1.4: la corrida de DSC uniforme (λ = 0), como en la fuente; las demás, PERT clásica
    t = du.muestrear_ensayo("M10", 3, n=20_000, rng=2026, forma={"t_corrida": 0})
    lo, _, hi = registro.corrida_tres_puntos("M10")
    assert stats.kstest(t["t_corrida"], "uniform", args=(lo, hi - lo)).pvalue > 0.01
    a, m, b = registro.tres_puntos_tareas("M10", 3)["t_preparacion"]
    assert stats.kstest((t["t_preparacion"] - a) / (b - a), "beta", args=du.parametros_beta(a, m, b)).pvalue > 0.01
    with pytest.raises(ValueError):
        du.muestrear_ensayo("M10", 3, forma={"corrida": 0})


def test_ensayo_misma_semilla_mismos_valores():
    uno = du.muestrear_ensayo("M07", 3, n=5, rng=2026)
    pd.testing.assert_frame_equal(uno, du.muestrear_ensayo("M07", 3, n=5, rng=2026))
    assert not uno.equals(du.muestrear_ensayo("M07", 3, n=5, rng=2027))


@pytest.mark.parametrize("metodo, n_soluciones", [("M02", 2), ("M02", 5), ("M01", 1.5), ("M12", 4)])
def test_soluciones_fuera_del_rango_de_la_complejidad(metodo, n_soluciones):
    with pytest.raises(ValueError):
        du.muestrear_ensayo(metodo, n_soluciones)
