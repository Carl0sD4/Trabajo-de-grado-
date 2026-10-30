"""Unidad de análisis: un registro es una ejecución de un ensayo sobre una muestra.

Diseño: docs/Diccionario_de_datos.docx, sección 1; el ejemplo es su Tabla 4.
"""
import pandas as pd

from src.generador import registro


def ejemplo() -> pd.DataFrame:
    """Una muestra de producto terminado con dos ensayos y un reanálisis, y una de control en
    proceso con un ensayo interrumpido por una falla del equipo y su repetición."""
    filas = [
        ("PT-000123", "PT", "M02", 1, "fuera de especificación", None, None),
        ("PT-000123", "PT", "M07", 1, "cumple", None, None),
        ("PT-000123", "PT", "M02", 2, "cumple", "PT-000123-M02-1", "reanálisis"),
        ("IN-000457", "IN", "M04", 1, "interrumpido", None, None),
        ("IN-000457", "IN", "M04", 2, "cumple", "IN-000457-M04-1", "falla de equipo"),
    ]
    df = pd.DataFrame(filas, columns=["id_muestra", "tipo_muestra", "metodo", "intento", "resultado",
                                      "id_actividad_origen", "causa_repeticion"])
    df.insert(0, "escenario", "base")
    df.insert(1, "semilla", 2026)
    df["id_actividad"] = [registro.id_actividad(*f) for f in zip(df["id_muestra"], df["metodo"], df["intento"])]
    return df


def test_el_ejemplo_cumple_todas_las_reglas():
    assert registro.cumple_unidad(ejemplo()).all()


def test_tres_ensayos_pedidos_y_cinco_registros():
    df = ejemplo()
    assert len(df) == 5
    assert (df["intento"] == 1).sum() == 3   # un registro por ensayo pedido; las repeticiones suman los otros dos
    assert df.groupby("id_muestra")["metodo"].nunique().to_dict() == {"IN-000457": 1, "PT-000123": 2}


def test_clave_repetida():
    df = pd.concat([ejemplo(), ejemplo().iloc[[1]]], ignore_index=True)
    r = registro.chequear_unidad(df)
    assert not r.loc[[1, 5], "clave_unica"].any()
    assert r.drop(index=[1, 5])["clave_unica"].all()


def test_repeticion_mal_enlazada():
    df = ejemplo()
    df.loc[2, "causa_repeticion"] = "falla de equipo"   # su origen salió fuera de especificación
    df.loc[4, "id_actividad_origen"] = None             # repetición sin origen
    r = registro.chequear_unidad(df)
    assert not r.loc[[2, 4], "repeticion_enlazada"].any()
    assert r.loc[[0, 1, 3], "repeticion_enlazada"].all()


def test_a_lo_sumo_un_reanalisis_por_ensayo():
    df = ejemplo()
    df.loc[2, "resultado"] = "fuera de especificación"
    segundo = df.loc[[2]].assign(intento=3, id_actividad="PT-000123-M02-3", id_actividad_origen="PT-000123-M02-2")
    r = registro.chequear_unidad(pd.concat([df, segundo], ignore_index=True))
    assert not r.loc[[0, 2, 5], "reanalisis_por_ensayo"].any()
    assert r.loc[[1, 3, 4], "reanalisis_por_ensayo"].all()


def test_muestra_con_cuatro_ensayos_o_dos_tipos():
    df = ejemplo()
    nuevos = pd.DataFrame([("base", 2026, "PT-000123", "PT", m, 1, "cumple", None, None, f"PT-000123-{m}-1")
                           for m in ("M01", "M06")], columns=df.columns)
    r = registro.chequear_unidad(pd.concat([df, nuevos], ignore_index=True))
    assert not r.loc[[0, 1, 2, 5, 6], "ensayos_por_muestra"].any()
    assert r.loc[[3, 4], "ensayos_por_muestra"].all()
    df.loc[1, "tipo_muestra"] = "GR"
    r = registro.chequear_unidad(df)
    assert not r.loc[[0, 1, 2], "muestra_coherente"].any()
    assert r.loc[[3, 4], "muestra_coherente"].all()


def test_valores_fuera_de_las_listas():
    df = ejemplo()
    df.loc[1, "resultado"] = "aprobado"
    df.loc[3, "metodo"] = "M13"
    r = registro.chequear_unidad(df)
    assert not r.loc[[1, 3], "valores_validos"].any()
    assert r.loc[[0, 2, 4], "valores_validos"].all()
