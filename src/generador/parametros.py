"""Lectura de la hoja de parámetros (datos/parametros/Parametros_generador_tesis.xlsx).

Se usa en todo el generador. Cada valor de la hoja tiene su fuente y página.
Anteproyecto: §1.4 (calibración con literatura) e indicador «% de parámetros con
trazabilidad bibliográfica» del sistema de variables.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

RUTA_PARAMETROS = (Path(__file__).resolve().parents[2]
                   / "datos" / "parametros" / "Parametros_generador_tesis.xlsx")

# En todas las hojas el encabezado de la tabla está en la fila 4 de Excel.
FILA_ENCABEZADO = 3


def cargar_hoja(nombre_hoja: str) -> pd.DataFrame:
    """Devuelve una hoja como DataFrame: «Parámetros», «Duraciones», «Demanda», «Supuestos»..."""
    return pd.read_excel(RUTA_PARAMETROS, sheet_name=nombre_hoja, header=FILA_ENCABEZADO)


@lru_cache(maxsize=None)
def _hoja_cruda(nombre_hoja: str) -> pd.DataFrame:
    return pd.read_excel(RUTA_PARAMETROS, sheet_name=nombre_hoja, header=None)


def valor(nombre_hoja: str, etiqueta: str, columna: int = 1):
    """Valor de la celda que está a la derecha de una etiqueta de la primera columna.

    Sirve para los bloques sueltos de las hojas de cálculo, por ejemplo la fracción de tiempo en
    falla (S02) del bloque D de la hoja «Calibración de turnos».
    """
    hoja = _hoja_cruda(nombre_hoja)
    filas = hoja.index[hoja[0].astype(str).str.strip().eq(etiqueta)]
    if len(filas) != 1:
        raise KeyError(f"La etiqueta {etiqueta!r} aparece {len(filas)} veces en la hoja {nombre_hoja!r}.")
    return hoja.iat[filas[0], columna]


def bloque(nombre_hoja: str, encabezado: str, filas: int) -> pd.DataFrame:
    """Tabla de `filas` filas cuyo encabezado empieza con el texto `encabezado` en la primera columna."""
    hoja = _hoja_cruda(nombre_hoja)
    inicio = hoja.index[hoja[0].astype(str).str.strip().eq(encabezado)]
    if len(inicio) != 1:
        raise KeyError(f"El encabezado {encabezado!r} aparece {len(inicio)} veces en la hoja {nombre_hoja!r}.")
    k = inicio[0]
    tabla = hoja.iloc[k + 1:k + 1 + filas].copy()
    tabla.columns = hoja.iloc[k]
    return tabla.dropna(axis=1, how="all").reset_index(drop=True)


def supuesto(id_supuesto: str):
    """Propuesta inicial de un supuesto de la hoja «Supuestos» (por ejemplo, S01 → 0,02)."""
    hoja = cargar_hoja("Supuestos").set_index("ID")
    return hoja.at[id_supuesto, "Propuesta inicial (supuesto)"]
