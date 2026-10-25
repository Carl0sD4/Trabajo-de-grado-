"""Lectura de la hoja de parámetros (datos/parametros/Parametros_generador_tesis.xlsx).

Se usa en todo el generador. Cada valor de la hoja tiene su fuente y página.
Anteproyecto: §1.4 (calibración con literatura) e indicador «% de parámetros con
trazabilidad bibliográfica» del sistema de variables.
"""
from pathlib import Path

import pandas as pd

RUTA_PARAMETROS = (Path(__file__).resolve().parents[2]
                   / "datos" / "parametros" / "Parametros_generador_tesis.xlsx")

# En todas las hojas el encabezado de la tabla está en la fila 4 de Excel.
FILA_ENCABEZADO = 3


def cargar_hoja(nombre_hoja: str) -> pd.DataFrame:
    """Devuelve una hoja como DataFrame: «Parámetros», «Duraciones», «Demanda», «Supuestos»..."""
    return pd.read_excel(RUTA_PARAMETROS, sheet_name=nombre_hoja, header=FILA_ENCABEZADO)
