# Generador de datos sintéticos (Objetivo 1)

Genera el dataset del trabajo de grado: un registro por cada ejecución de un ensayo sobre una muestra en el
laboratorio de control de calidad de referencia, con sus 51 campos (diccionario de datos). Todo sale de la hoja de
parámetros `datos/parametros/Parametros_generador_tesis.xlsx`, donde cada valor tiene su fuente y página o está
declarado como supuesto (hoja «Supuestos»).

## Cómo generar y validar

```bash
python -m src.generador.generar --version v0.2                      # 3 escenarios × 3 semillas × 52 semanas (unos 10 minutos)
python -m src.generador.generar --version v0.2 --escenario base --semilla 2026 --semanas 8   # prueba corta
```

Cada corrida deja tres archivos en `datos/sinteticos/<versión>/`. El v0.1, de prueba, va a `datos/sinteticos/borradores/v0.1/`, y una corrida con otro número de semanas, como la prueba corta, a `datos/sinteticos/borradores/<versión>_<semanas>_semanas/`, para no reemplazar los archivos del dataset:

| Archivo | Qué tiene |
|---|---|
| `registros_<escenario>_<semilla>.csv.gz` | Los registros del dataset, con los 51 campos del diccionario |
| `eventos_<escenario>_<semilla>.csv.gz` | Ventanas de mantenimiento y fallas de cada equipo (diccionario, Tabla 17) |
| `resumen_<escenario>_<semilla>.json` | Configuración, conteos y lo que quedó en cola o en curso al cierre |

Para leer una versión y calcular los 22 indicadores del sistema de variables:

```python
from src.generador import generar, validacion
registros, eventos, resumenes = generar.cargar_version("v0.2")
v = validacion.validar(registros, eventos, "v0.2")
v["resumen"]                  # una fila por indicador: valor, referencia y veredicto
validacion.guardar(v, "v0.2") # resultados/tablas/validacion_v0.2_resumen.csv y validacion_v0.2.xlsx
```

El cuaderno `notebooks/03_validacion_dataset.ipynb` hace eso para el v0.1 y el v0.2 y dibuja las figuras del informe
(`docs/Validacion_dataset.docx`).

## Módulos

En el orden en que intervienen en una corrida:

| Módulo | Qué hace | Documento |
|---|---|---|
| `parametros.py` | Lee las hojas del Excel de parámetros | Generador, sección 1 |
| `registro.py` | Define los 51 campos y verifica las 17 reglas de un registro (`chequear_registro`) | Diccionario, secciones 1 y 2 |
| `recursos.py` | Analistas, equipos, matrices de calificación y compatibilidad, calendario de turnos y ventanas de mantenimiento (`Laboratorio`) | Generador, sección 3 |
| `disponibilidad.py` | Combinaciones analista–equipo factibles, cobertura en el tiempo e indicadores C1 a C4 | Generador, sección 4 |
| `llegadas.py` | Llegada de las muestras, ensayos de cada muestra, plazos y urgentes | Generador, sección 5 |
| `perturbaciones.py` | Urgentes (S03), probabilidad de reanálisis (S01) y fallas de equipos (S02) | Generador, sección 6 |
| `duraciones.py` | Duraciones BetaPERT de las tareas, forma λ, alistamiento (S29) | Generador, sección 1 |
| `correlacion.py` | Correlación entre las tareas de una ejecución (cópula gaussiana) e indicador V2 | Generador, sección 2 |
| `escenarios.py` | Escenarios de carga, semillas, réplicas y configuración de cada versión | Generador, sección 7 |
| `generar.py` | Arma una corrida, la verifica y la guarda; lee una versión | Generador, sección 8 |
| `validacion.py` | Los 22 indicadores del sistema de variables con su veredicto | Validación del dataset |
| `contraste.py` | Contraste estructural con las instancias de Martins et al. (2023) y del TLSP: trabajos y tareas, recursos, densidad de elegibilidad y duraciones | Contraste estructural |

El motor de eventos discretos que ejecuta los ensayos con sus recursos está en `src/simulacion/entorno.py`, y la regla
de despacho del generador (S30), en `src/simulacion/politicas.py`. Es el mismo motor que usará el O3 para comparar
reglas de despacho con las mismas muestras.

## Reproducibilidad

Una corrida es un escenario con una semilla. La semilla alimenta seis generadores de números aleatorios
independientes (llegadas, urgentes, ensayos, duraciones, resultados y fallas), así que con la misma semilla salen los
mismos registros. Las uniformes de las tareas de cada ensayo y la que decide si su resultado sale fuera de
especificación se sacan antes de empezar, en el orden de los ensayos, y cada repetición usa un generador propio,
derivado de la semilla, del ensayo y del intento; así, al cambiar la regla de despacho cambian las asignaciones,
pero no las muestras, la corrida de cada ensayo, sus resultados ni las fallas. Los tiempos se guardan con seis
decimales (minutos). Los tres escenarios usan las mismas
semillas (2026, 2027 y 2028): números aleatorios comunes.

## Versiones del dataset

`escenarios.VERSIONES` dice con qué configuración se genera cada versión:

| Versión | Forma λ | ρ (S04) | Alistamiento | Estado |
|---|---|---|---|---|
| v0.1 | 4 (PERT clásica) | 0,5 | Sin preparación ni idoneidad del sistema | Primera corrida completa (borrador) |
| v0.2 | Columna «Forma λ» de la hoja Duraciones (1; 0 en las tareas uniformes) | 0,3 | Regla de 24 horas (S29), medida al empezar la corrida con las duraciones medias | Validada; decisiones por confirmar con el tutor |

Para una versión nueva, se agrega su configuración a `VERSIONES` y se genera con `--version`.

## Pruebas

```bash
pytest tests/test_calendarios.py tests/test_llegadas.py tests/test_perturbaciones.py tests/test_simulacion.py tests/test_validacion.py tests/test_contraste.py
```

Comprueban los calendarios y las ventanas, las tasas de llegada y la mezcla de ensayos, las fallas y los urgentes, que
una corrida corta cumpla las 17 reglas, que sea reproducible, que otra regla de despacho deje las mismas muestras,
corridas, resultados y fallas, que respete los turnos, la regla de 24 horas de la idoneidad y la ocupación de cada
equipo, que despache el lunes a las 0:00, que una corrida de prueba no reemplace el dataset, y que la validación
detecte valores fuera de rango y una forma λ equivocada. Las del contraste leen una instancia mínima con el formato
de Martins et al. (2023) y corridas cortas, y revisan las 54 instancias publicadas cuando están bajadas.
