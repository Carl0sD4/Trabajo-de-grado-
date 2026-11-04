# Modelo predictivo con datos sintéticos para laboratorios de control de calidad farmacéuticos

Trabajo de grado · Maestría en Ciencia de Datos · Universidad Simón Bolívar (Barranquilla, 2026)

**Autores:** Aldair Rafael Cétares Barrios · Carlos David Campo Urzola
**Tutor:** Dr. Antonio José Bravo Valero

Prototipo basado en aprendizaje automático y datos sintéticos que predice la duración de las
actividades analíticas y recomienda la asignación de analistas y equipos en un laboratorio de
control de calidad farmacéutico de referencia.

| Objetivo | Producto | Dónde |
|---|---|---|
| O1 · Datos sintéticos | Conjunto de datos sintéticos documentado, con la fuente de cada supuesto | `src/generador/`, `datos/`, `docs/` |
| O2 · Estimación de duraciones | Comparación de algoritmos con protocolo y métricas declarados | `src/modelos/` |
| O3 · Asignación de recursos | Prototipo con regla de despacho informada vs. FIFO, aleatoria y regla experta | `src/simulacion/`, `app/` |

## Estructura

```
modelo-predictivo-lab-cc/
├── app/                  Prototipo funcional (Streamlit) · O3
├── datos/
│   ├── parametros/       Hoja de parámetros: 21 hojas, cada valor con su fuente y página
│   ├── sinteticos/       Datasets generados: v0.2 (el v0.1 y las corridas de prueba quedan en borradores/, sin subir)
│   ├── instancias_tlsp/     Instancias del TLSP (TU Wien) para el contraste estructural
│   └── instancias_martins/  Instancias de Martins et al. (2023) para el contraste estructural
├── docs/
│   ├── Diseno_laboratorio_referencia.docx   O1 · Catálogo, demanda, equipos, analistas y turnos
│   ├── Diccionario_de_datos.docx            O1 · Unidad de análisis, los 51 campos y su amarre con los indicadores
│   ├── Generador_de_datos.docx              O1 · Cómo funciona el generador, bloque por bloque, y versiones del dataset
│   ├── Validacion_dataset.docx              O1 · Primera corrida, validación con los 22 indicadores y ajuste (v0.1 y v0.2)
│   ├── Contraste_estructural.docx           O1 · El dataset v0.2 frente a las instancias de Martins et al. (2023) y del TLSP
│   ├── Resultados_O1_borrador.docx          O1 · Borrador de la sección de resultados del O1 para la tesis
│   ├── Plan_trabajo_de_grado.pdf            Plan del 25 de octubre al 30 de noviembre
│   ├── Bitacora_asesorias.docx              Qué se lleva a cada asesoría con Bravo y qué se decide
│   ├── fuentes/          PDF de las cinco fuentes de calibración
│   └── tesis/            Documento de tesis
├── notebooks/
│   ├── 01_forma_duraciones_ensayos.ipynb    Forma λ de la BetaPERT frente a la fuente
│   ├── 02_correlacion_duraciones.ipynb      Correlación entre las tareas e indicador V2
│   ├── 03_validacion_dataset.ipynb          Validación de los datasets v0.1 y v0.2 y figuras del informe
│   └── 04_contraste_estructural.ipynb       Contraste estructural con Martins et al. (2023) y el TLSP
├── resultados/
│   ├── tablas/           Tablas de los cuadernos (CSV)
│   └── figuras/          Figuras de los documentos (PNG)
├── src/
│   ├── generador/        O1 · registro, duraciones, correlación, recursos, disponibilidad, llegadas,
│   │                     perturbaciones, escenarios, generación, validación y contraste estructural
│   │                     (src/generador/README.md)
│   ├── modelos/          O2 · preprocesamiento, entrenamiento y evaluación
│   └── simulacion/       Motor de eventos discretos (O1 y O3), reglas de despacho e indicadores del O3
└── tests/                Pruebas (181)
```

## Cómo empezar

```bash
conda create -n tesis python=3.11 -y
conda activate tesis
pip install -r requirements.txt
pytest                # 181 pruebas; sin las instancias de Martins et al., una se salta
```

En Mac, si `xgboost` da un error de `libomp`, instálalo con `brew install libomp`
o instala el paquete con `conda install -c conda-forge xgboost`.

La cobertura de los módulos de duraciones y de correlación, que sus pruebas ejecutan completos, se mide con:

```bash
pytest --cov=src.generador.duraciones --cov=src.generador.correlacion --cov-branch
```

Los cuadernos se corren en orden: primero `01_forma_duraciones_ensayos.ipynb`, después
`02_correlacion_duraciones.ipynb`, que usa una tabla del 01 y tarda unos 10 minutos, y al final
`03_validacion_dataset.ipynb`, que necesita los datasets v0.1 y v0.2 y tarda unos 17 minutos. El
`04_contraste_estructural.ipynb` necesita el v0.2 y las instancias de referencia (`python datos/descargar_instancias.py`)
y tarda menos de un minuto. Todos guardan sus tablas en `resultados/tablas/` y sus figuras en `resultados/figuras/`,
y con la misma semilla las rehacen idénticas.

## Generar el dataset

```bash
python -m src.generador.generar --version v0.2     # 3 escenarios × 3 semillas × 52 semanas, unos 10 minutos
python -m src.generador.generar --version v0.1     # la primera corrida, para comparar (va a borradores/v0.1/)
```

Cada corrida deja en `datos/sinteticos/<versión>/` sus registros (51 campos), la tabla de eventos de los
equipos y un resumen; las corridas de prueba con otro número de semanas van a `datos/sinteticos/borradores/`. Cómo leerlos, validarlos y qué hace cada módulo está en `src/generador/README.md`.

## Documentos del Objetivo 1

| Documento | Qué tiene | Código y datos |
|---|---|---|
| `docs/Diseno_laboratorio_referencia.docx` | El laboratorio de referencia: el catálogo de 12 métodos, los tipos de muestra y su demanda, los equipos y analistas, y los turnos y ventanas de mantenimiento | Hojas del Excel de la «Catálogo» a la «Calibración de turnos» |
| `docs/Diccionario_de_datos.docx` | Qué es un registro del dataset, sus 51 campos y qué indicador alimenta cada uno | `src/generador/registro.py`, hojas «Diccionario» e «Indicadores» |
| `docs/Generador_de_datos.docx` | Duraciones BetaPERT y factores de correlación (secciones 1 y 2); analistas, equipos y calendarios, disponibilidad cruzada, llegadas y plazos, perturbaciones, motor de simulación, escenarios y semillas (secciones 3 a 7), y versiones del dataset (sección 8) | `src/generador/`, `src/simulacion/entorno.py` y `politicas.py`, cuadernos 01 y 02 |
| `docs/Validacion_dataset.docx` | Primera corrida (v0.1), los 22 indicadores del sistema de variables con su veredicto, el ajuste (v0.2), la capacidad del laboratorio y las decisiones para la asesoría | `src/generador/validacion.py`, cuaderno 03, `resultados/tablas/validacion_*` |
| `docs/Contraste_estructural.docx` | El dataset v0.2 frente a las instancias publicadas de Martins et al. (2023) y del TLSP: trabajos y tareas por trabajo, tipos de recurso, densidad de elegibilidad, rangos de duración y contexto | `src/generador/contraste.py`, cuaderno 04, `resultados/tablas/contraste_estructural*`, hoja «Contraste estructural» |
| `docs/Resultados_O1_borrador.docx` | Borrador de la sección «Objetivo 1: conjunto de datos sintéticos» del capítulo RESULTADOS: laboratorio de referencia y parámetros con sus fuentes, generador, validación con los 22 indicadores y contraste estructural, con 12 tablas y 6 figuras, para pegar en `docs/tesis/Trabajo_de_grado.docx` | Las cifras de los otros cinco documentos |

Los seis están en limpio, sin marcas de revisión ni comentarios, y cada uno dice en su primera página la fecha de su
versión. El dataset v0.2 aplica como propuesta cuatro decisiones de la validación: λ = 1 (columna «Forma λ»), S04 = 0,3, el
alistamiento con la regla de 24 horas (S29) y la regla de despacho del generador (S30); falta confirmarlas con
el tutor. Lo demás que falta decidir está en la sección «Decisiones abiertas» de cada documento y en la sección 6
de la validación; lo más importante es la división por áreas de los analistas (S20), que deja el escenario base
al límite de su capacidad; el escenario alto, en el que con 50 % más de carga la cola crece todo el año, y el
plazo de 4 horas del control en proceso (S06), que en el escenario base se incumple en la mayoría de los casos.

## Datos

No se usan registros reales de ninguna empresa (anteproyecto, §1.4): todo se genera por
simulación calibrada con literatura. Cada parámetro y su fuente (artículo y página) están en
`datos/parametros/Parametros_generador_tesis.xlsx`; su hoja «Léeme» explica qué hay en cada hoja
y qué celdas son decisiones (las amarillas). Los supuestos que no salen de las fuentes están en la
hoja «Supuestos», con sus escenarios de sensibilidad.

El documento de tesis está en `docs/tesis/`. Es un archivo de Word y git no combina cambios de dos personas:
hagan pull antes de abrirlo, push apenas lo cierren y nunca lo editen al mismo tiempo. Lo mismo vale
para los otros documentos de Word y para el Excel de parámetros.

Las instancias de referencia para el contraste estructural (TLSP y Martins et al., 2023) no se suben
al repositorio. Para bajarlas:

```bash
python datos/descargar_instancias.py
```

## Avance

- [ ] O1 · Generador y dataset v1 (meta: asesoría del 4 de noviembre)
  - [x] Extraer parámetros de la literatura (25 oct)
  - [x] Diseñar el laboratorio de referencia (26 oct)
  - [x] Estructura del dataset y diccionario de datos (26 oct)
  - [x] Generador: duraciones BetaPERT (27 oct)
  - [x] Generador: factores de correlación (27 oct)
  - [x] Generador: analistas, equipos y calendarios (28 oct)
  - [x] Generador: disponibilidad cruzada (28 oct)
  - [x] Generador: llegadas de muestras y plazos (29 oct)
  - [x] Generador: perturbaciones (29 oct)
  - [x] Escenarios de carga y semillas, y primera corrida: dataset v0.1 (30 oct)
  - [x] Validar con el sistema de variables y ajustar: dataset v0.2 (31 oct)
  - [ ] Contraste estructural con TLSP y Martins (1 nov): el dataset y Martins, listos; falta medir el TLSP en sus archivos
  - [x] Resultados del O1 en borrador (2 nov): `docs/Resultados_O1_borrador.docx`
  - [ ] Dataset v1 y diccionario final (3 nov): se genera con lo que se decida en la asesoría del 4 de noviembre
- [ ] O2 · Comparación de modelos (meta: asesoría del 18 de noviembre)
- [ ] O3 · Simulador y prototipo (meta: asesoría del 25 de noviembre)
- [ ] Entrega del trabajo de grado (30 de noviembre)
