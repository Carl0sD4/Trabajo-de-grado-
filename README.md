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
│   ├── sinteticos/       Datasets generados (v0.1, v0.2, v1...)
│   ├── instancias_tlsp/     Instancias del TLSP (TU Wien) para el contraste estructural
│   └── instancias_martins/  Instancias de Martins et al. (2023) para el contraste estructural
├── docs/
│   ├── Diseno_laboratorio_referencia.docx   O1 · Catálogo, demanda, equipos, analistas y turnos
│   ├── Diccionario_de_datos.docx            O1 · Unidad de análisis, los 51 campos y su amarre con los indicadores
│   ├── Generador_de_datos.docx              O1 · Duraciones BetaPERT y factores de correlación
│   ├── Plan_trabajo_de_grado.pdf            Plan del 25 de octubre al 30 de noviembre
│   ├── Bitacora_asesorias.docx              Qué se lleva a cada asesoría con Bravo y qué se decide
│   ├── fuentes/          PDF de las cinco fuentes de calibración
│   └── tesis/            Documento de tesis
├── notebooks/
│   ├── 01_forma_duraciones_ensayos.ipynb    Forma λ de la BetaPERT frente a la fuente
│   └── 02_correlacion_duraciones.ipynb      Correlación entre las tareas e indicador V2
├── resultados/
│   ├── tablas/           Tablas de los cuadernos (CSV)
│   └── figuras/          Figuras de los documentos (PNG)
├── src/
│   ├── generador/        O1 · registro, duraciones y correlación (listos); recursos, disponibilidad,
│   │                     llegadas, perturbaciones, escenarios, generación y validación (por hacer)
│   ├── modelos/          O2 · preprocesamiento, entrenamiento y evaluación
│   └── simulacion/       O3 · simulador de eventos discretos, políticas e indicadores
└── tests/                Pruebas (114)
```

## Cómo empezar

```bash
conda create -n tesis python=3.11 -y
conda activate tesis
pip install -r requirements.txt
pytest                # 114 pruebas
```

En Mac, si `xgboost` da un error de `libomp`, instálalo con `brew install libomp`
o instala el paquete con `conda install -c conda-forge xgboost`.

La cobertura de los módulos del generador que ya están listos se mide con:

```bash
pytest --cov=src.generador.duraciones --cov=src.generador.correlacion --cov-branch
```

Los cuadernos se corren en orden: primero `01_forma_duraciones_ensayos.ipynb` y después
`02_correlacion_duraciones.ipynb`, que usa una tabla del 01 y tarda unos 10 minutos. Los dos guardan
sus tablas en `resultados/tablas/` y sus figuras en `resultados/figuras/`, y con la misma semilla
las rehacen idénticas.

## Documentos del Objetivo 1

| Documento | Qué tiene | Código y datos |
|---|---|---|
| `docs/Diseno_laboratorio_referencia.docx` | El laboratorio de referencia: el catálogo de 12 métodos, los tipos de muestra y su demanda, los equipos y analistas, y los turnos y ventanas de mantenimiento | Hojas del Excel de la «Catálogo» a la «Calibración de turnos» |
| `docs/Diccionario_de_datos.docx` | Qué es un registro del dataset, sus 51 campos y qué indicador alimenta cada uno | `src/generador/registro.py`, hojas «Diccionario» e «Indicadores» |
| `docs/Generador_de_datos.docx` | Duraciones BetaPERT (sección 1) y factores de correlación entre las tareas (sección 2) | `src/generador/duraciones.py`, `src/generador/correlacion.py`, hojas «Duraciones» y «Correlaciones», cuadernos 01 y 02 |

Los tres están en limpio desde el 29 de octubre, con las correcciones de la revisión y el bloque de
factores de correlación ya aceptados. Lo que falta decidir está en la sección «Decisiones abiertas»
de cada uno; lo más importante es la forma λ de la BetaPERT junto con la correlación ρ del supuesto
S04, y la tolerancia del indicador V2.

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
  - [ ] Generador: analistas, equipos y calendarios (28 oct)
  - [ ] Generador: disponibilidad cruzada (28 oct)
  - [ ] Generador: llegadas de muestras y plazos (29 oct)
  - [ ] Generador: perturbaciones (29 oct)
  - [ ] Escenarios de carga y semillas, y primera corrida: dataset v0.1 (30 oct)
  - [ ] Validar con el sistema de variables y ajustar: dataset v0.2 (31 oct)
  - [ ] Contraste estructural con TLSP y Martins (1 nov)
  - [ ] Dataset v1 y diccionario final (3 nov)
- [ ] O2 · Comparación de modelos (meta: asesoría del 18 de noviembre)
- [ ] O3 · Simulador y prototipo (meta: asesoría del 25 de noviembre)
- [ ] Entrega del trabajo de grado (30 de noviembre)
