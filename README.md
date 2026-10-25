# Modelo predictivo con datos sintéticos para laboratorios de control de calidad farmacéuticos

Trabajo de grado · Maestría en Ciencia de Datos · Universidad Simón Bolívar (Barranquilla, 2026)

**Autores:** Aldair Rafael Cétares Barrios · Carlos David Campo Urzola
**Tutor:** Dr. Antonio José Bravo Valero

Prototipo basado en aprendizaje automático y datos sintéticos que predice la duración de las
actividades analíticas y recomienda la asignación de analistas y equipos en un laboratorio de
control de calidad farmacéutico de referencia.

| Objetivo | Producto | Dónde |
|---|---|---|
| O1 · Datos sintéticos | Conjunto de datos sintéticos documentado, con la fuente de cada supuesto | `src/generador/`, `datos/` |
| O2 · Estimación de duraciones | Comparación de algoritmos con protocolo y métricas declarados | `src/modelos/` |
| O3 · Asignación de recursos | Prototipo con regla de despacho informada vs. FIFO, aleatoria y regla experta | `src/simulacion/`, `app/` |

## Estructura

```
modelo-predictivo-lab-cc/
├── app/                  Prototipo funcional (Streamlit) · O3
├── datos/
│   ├── parametros/       Hoja de parámetros con fuente y página
│   ├── sinteticos/       Datasets generados (v0.1, v0.2, v1...)
│   └── instancias_tlsp/  Instancias públicas de TU Wien para el contraste estructural
├── docs/                 Plan de trabajo y bitácora de asesorías
├── notebooks/            Exploración
├── resultados/           Tablas y figuras para el documento
├── src/
│   ├── generador/        O1 · BetaPERT, correlaciones, recursos, llegadas, perturbaciones, validación
│   ├── modelos/          O2 · preprocesamiento, entrenamiento y evaluación
│   └── simulacion/       O3 · simulador de eventos discretos, políticas e indicadores
└── tests/                Pruebas
```

## Cómo empezar

```bash
conda create -n tesis python=3.11 -y
conda activate tesis
pip install -r requirements.txt
pytest
```

En Mac, si `xgboost` da un error de `libomp`, instálalo con `brew install libomp`
o instala el paquete con `conda install -c conda-forge xgboost`.

## Datos

No se usan registros reales de ninguna empresa (anteproyecto, §1.4): todo se genera por
simulación calibrada con literatura. Cada parámetro y su fuente (artículo y página) están en
`datos/parametros/Parametros_generador_tesis.xlsx`.

El documento de tesis (Word) no va en este repositorio: se edita en la carpeta compartida.

## Avance

- [ ] O1 · Generador y dataset v1 (meta: asesoría del 4 de noviembre)
- [ ] O2 · Comparación de modelos (meta: asesoría del 18 de noviembre)
- [ ] O3 · Simulador y prototipo (meta: asesoría del 25 de noviembre)
- [ ] Entrega del trabajo de grado (30 de noviembre)
