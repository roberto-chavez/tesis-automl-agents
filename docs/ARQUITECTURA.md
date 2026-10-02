# Arquitectura y trazabilidad con la tesis

```
ZIP/carpeta ─► [perfil] ─► [rag] ─► [candidatas] ─► [entrenar] ◄─┐
                                                        │        │ (rondas)
                                                   [controlar] ──┘
                                                        │ stop
                     [empaquetar] ◄─ [reportar] ◄─ [benchmark] ◄─ [evaluar]
```
Los nodos son un grafo de estados LangGraph (`pipeline.py`). Los objetos pesados (modelos, datasets,
Chroma) viven en el contexto `Pipeline`; el estado del grafo solo lleva la ronda y la bandera de parada.

## Mapeo resultado → código → evidencia

| Resultado | Módulo | Evidencia generada | Cómo se valida |
|---|---|---|---|
| R1.1 Base RAG | `kb.py`, `knowledge/*.md` | Colección Chroma | `automl-agents kb-eval` (precisión/exhaustividad@k), `tests/test_kb_catalog.py` |
| R1.2 Candidatas acotadas | `candidates.py`, `catalog.py` | `reports/01_criterios_acotacion.json` | % de reducción del espacio (`space.reduction_pct`) |
| R1.3 Reporte de criterios | `reporting.write_criteria_report` | `reports/01_criterios_acotacion.md` | Test e2e: se genera en cada ejecución |
| R2.1 Agente de entrenamiento | `trainer.py`, `search.py` | `events.jsonl`, `models/` | Trials sin intervención / trials totales |
| R2.2 Agente de evaluación | `evaluation.py` | `reports/03_evaluacion.*` | Métricas para todos los modelos guardados |
| R2.3 Iteración automática | `controller.py` | `reports/02_historial_iteraciones.json` | Nº de decisiones/rondas automáticas |
| R3.1 Interfaz de carga | `api.py`, `web/index.html` | Formulario + progreso | SUS con usuarios (pendiente: pruebas con personas) |
| R3.2 Reporte comprensible | `reporting.write_final_report` | `reports/06_reporte_final.md` (model card) | Encuesta de comprensión (pendiente) |
| R3.3 Artefacto desplegable | `packaging.py` | `artifact/` (FastAPI + Dockerfile) | `artifact/smoke_test.py`; `docker build` (pendiente) |
| R4.1 Descarte temprano | `search.py` (Hyperband), `controller.py` | `trials_pruned`, `saving_from_pruning_pct` | `reports/05_benchmark.md` |
| R4.2 Registro de recursos | `resources.py` | `reports/04_consumo_recursos.csv` | Completitud por configuración |
| R4.3 Comparación con línea base | `benchmark.py` | `reports/05_benchmark.md` | Líneas base estimadas; medir una real |

## Decisiones de diseño
- **Espacio discretizado** (`catalog.py`): 11 arquitecturas × 90 combinaciones = 990 configuraciones; el mismo
  espacio se usa para medir la reducción (R1.2), muestrear con Optuna y estimar líneas base.
- **El LLM no manda**: sus candidatas se validan contra catálogo/rejilla y el sistema cae a reglas ante cualquier
  fallo (mitiga alucinaciones y dependencia de red, cf. sección 3.6.2 de la tesis).
- **Selección por validación, reporte en prueba**: el ganador se elige con `val_acc`; el conjunto de prueba solo
  se reporta (evita sobreajustar la selección, cf. Wang et al., 2026).
- **Una nueva instancia de Optuna por ronda**: permite acotar lr/dropout alrededor del mejor trial ("ajustar").
- **Preprocesamiento fuera del modelo**: se guarda `preprocess_module` en `metadata.json` y la API lo reaplica,
  evitando capas `Lambda` no portables entre versiones de Python.
- **Riesgo y datos personales** (`risk.py`): señal heurística ligada a Ley 29733 / Ley 31814 (secciones 2.2 y 2.3.3).

## Próximos pasos sugeridos
1. Validar con pesos ImageNet en GPU y un dataset industrial (defectos / residuos).
2. Medir una línea base real (búsqueda exhaustiva reducida) para R4.3.
3. Enriquecer el KB (campos por modelo, latencia/VRAM medidos) y agregar el agente recomendador de métricas.
4. Fine-tuning parcial del backbone; API web con autenticación y cola de trabajos.
5. Pruebas de usabilidad (SUS) y de comprensión del reporte con usuarios sin experiencia en ML.
