# RAG dinámico y contexto acotado (etapa 8)

`query_evidence` está disponible durante perfil y búsqueda para el orquestador,
configuración y diagnóstico. Admite `query`, `k` (1–10) y `kinds` opcional
(por ejemplo `["guide"]` o `["arch"]`). El orquestador puede recuperar una guía,
leerla y después ajustar una candidata. Antes de diagnosticar, el host consulta
guías sobre memoria/lote si hay OOM, o sobre curvas, regularización y tasa de
aprendizaje en los demás casos. Esta consulta usa una decisión del presupuesto
global, es idempotente y no añade niveles de delegación. Los especialistas
mantienen una sola llamada LLM; el host realiza la recuperación por su rol.

## Evidencia y límites persistentes

`EvidenceManager` conserva el texto íntegro y los metadatos de los documentos en
SQLite y entrega referencias. `rag_ledger` cobra cada combinación de ID y versión
una sola vez durante toda la ejecución, compartida entre roles. La versión es
SHA-256 del documento y sus metadatos, excluyendo el score. Un cambio de contenido
o procedencia constituye una versión nueva; las referencias anteriores conservan
su contenido. Consultar de nuevo la misma versión no consume más documentos ni
tokens. Las decisiones de herramientas sí conservan su cargo habitual.

El presupuesto RAG cuenta documentos íntegros, no sólo sus extractos enviados al
modelo. Se omiten documentos que no caben y se devuelve `omitted`, `usage` y
`limits`; el agotamiento RAG permite continuar entrenando y diagnosticando con la
evidencia ya disponible. Los tokens RAG se **estiman** como bytes UTF-8 / 3,
redondeados hacia arriba. Este valor no representa consumo real del proveedor.

Cada `rag_retrieval:<retrieval_id>` registra consulta, filtros, rol, action ID,
IDs recuperados, versiones, metadatos, artefactos y consumo. Los originales,
ledger y registro se publican juntos mediante una transacción SQLite bajo el
bloqueo de la ejecución. Repetir una acción confirmada devuelve el mismo resultado.
`automl-agents status` incluye límites/consumo RAG y contabilidad del último
contexto de cada rol.

Los Markdown existentes siguen funcionando. Los documentos nuevos pueden añadir
un encabezado opcional con estos campos escalares:

```markdown
---
source: https://example.org/manual
date: 2026-10-21
scope: train
kind: guide
---
# Memoria y tamaño de lote
Contenido de la guía...
```

Sin `kind` se usa el prefijo del nombre de archivo. Los ámbitos permitidos para
RAG son ausencia de ámbito, `general`, `train` y `validation`; configuración no
recibe documentos con ámbito `validation`. Documentos con ámbito `test` u otro
valor quedan excluidos. Se retiran del índice los documentos eliminados del
directorio; las versiones recuperadas antes permanecen en la ejecución.

## Construcción de contexto

`BudgetedLLM` prepara el prompt en el hilo del host antes de reservar una llamada
y antes de iniciar el worker del proveedor. `ContextManager` conserva en
`context_policy` objetivo, restricciones, rejilla, catálogo permitido y política
de entrada/salida por modelo. Plan, límites y consumo proceden de SQLite en cada
llamada. Estos campos críticos nunca se truncan para hacer espacio. El perfil
del orquestador se proyecta sobre campos de entrenamiento, sin rutas de archivos.

Los resúmenes son deterministas y no consumen llamadas LLM:

- Curvas: cantidad, primer/último valor, mínimo y máximo por serie.
- Trials: cantidad y estados, mejores métricas, resumen por candidata y últimos intentos.
- Plan/historial: cantidades, estados, tareas recientes y referencias a originales.
- Documentos: extractos cortos, metadatos y referencias a textos íntegros.
- Matrices de confusión: dimensiones y referencia al original.

Cada solicitud original queda como artefacto recuperable con los permisos del
rol. El historial íntegro también se conserva como referencia del orquestador.
Si el primer resumen no cabe, se reducen extractos y ventanas recientes. Si aun
así no cabe, se devuelve `context_limit` sin contactar al proveedor: el orquestador
o especialista aplica su respaldo por reglas. No se eliminan restricciones.

La contabilidad incluye instrucciones de sistema, datos y esquemas JSON. Sin
tokenizer local disponible se identifica como `estimated_utf8_bytes` y
`estimated: true`. Puede usarse un cliente con `tokenizer.encode(text)` para un
conteo local identificado como `tokenizer`. La auditoría de cada `llm_calls`
incluye modelo, rol, tokens de entrada, método de conteo, reserva de salida y
referencia del original. El SDK también verifica entrada y salida antes del envío.

## Referencias y permisos

`read_artifact` recibe sólo `artifact_id`, `offset` y `length`. No acepta rutas.
Devuelve `text`, `total_chars`, `next_offset` y versión del contenido; los fragmentos
son segmentos de la representación JSON del original. Se pueden concatenar hasta
`next_offset: null` y reconstruir con `json.loads`. Cada lectura, incluso una
acción repetida, vuelve a comprobar permisos. La longitud se limita por código.

| Alcance del artefacto | Roles autorizados |
|---|---|
| Evidencia general/entrenamiento | Orquestador, configuración, diagnóstico |
| Contexto de configuración | Orquestador, configuración |
| Validación/curvas/contexto de diagnóstico | Orquestador, diagnóstico |
| Historial/contexto del orquestador | Orquestador |
| Test/manifiestos/checkpoints/rutas | Ninguno mediante estas referencias |

Externalizar un fragmento conserva el alcance del rol; no convierte datos de
validación en evidencia legible por configuración. El entrenamiento sigue siendo
secuencial y test continúa fuera de las decisiones de búsqueda.

## Configuración

| Campo / CLI | Valor predeterminado |
|---|---:|
| `rag_max_documents` / `--rag-max-documents` | 32 versiones únicas |
| `rag_max_tokens` / `--rag-max-tokens` | 32000 tokens estimados acumulados |
| `context_read_max_chars` / `--context-read-max-chars` | 4000 caracteres por lectura |
| `llm_context_budget.input_tokens` / `--llm-input-tokens` | 16000 |
| `llm_context_budget.output_tokens` / `--llm-output-reserve` | 2000 |
| `llm_context_budget.window_tokens` / `--llm-context-window` | 24000 |

Son límites de la aplicación, no especificaciones de un proveedor. Entrada y
reserva de salida deben caber juntas en la ventana configurada. La petición de
salida se limita al menor valor entre el solicitado y la reserva. En JSON/Python,
`llm_model_budgets` permite sobreescribir estos tres campos por nombre de modelo:

```python
RunConfig(dataset_dir="data/demo", llm_model_budgets={
    "mi-modelo": {"input_tokens": 8000, "output_tokens": 1500, "window_tokens": 10000}
})
```

Los límites persistidos no se amplían al reconstruir servicios con otra
configuración. `--no-llm` y `--mode rules` mantienen la recuperación local y los
diagnósticos deterministas sin contactar al proveedor.

## Verificación

`tests/test_context_rag.py` verifica límites globales, versiones, filtros,
metadatos, reanudación, permisos, fragmentos recuperables, contextos extensos,
restricciones intactas y presupuestos por modelo. `tests/test_orchestrator.py`
incluye un recorrido con proveedor simulado: fallo → recuperación RAG → lectura
por referencia → entrenamiento → cierre. El smoke del pipeline comprueba el
recorrido sin LLM ni descarga de pesos.
