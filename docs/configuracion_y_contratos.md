# Configuración y contratos (etapa 1)

`RunConfig` conserva su interfaz dataclass y valida al construirse. El pipeline
revalida la configuración antes de crear directorios, clientes o iniciar trabajo.
Las fracciones de validación y test deben ser positivas y sumar menos de uno;
las cantidades de trabajo deben ser enteros positivos. Llamadas LLM y reintentos
admiten cero. Las arquitecturas deben existir en el catálogo y debe quedar al
menos una compatible con el nivel de recursos. `None` permite todo el catálogo;
una lista vacía se considera un error.

`mode="agentic"` conserva el comportamiento anterior de propuestas mediante LLM
con respaldo por reglas. `mode="rules"`, `use_llm=False` o `max_llm_calls=0`
desactivan el cliente. El orquestador adaptativo corresponde a la etapa 6.

La política inicial admite únicamente `accuracy`, dirección `maximize` y desempate
por menor número de parámetros (`smaller_model`). Su `threshold` se refleja en
`target_accuracy` para mantener el controlador existente. Si se proporciona una
política explícita, su umbral tiene prioridad. Sin política, se utiliza el valor
de `target_accuracy`.

Los límites globales se validan y serializan en esta etapa. El máximo de intentos
predeterminado es `max_candidates * rounds * trials_per_round`. El presupuesto
inicial se obtiene con `cfg.initial_budget()`. La reserva, consumo y detención
global por límites se aplican mediante `BudgetManager` desde la etapa 5.

Ejemplo de CLI:

```powershell
automl-agents run --dataset data/formas --mode rules --no-llm --no-pretrained --archs MobileNetV3Small --max-candidates 1 --rounds 1 --trials 2 --epochs 2 --target-accuracy 0.95 --max-training-attempts 2 --max-search-seconds 300
```

`schemas.py` contiene `RunState`, `Plan`, `PlanTask`, `Candidate`, `CandidateRef`,
`TrialRef`, `Decision`, `ToolResult`, `TrialResult`, `Budget`, `MetricPolicy` y
`DomainError`. No admiten objetos ML ni clientes externos. Los contratos Pydantic
se exportan con `to_dict()` o `model_dump_json()` y se reconstruyen con
`model_validate()` o `model_validate_json()`. Las dataclasses `Candidate` y
`RunConfig` ofrecen `to_dict()` y `from_dict()` para reconstrucción desde JSON.

Una candidata conserva `id` entre revisiones y aumenta `version` cuando el
controlador cambia su espacio. Cada estudio pertenece a una candidata y versión.
`trial_id` identifica el trial y `attempt_id` una ejecución concreta; un futuro
reintento conservará el trial y tendrá un intento nuevo. El campo histórico
`trial` sigue siendo el número local de Optuna. Los registros de búsqueda y el
ganador incluyen estos identificadores. Los errores estructurados contienen
`code`, `message` y `recoverable`; el respaldo del generador ya los devuelve en
el campo `error`, conservando también `note` para los reportes existentes.

## Entrenamiento por bloques (etapa 3)

`SearchEngine.run_block(candidates, max_attempts, cancellation=None)` recibe
candidatas completas con su versión y devuelve `list[TrialResult]`. El límite
es total para el bloque, incluyendo fallos y podas; las candidatas se alternan
secuencialmente. Un bloque vacío o con límite cero no inicia intentos.
`run_round` conserva el presupuesto previo por candidata y usa esta operación.
Los estudios se reutilizan en memoria por `(id, version)`; su persistencia y la
reanudación del proceso corresponden a la etapa 4.

Cada intento publica `attempts/<attempt_id>/parameters.json`, `curves.json` y
`result.json`. Las curvas de `loss`, `accuracy`, `val_loss` y `val_accuracy` se
actualizan tras cada época, incluso si luego falla el entrenamiento. El resultado
incluye referencia, estado, semilla, parámetros, épocas, recursos, motivo de parada
y error estructurado cuando corresponde. Los estados son `completed`, `pruned`,
`failed`, `cancelled` e `interrupted`. Una interrupción por KeyboardInterrupt se
conserva; reconciliar cierres forzosos queda para la etapa 4.

Los intentos completados evalúan accuracy, F1 macro y matriz de confusión sobre
validación usando los mejores pesos restaurados por `val_accuracy` (modo máximo).
Optuna, poda, early stopping y selección usan accuracy de validación. Cada modelo
completado tiene un checkpoint `models/<trial_id>_<attempt_id>.keras`; el ganador
y el empaquetado referencian ese archivo. Test no se consulta en el bloque.

`engine.cancel()` o `CancellationToken.cancel()` solicitan cancelación cooperativa:
se comprueba entre trials y al finalizar cada época, con esa granularidad. El
pipeline finaliza sin abrir test cuando la búsqueda se cancela o interrumpe.
OOM, errores de datos/operaciones TensorFlow y errores de archivos se conservan
como fallos recuperables y permiten continuar con el siguiente intento. Los errores
inesperados se registran antes de propagarse. La etapa 5 permite un reintento
acotado por OOM con batch menor dentro del espacio permitido.
Los recursos se liberan en `finally`; el caché se limita a dos pipelines y un
presupuesto agregado de memoria, y se vacía ante OOM.

Las herramientas tipadas, permisos y reservas globales persistentes se describen
en [Herramientas y políticas](herramientas_y_politicas.md).
