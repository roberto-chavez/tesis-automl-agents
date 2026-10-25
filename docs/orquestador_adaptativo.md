# Orquestador adaptativo (etapa 6)

El pipeline usa `Orchestrator` sobre `DomainTools`. El perfilado y la recuperación
inicial preparan el contexto; la búsqueda ya no depende de ejecutar rondas fijas.
`rounds` conserva su función de calcular el límite inicial de intentos cuando no
se especifica `max_training_attempts`. `trials_per_round` fija el tamaño máximo de
cada bloque secuencial, tras el cual se vuelve a observar.

## Decisiones y plan

`Policy.decide(observation)` devuelve un `Decision` validado. `RulesPolicy` explora
las candidatas menos observadas, inspecciona fallos/poda, descarta candidatas con
fallos o fuera del margen del líder y acota la rejilla alrededor de resultados
completados. Las revisiones incrementan la versión y conservan el historial.
`LLMPolicy` utiliza el proveedor Gemini existente y la misma interfaz. Puede
consultar evidencia, leer fragmentos seguros, recuperar validación, diagnosticar,
entrenar, revisar, descartar o solicitar cierre. Los [especialistas de la etapa 7](especialistas.md)
atienden propuestas y diagnósticos con contratos y contextos separados. El
orquestador revalida sus recomendaciones antes de revisar o descartar candidatas.

El ciclo observa SQLite, decide, valida, ejecuta la herramienta, comprueba la
salida y persiste. El contexto incluye perfil de entrenamiento, candidatas,
validación, plan, presupuesto y respuestas recientes de herramientas. No incluye
checkpoints, rutas de datos ni resultados de test. Las herramientas conservan la
autoridad para rechazar versiones, fuentes, parámetros, roles y fases inválidos.

El plan crece con tareas concretas, dependencias, estado y referencia al resultado.
Antes de ejecutar se guarda la tarea en curso y su decisión pendiente. Sólo pasa
a `completed` después de validar la salida; un bloque necesita resultados con
intentos realmente registrados. Una herramienta fallida queda como tarea fallida.
Los resultados completos se conservan por ID, además del historial de decisiones,
justificación breve, fuentes y contadores LLM. `status` expone el plan y selección.

## Recuperación y terminación

Las decisiones reciben IDs generados por el host, usados como claves idempotentes.
Al reiniciar, se ejecuta la intención pendiente con el mismo ID: las herramientas
confirmadas devuelven su resultado y los bloques parciales recuperan el journal.
Una cancelación/interrupción pausa la búsqueda sin abrir test y conserva la tarea
pendiente para `resume`, como en la etapa 4. El tiempo de pared sigue consumiéndose.

El host impone cierre por objetivo, presupuesto, estancamiento o falta de progreso.
La política también puede terminar al agotar candidatas. La ausencia de checkpoint
válido produce `no_model` con causa de parada, sin evaluar ni empaquetar.

Se detectan acciones idénticas sin nuevos intentos/descartes. Además, un contador
acota secuencias de consultas o revisiones distintas sin observaciones nuevas; los
bloques sin mejora tienen su propio límite de estancamiento. Estos contadores y
huellas sobreviven a la reanudación.

`close_search` congela atómicamente la selección con accuracy de validación y
menor tamaño del catálogo como desempate, junto con el cierre y el reloj. Repetir
el cierre no cambia ganador ni motivo. El pipeline evalúa sólo ese checkpoint en
test. Después del cierre no se admiten nuevas operaciones de búsqueda.

## LLM y parámetros

El cliente reserva cada llamada antes de contactar al proveedor. El SDK tiene
un timeout HTTP y un único intento interno; el wrapper limita además la espera.
Ambos usan `llm_timeout_seconds`, con 240 segundos por defecto para permitir
respuestas de Gemini de dos a tres minutos. La configuración también alcanza
al cliente inyectado cuando se envuelve con `BudgetedLLM`.
Un worker que excede el plazo no accede a SQLite y su respuesta tardía se ignora.
No se puede garantizar la cancelación remota ni recuperar su coste. Las llamadas
registran duración, error y metadatos de uso cuando el proveedor los entrega.

Hay como máximo un reintento de proveedor/formato por decisión, sujeto al
presupuesto global `max_retries`. Un timeout cae directamente a reglas. Tras
agotar esa recuperación, el respaldo por reglas queda persistido para la búsqueda.
Los rechazos de herramientas también activan reglas; no autorizan efectos inválidos.

| Configuración / CLI | Predeterminado | Significado |
|---|---:|---|
| `llm_timeout_seconds` / `--llm-timeout-seconds` | 240 | Espera máxima por llamada, compartida con HTTP |
| `max_stagnant_blocks` / `--max-stagnant-blocks` | 3 | Bloques consecutivos sin mejora |
| `max_no_progress_actions` / `--max-no-progress-actions` | 6 | Acciones sin nuevas observaciones |

`--mode rules` y `--no-llm` mantienen ejecución totalmente local. Los límites
existentes de intentos, tiempo, decisiones, llamadas y reintentos siguen vigentes.
El cierre siempre es posible aunque se agote el presupuesto.

## Verificación

`tests/test_orchestrator.py` usa SQLite y herramientas reales con entrenamiento
simulado: éxito, fallo, diagnóstico, descarte, revisiones, estancamiento,
repeticiones, presupuesto, congelación, errores/timeout LLM y caídas después de
confirmar herramientas. Un cliente simulado produce distintas trayectorias según
las observaciones. `tests/test_persistence.py` verifica reanudación con Optuna y
LangGraph; `tests/test_pipeline_e2e.py` ejecuta entrenamiento real sin descargas.

La prueba `test_live_provider_smoke` es optativa: requiere autorización para usar
las credenciales configuradas y `AUTOML_TEST_LIVE_LLM=1`. Envía observaciones
sintéticas y comprueba una respuesta real y sus metadatos de uso; no considera el
respaldo por reglas como una integración exitosa con el proveedor.

### Resultado de verificación de esta implementación

La validación inicial aprobó 168 pruebas locales, incluyendo entrenamiento real
sin LLM ni descarga de pesos. La integración remota inicial recibió errores
404/504 y sólo pudo comprobar la continuación por reglas.

Después de unificar los timeouts a 240 segundos, la prueba remota autorizada con
`gemini-3.8-flash` **aprobó**: una llamada real respondió en 28.61 segundos,
solicitó `train_block` y la ejecución cerró con `close_search` por objetivo
alcanzado. Se verificaron la decisión, la ejecución de herramientas, la selección
persistida y los metadatos de uso: 2278 tokens de entrada, 144 de salida y
599 de razonamiento (3021 totales). No hubo errores ni respaldo por reglas.

Esta prueba usa entrenamiento simulado para aislar el proveedor; el entrenamiento
real sin LLM se comprueba por separado. El informe reproducible de esta ejecución
se guardó en `runs/validation/gemini-stage6.json`. La prueba optativa exige ahora
al menos un bloque de entrenamiento solicitado por Gemini, una selección al cerrar
y resultados verificados de todas las herramientas. Para guardar un informe de
otra ejecución, definir `AUTOML_LIVE_LLM_REPORT` con la ruta de salida.

`tests/test_llm_timeout.py` comprueba sin red que el timeout configurado alcanza
tanto al wrapper como a las peticiones del SDK, incluso con un valor distinto del
predeterminado. Las ejecuciones antiguas conservan el timeout guardado en su
`config.json`; el nuevo valor predeterminado se aplica a ejecuciones nuevas.
