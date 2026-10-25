# Hallazgos y propuestas a partir de los documentos de papers/

Fecha de revisión: 19 de octubre de 2026. Estado: análisis documental y propuestas; los beneficios para este proyecto todavía deben medirse.

Se revisaron los **cuatro archivos** presentes en `papers/`: tres PDF y un Markdown. Se extrajo el texto de los PDF y se contrastaron visualmente la tabla principal de resultados de ReAct y las figuras de planificación y compresión de los otros dos PDF. Las referencias de páginas corresponden a la numeración física del PDF, empezando en 1; el Markdown se referencia por sección.

La propuesta conjunta es incorporar un ciclo de decisión basado en observaciones, un plan persistente, gestión de contexto recuperable y herramientas especializadas. Su diseño se desarrolla en [la arquitectura](01_arquitectura_orquestador_y_flujo.md); las dependencias y pruebas están en [la hoja de ruta](03_hoja_de_ruta_y_pasos.md).

## 1. Inventario y calidad de la evidencia

| ID | Archivo y referencia | Naturaleza | Qué permite sostener |
|---|---|---|---|
| P1 | [2210.03629v3.pdf](../papers/2210.03629v3.pdf): Yao et al., *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR 2023; arXiv v3, 10-03-2023; 33 páginas | Paper académico con experimentos y ablaciones | Evidencia del patrón de razonamiento/acción en los benchmarks estudiados |
| P2 | [Introduction to LangChain Deep Agents and the Shift to Agent 2.0.pdf](<../papers/Introduction to LangChain Deep Agents and the Shift to Agent 2.0.pdf>): Rick Hightower, 15-03-2026; 22 páginas | Artículo técnico/divulgativo de Medium | Patrones de planificación, archivos, delegación y contexto; no prueba cuantitativa de superioridad en AutoML |
| P3 | [Context Management for Deep Agents.pdf](<../papers/Context Management for Deep Agents.pdf>): Chester Curme y Mason Daugherty, LangChain, 28-01-2026; 6 páginas | Artículo técnico del equipo desarrollador con trazas y evaluaciones dirigidas | Mecanismos concretos de compresión y cómo verificar recuperabilidad |
| P4 | [building-deep-agents-with-langchain.md](../papers/building-deep-agents-with-langchain.md): *Building Deep Agents with LangChain: A Complete Hands-On Tutorial* | Tutorial; la copia local no identifica inequívocamente autor, fecha o URL original | Ejemplo práctico de herramientas, prompts e integración; no experimento controlado |

No se debe presentar el conjunto como cuatro papers arbitrados. P2, P3 y P4 sirven como literatura técnica de apoyo a la implementación. “Agent 2.0” es el marco narrativo de P2, no una categoría científica establecida por estos documentos. Los trabajos mencionados en el estado del arte de la tesis y ausentes de `papers/` son contexto secundario, no fuentes primarias revisadas en este análisis.

## 2. P1 — ReAct: cerrar el ciclo entre decisión, acción y observación

### 2.1 Hallazgos de la fuente

ReAct intercala decisiones expresadas en lenguaje con acciones sobre un entorno y observaciones devueltas por ese entorno. El resultado de una acción puede cambiar el plan o motivar una consulta adicional. El diseño incluye acciones específicas del dominio y ejemplos de trayectorias; no presupone un sistema multiagente. Véanse §2, páginas 3–4, y Figura 1, página 2.

La evidencia no muestra una superioridad universal. En la Tabla 1, página 5, ReAct obtiene 27,4 de exact match en HotpotQA frente a 29,4 de CoT; en FEVER obtiene 60,9 de accuracy frente a 56,3. En WebShop, Tabla 4, página 8, la tasa de éxito es 40,0 para ReAct y 30,1 para Act. Son métricas y entornos distintos, no ganancias esperables de accuracy de un clasificador de imágenes.

El análisis de errores (§3.3, página 6) identifica repeticiones de acciones y búsquedas que no aportan información. La combinación con herramientas reduce algunos errores, pero también introduce dependencia de la recuperación y problemas de control del bucle. Los apéndices documentan prompts y trayectorias de éxito/fallo; sirven como referencia para construir casos de evaluación reproducibles.

### 2.2 Brecha específica del proyecto

Actualmente `Pipeline.n_retrieve()` y `n_propose()` se ejecutan una vez. Después, `controller.decide()` solo utiliza puntuaciones, margen de descarte, ronda y parámetros del mejor trial. No puede pedir más evidencia ni razonar sobre un error observado. `trainer.train_trial()` devuelve máximo de exactitud, épocas y poda, pero no guarda curvas completas para diagnosticar sobreajuste.

### 2.3 Mejora propuesta P1: controlador adaptativo con acciones verificables

Implementar una política de orquestación que reciba un resumen estructurado del estado y elija una acción entre recuperar evidencia, revisar candidatas, entrenar otro bloque, solicitar diagnóstico o cerrar búsqueda. Cada acción produce una observación real y verificable antes de decidir de nuevo. Mantener `controller.decide()` como baseline y respaldo.

Ejemplo hipotético, sin atribuirlo a una corrida existente:

1. El entrenamiento informa OOM con batch 32; registra intento y recursos consumidos.
2. El orquestador consulta la guía de tamaño de lote y el espacio permitido.
3. Propone una nueva versión de la candidata con batch 16.
4. El runtime valida el cambio y el presupuesto; ejecuta un reintento limitado.
5. El diagnóstico utiliza el resultado medido; no supone que reducir batch siempre resuelva el error.

**Cambios:** `pipeline.py`, `controller.py`, `search.py`, `trainer.py`, `events.py`; nuevos contratos y políticas del orquestador. Cada decisión conserva acción, resumen de justificación, fuentes, resultado y motivo de parada. No es necesario almacenar razonamiento interno extenso ni entrenar un LLM.

**Validación:** comparar la política actual y la propuesta con el mismo catálogo, splits, presupuesto y proveedor configurado. Incluir casos de estancamiento, evidencia insuficiente, OOM simulado y respuesta inválida. Medir porcentaje de ejecuciones completadas, recuperaciones, acciones inválidas, repeticiones, tiempo total, tokens y desempeño final. Separar pruebas simuladas de resultados de entrenamiento real.

**Riesgo y límite:** más consultas pueden aumentar costo o sobreajuste a validación. Usar topes de pasos, consultas y reintentos; test inaccesible durante la búsqueda. Hipótesis: mejor recuperación y asignación del presupuesto; no se garantiza mejor accuracy.

**Prioridad:** P0, núcleo de R2.3 y apoyo a R1.3/R4.1.

## 3. P2 — Deep Agents y “Agent 2.0”: hacer explícito el plan

### 3.1 Hallazgos de la fuente

El artículo describe cuatro pilares alrededor del bucle de herramientas: planificación explícita, archivos como memoria, delegación con contexto aislado e instrucciones de comportamiento. Las páginas 10–11 explican la lista de tareas como recordatorio del objetivo; las páginas 11–13 tratan almacenamiento; las páginas 13–14, subagentes; las páginas 15–16, instrucciones.

La idea transferible es que un plan y los artefactos permiten mantener continuidad en tareas largas. El artículo no presenta un ensayo controlado sobre clasificadores, ahorro de GPU o precisión de selección de modelos. Sus afirmaciones sobre fiabilidad deben tratarse como motivación técnica.

Hay que distinguir una lista escrita por el modelo de una prueba de avance. También hay que configurar realmente el backend: un archivo virtual en memoria no demuestra persistencia tras reiniciar. Los ejemplos de backends y versiones del artículo no deben copiarse como API vigente sin verificación.

### 3.2 Brecha específica del proyecto

El flujo conoce una ronda y una bandera de parada, pero no un plan durable. Si termina el proceso, se pierden las candidatas activas, decisiones y registros en memoria. `events.jsonl` narra eventos, pero no es suficiente para reconstruir de forma inequívoca todos los efectos de entrenamiento.

### 3.3 Mejora propuesta P2: plan versionado y delegación de un nivel

Crear `ExperimentPlan` con tareas, dependencias, estado y criterio de aceptación. El orquestador actualiza el plan tras verificar observaciones. Las tareas típicas son perfilar, recuperar guías, proponer candidatas, ejecutar búsquedas acotadas, diagnosticar validación, congelar selección y verificar la entrega.

Cada tarea tiene `task_id`, `status`, `depends_on`, `assigned_role`, `artifact_refs` y `acceptance_check`. El estado completo se respalda con checkpoint y manifiestos. El especialista de configuración recibe perfil, catálogo y RAG; el de diagnóstico recibe curvas, métricas y recursos. Devuelven contratos pequeños al orquestador. No se habilitan agentes recursivos ni entrenamiento concurrente sobre la misma GPU.

**Cambios:** ampliar estado en `pipeline.py`; introducir `state_store.py`, `schemas.py` y roles especializados; proyectar el plan en la API existente. Los agentes no comparten una conversación completa ni pueden editar libremente el estado canónico.

**Validación:** detener el proceso entre trials y reanudarlo. Deben preservarse objetivo, split, presupuesto consumido, plan y mejores modelos. Ninguna acción completada debe ejecutarse otra vez por pérdida del checkpoint. Inyectar una respuesta que declare una tarea completada sin artefacto: el runtime debe rechazarla.

**Riesgo y límite:** un plan puede quedar desactualizado y una especialización excesiva incrementa llamadas. Un solo escritor del estado y delegaciones con objetivos concretos limitan ambos problemas. El plan se puede generar inicialmente desde una plantilla determinista y adaptarse después.

**Prioridad:** P0 para persistencia/plan; P1 para separar los dos especialistas. Vinculación: R2.1, R2.3 y R3.1.

## 4. P3 — Context Management: conservar la información fuera del prompt

### 4.1 Hallazgos de la fuente

El artículo describe tres mecanismos: externalizar respuestas grandes de herramientas, sustituir argumentos antiguos de escritura/edición por referencias y resumir el historial cuando lo anterior no basta (páginas 1–3). Se conserva el historial original para poder recuperar información posteriormente.

La versión descrita externaliza respuestas que exceden 20 000 tokens, con referencia y vista previa; menciona un umbral de 85 % de ventana para otras medidas de compresión. Son parámetros del material fechado en enero de 2026, no valores óptimos demostrados para este proyecto ni promesas sobre versiones futuras.

La Figura de la página 5 muestra disminuciones de tokens por resumen y descarte de contenido redundante con un umbral experimental de 25 %. No demuestra por sí sola mayor éxito en AutoML. Las páginas 4–6 recomiendan forzar compresión en evaluaciones dirigidas, comprobar continuidad del objetivo y recuperar un hecho que ya no aparece en el contexto activo.

### 4.2 Brecha específica del proyecto

`candidates.propose_llm()` incorpora el texto de todos los documentos recuperados. `retrieve_many()` deduplica pero no aplica un presupuesto global de tokens. Con la KB pequeña y una sola propuesta esto puede ser suficiente; el problema aumenta al añadir múltiples diagnósticos, curvas e historial al nuevo orquestador.

Guardar JSON en disco ya existe parcialmente, pero todavía no hay una herramienta controlada para recuperar fragmentos por ID ni una política para decidir qué entra en cada llamada al LLM.

### 4.3 Mejora propuesta P3: contexto por referencias con recuperación comprobada

Implementar un `ContextManager` que construya cada entrada con el objetivo fijado, restricciones, resumen del plan, observaciones recientes pertinentes y referencias de evidencia. Curvas completas, errores extensos y documentos permanecen en artefactos. Una herramienta de lectura devuelve rangos limitados y verifica el rol: un agente de búsqueda no puede recuperar métricas test mediante una ruta indirecta.

Adoptar dos niveles:

1. **MVP:** resúmenes estructurados deterministas de trials, top-k con presupuesto global y referencias recuperables. Evitar enviar todas las épocas o todo el historial.
2. **Si las mediciones lo justifican:** resumen LLM del historial antiguo, con retención de originales y validación de campos obligatorios.

Los campos críticos —objetivo, métrica, split, límites, bloqueo del test— se leen siempre del estado canónico; no se reconstruyen a partir de un resumen libre. La ventana disponible y los tokens reservados para salida deben configurarse para el modelo efectivo. Si no hay contador exacto, declarar la estimación y aplicar un margen conservador.

**Cambios:** `kb.py`, `llm.py`, `events.py`, almacenamiento de artefactos y nuevo `context.py`. Medir tokens de entrada/salida, latencia, cantidad de externalizaciones, lecturas de recuperación y eventos de resumen.

**Validación:** crear una trayectoria larga que incluya una restricción poco frecuente al inicio. Forzar compresión y exigir que el sistema la respete; para un dato no crítico externalizado, exigir recuperación desde su referencia. Comprobar objetivo, siguiente tarea y presupuesto después del resumen. Comparar consumo con y sin gestión de contexto manteniendo la tarea y el proveedor constantes.

**Riesgo y límite:** los resúmenes pueden omitir restricciones; los archivos también pueden volverse inaccesibles o ser modificados. Usar versiones/hashes y comprobaciones de existencia. No adoptar umbrales del artículo sin medir.

**Prioridad:** P0 para referencias y contexto acotado; P2 para resumen LLM avanzado. Vinculación: R1.1, R2.3 y R4.2.

## 5. P4 — Tutorial de Deep Agents: convertir módulos en herramientas de dominio

### 5.1 Hallazgos de la fuente

Las secciones “Step 2: Create a Web Search Tool”, “Customization 3: Custom Tools” y “Best Practices” muestran cómo exponer capacidades con nombres, parámetros y descripciones que orienten al modelo. “Step 4” y “The Architecture Behind Deep Agents” ilustran el uso de un runtime para combinar planificación, llamadas a herramientas y artefactos.

El tutorial facilita una prueba de integración, pero sus ejemplos se centran en investigación web. La presencia de una herramienta de planificación no garantiza que se use correctamente en cada invocación; tampoco una llamada a `create_deep_agent` prueba persistencia, calidad o ahorro. El texto afirma que el prompt personalizado se añade al incorporado, mientras P2 lo describe como fundamento del comportamiento: esa diferencia es una razón para verificar la versión real en lugar de inferir contratos del texto divulgativo.

### 5.2 Brecha específica del proyecto

Las operaciones existen como métodos y funciones Python, pero el LLM solo dispone de llamadas de generación de texto/JSON. No hay un registro de acciones con precondiciones, efectos, costo y formato uniforme de resultado. Una migración directa que exponga todo `run_pipeline()` como una sola herramienta mantendría el flujo fijo y ocultaría las decisiones intermedias.

### 5.3 Mejora propuesta P4: registro de herramientas con contratos y pruebas de integración

Envolver las funciones existentes en herramientas pequeñas, por ejemplo `retrieve_evidence`, `propose_candidates`, `run_search_batch`, `evaluate_validation` y `package_and_verify`. Definir para cada una esquema de entrada, esquema de resultado, permisos por fase, errores recuperables, límites y clave de idempotencia.

Un resultado uniforme contendría `status`, `artifact_refs`, `metrics`, `resource_usage`, `warnings` y `error`. El LLM selecciona una acción tipada y el runtime la ejecuta. Se puede empezar con JSON validado usando el cliente actual, sin exigir function calling nativo ni una migración de proveedor.

**Cambios:** crear `tools.py` y `schemas.py`; adaptar `llm.py` a una interfaz de políticas que reporte uso del proveedor cuando esté disponible; extraer operaciones actualmente ligadas a atributos mutables de `Pipeline`.

**Validación:** pruebas de contratos con herramientas falsas; rechazo de arquitectura inexistente, parámetros fuera de rejilla, rutas externas, test antes del cierre y llamadas duplicadas. Después, una ejecución pequeña sin red y una integración explícita con LLM real. Las pruebas simuladas verifican control de flujo, no capacidad del modelo real.

**Riesgo y límite:** un framework con herramientas genéricas de shell o escritura libre ampliaría innecesariamente el alcance. Mantener herramientas de dominio y una única capa dueña de la persistencia. Si se evalúa Deep Agents posteriormente, usar exactamente estos contratos para comparar implementaciones.

**Prioridad:** P0, habilitador de las otras propuestas. Vinculación: R2.1–R2.3, R3.3.

## 6. Síntesis: una mejora principal por documento

| Fuente | Mejora principal | Entregable verificable | Dependencia |
|---|---|---|---|
| P1 ReAct | Decidir el siguiente paso usando observaciones reales | Ciclo acotado con decisiones/evidencias y recuperación de fallos | Herramientas P4 y estado P2 |
| P2 Introducción a Deep Agents | Plan persistente y especialistas con contexto separado | Plan versionado, checkpoints y reanudación | Contratos P4 |
| P3 Gestión de contexto | Memoria externa recuperable y prompt acotado | Referencias, presupuesto de tokens y pruebas de recuperación | Registro de artefactos P2 |
| P4 Tutorial | Herramientas de AutoML con contratos explícitos | Registro de herramientas y pruebas por fase | Refactor mínimo del flujo actual |

Orden de implementación recomendado: **P4 → persistencia de P2 → P1 → especialistas de P2 → P3 avanzada**. Las referencias y límites básicos de P3 deben existir desde el inicio; no requieren una fase avanzada de resumen.

## 7. Hallazgos del repositorio que condicionan las propuestas

Estos hallazgos provienen de la lectura del código y de la tesis, no de los resultados publicados en los papers.

| Brecha | Evidencia | Tratamiento propuesto |
|---|---|---|
| Meta RAG incompatible con etiquetas actuales | Tesis R1.1: precisión@5 ≥ 0,70; `reference_queries.json`: 10 consultas, 2–3 relevantes cada una | Con esas etiquetas, el techo de la media es 23/50 = **0,46**. Reanotar exhaustivamente relevancia; si no hay suficientes relevantes, revisar el indicador con el asesor. No inventar relevancia para cumplirlo |
| Rondas insuficientes para el indicador actual | Tesis R2.3: ≥3 ciclos; `RunConfig.rounds=2` y presets rápido/estándar con 2 | Definir qué es un ciclo y un protocolo de evaluación con presupuesto suficiente; discutir la tensión con parada anticipada |
| Confusión entre arquitecturas y configuraciones | `Candidate` contiene listas; `space_size()` cuenta su producto cartesiano | Informar nº de arquitecturas y nº de configuraciones. Tres candidatas no equivalen a tres entrenamientos posibles |
| Cobertura de evaluación parcial | Se evalúa en test el mejor checkpoint por candidata, no cada trial completo | Producir métricas de validación por trial completado y definir la cohorte final de test antes de abrirlo |
| Ahorro no medido frente a baseline real | `benchmark.py` extrapola costos | Ejecutar una rejilla pequeña y fija, además de comparar el controlador actual y el propuesto |
| Registros insuficientes para diagnosticar | Solo se devuelve máximo de `val_acc`, épocas y poda | Guardar curvas, errores, tiempo total y estado de cada intento |
| Calidad del catálogo como restricción | `params_m` es un valor estático usado para filtrar y desempatar | Verificar metadatos con el modelo real; medir memoria/latencia en el hardware objetivo |

La reducción del espacio debe distinguir: espacio global (hoy 990 combinaciones), espacio factible por hardware y espacio final de la propuesta. Si la exclusión por recursos ya reduce el universo, atribuir toda la diferencia al RAG sobreestimaría su contribución.

## 8. Relación con Claude Code y fuentes complementarias

La arquitectura propuesta toma de la documentación pública de Claude Code el ciclo de contexto/acción/verificación y la delegación con contexto separado. La adaptación a presupuesto GPU, protocolo train/val/test y métricas de AutoML es elaboración de este análisis. Las capacidades públicas no prueban qué algoritmos privados usa Claude Code ni requieren replicarlo completo.

Fuentes primarias consultadas para contrastar patrones y APIs, 19-10-2026:

- [Claude Code: funcionamiento del ciclo y contexto](https://code.claude.com/docs/en/how-claude-code-works).
- [Claude Code: subagentes y permisos](https://code.claude.com/docs/en/sub-agents).
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents).
- [LangGraph: persistencia y diferencia entre checkpointer/store](https://docs.langchain.com/oss/python/langgraph/persistence).
- [Deep Agents: capacidades documentadas](https://docs.langchain.com/oss/python/deepagents/overview).
- [Deep Agents: backends](https://docs.langchain.com/oss/python/deepagents/backends).

Las páginas de documentación evolucionan; los PDF locales representan versiones fechadas. Las propuestas anteriores se apoyan en patrones, y los nombres concretos de clases y opciones deberán verificarse al fijar dependencias. No hay evidencia en estos cuatro documentos de que la arquitectura propuesta alcance por sí sola las metas numéricas de la tesis.
