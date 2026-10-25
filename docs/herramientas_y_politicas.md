# Herramientas y políticas (etapa 5)

`DomainTools` (`src/automl_agents/tools.py`) adapta los servicios existentes y sirve al
[orquestador adaptativo de la etapa 6](orquestador_adaptativo.md). `REGISTRY` describe
entrada y salida Pydantic, fases, roles, efectos y errores recuperables. Cada
contrato publica `model_json_schema()`; `call()` devuelve `ToolResult`, con un
`DomainError` cuando se rechaza una operación.

| Operación | Fases | Roles | Efecto |
|---|---|---|---|
| `query_evidence` | profile, search | todos | Consultar KB y persistir evidencia |
| `propose_candidates` | search | orchestrator | Delegar configuración y publicar candidatas validadas |
| `revise_candidate` | search | orchestrator | Publicar la siguiente versión de una candidata |
| `diagnose_trial` | search | orchestrator | Delegar diagnóstico y guardar hechos, hipótesis y recomendación |
| `discard_candidate` | search | orchestrator | Descartar una candidata con versión esperada |
| `train_block` | search | orchestrator | Entrenar secuencialmente dentro de los límites globales |
| `evaluate_validation` | search | orchestrator, diagnosis | Recuperar la evaluación de validación ya realizada para un intento |
| `read_artifact` | profile, search, evaluation | todos, según ACL | Leer un fragmento acotado de una proyección segura |
| `close_search` | search, evaluation | orchestrator | Cerrar permanentemente la búsqueda y habilitar evaluación |

El host construye `DomainTools(cfg, store, kb=..., profile=..., engine=...,
llm=...)` y asigna el rol. No debe tomar el rol de un argumento generado por el
agente. Ejemplo de llamada:

```python
result = tools.call(
    "train_block",
    {"candidates": [{"candidate_id": "c1", "candidate_version": 1}],
     "max_attempts": 2},
    role="orchestrator", action_id="block-1",
)
```

Desde la [etapa 7](especialistas.md), sólo el orquestador publica cambios y solicita
delegaciones. Configuración tiene permisos para consultar/leer evidencia;
diagnóstico puede además consultar validación y sus artefactos. Ningún especialista
puede entrenar, revisar, descartar, cerrar búsqueda o invocar otro especialista.
Las ACL impiden que configuración lea artefactos de validación, incluso si fueron
publicados antes de este cambio con permisos más amplios.

Las candidatas y fuentes del adaptador se guardan en `tool_candidates` y
`tool_evidence`. Se validan catálogo, recursos, rejillas, resolución configurada,
fuentes recuperadas y versión esperada antes de cualquier cambio. Una revisión
conserva el ID e incrementa exactamente una versión. Para empezar una ejecución,
se consulta evidencia y se proponen candidatas; para adaptar una ejecución ya
existente, el host puede inicializar estos registros con su contexto validado.

Los artefactos accesibles son proyecciones JSON de evidencia, curvas y validación,
registradas en SQLite con ID opaco, ámbito y ACL. Las entradas no aceptan rutas.
No hay herramientas para registrar archivos arbitrarios, abrir checkpoints,
manifiestos, reportes mixtos o test. `SearchEngine.base_ds("test", ...)` también
exige cierre persistido. Las respuestas de entrenamiento omiten las rutas de los
checkpoints. La lectura permite hasta 8000 caracteres y revalida ACL incluso al
recuperar una respuesta confirmada.

`action_id` identifica una invocación. Repetir la misma llamada confirmada devuelve
su resultado sin gastar presupuesto; reutilizar el ID con otro rol, herramienta
o entrada devuelve `idempotency_conflict`. El bloqueo por ejecución evita dos
herramientas simultáneas. Los bloques usan además el journal por intento de la
etapa 4 para reanudar trabajo parcial. Una revisión cuya publicación fue realizada
antes de una caída, pero cuya respuesta aún no se confirmó, debe reconciliarse
leyendo la versión actual; una versión antigua se rechaza.

## Presupuesto compartido

`BudgetManager` reserva usando `BEGIN IMMEDIATE` antes de llamar a entrenamiento
o al proveedor. Los contadores están en SQLite y se confirman al terminar, también
si el servicio falla. Una reserva abandonada conserva su cargo: cerrar un proceso
no devuelve cupo. La migración desde la etapa 4 conserva intentos y reintentos
históricos. Los límites guardados no se amplían al reconstruir el gestor.

Todos los roles comparten intentos, decisiones, llamadas LLM y reintentos. Cada
invocación nueva de herramienta consume una decisión; el orquestador reemplaza
al controlador por rondas. Entrenar y llamar al LLM reservan además sus contadores
específicos. Una decisión admitida puede ejecutar su trabajo aunque sea la última
disponible. En modo sin LLM, un máximo cero desactiva las llamadas y permite buscar
por reglas. Al agotar cualquier otro límite global no se admite nuevo trabajo de
búsqueda. Cerrar siempre es posible, incluso sin presupuesto.

`max_search_seconds` mide exclusivamente búsqueda mediante reloj de pared desde
la primera reserva. Incluye tiempo de caída/pausa y no vuelve a empezar al
reanudar. Un intento activo comprueba el plazo al finalizar cada época; una época
larga puede superar el límite. También se comprueba antes de crear nuevos trials.
El cierre congela el tiempo consumido y registra `closing_deadline` con el margen
separado `closing_reserve_seconds` (CLI: `--closing-reserve-seconds`, 60 por defecto).
Este margen planifica tiempo adicional para evaluación y entrega: no es un timeout
duro de operaciones de cierre, que actualmente no son interrumpibles.

Un OOM puede usar el siguiente intento disponible del bloque para **un solo**
reintento del mismo trial, con el batch inmediatamente menor entre los valores
permitidos. Mantiene semilla y demás parámetros. Requiere cupo de intentos y
reintentos, conserva el fallo original y no amplía el tamaño del bloque. Sin un
batch menor, no se reintenta. Los reintentos de interrupciones también consumen
el límite global; una segunda recuperación OOM del mismo trial está prohibida.

## Verificación

`tests/test_tools_policies.py` verifica contratos, rechazo sin efectos, ACL,
versiones, fuentes, idempotencia, persistencia, agotamiento, reloj y reservas
concurrentes. `tests/test_search_policies.py` conecta las políticas a entrenamiento
simulado, al journal real, a Optuna y al cliente LLM: las llamadas sin cupo no
alcanzan los servicios, y el reintento OOM consume recursos globales.
