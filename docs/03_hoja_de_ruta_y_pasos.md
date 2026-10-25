# Hoja de ruta de implementación de la aplicación

Esta hoja de ruta contiene tareas de código para evolucionar la aplicación actual hacia un sistema con orquestador, especialistas, herramientas de ML y ejecución recuperable. Cada etapa está preparada para trabajarse en un chat separado y termina con una funcionalidad verificable.

Referencia de diseño: [arquitectura del orquestador](01_arquitectura_orquestador_y_flujo.md). Los nombres de módulos nuevos, comandos y endpoints que aparecen aquí son propuestas que se crearán durante la implementación.

## Cómo trabajar las etapas en otros chats

Ejecutar las etapas en orden. Al comenzar cada chat, revisar el código existente: una etapa anterior puede haber cambiado nombres o resuelto parte de las tareas. Reutilizar lo implementado y completar solamente lo pendiente.

Cada etapa incluye un mensaje que se puede copiar al nuevo chat. Su alcance comprende implementar, integrar y probar la funcionalidad indicada. Los registros, manifiestos y reportes mencionados son salidas automáticas de la aplicación.

## Orden y alcance

| Etapa | Funcionalidad al terminar | Depende de |
|---|---|---|
| 1 | Configuración validada y contratos comunes | — |
| 2 | Dataset validado y particiones persistentes | 1 |
| 3 | Entrenamiento por bloques con resultados completos | 1–2 |
| 4 | Ejecución persistente y reanudación por CLI | 1–3 |
| 5 | Herramientas y límites globales de ejecución | 1–4 |
| 6 | Orquestador adaptativo con respaldo por reglas | 1–5 |
| 7 | Especialistas de configuración y diagnóstico | 6 |
| 8 | RAG dinámico y contexto acotado | 6–7 |
| 9 | Selección final y artefacto verificado | 6–8 |
| 10 | API integrada y web rediseñada para carga, seguimiento y resultados | 4–9 |
| 11 | Integración completa de la aplicación | 1–10 |

Reutilizar `dataset.py`, `kb.py`, `candidates.py`, `search.py`, `trainer.py`, `evaluation.py`, `reporting.py` y `packaging.py`. Refactorizar `pipeline.py` y `controller.py` a medida que se incorporen las capacidades nuevas. El entrenamiento seguirá siendo secuencial y el modo sin LLM continuará disponible.

## Etapa 1 — Configuración y contratos comunes

**Objetivo:** que todos los componentes intercambien entradas, decisiones y resultados validados.

**Código:** modificar `config.py`, `cli.py`, `candidates.py` y `pyproject.toml`; crear `schemas.py`.

### Implementar

- Validar fracciones de partición, cantidades positivas, arquitecturas permitidas y niveles de recursos antes de iniciar trabajo.
- Añadir configuración para modo `rules` o `agentic`, métrica primaria, máximo global de intentos de entrenamiento, tiempo de búsqueda, decisiones, llamadas LLM y reintentos.
- Mantener accuracy como métrica primaria inicial. Representar la política de métricas con nombre, dirección, umbral y desempate; conectar otras métricas cuando estén implementadas en todas las capas.
- Crear contratos para estado de ejecución, plan, candidata versionada, decisión, resultado de herramienta, resultado de trial y presupuesto.
- Separar identificadores de candidata, versión, estudio, trial e intento. Mantener referencias inequívocas entre ellos.
- Representar errores con código, mensaje y condición recuperable, en lugar de depender de texto libre.
- Mantener operativos los comandos actuales y `--no-llm`; añadir las opciones nuevas a la CLI.

**Terminado cuando:** una configuración inválida falla antes de entrenar y los contratos pueden serializarse y reconstruirse sin objetos TensorFlow o clientes externos. Las pruebas actuales de configuración/candidatas siguen funcionando.

**Mensaje para otro chat:**

> Implementa la etapa 1 de docs/03_hoja_de_ruta_y_pasos.md. Revisa el código actual, añade configuración validada y contratos comunes, intégralos con la CLI y las candidatas existentes y ejecuta las pruebas pertinentes.

## Etapa 2 — Validación de imágenes y particiones persistentes

**Objetivo:** utilizar las mismas imágenes y particiones durante toda la ejecución y sus reanudaciones.

**Código:** modificar `dataset.py` y su integración con `search.py`; crear utilidades de manifiestos si hacen falta.

### Implementar

- Validar las imágenes que se utilizarán, identificar archivos corruptos y devolver errores comprensibles.
- Comprobar que el número de imágenes por clase permite la partición solicitada; evitar que el mínimo fijo actual permita splits inviables.
- Guardar un manifiesto con rutas relativas, etiquetas, clases y huellas de archivos, y otro con train/val/test y semilla.
- Cargar las particiones desde esos manifiestos al continuar una ejecución.
- Detectar cambios en datos o clases al reanudar y detener la operación si son incompatibles.
- Calcular el perfil que recibe el agente sobre entrenamiento. Reservar la inspección global inicial para integridad y construcción de la partición.
- Detectar duplicados exactos y evitar que una misma imagen aparezca en particiones distintas; informar conflictos de etiqueta.

**Terminado cuando:** reiniciar la aplicación conserva las particiones y una modificación del dataset se detecta. Añadir pruebas de imágenes corruptas, clases pequeñas y duplicados entre splits.

**Mensaje para otro chat:**

> Implementa la etapa 2 de docs/03_hoja_de_ruta_y_pasos.md sobre los contratos existentes. Añade validación de imágenes, manifiestos y particiones persistentes e integra su lectura con el motor de búsqueda. Prueba los casos de error y reanudación.

## Etapa 3 — Motor de entrenamiento por bloques

**Objetivo:** ejecutar un bloque limitado de trials y obtener resultados suficientes para decidir el siguiente paso.

**Código:** modificar `search.py`, `trainer.py`, `evaluation.py`, `resources.py` y `events.py`.

### Implementar

- Separar la ejecución de un trial de la coordinación de rondas. Exponer una operación que reciba candidata/versiones y un máximo de intentos.
- Devolver y guardar inmediatamente el resultado de cada intento: completado, podado, fallido, cancelado o interrumpido.
- Guardar curvas de pérdida y accuracy de entrenamiento/validación por época, parámetros, semilla, recursos y motivo de parada.
- Mantener alineadas la métrica de Optuna, poda, early stopping, checkpoint y selección del mejor modelo.
- Calcular accuracy, F1 macro y matriz de confusión de validación para cada trial completado.
- Guardar checkpoints con identificadores de trial, evitando sobrescribir el modelo de otro intento.
- Liberar recursos en `finally`, capturar OOM y evitar que un fallo recuperable aborte todos los candidatos.
- Añadir cancelación cooperativa al finalizar cada época y comprobarla también entre trials.
- Limitar el caché acumulado de datasets para que no crezca indefinidamente entre candidatas.

**Terminado cuando:** un bloque devuelve resultados estructurados, conserva los intentos fallidos y puede cancelarse. Probar errores con simulaciones y verificar una ejecución pequeña sin pesos descargados.

**Mensaje para otro chat:**

> Implementa la etapa 3 de docs/03_hoja_de_ruta_y_pasos.md. Refactoriza entrenamiento y búsqueda para ejecutar bloques limitados de trials, guardar resultados y curvas por intento, evaluar validación y admitir cancelación y errores recuperables. Reutiliza los contratos y manifiestos ya implementados.

## Etapa 4 — Persistencia y reanudación por CLI

**Objetivo:** recuperar una ejecución entre trials después de cerrar o reiniciar el proceso.

**Código:** crear `state_store.py`; modificar `pipeline.py`, `search.py`, `cli.py` y dependencias.

### Implementar

- Añadir almacenamiento SQLite para estado operativo, acciones, intentos y presupuesto consumido.
- Configurar persistencia de LangGraph con identidad por ejecución y reconstruir los objetos pesados desde referencias.
- Persistir estudios Optuna por candidata y versión del espacio. Crear una nueva versión al cambiar opciones categóricas incompatibles con el estudio anterior.
- Identificar acciones con claves de idempotencia: una acción confirmada devuelve su resultado existente al repetirse.
- Guardar artefactos mediante archivos temporales y publicación atómica; reconciliar resultados guardados antes de un cierre inesperado.
- Marcar los intentos que quedaron activos como interrumpidos. Un reintento comienza un intento nuevo y conserva el consumo anterior.
- Evitar configuraciones repetidas accidentalmente mediante huellas que incluyan parámetros, datos, split y semilla.
- Añadir `automl-agents resume <run_id>` y consulta del estado por CLI, con una opción consistente para localizar el directorio de ejecuciones.

**Terminado cuando:** una ejecución con dos trials completados y un tercero interrumpido puede reanudarse sin repetir los dos primeros. Probar caídas antes y después de confirmar resultados.

**Mensaje para otro chat:**

> Implementa la etapa 4 de docs/03_hoja_de_ruta_y_pasos.md. Añade persistencia SQLite, checkpoints del grafo, estudios Optuna persistentes y reanudación por CLI. Comprueba que las acciones confirmadas no se repiten y que los intentos interrumpidos se reconcilian correctamente.

## Etapa 5 — Herramientas de dominio y presupuesto global

**Objetivo:** ofrecer al orquestador operaciones ejecutables con límites controlados por código.

**Código:** crear `tools.py` y `policies.py`; adaptar los servicios existentes.

### Implementar

- Registrar herramientas para consultar evidencia, proponer/revisar candidatas, entrenar un bloque, evaluar validación, leer fragmentos de artefactos y solicitar cierre de búsqueda.
- Definir entrada, salida, fase permitida, errores recuperables y efectos de cada herramienta.
- Resolver artefactos mediante IDs y permisos por rol, evitando rutas arbitrarias y acceso indirecto a test durante la búsqueda.
- Validar arquitectura, hiperparámetros, fuentes y versión esperada antes de ejecutar cambios.
- Reservar presupuesto antes de lanzar trabajo y confirmar el consumo al terminar. Los especialistas y reintentos comparten los límites globales.
- Bloquear nuevos intentos cuando se agoten tiempo, intentos, llamadas LLM o decisiones.
- Definir claramente el límite de tiempo de búsqueda y reservar tiempo para cierre/evaluación. Mostrar que la cancelación por época tiene esa granularidad.
- Permitir como máximo un reintento inicial de recuperación por OOM con batch menor dentro del espacio permitido.

**Terminado cuando:** una llamada inválida no produce efectos, el presupuesto se conserva al reanudar y ninguna herramienta puede abrir test durante la búsqueda. Añadir pruebas de rechazo, duplicación y agotamiento.

**Mensaje para otro chat:**

> Implementa la etapa 5 de docs/03_hoja_de_ruta_y_pasos.md. Expón las operaciones existentes como herramientas tipadas y añade políticas de fase, permisos y presupuesto global persistente. Prueba que las acciones inválidas o sin presupuesto no se ejecutan.

## Etapa 6 — Orquestador adaptativo

**Objetivo:** decidir el siguiente paso a partir de resultados reales de la aplicación.

**Código:** crear `orchestrator.py`; refactorizar `pipeline.py`, `controller.py` y `llm.py`.

### Implementar

- Crear una interfaz de política con implementación por reglas y por LLM usando el proveedor actual.
- Construir el ciclo observar → decidir → validar → ejecutar herramienta → verificar resultado → persistir.
- Dar al orquestador un plan con tareas, dependencias y estados. Marcar una tarea completa únicamente cuando su resultado exista y supere su comprobación.
- Permitir consultar evidencia, revisar candidatas, continuar entrenando, descartar, solicitar diagnóstico y cerrar búsqueda.
- Permitir ajustes dentro del catálogo/rejilla y crear versiones de candidatas al aplicarlos.
- Incorporar timeout LLM, reintentos limitados y respaldo por reglas ante fallos de proveedor o formato.
- Detectar repetición de acciones sin nueva información y detener bucles sin progreso.
- Implementar paradas por objetivo, presupuesto, estancamiento, cancelación o ausencia de modelo.
- Congelar la selección al cerrar búsqueda; impedir que la búsqueda se reactive después de abrir test.
- Registrar acción, justificación breve, fuentes, resultado y uso LLM disponible por llamada.

**Terminado cuando:** resultados diferentes producen secuencias de herramientas diferentes y el sistema también termina sin LLM. Probar trayectorias con un cliente simulado y una integración pequeña con proveedor real cuando esté disponible.

**Mensaje para otro chat:**

> Implementa la etapa 6 de docs/03_hoja_de_ruta_y_pasos.md. Construye el orquestador adaptativo sobre las herramientas, persistencia y políticas existentes. Incluye plan operativo, respaldo por reglas, prevención de bucles y cierre de búsqueda. Verifica trayectorias distintas según las observaciones.

## Etapa 7 — Especialistas de configuración y diagnóstico

**Objetivo:** delegar decisiones concretas con contexto y herramientas propios.

**Código:** crear módulos de especialistas; adaptar `candidates.py` y `orchestrator.py`.

### Implementar

- Extraer el especialista de configuración a partir del generador actual: recibe perfil de entrenamiento, restricciones, catálogo y evidencia; devuelve candidatas validadas.
- Crear el especialista de diagnóstico: recibe curvas, métricas de validación, errores y recursos; devuelve hallazgos y una recomendación estructurada.
- Diferenciar observaciones comprobadas de hipótesis como sobreajuste o capacidad insuficiente.
- Dar a cada rol herramientas específicas y excluir test de ambos contextos.
- Ejecutar delegaciones de un nivel, con presupuesto compartido y sin entrenamiento paralelo.
- Invocar especialistas solo cuando su tarea sea necesaria. El orquestador aplica sus recomendaciones después de validarlas.
- Conservar el respaldo por reglas de propuestas y diagnóstico básico.

**Terminado cuando:** cada especialista devuelve un resultado útil y validado y una recomendación inválida no cambia el estado. El orquestador sigue siendo el único responsable de coordinar acciones.

**Mensaje para otro chat:**

> Implementa la etapa 7 de docs/03_hoja_de_ruta_y_pasos.md. Añade especialistas de configuración y diagnóstico con contratos, contexto y permisos separados; intégralos al orquestador existente con límites de llamadas y respaldo por reglas.

## Etapa 8 — RAG dinámico y gestión de contexto

**Objetivo:** recuperar conocimiento cuando haga falta y mantener pequeñas las entradas al LLM.

**Código:** modificar `kb.py`, `llm.py` y especialistas; crear `context.py`.

### Implementar

- Permitir consultas RAG durante diagnóstico y ajustes, además de la recuperación inicial.
- Añadir límite global de documentos y tokens, filtros por tipo y deduplicación.
- Guardar consulta, IDs recuperados y versión de la evidencia utilizada.
- Incorporar metadatos opcionales de fuente, fecha y alcance a los documentos de conocimiento, manteniendo compatibilidad con los actuales.
- Mantener documentos extensos, curvas e historial en artefactos; entregar al LLM resúmenes y referencias.
- Implementar lectura acotada por referencia y permisos de rol.
- Construir cada contexto desde objetivo, restricciones, plan, presupuesto y observaciones pertinentes. Obtener campos críticos del estado persistente.
- Configurar un presupuesto de entrada y reserva de salida por modelo. Identificar como estimación el conteo cuando no haya tokenizer disponible.
- Empezar con resúmenes estructurados deterministas del historial; conservar los originales recuperables.

**Terminado cuando:** el orquestador puede recuperar una guía tras un fallo o diagnóstico y consultar un dato externalizado sin enviar todo el historial. Probar límites de tamaño y conservación de restricciones.

**Mensaje para otro chat:**

> Implementa la etapa 8 de docs/03_hoja_de_ruta_y_pasos.md. Habilita RAG durante el ciclo y crea gestión de contexto por resúmenes estructurados y referencias recuperables, con límites globales y permisos por rol.

## Etapa 9 — Evaluación final, reportes de la aplicación y empaquetado

**Objetivo:** entregar un modelo seleccionado y una API cuyo funcionamiento haya sido comprobado.

**Código:** modificar `evaluation.py`, `reporting.py`, `packaging.py`, `benchmark.py` y el cierre del pipeline.

### Implementar

- Seleccionar el ganador con la política fijada y persistir su checkpoint y versión antes de evaluar test.
- Ejecutar test después del cierre; impedir que sus resultados vuelvan al contexto de búsqueda.
- Generar métricas finales y tablas desde los resultados persistidos.
- Adaptar los reportes automáticos a candidatas versionadas, decisiones, errores, recursos y motivos de parada.
- Insertar cifras mediante plantillas; limitar el LLM a texto explicativo basado en hechos.
- Mantener las estimaciones de ahorro identificadas como estimaciones y añadir tiempos reales de las etapas de la ejecución.
- Compartir o verificar equivalencia del resize/preprocesamiento entre entrenamiento y API generada.
- Ejecutar automáticamente pruebas de carga del modelo y endpoints `/health`, `/labels` y `/predict`, incluyendo imagen inválida.
- Añadir una operación de verificación Docker con timeout y resultado explícito: aprobado, fallido o no ejecutado si Docker no está disponible.
- Distinguir entrega completada, objetivo no alcanzado, ausencia de modelo y fallo de artefacto; conservar salidas útiles en los casos parciales.

**Terminado cuando:** la aplicación produce un artefacto con resultados de verificación y no lo marca verificado si sus pruebas fallan o no se ejecutaron. Un test de integración comprueba clases y predicciones de la API generada.

**Mensaje para otro chat:**

> Implementa la etapa 9 de docs/03_hoja_de_ruta_y_pasos.md. Integra selección congelada, evaluación final, reportes desde estado persistente y empaquetado con pruebas automáticas. Revisa la equivalencia del preprocesamiento y distingue verificación aprobada, fallida y no ejecutada.

## Etapa 10 — API y rediseño de la web de la aplicación

**Objetivo:** crear una versión mejorada y coherente de la web que permita a una persona sin experiencia en ML completar el proceso de carga, seguimiento y obtención de resultados.

**Código:** modificar `api.py`, el frontend en `web/` y el servicio de ejecución persistente. Crear los componentes, estilos y scripts necesarios para el rediseño, aprovechando FastAPI y el frontend existente; una migración a React no es un requisito de esta etapa.

### Implementar

- Rediseñar la navegación, distribución visual y componentes con una jerarquía clara, estilos consistentes y textos comprensibles para usuarios sin conocimientos de ML.
- Organizar el recorrido en tres vistas o secciones: inicio/carga, seguimiento de ejecución y resultados. Mantener visible la ejecución actual y permitir volver a consultar sus resultados.
- En inicio/carga, permitir seleccionar el dataset, describir el objetivo y elegir límites de ejecución mediante presets comprensibles. Mostrar requisitos del dataset, validaciones y errores junto al campo correspondiente.
- En seguimiento, presentar la etapa actual, el plan, el tiempo consumido y los límites de ejecución, las mejores métricas de validación disponibles y un resumen de decisiones relevantes. Mostrar progreso medido sin inventar porcentajes cuando no pueda calcularse un avance fiable.
- En resultados, destacar el modelo elegido, métricas con explicaciones breves, limitaciones, motivo de finalización y descargas disponibles. Diferenciar calidad alcanzada, entrega parcial y estado de verificación del artefacto.
- Diseñar estados de carga, espera, ejecución, cancelación solicitada, interrupción, error y finalización, con una acción siguiente clara cuando corresponda. Evitar controles activos para operaciones incompatibles con el estado actual.
- Adaptar la interfaz a escritorio y móvil; incluir etiquetas de formulario, navegación por teclado, foco visible y mensajes de estado accesibles.
- Conectar la carga y los presets con la configuración y el orquestador nuevos.
- Reemplazar la cola exclusivamente en memoria por jobs recuperables con un único entrenamiento activo.
- Al iniciar el servidor, reconciliar jobs pendientes/interrumpidos y evitar dos workers para la misma ejecución.
- Extender `GET /api/runs/{run_id}` con plan, tarea activa, progreso, presupuesto, mejores métricas de validación y causa de parada.
- Añadir `POST /api/runs/{run_id}/cancel` y `POST /api/runs/{run_id}/resume`, con validación de estados y solicitudes persistidas.
- Integrar botones de cancelar/reanudar y actualizar el seguimiento sin recargar toda la página. Manejar fallos de conexión sin perder la referencia a la ejecución.
- Conectar descarga de modelo, reportes y artefacto con los estados de entrega.
- Consolidar las variantes HTML actuales en una única web activa. Ajustar la ruta servida y los recursos incluidos en el paquete para que la interfaz rediseñada funcione también al instalar la aplicación.

**Terminado cuando:** la web rediseñada permite cargar, ejecutar, seguir el progreso, cancelar, reanudar y descargar resultados con un recorrido coherente; reiniciar el servidor no pierde los jobs. Probar endpoints y recorrer las tres vistas con un dataset pequeño, incluyendo errores y entrega parcial. Comprobar la presentación en escritorio y móvil, la navegación por teclado y la correspondencia entre los estados mostrados y los datos reales de la API.

**Mensaje para otro chat:**

> Implementa la etapa 10 de docs/03_hoja_de_ruta_y_pasos.md. Rediseña la web para usuarios sin experiencia en ML, con vistas de inicio/carga, seguimiento y resultados, diseño adaptable y estados claros. Consolida las variantes actuales en una única interfaz activa e intégrala con FastAPI, jobs persistentes, progreso real, cancelación, reanudación y descargas según estado. Aprovecha el stack existente y mantén un solo entrenamiento activo. Verifica el recorrido completo, los errores y la presentación en escritorio y móvil.

## Etapa 11 — Integración completa y corrección de fallos

**Estado:** implementada y verificada el 21 de octubre de 2026. Resultados,
correcciones, límites del entorno y comandos reproducibles en
[integración completa](integracion_completa.md). Gemini real y pesos ImageNet
locales fueron ejecutados. La comprobación Docker pendiente se ejecutó y aprobó
el 24 de octubre de 2026, incluida la API Uvicorn mediante HTTP real.

**Objetivo:** dejar funcionando el recorrido completo en los dos modos de ejecución.

**Código:** ajustes de integración en los módulos anteriores y pruebas en `tests/`.

### Implementar y comprobar

- Ejecutar el recorrido completo con el dataset sintético, sin LLM ni descargas, y corregir fallos.
- Ejecutar un recorrido pequeño con LLM real cuando haya credenciales y otro con pesos preentrenados cuando estén disponibles.
- Probar caída del proveedor, respuesta inválida, OOM simulado, cancelación y reinicio durante búsqueda.
- Comprobar que los intentos confirmados no se repiten, el presupuesto no se reinicia y test permanece fuera de las decisiones.
- Comprobar que reportes, modelo ganador, metadatos y paquete referencian el mismo checkpoint.
- Verificar el recorrido por CLI y por web y corregir diferencias de comportamiento.
- Ejecutar las pruebas pertinentes de la aplicación y las del artefacto generado; ejecutar la verificación Docker en el entorno disponible.

**Terminado cuando:** el recorrido completo funciona y los fallos previstos terminan en estados recuperables o errores comprensibles. Cada capacidad dependiente de credenciales, pesos o Docker muestra su resultado real de ejecución.

**Mensaje para otro chat:**

> Completa la etapa 11 de docs/03_hoja_de_ruta_y_pasos.md. Revisa la integración de las etapas implementadas, ejecuta los recorridos disponibles por CLI/web, prueba recuperación y empaquetado y corrige los fallos encontrados hasta dejar funcional el flujo completo.

## Primera tarea para comenzar

Iniciar con la **etapa 1** en un chat nuevo. Cada chat debe implementar una etapa sobre el estado actual del repositorio y dejar sus pruebas pertinentes funcionando antes de pasar a la siguiente.
