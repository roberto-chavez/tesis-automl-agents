# Especialistas de configuración y diagnóstico (etapa 7)

`specialists/configuration.py` y `specialists/diagnosis.py` reciben exclusivamente
contratos JSON de `specialists/contracts.py`. No reciben el motor de entrenamiento,
el almacén operativo ni herramientas para coordinar otras tareas. La delegación
tiene un nivel y cada invocación realiza como máximo una llamada LLM, sin reintentos.
El entrenamiento continúa secuencial y bajo control del orquestador.

## Entradas y resultados

Configuración recibe `ConfigurationContext`: perfil con ámbito obligatorio `train`,
restricciones, catálogo filtrado por recursos y evidencia recuperada. Reutiliza los
generadores de `candidates.py` y devuelve `ConfigurationResult` con candidatas,
descartes, modo y error recuperable cuando corresponde. El host comprueba cantidad,
IDs/arquitecturas únicos, versión inicial, resolución, rejilla y fuentes antes de
publicarlas. La función pública anterior `candidates.propose()` conserva compatibilidad.
El pipeline utiliza el especialista a través de `propose_candidates`.

Diagnóstico recibe `DiagnosisContext` para un intento: candidata/versiones, curvas
de entrenamiento y validación, accuracy/F1 macro, matriz de confusión, error y
mediciones de recursos. Se proyectan explícitamente las claves admitidas, también
en métricas, curvas, validación y recursos. No se incluyen test, checkpoints,
manifiestos, rutas ni reportes finales.

`DiagnosisResult.findings` contiene observaciones calculadas por código: estado,
error, métricas, recursos y brecha final de accuracy. El LLM sólo puede añadir
`hypotheses`, con incertidumbre explícita, y una `recommendation`. Sobreajuste y
capacidad insuficiente son hipótesis; no se aceptan como hechos del proveedor.
Las reglas señalan una posible situación de sobreajuste cuando la brecha final
train/val supera 0.15 y permiten continuar aun sin proveedor.

La recomendación contiene `action`, `reason` y, para ajustes/descartes, la referencia
`expected`. Un ajuste lleva una candidata nueva: conserva ID, arquitectura y
resolución, incrementa exactamente una versión y utiliza la rejilla existente.
No crea ni ejecuta un trial. Ante formato inválido, fuentes inventadas, versión
incorrecta o parámetros fuera de catálogo, el especialista devuelve diagnóstico
por reglas. El host vuelve a validar el contrato y las observaciones antes de
guardarlo.

## Coordinación y permisos

El orquestador solicita diagnóstico por fallo, poda o brecha de generalización;
las reglas no vuelven a diagnosticar un intento ya observado. Una recomendación
`review` deja la siguiente decisión a la política; `continue` permite el flujo
normal. Para `adjust` y `discard`, el orquestador comprueba otra vez la versión
actual y crea una acción mediante las herramientas existentes. Consejos obsoletos
o manipulados se ignoran. Las paradas globales tienen prioridad sobre cualquier
recomendación. Los diagnósticos básicos de ejecuciones anteriores siguen siendo
legibles y no se interpretan como recomendaciones estructuradas.

| Rol | Herramientas permitidas |
|---|---|
| configuration | `query_evidence`, `read_artifact` de evidencia |
| diagnosis | `query_evidence`, `read_artifact` de evidencia/validación, `evaluate_validation` |
| orchestrator | Herramientas de búsqueda, publicación, entrenamiento y delegación |

Los especialistas actuales trabajan con el contexto preparado por el host; no
ejecutan bucles de herramientas. Los permisos anteriores también se aplican si
el host solicita una lectura por rol. Ambos roles carecen de acceso a test.

## Límites y recuperación

| Configuración / CLI | Predeterminado | Alcance |
|---|---:|---|
| `max_configuration_llm_calls` / `--max-configuration-llm-calls` | 1 | Llamadas de configuración por ejecución |
| `max_diagnosis_llm_calls` / `--max-diagnosis-llm-calls` | 3 | Llamadas de diagnóstico por ejecución |
| `max_llm_calls` / `--max-llm-calls` | 20 | Total compartido con el orquestador |

Un límite por especialista de cero lo mantiene en reglas. Agotar su límite local
también activa reglas y conserva el presupuesto restante para otras funciones.
Los límites globales de decisiones, tiempo y llamadas siguen bloqueando nuevas
herramientas; cerrar búsqueda siempre es posible. Los especialistas utilizan el
mismo timeout del proveedor y no reintentan tras error o respuesta inválida.

SQLite conserva `specialist_limits`, `specialist_usage` y `llm_calls` con rol,
duración, error del proveedor y uso disponible. Los fallos/timeout consumen su
llamada. Los límites locales guardados se mantienen al reconstruir el cliente.
Cada herramienta guarda su contexto y resultado en `specialist:<action_id>`;
repetir una acción confirmada devuelve su resultado sin otra delegación o cargo.
Un fallo de formato queda en el error del resultado del especialista.

## Pruebas

`tests/test_specialists.py` verifica contextos, contratos, hechos/hipótesis,
permisos, parámetros/fuentes/versiones inválidas, respaldo sin red, presupuesto
compartido, límites locales y recuperación. `tests/test_orchestrator.py` prueba
el recorrido entrenamiento → diagnóstico → revisión validada → entrenamiento,
y el rechazo de consejos obsoletos/manipulados. El smoke del pipeline comprueba
también la propuesta por reglas con entrenamiento real sin pesos descargados.

La suite local se ejecuta con `python -m pytest --ignore=tests/test_gemini.py -q`.
Ese archivo es un script manual con una petición remota al importarse; no es una
prueba local aislada. La prueba remota optativa del orquestador sigue desactivada
salvo que se configure `AUTOML_TEST_LIVE_LLM=1`.
