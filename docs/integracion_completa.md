# Integración completa (etapa 11)

Verificación del 21 de octubre de 2026, en Windows con Python 3.10 y TensorFlow
CPU. Las salidas de esta sesión están en `runs/stage11/` (directorio ignorado por
Git). `runs/stage11/verification.json` resume los cinco recorridos y comprueba
automáticamente la identidad del checkpoint en sus entregas. Se usaron únicamente
imágenes sintéticas para los recorridos descritos.

**Actualización del 24 de octubre:** Docker ya está instalado y su comprobación
real aprobó. Los resultados originales del día 21 se conservan como historial;
el cierre de la comprobación Docker se detalla al final de este documento.

## Correcciones

- **Reanudación CLI de una cancelación:** antes conservaba `cancel.requested` y
  volvía a cancelar. `resume_pipeline` solicita ahora explícitamente su limpieza
  dentro de los bloqueos de entrenamiento y ejecución. Una llamada que encuentra
  otro proceso activo no puede borrar su cancelación. La web conserva su
  reanudación transaccional existente.
- **Verificación CLI de un job web:** el nuevo resultado de `verify-artifact`
  actualiza también `jobs.sqlite`. Un artefacto que falla deja de aparecer como
  entrega completada y una verificación posterior aprobada restaura el estado
  correspondiente. Se conservan selección, evaluación y reportes.
- **Errores operativos CLI:** datos inválidos, errores de archivos y bloqueos de
  ejecución se muestran con mensajes legibles y salida 2. Los errores inesperados
  siguen disponibles para diagnóstico.
- El README identifica ahora el proveedor efectivo (Google Gemini) y las claves
  admitidas por el cliente, e incorpora los resultados externos comprobados.

## Recorridos ejecutados

| Comprobación | Resultado y evidencia |
|---|---|
| CLI por reglas, sin LLM ni pesos | Dos intentos reales, evaluación, reportes y API exportada. `cli/20261021-204153-8b4c/`; log `cli.log`. |
| CLI agentic y pesos ImageNet | MobileNetV2, un intento real, pesos locales en caché. Gemini `gemini-3.8-flash`: dos respuestas válidas (configuración y orquestador), metadatos de uso, sin respaldo por reglas. `agentic/20261021-204539-ec4a/`; log `agentic.log`. |
| Proveedor real aislado | Prueba optativa aprobada; una llamada real, herramientas y entrenamiento simulado. `live-provider.json` y `live-provider.log`. |
| Caída real y reanudación CLI | Proceso eliminado durante el segundo intento después de confirmar el primero. Se conserva el primero, el segundo queda `interrupted`, se reintenta con el mismo trial, parámetros y semilla y distinto attempt. Cuatro intentos consumidos, un reintento. Repetir `resume` no añade intentos ni consumo. `recovery/verification.json`. |
| Web en Edge headless | Carga y ZIP inválido, seguimiento, cancelación por época, reanudación, resultados, reporte, todas las descargas, desconexión y restauración por URL. Sin errores JavaScript. `browser/verification.json`; job `20261021-204529-05a2` en `runs/web-smoke/jobs/`. |
| Reinicio real de la web | Servidor eliminado con un intento activo y reiniciado sobre el mismo SQLite. Job recuperado como `interrupted`, consumo conservado, reanudación y descargas aprobadas; un reintento dentro del límite global. Servidor aislado en puerto 8766. `web-restart/verification.json`, job `20261021-205358-04d6`. |
| Presentación web | Capturas de las tres vistas y cancelación en 1440/390 px, sin desbordamiento a 320/768 px; foco y errores de formulario. Revisión visual de los resultados móviles. El fallo de empaquetado mostrado en el navegador es una simulación identificada; los estados fallidos reales se verifican en Python. |
| Consistencia de selección | La selección congelada, el ganador del resumen, la evaluación, `metadata.json`, `reports/07_entrega.json` y el SHA-256 de `artifact/model.keras` corresponden al mismo checkpoint. Comprobado automáticamente en el recorrido de recuperación. |
| API exportada | Integridad, carga, preprocesamiento, `/health`, `/labels`, `/predict` e imagen inválida aprobados. Se volvió a ejecutar `verify-artifact` tras la recuperación. |
| Wheel Python | Construido sin descargar dependencias, extraído en un directorio aislado; HTML, CSS y JS servidos correctamente desde esa copia. Una sola variante HTML; metadatos incluyen las dependencias de persistencia y contratos. `dist/`, `installed/`, `wheel.log`. |
| Docker | `not_run`, motivo `docker_not_installed`. `verified=false` en todas las entregas; las pruebas locales aprobadas no sustituyen la comprobación Docker. |

Las entregas terminaron en `target_not_reached`: el flujo completó la entrega,
pero no alcanzó el objetivo de validación de 0.99. Las métricas sobre este pequeño
dataset sintético sólo sirven para comprobar la integración. No se ejecutó GPU.

La primera llamada remota encontró un bloqueo de red del sandbox y continuó por
reglas. Tras autorizar explícitamente el envío del contexto sintético a Gemini,
las pruebas reales anteriores aprobaron. La evidencia aprobada registra llamadas
con uso y sin errores; no se contabiliza el respaldo como éxito del proveedor.

## Pruebas y reproducción

La suite final aprobó **292 pruebas y omitió 1** en 341.53 segundos. La omitida
es la integración remota optativa, ejecutada también de forma independiente:
**1 aprobada** con Gemini real. La suite inicial aprobaba 286 pruebas; se añadieron
regresiones para cancelación persistida en CLI, exclusión mutua y mensajes de
errores operativos. La regresión de verificación de entrega cubre también la
sincronización de estados con los jobs web.

Las pruebas cubren proveedor caído, JSON inválido, timeout, OOM simulado con lote
menor y presupuesto compartido, cancelación, caída antes/después de publicar o
confirmar, reconciliación de jobs al reiniciar, exclusión mutua de workers/CLI,
test fuera de las decisiones, selección congelada, corrupción de checkpoint y
errores de empaquetado. SQLite, Optuna y LangGraph se usan realmente en las
pruebas de recuperación; los fallos se inyectan de forma controlada.

```powershell
# Suite local; la integración externa es optativa.
.venv/Scripts/python.exe -m pytest -q --disable-warnings

# Recuperación real y comandos status/resume/report/verify-artifact.
.venv/Scripts/python.exe scripts/verify_cli_recovery.py runs/integration-recovery

# Reiniciar de verdad un servidor aislado mientras un intento está activo.
.venv/Scripts/python.exe scripts/verify_web_restart.py runs/integration-web-restart --port 8766

# CLI por reglas, sin red.
.venv/Scripts/python.exe -m automl_agents.cli demo-dataset runs/integration-data --per-class 10
$env:AUTOML_EMBEDDING='hash'
.venv/Scripts/python.exe -m automl_agents.cli run --dataset runs/integration-data --runs-dir runs/integration-cli --mode rules --no-llm --no-pretrained --archs MobileNetV3Small --max-candidates 1 --rounds 1 --trials 2 --epochs 1 --img-size 32

# Requiere credenciales, autorización de envío al proveedor y pesos disponibles.
.venv/Scripts/python.exe -m automl_agents.cli run --dataset runs/integration-data --runs-dir runs/integration-agentic --mode agentic --archs MobileNetV2 --max-candidates 1 --rounds 1 --trials 1 --epochs 1 --img-size 32 --max-llm-calls 8 --max-search-seconds 300 --llm-timeout-seconds 60

# Proveedor real con entrenamiento simulado y resultado independiente.
$env:AUTOML_TEST_LIVE_LLM='1'
$env:AUTOML_LIVE_LLM_REPORT='runs/integration-live-provider.json'
.venv/Scripts/python.exe -m pytest tests/test_orchestrator.py::test_live_provider_smoke -q
Remove-Item Env:AUTOML_TEST_LIVE_LLM

# Web: mantener servidor en una terminal y navegador en otra.
.venv/Scripts/python.exe scripts/serve_web_smoke.py
node scripts/verify_web.cjs runs/integration-browser

# Empaquetado Python con el backend ya instalado, sin caché de pip externa.
.venv/Scripts/python.exe -m pip wheel . --no-deps --no-build-isolation --no-cache-dir --wheel-dir runs/integration-dist
```

Playwright es una herramienta de desarrollo para el recorrido del navegador;
no se requiere al instalar o utilizar la aplicación. El script utiliza Edge por
defecto. Docker se vuelve a intentar mediante
`automl-agents verify-artifact <run_id> --runs-dir <directorio>` cuando esté
instalado; `--skip-docker` deja explícitamente su resultado como no ejecutado.

`WEB_SMOKE_ROOT` y `WEB_SMOKE_PORT` permiten aislar el servidor de humo en otro
directorio/puerto. En el script de navegador, `WEB_SMOKE_URL` y
`WEB_SMOKE_DATASET` seleccionan ese servidor y su ZIP sintético. Los valores por
defecto conservan el recorrido documentado en `web_y_jobs.md`.

## Verificación Docker del 24 de octubre de 2026

Docker Desktop 4.96.3, Engine 29.9.2, contexto `desktop-linux` y contenedores
`linux/amd64`. El modelo del recorrido agentic MobileNetV2 superó integridad,
smoke local, construcción de imagen y smoke dentro del contenedor:
`status=passed` y `verified=true`. La primera construcción requirió 423.23
segundos y su smoke 23.62 segundos; se usó `--timeout 1800` para permitir descargar
e instalar la imagen base y las dependencias. Las construcciones siguientes
reutilizan las capas de dependencias.

Se revalidaron los cinco artefactos guardados (CLI por reglas, agentic, recuperación
CLI, recorrido web y reinicio web). Los cinco registran ahora `verified=true` en
SQLite, el resumen y los reportes. Se comprobó que intentos, presupuesto, selección
y checkpoint permanecen iguales. Las dos ejecuciones web muestran el estado
actualizado y ofrecen sus descargas; el ZIP de la API contiene la verificación
aprobada y el mismo modelo seleccionado. No quedaron contenedores ni imágenes
temporales de estas pruebas. El resumen de evidencia está en
`runs/stage11/docker-20261024/verification.json`; la evidencia general de
`runs/stage11/verification.json` también fue actualizada y su versión anterior
se conserva en `docker-20261024/previous-integration-summary.json`.

Además del TestClient interno, `scripts/verify_docker_http.py` ejecutó el comando
Uvicorn publicado en el Dockerfile y peticiones HTTP reales desde Windows.
`/health`, `/labels`, `/predict` e imagen inválida aprobaron. El puerto se publicó
sólo en `127.0.0.1`; el contenedor y la imagen temporales se eliminaron al terminar.
Runtime comprobado: Python 3.11.17, TensorFlow CPU 2.21.0 y Keras 3.15.1.
Evidencia: `runs/stage11/docker-20261024/http-agentic/verification.json`.

La suite se volvió a ejecutar con `AUTOML_TEST_LIVE_LLM=1`: **293 aprobadas y
ninguna omitida** en 846.30 segundos, incluida la integración real con Gemini.
El informe remoto vuelve a registrar `validated=true` y una llamada sin error.
Después de corregir la salida Unicode, las **52 pruebas focalizadas de CLI**
aprobaron, incluida la nueva regresión. Los registros son `all-tests.log`,
`live-provider.json` y `cli-unicode-tests.log` dentro de
`runs/stage11/docker-20261024/`.

Se detectaron y corrigieron dos detalles del recorrido de verificación:

- Los registros Docker incluyen Unicode que no cabe en la codificación cp1252
  predeterminada de algunas salidas redirigidas de Windows. La CLI configura ahora
  stdout como UTF-8. Se añadió una regresión con una salida inicialmente cp1252,
  caracteres de progreso Docker y etiquetas Unicode.
- Los scripts de humo usaban un plazo Docker de dos segundos. Se amplió a 300
  segundos, con esperas de cierre de hasta 600 segundos en sus clientes. Las
  pruebas de pipeline/API que prometen funcionar sin red deshabilitan Docker
  explícitamente; su verificación real se ejecuta por separado.

Un primer intento encontró el archivo de contexto Docker ocupado por otro
proceso. La repetición con el mismo motor aprobó; se conservan ambos registros.
El artefacto ya estaba verificado cuando apareció el error de impresión Unicode;
la corrección de la CLI permite devolver también su resultado sin ese error.

```powershell
.venv/Scripts/python.exe -m automl_agents.cli verify-artifact 20261021-204539-ec4a --runs-dir runs/stage11/agentic --timeout 1800
.venv/Scripts/python.exe scripts/verify_docker_http.py runs/stage11/agentic/20261021-204539-ec4a/artifact runs/docker-http-check --timeout 1800
```
