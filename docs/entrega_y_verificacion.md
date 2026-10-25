# Evaluación y entrega (etapa 9)

Al cerrar la búsqueda, la aplicación elige entre intentos completados con checkpoint
existente, usando accuracy de validación y, en empate, el tamaño del catálogo. Guarda
la referencia completa (candidata, versión, estudio, trial e intento), la configuración
exacta de la candidata, la política de métricas y SHA-256 del checkpoint. El cierre
es irreversible. Test nunca participa en la selección ni se agrega a los registros
de entrenamiento o al contexto de los especialistas.

La evaluación comprueba el checkpoint antes de abrir test y publica
`final_evaluation.json` atómicamente. SQLite confirma ese resultado: si el proceso
cae entre publicación y confirmación, la reanudación recupera el archivo sin repetir
la evaluación. Si cae antes de publicar, la operación puede repetirse, siempre sobre
la misma selección. Una modificación del checkpoint o ausencia de su candidata
versionada produce una entrega fallida con resultados parciales conservados.

## Reportes desde persistencia

```powershell
automl-agents report <run_id> --runs-dir runs
```

Este comando reconstruye los reportes desde SQLite, incluso si los datos originales
ya no están disponibles. No inicializa TensorFlow, el motor de entrenamiento o un
cliente LLM. Las cifras y el resumen se insertan con plantillas deterministas.

`reports/07_entrega.json` contiene selección, versiones de candidatas, decisiones,
intentos con errores y recursos, motivo de parada, evaluación final, estado de entrega
y verificaciones. Los CSV incluyen versión e IDs de cada intento. Los tiempos reales
por etapa se acumulan al ejecutar nodos, sin sumar otra vez los nodos recuperados.
Las líneas base exhaustivas siguen identificadas como estimaciones; fallos y
cancelaciones no se atribuyen al ahorro por poda/early stopping.

## Preprocesamiento y pruebas del paquete

Entrenamiento y API utilizan el mismo `preprocessing.py`, copiado al paquete:
decodificación TensorFlow con tres canales, primer frame en animaciones, resize
bilineal sin antialias, float32 en rango 0–255 y `preprocess_input` del backbone.
La aumentación pertenece al entrenamiento y se desactiva durante inferencia.

El paquete conserva la selección en `metadata.json` y registra hashes en
`publication.json`. Antes de publicar una aprobación, un subproceso con timeout:

- Carga el modelo y comprueba forma de entrada y número de clases.
- Compara el preprocesamiento exportado con una referencia rectangular no uniforme.
- Prueba `/health`, `/labels` y `/predict`, comparando etiquetas, orden y confianza
  con inferencia directa del checkpoint.
- Comprueba que una imagen inválida devuelve HTTP 400.

`artifact/smoke_result.json` contiene cada comprobación. Docker se detecta
automáticamente; si está disponible se construye una imagen y se repiten las pruebas
HTTP dentro del contenedor. El plazo cubre detección, build y ejecución; la limpieza
de contenedor e imagen tiene un plazo adicional acotado. Se pueden repetir las pruebas:

```powershell
automl-agents verify-artifact <run_id> --runs-dir runs --timeout 300
automl-agents verify-artifact <run_id> --runs-dir runs --skip-docker
```

`--timeout` controla Docker; los valores iniciales de una ejecución se configuran
con `--artifact-smoke-timeout-seconds` (120) y `--docker-timeout-seconds` (300).
La operación actualiza SQLite, el resumen y los reportes con el resultado nuevo.

## Estados

| Verificación | Significado |
|---|---|
| `passed` (aprobada) | Todas las comprobaciones locales y Docker pasaron. |
| `failed` (fallida) | Alguna comprobación falló, hubo timeout o cambió un archivo publicado. |
| `not_run` (no ejecutada) | Falta Docker/daemon o se omitió; no equivale a aprobación. |

`verified` sólo es `true` cuando el resultado agregado es `passed`. Un fallo tiene
prioridad sobre una comprobación no ejecutada. Una entrega puede estar completada
y tener verificación `not_run` si sus pruebas locales pasaron pero falta Docker.

| Entrega | Significado |
|---|---|
| `completed` | Se entregó el paquete y el ganador alcanzó el umbral de validación. |
| `target_not_reached` | Hay modelo, métricas y paquete, pero no se alcanzó el objetivo de validación. |
| `no_model` | No hubo un modelo elegible; se conservan historial, recursos y reportes. |
| `artifact_failed` | Falló la evaluación, construcción, integridad o verificación del artefacto. |

La exactitud de test se reporta después del cierre y no cambia el estado de objetivo
alcanzado. Un paquete con pruebas fallidas se conserva para diagnóstico y no se
anuncia como verificado. Cancelación e interrupción mantienen sus estados recuperables.

## Pruebas

`tests/test_final_delivery.py` cubre selección congelada, cambios del checkpoint,
recuperación, equivalencia del preprocesamiento para varios backbones y formatos,
una API exportada real, reportes sin dataset, estados de entrega, agregación de
verificaciones y Docker simulado con timeout/limpieza. `tests/test_pipeline_e2e.py`
ejecuta entrenamiento real sin pesos descargados y comprueba que SQLite, reportes,
metadatos y paquete refieren al mismo checkpoint.

Para comprobar también el servidor Uvicorn con HTTP real desde Windows:

```powershell
python scripts/verify_docker_http.py runs/<run_id>/artifact runs/docker-http-check --timeout 1800
```

El script publica un puerto temporal exclusivamente en `127.0.0.1`, prueba
`/health`, `/labels`, `/predict` e imagen inválida, guarda las versiones del runtime
y elimina su propio contenedor e imagen. La primera construcción puede descargar
la imagen base y las dependencias. Esta comprobación complementa el smoke interno
que ejecuta `verify-artifact` y guarda su evidencia en un directorio separado.

Las pruebas locales de pipeline y API deshabilitan expresamente la detección de
Docker para mantener su ejecución sin red incluso cuando Docker está instalado.
La integración real con Docker se ejecuta mediante los comandos anteriores.
