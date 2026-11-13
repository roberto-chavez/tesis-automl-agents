# AutoML multiagente + RAG para clasificación de imágenes (v0.1)

Primera versión del sistema de la tesis *"Arquitectura multiagente con RAG para automatizar la construcción,
selección, entrenamiento y preparación para el despliegue de modelos de clasificación de imágenes en pymes"*
(PUCP, FCI). Dado un dataset etiquetado (una carpeta por clase), el sistema:

1. **Perfila** el dataset y evalúa una señal de riesgo / datos personales del caso de uso.
2. **Recupera** buenas prácticas de una base de conocimiento vectorial (ChromaDB) — **RAG**.
3. **Acota** el espacio de arquitecturas (Keras Applications) e hiperparámetros a pocas candidatas (LLM o reglas).
4. **Entrena, evalúa e itera** con Optuna (TPE) + poda Hyperband; un controlador decide continuar/ajustar/descartar.
5. **Evalúa** con scikit-learn sobre un conjunto de prueba separado y compara contra líneas base.
6. **Reporta** (criterios, historial, consumo, benchmark, ficha del modelo en lenguaje no técnico).
7. **Empaqueta** una API FastAPI dockerizada lista para desplegar.

Trazabilidad con la tesis (secciones 1.2.3 / 1.3): ver [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

## Stack
Python 3.10+ · LangGraph (orquestación) · ChromaDB (RAG) · TensorFlow/Keras · Optuna · scikit-learn ·
FastAPI + página web sin build · Google Gemini (opcional) · psutil/nvidia-smi.
La web adaptable usa HTML, CSS y JavaScript locales servidos por FastAPI, sin build ni Node.
Ofrece carga, seguimiento y resultados, con jobs SQLite, cancelación, reanudación y descargas
según la entrega. Véase [web y jobs persistentes](docs/web_y_jobs.md).

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                # núcleo (TensorFlow CPU)
pip install -e ".[embeddings]"                         # opcional: embeddings multilingües para el RAG
cp .env.example .env                                   # opcional: GOOGLE_API_KEY o GEMINI_API_KEY
```
GPU (p. ej. GTX 1650): instale `tensorflow[and-cuda]` en lugar de `tensorflow-cpu` según su sistema.

## Uso rápido
```bash
# 1) Dataset de prueba sintético (sin descargas) y ejecución de humo
automl-agents demo-dataset data/demo
automl-agents run --dataset data/demo --no-pretrained --img-size 96 --epochs 3 --trials 2 --no-llm

# 2) Con su dataset real (transferencia de aprendizaje con pesos ImageNet; requiere internet la 1.ª vez)
automl-agents run --dataset data/mi_dataset --use-case "Clasificar residuos reciclables" --tier low

# 3) Plataforma web (http://127.0.0.1:8000)
automl-agents serve

# 4) Evaluar la recuperación del RAG (precisión@k / exhaustividad@k, R1.1)
automl-agents kb-eval --k 3
```
Formato del dataset: `mi_dataset/<clase>/*.jpg|png` (2+ clases). Cada clase debe permitir
train/val/test según las fracciones configuradas: con 15 %/15 % se requieren al menos
7 imágenes únicas por clase. Las copias exactas se mantienen juntas; una copia con
etiquetas diferentes produce un error.
Cada ejecución crea `runs/<id>/` con `reports/`, `artifact/` (API lista), `models/` y `events.jsonl`.

Las imágenes se validan por completo antes de la búsqueda. En `data/dataset_manifest.json`
se guardan clases, etiquetas, rutas relativas, dimensiones y SHA-256 de cada archivo;
`data/split_manifest.json` guarda train/val/test, semilla, fracciones y la huella del primer
manifiesto. Las fracciones se aplican por clase a grupos de archivos idénticos, redondeando
val/test hacia abajo y asignando el resto a entrenamiento. El perfil para RAG y candidatas
se calcula solo sobre entrenamiento.

### MNIST e imágenes pequeñas

Para imágenes con mediana de dimensiones de hasta 32×32, la propuesta reserva una
candidata `SmallCNN` (si las arquitecturas permitidas la incluyen), tanto con reglas
como con LLM. Se entrena desde cero a 28×28, normaliza a [0,1] y conserva información
espacial. No necesita pesos ImageNet. Los aumentos no reflejan horizontalmente las
imágenes; para dígitos se usan rotaciones, traslaciones y zoom moderados.
La carga web permite hasta 20 épocas por intento para estas imágenes, conservando
el límite de tiempo del modo elegido. En CLI, usar `--epochs 20`.

Para verificar el objetivo mínimo de 95 % con hasta 20 épocas por intento:

```bash
python scripts/verify_mnist.py --dataset ruta/a/mnist
# Comparación con exactamente los manifiestos de una ejecución anterior:
python scripts/verify_mnist.py --dataset ruta/a/mnist --reference-run runs/ID_ANTERIOR
```

El script crea una ejecución nueva, conserva los reportes y escribe `acceptance.json`.
Termina con error si la mejor accuracy de validación queda por debajo de 0.95.
El test permanece separado y sólo se evalúa tras cerrar la búsqueda. El umbral es
un criterio de aceptación medido, no una garantía para cualquier muestra o presupuesto.

Al reutilizar el mismo `RunConfig.run_id`, el pipeline y el motor leen las particiones
guardadas y verifican los archivos actuales. Cambios en imágenes, rutas, clases, semilla
o fracciones detienen la operación. Un par de manifiestos incompleto o inválido también
se rechaza; no se generan particiones nuevas sobre esa ejecución. Esto conserva los datos
y splits durante la reanudación.

## Persistencia y reanudación (etapa 4)

```bash
automl-agents run --dataset data/demo --runs-dir runs --no-llm --no-pretrained
automl-agents status <run_id> --runs-dir runs
automl-agents resume <run_id> --runs-dir runs
```

`--runs-dir` y `--output` son alias en los tres comandos. Su valor por defecto es
`AUTOML_RUNS_DIR` o `runs`. La configuración conserva rutas absolutas para reanudar
desde otro directorio de trabajo. La reanudación utiliza la configuración original;
los cambios incompatibles en el dataset se rechazan.

Cada ejecución guarda estado, acciones, intentos y consumo en `state.sqlite`, los
checkpoints de LangGraph en `graph.sqlite` (identidad `thread_id = run_id`) y los
estudios de Optuna en `optuna.sqlite`. Las candidatas y documentos se reconstruyen
desde datos JSON; los modelos y datasets no se serializan en el grafo. Los espacios
categóricos incompatibles reciben una versión y un estudio nuevos.

Una acción confirmada devuelve su resultado guardado. Al reanudar se importan los
resultados publicados antes de una caída y se sincronizan los trials de Optuna que
quedaron activos. Los intentos sin resultado se marcan `interrupted`; su reintento
conserva trial, parámetros y semilla, crea otro `attempt_id` y mantiene el consumo
anterior. Se conservan las curvas hasta la última época publicada. Los checkpoints,
JSON y el paquete final se publican mediante reemplazo atómico. Un bloqueo del
sistema operativo impide dos procesos simultáneos sobre la misma ejecución.

La recuperación continúa entre trials; el entrenamiento de un intento interrumpido
comienza desde el inicio. `status` muestra el consumo persistido de intentos, tiempo
medido, decisiones, llamadas LLM y reintentos. Los límites globales se reservan
atómicamente antes del trabajo y se conservan al reanudar. Las ejecuciones anteriores sin este journal no
permiten recuperar retrospectivamente sus acciones de entrenamiento.

Las [herramientas de dominio y políticas](docs/herramientas_y_politicas.md) ofrecen
contratos tipados, permisos por rol, acceso a artefactos por ID y cierre permanente
de búsqueda antes de abrir test. El plazo de búsqueda es cooperativo por época;
`--closing-reserve-seconds` configura un margen adicional para cierre/evaluación.

## Inspeccionar la base RAG en Google Colab

Abre [`notebooks/Explorar_RAG_Colab.ipynb`](notebooks/Explorar_RAG_Colab.ipynb) en Colab
con **Archivo → Subir notebook** y ejecuta las celdas en orden. Incluye una copia de los 23
documentos de `src/automl_agents/knowledge/` y del módulo `kb.py`: permite leerlos, reconstruir
el índice Chroma, ver los vectores, probar consultas y evaluar la recuperación. Usa CPU y no
requiere claves de API ni subir el repositorio. La instalación inicial necesita internet.

El notebook reconstruye un índice independiente; no sincroniza `runs/.chroma/` de tu PC.
Arranca con embeddings hash y permite probar `sentence-transformers` como opción.
Para actualizar la copia incluida después de editar la KB, ejecuta
`python notebooks/_build_rag_inspector.py` y vuelve a subir el notebook generado.

## Modos de operación
| Componente | Con `GOOGLE_API_KEY` o `GEMINI_API_KEY` | Sin clave |
|---|---|---|
| Generador de candidatas (R1.2) | LLM razona sobre RAG; salida **validada** contra catálogo/rejilla | Reglas deterministas |
| Resumen del reporte final (R3.2) | Plantilla con cifras persistidas | Plantilla con cifras persistidas |
| Embeddings del RAG | `sentence-transformers` si está instalado | Hash léxico (offline) |

Toda ejecución funciona sin LLM; si el LLM falla, se cae a reglas y se deja constancia en el reporte.

## Pruebas
```bash
pytest -q        # Incluye entrenamiento y API exportada reales, sin descargar pesos
```

## Estado y límites de esta versión (leer antes de usar)
- **Verificado en entorno de desarrollo:** pipeline completo, API de carga, artefacto FastAPI (smoke test) y pruebas.
- **Integración verificada (etapa 11):** CLI, web, recuperación tras matar el proceso,
  Gemini real y entrenamiento MobileNetV2 con pesos ImageNet en caché. Evidencia y
  comandos en [integración completa](docs/integracion_completa.md).
- **Docker verificado el 24 de octubre de 2026:** construcción, carga del modelo,
  pruebas internas y API Uvicorn mediante HTTP real desde Windows aprobadas.
- **No ejecutado en este entorno:** GPU GTX 1650.
- Las líneas base de R4.3 son **estimaciones** (config × épocas × costo medio por época medido), no corridas reales.
  Para la tesis conviene medir una línea base real (p. ej. búsqueda exhaustiva en un dataset pequeño).
- El KB tiene 24 documentos redactados a mano (12 arquitecturas + 12 guías); ampliarlo con más campos
  por modelo es una tarea pendiente del asesor. Con embeddings hash la exhaustividad@3 es ~0.6; con
  `sentence-transformers` debería mejorar (medirlo con `kb-eval`).
- Para Keras Applications, el modo `--no-pretrained` es solo para pruebas: con pocos pasos por época, las estadísticas de BatchNorm de una red
  entrenada desde cero no convergen y la exactitud de validación se queda en azar (el entrenamiento sí aprende). No use
  sus métricas para juzgar el sistema; la ruta real usa backbone preentrenado y congelado.
- Sin *fine-tuning* del backbone (solo cabeza), sin recomendación de métricas por el agente, sin autenticación
  en la API web y una sola ejecución concurrente. La heurística de riesgo/datos personales es una señal, no asesoría legal.

### Orquestación adaptativa (etapa 6)

La búsqueda usa políticas por reglas o LLM sobre herramientas tipadas, con plan
persistente, diagnóstico básico, revisiones versionadas, detección de bucles y
selección congelada antes de test. Véase [orquestador adaptativo](docs/orquestador_adaptativo.md)
para los límites configurables, recuperación y pruebas.

### Especialistas (etapa 7)

La propuesta de candidatas y el diagnóstico usan especialistas con contratos,
contextos y permisos separados. El orquestador valida y aplica sus recomendaciones;
los especialistas no entrenan ni delegan. `--max-configuration-llm-calls` (1) y
`--max-diagnosis-llm-calls` (3) limitan sus llamadas dentro del presupuesto global.
Al agotar un límite local o fallar el proveedor se utilizan reglas. Véase
[especialistas](docs/especialistas.md) para contratos, persistencia y pruebas.

### RAG dinámico y contexto (etapa 8)

La búsqueda puede recuperar evidencia y leer fragmentos por referencia; el host
consulta guías específicas durante el diagnóstico. Los límites RAG de documentos
y tokens estimados se comparten entre roles y se conservan al reanudar. El LLM
recibe resúmenes deterministas de curvas e historial, restricciones persistentes
y referencias con permisos, con presupuesto de entrada y reserva de salida por
modelo. Véase [RAG y gestión de contexto](docs/rag_y_contexto.md) para configuración,
metadatos, límites, permisos y trazabilidad.

### Evaluación y entrega (etapa 9)

El cierre congela versión, checkpoint, política y huella SHA-256 antes de abrir test.
La evaluación final se publica de forma recuperable; los reportes se reconstruyen
desde SQLite con `automl-agents report <run_id> --runs-dir runs`. El paquete comparte
el preprocesamiento del entrenamiento y ejecuta automáticamente carga del modelo y
pruebas de `/health`, `/labels`, `/predict` e imagen inválida. Docker también se
comprueba con timeout y se puede repetir mediante `verify-artifact <run_id>`.

`verification.json` distingue `passed`, `failed` y `not_run`; sólo `passed` permite
`verified=true`. La entrega distingue `completed`, `target_not_reached`, `no_model`
y `artifact_failed`, conservando reportes y resultados parciales. Véase
[evaluación y verificación](docs/entrega_y_verificacion.md).

### Integración completa (etapa 11)

`automl-agents resume` revoca la solicitud de cancelación anterior bajo los
bloqueos de ejecución y entrenamiento. Reanudar una entrega confirmada conserva
los intentos y el presupuesto. `verify-artifact` actualiza también el estado de
un job creado desde la web. Los errores de datos, almacenamiento y bloqueos se
muestran como errores legibles en CLI.

Para repetir la recuperación real, ejecutar
`python scripts/verify_cli_recovery.py runs/integration-recovery`. El script mata
únicamente el proceso de prueba y sus hijos, recupera un intento interrumpido,
verifica los comandos CLI y comprueba la identidad del checkpoint en modelo,
metadatos y reportes. Véase [verificación de la etapa 11](docs/integracion_completa.md).
