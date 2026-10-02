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
FastAPI + página web sin build · Anthropic API (opcional) · psutil/nvidia-smi.
> La tesis menciona React/Node para la interfaz; en esta primera versión la UI es una página HTML
> servida por FastAPI para no exigir toolchain de Node. Cambiar a React es un reemplazo aislado (`web/`).

## Instalación
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                # núcleo (TensorFlow CPU)
pip install -e ".[embeddings]"                         # opcional: embeddings multilingües para el RAG
cp .env.example .env                                   # opcional: ANTHROPIC_API_KEY
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
Formato del dataset: `mi_dataset/<clase>/*.jpg|png` (mín. 6 imágenes por clase, 2+ clases).
Cada ejecución crea `runs/<id>/` con `reports/`, `artifact/` (API lista), `models/` y `events.jsonl`.

## Modos de operación
| Componente | Con `ANTHROPIC_API_KEY` | Sin clave |
|---|---|---|
| Generador de candidatas (R1.2) | LLM razona sobre RAG; salida **validada** contra catálogo/rejilla | Reglas deterministas |
| Resumen del reporte final (R3.2) | LLM redacta en lenguaje sencillo | Plantilla |
| Embeddings del RAG | `sentence-transformers` si está instalado | Hash léxico (offline) |

Toda ejecución funciona sin LLM; si el LLM falla, se cae a reglas y se deja constancia en el reporte.

## Pruebas
```bash
pytest -q        # 17 pruebas; la de extremo a extremo usa pesos aleatorios (sin red) y tarda ~40 s en CPU
```

## Estado y límites de esta versión (leer antes de usar)
- **Verificado en entorno de desarrollo:** pipeline completo, API de carga, artefacto FastAPI (smoke test) y pruebas.
- **No verificado aún:** entrenamiento con pesos ImageNet (requiere descarga desde internet), ruta con LLM real
  (probada solo con un LLM simulado), `docker build` del artefacto, ejecución en GPU GTX 1650.
- Las líneas base de R4.3 son **estimaciones** (config × épocas × costo medio por época medido), no corridas reales.
  Para la tesis conviene medir una línea base real (p. ej. búsqueda exhaustiva en un dataset pequeño).
- El KB inicial tiene 23 documentos redactados a mano (11 arquitecturas + 12 guías); ampliarlo con más campos
  por modelo es una tarea pendiente del asesor. Con embeddings hash la exhaustividad@3 es ~0.6; con
  `sentence-transformers` debería mejorar (medirlo con `kb-eval`).
- El modo `--no-pretrained` es solo para pruebas: con pocos pasos por época, las estadísticas de BatchNorm de una red
  entrenada desde cero no convergen y la exactitud de validación se queda en azar (el entrenamiento sí aprende). No use
  sus métricas para juzgar el sistema; la ruta real usa backbone preentrenado y congelado.
- Sin *fine-tuning* del backbone (solo cabeza), sin recomendación de métricas por el agente, sin autenticación
  en la API web y una sola ejecución concurrente. La heurística de riesgo/datos personales es una señal, no asesoría legal.
