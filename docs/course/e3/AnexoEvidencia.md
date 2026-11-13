# Anexo C. Evidencias y trazabilidad de la evaluación

## C.1 Registro de evidencias

Las rutas de la tabla son relativas a la raíz del repositorio. El archivo `.gitignore` permite incluir en Git únicamente los artefactos citados aquí que se consideran necesarios para revisar la evidencia. El resto del contenido de `runs/` —por ejemplo, modelos, bases de datos, conjuntos de imágenes, archivos temporales y registros de otras ejecuciones— permanece excluido. Por ello, la disponibilidad de esos artefactos depende de que se hayan incluido en el repositorio o se entreguen por separado.

La tabla delimita el alcance de cada evidencia y distingue entre resultados de MNIST, pruebas históricas de integración y la evaluación de la base de conocimiento.

**Tabla C1**  
*Evidencias consultadas y alcance*

| ID | Archivo o documento | Qué sustenta |
|---|---|---|
| V1 | `runs/mnist-validation/20261108-235255-e1aa/acceptance.json` | Exactitud de validación de 0,9726, umbral de 0,95 y decisión de aceptación. |
| V2 | Misma ejecución: `final_evaluation.json` | Evaluación de prueba, matriz de confusión, clases y selección congelada. |
| V3 | Misma ejecución: `attempts/8423a0feb53f4f2c8ea7b07e362d765a/result.json` | Parámetros, semilla, curvas, 18 épocas y F1 macro de validación. |
| V4 | Misma ejecución: `reports/01_criterios_acotacion.json` | Modo basado en reglas, restricción a SmallCNN y espacio de búsqueda de 1 080/8. |
| V5 | Misma ejecución: `reports/05_benchmark.json` | Dos intentos, 28 épocas y líneas base identificadas como estimaciones. |
| V6 | Misma ejecución: `artifact/verification.json` | Resultado de las pruebas locales y constancia de que Docker no se ejecutó. |
| V7 | `runs/stage11/recovery/verification.json` | Interrupción y reintento reales, consumo conservado y coherencia de la selección. |
| V8 | `runs/stage11/docker-20261024/http-agentic/verification.json` | Construcción Docker y comprobación de la API HTTP de MobileNetV2. |
| V9 | `runs/stage11/docker-20261024/all-tests.log` | Resultado histórico de 293 pruebas aprobadas. |
| V10 | `docs/integracion_completa.md` y los reportes `01_criterios_acotacion`, `03_evaluacion` y `05_benchmark` de `runs/stage11/agentic/20261021-204539-ec4a/`; también `artifact/verification.json` | Recorrido con Gemini y MobileNetV2 usando datos sintéticos, con sus límites declarados. Solo se incluyen los archivos de evidencia enumerados; el resto de esa ejecución sigue ignorado. |
| V11 | `src/automl_agents/knowledge/reference_queries.json` y salida de `kb-eval` del 10/11/2026 | Diez consultas y 23 asociaciones relevantes; precisión@3 de 0,467 y exhaustividad@3 de 0,617 con el backend hash. |
| V12 | `docs/mnist_validacion.md` | Contexto del experimento MNIST y diferencias frente a la ejecución anterior. |

La huella SHA-256 registrada para el modelo ganador de MNIST es `99f106e7181c0a367ffda34bb6a866fab32553871109220dea72e29fe419b9fb`. El archivo del modelo no se incluye en Git; la huella permite identificarlo y contrastar la selección, la evaluación y el artefacto empaquetado cuando se dispone de ellos. El identificador completo del ensayo es `d4534790c3974c4b9270ab919dbe8876`.

## C.2 Comandos de reproducción

Los comandos y las notas sobre requisitos y límites de reproducción se encuentran en el archivo [`ComandosReproduccion.md`](../../../ComandosReproduccion.md), en la raíz del repositorio. Deben ejecutarse desde esa raíz con el entorno instalado. La repetición de los procedimientos no garantiza cifras idénticas en otros entornos; la muestra y los modelos descargables pueden no estar disponibles en un clon nuevo.
