# Validación de MNIST

## Resultado medido

La ejecución `runs/mnist-validation/20261108-235255-e1aa` completó la búsqueda
con `target_reached`. Se utilizó la muestra local `small_mnist` de 1 000 imágenes,
con los mismos manifiestos de `runs/20261102-133428-608f`: 708 imágenes de
entrenamiento, 146 de validación y 146 de test; semilla 42. No es una medición
sobre las 70 000 imágenes del benchmark oficial.

| Métrica | Anterior | SmallCNN |
| --- | ---: | ---: |
| Mejor accuracy de validación | 88.3562 % | 97.2603 % |
| F1 macro de validación del ganador | — | 97.2779 % |
| Accuracy de test | — | 95.21 % |
| F1 macro de test | — | 95.34 % |

El resultado anterior es el máximo de los ocho intentos persistidos de la
ejecución de referencia. La nueva búsqueda completó dos intentos y seleccionó
el primero por validación; el test se abrió sólo después de congelar la selección.
La comparación incluye un mayor presupuesto de épocas, además del cambio de
arquitectura y aumentos: no constituye una ablación de un único factor.

Configuración ganadora: SmallCNN a 28×28, Adam 0.003, dropout 0.2, batch 32 y
aumentos de rotación/traslación/zoom sin reflejos. Hasta 20 épocas por intento;
el ganador terminó en la época 18 por parada temprana, restaurando la época 13.
La validación superó 95 % desde la época 10. No se modificaron las etiquetas,
particiones ni métricas para alcanzar el umbral.

## Reproducción

```powershell
.venv/Scripts/python.exe scripts/verify_mnist.py --dataset runs/20261102-133428-608f/dataset --reference-run runs/20261102-133428-608f
```

El script crea una ejecución independiente y falla si su mejor accuracy de
validación no alcanza 0.95. El resultado puede variar entre entornos; esta muestra
de validación es pequeña y el resultado no garantiza el mismo porcentaje en otros
datos. En la web, las imágenes de baja resolución reciben hasta 20 épocas,
manteniendo el plazo del preset. En CLI se debe indicar `--epochs 20`.

## Evidencia

- `acceptance.json`: accuracy medida, umbral y `passed: true`.
- `data/dataset_manifest.json` y `data/split_manifest.json`: integridad y particiones.
- `attempts/*/result.json`: hiperparámetros, curvas, semillas y validación.
- `final_evaluation.json`: selección congelada y evaluación de test.
- `artifact/verification.json`: carga, preprocesamiento y endpoints de la API
  exportada aprobados. Docker no se ejecutó porque su daemon no estaba disponible.

Los archivos anteriores están dentro de la nueva ejecución. Durante la comprobación
hubo demoras de carga de bibliotecas nativas en Windows; se usaron copias temporales
de dependencias para diagnóstico, sin cambiar las versiones del entorno del proyecto.

Pruebas en el entorno original: 156 aprobadas (97 de candidatas, CNN, especialistas,
catálogo y configuración; 59 de web/API y entrega). Las copias temporales se retiraron.

```powershell
.venv/Scripts/python.exe -m pytest tests/test_candidates.py tests/test_small_cnn.py tests/test_specialists.py tests/test_kb_catalog.py tests/test_config_schemas_cli.py -q
.venv/Scripts/python.exe -m pytest tests/test_api.py tests/test_final_delivery.py -q
```
