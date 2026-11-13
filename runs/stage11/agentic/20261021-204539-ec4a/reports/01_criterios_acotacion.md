# Criterios de acotación del espacio de búsqueda (R1.3)

Generador: **llm**. 

- Dataset: 24 imágenes, 3 clases, razón de desbalance 1.0:1.
- Espacio total considerado: **990** configuraciones; candidato: **24** (reducción **97.6 %**).

## Evidencia recuperada (RAG)
| Documento | Tipo | Similitud |
|---|---|---|
| Datasets pequeños: pocas imágenes por clase | guide | 0.33 |
| Transferencia de aprendizaje con pesos ImageNet | guide | 0.27 |
| EfficientNetB0 (Keras Applications) | arch | 0.27 |
| Rangos de tasa de aprendizaje | guide | 0.41 |
| Dropout y regularización | guide | 0.15 |
| Tamaño de lote | guide | 0.46 |
| DenseNet121 (Keras Applications) | arch | 0.27 |
| Xception (Keras Applications) | arch | 0.20 |
| Entrenamiento con recursos computacionales limitados | guide | 0.37 |
| VGG16 (Keras Applications) | arch | 0.24 |
| EfficientNetV2B0 (Keras Applications) | arch | 0.22 |
| MobileNetV2 (Keras Applications) | arch | 0.23 |
| Descarte temprano y parada anticipada | guide | 0.18 |
| ConvNeXtTiny (Keras Applications) | arch | 0.21 |
| Métricas de evaluación en clasificación de imágenes | guide | 0.22 |
| Desbalance de clases | guide | 0.09 |

## Candidatas
| ID | Versión | Arquitectura | Justificación | Fuentes | lr | dropout | batch | aug |
|---|---|---|---|---|---|---|---|---|
| c1 | 1 | MobileNetV2 | MobileNetV2 es la única arquitectura permitida por las restricciones y se ajusta al nivel de recursos bajo con solo 3.5M de parámetros. Al disponer de un dataset sumamente pequeño (24 imágenes en total, 8 por clase), la evidencia indica utilizar transferencia de aprendizaje con backbone congelado, aumento de datos y regularización por dropout (0.2 a 0.4) para evitar sobreajuste. El tamaño de lote se restringe a 16 y 32 según las recomendaciones para recursos limitados, y el learning rate se enfo | arch_mobilenetv2, guide_small_dataset, guide_transfer_learning, guide_learning_rate, guide_dropout_regularization, guide_batch_size, guide_limited_resources | [0.0003, 0.001, 0.003] | [0.2, 0.4] | [16, 32] | [True, False] |

## Descartadas
| Arquitectura | Motivo |
|---|---|
| MobileNetV3Small | Excluida por el usuario (lista de arquitecturas permitidas). |
| EfficientNetB0 | Excluida por el usuario (lista de arquitecturas permitidas). |
| EfficientNetV2B0 | Excluida por el usuario (lista de arquitecturas permitidas). |
| NASNetMobile | Excluida por el usuario (lista de arquitecturas permitidas). |
| DenseNet121 | Excluida por el usuario (lista de arquitecturas permitidas). |
| ResNet50V2 | Excluida por el usuario (lista de arquitecturas permitidas). |
| Xception | Excluida por el usuario (lista de arquitecturas permitidas). |
| InceptionV3 | Excluida por el usuario (lista de arquitecturas permitidas). |
| ConvNeXtTiny | Excluida por el usuario (lista de arquitecturas permitidas). |
| VGG16 | Excluida por el usuario (lista de arquitecturas permitidas). |