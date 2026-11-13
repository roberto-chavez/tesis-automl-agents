# SmallCNN para imágenes de baja resolución

CNN local entrenada desde cero, sin descarga de pesos. Para imágenes de hasta
32x32, como dígitos MNIST, evita reescalar a 224x224 y depender de características
ImageNet congeladas. Dos bloques Conv2D/ReLU/MaxPooling (32 y 64 filtros), Flatten,
Dense de 128 unidades y dropout. Aproximadamente 0.42 millones de parámetros a
28x28. Entrada RGB normalizada a [0,1]. Nivel de recursos: bajo.

Usar Adam con learning rate 0.001 o 0.003 y batch 32. Los aumentos son rotaciones,
traslaciones y zoom moderados; no reflejar dígitos. Para muestras pequeñas permitir
20 épocas y seleccionar el checkpoint por validación. Medir el resultado con
particiones fijas y mantener test separado. No garantiza un umbral de accuracy.
